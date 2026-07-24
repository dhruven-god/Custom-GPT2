import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tokenizers import Tokenizer
import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Custom PyTorch Dataset for preparing training samples for GPT.
# 
# GPT training follows a next-token prediction objective:
# Given input tokens:
#     [t1, t2, t3, t4]
# The model learns to predict:
#     [t2, t3, t4, t5]
#
# The dataset creates many overlapping sequences using a sliding window.
class GPTDatasetV1(Dataset):

    def __init__(self, txt, tokenizer, max_length, stride):

        # Store tokenizer for future use
        self.tokenizer = tokenizer

        # Lists to store input sequences and their corresponding targets
        self.input_ids = []
        self.target_ids = []

        # Convert complete text into token IDs.
        #
        # Example:
        # Text:
        # "Hello world"
        #
        # Tokens:
        # [15496, 995]
        #
        # Each integer represents a token in the vocabulary.
        token_ids = tokenizer.encode(str(txt)).ids


        # Create training samples using a sliding window.
        #
        # Example:
        # token_ids:
        # [1,2,3,4,5,6,7]
        #
        # max_length = 4
        # stride = 2
        #
        # Input:
        # [1,2,3,4]
        # Target:
        # [2,3,4,5]
        #
        # Next sample:
        # Input:
        # [3,4,5,6]
        # Target:
        # [4,5,6,7]
        #
        # Overlapping sequences help the model learn contextual relationships.
        for i in range(0, len(token_ids) - max_length, stride):

            # Input sequence given to the transformer
            input_chunk = token_ids[i:i + max_length]

            # Target sequence shifted by one token.
            # The model tries to predict these tokens.
            target_chunk = token_ids[i + 1: i + max_length + 1]


            # Convert Python lists into PyTorch tensors.
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))


    # Returns number of training samples.
    def __len__(self):
        return len(self.input_ids)


    # Returns one training example.
    #
    # Output:
    # input:
    #   Tensor shape -> (sequence_length)
    #
    # target:
    #   Tensor shape -> (sequence_length)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]



# Creates PyTorch DataLoader for GPT training.
#
# DataLoader:
# - Groups samples into batches
# - Shuffles training examples
# - Loads data efficiently
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):


    # Load custom Byte Pair Encoding tokenizer.
    #
    # The tokenizer converts text into integer token IDs.
    tokenizer = Tokenizer.from_file("guj_bpe_tokenizer.json")


    # Create dataset containing input-target token pairs.
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)


    # Create batches for training.
    #
    # Example:
    #
    # batch_size = 4
    #
    # Input shape:
    # (4, 256)
    #
    # Meaning:
    # 4 training examples,
    # each containing 256 tokens.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# Multi-head self-attention layer used inside the Transformer.
#
# Attention allows every token to look at previous tokens and decide
# which tokens are important for understanding the current token.
#
# Example:
#
# "The animal didn't cross the street because it was tired"
#
# The word "it" attends strongly to "animal".
class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):

        super().__init__()

        # Output embedding dimension must be divisible by number of heads.
        #
        # Example:
        #
        # d_out = 768
        # num_heads = 12
        #
        # Each attention head gets:
        #
        # 768 / 12 = 64 dimensions
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads

        # Dimension handled by each attention head.
        #
        # Example:
        # embedding size = 512
        # heads = 8
        #
        # Each head:
        # 512/8 = 64
        self.head_dim = d_out // num_heads

        # Linear layers create:
        #
        # Query:
        # "What information am I looking for?"
        #
        # Key:
        # "What information do I contain?"
        #
        # Value:
        # "What information should be passed forward?"
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Combines outputs from all attention heads.
        self.out_proj = nn.Linear(d_out, d_out)

        # Prevents overfitting during training.
        self.dropout = nn.Dropout(dropout)

        # Creates causal attention mask.
        #
        # GPT is an autoregressive model.
        #
        # A token can only see previous tokens.
        #
        # Example:
        #
        # Token 3 can see:
        # Token 1, Token 2, Token 3
        #
        # But cannot see:
        # Token 4, Token 5
        #
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1
            )
        )

    def forward(self, x):

        # Input shape:
        #
        # batch_size,
        # number_of_tokens,
        # embedding_dimension
        #
        # Example:
        # (4, 256, 768)
        b, num_tokens, d_in = x.shape


        # Generate query, key and value matrices.
        #
        # Shape:
        #
        # (batch, tokens, embedding)
        keys = self.W_key(x)

        queries = self.W_query(x)

        values = self.W_value(x)



        # Split embedding dimension into multiple attention heads.
        #
        # Before:
        # (batch, tokens, 768)
        #
        # After:
        # (batch, tokens, heads, head_dimension)
        keys = keys.view(
            b,
            num_tokens,
            self.num_heads,
            self.head_dim
        )

        values = values.view(
            b,
            num_tokens,
            self.num_heads,
            self.head_dim
        )

        queries = queries.view(
            b,
            num_tokens,
            self.num_heads,
            self.head_dim
        )

        # Rearrange dimensions so attention is calculated independently
        # for every head.
        #
        # Before:
        # (batch,tokens,heads,head_dim)
        #
        # After:
        # (batch,heads,tokens,head_dim)
        keys = keys.transpose(1,2)

        queries = queries.transpose(1,2)

        values = values.transpose(1,2)
        # Compute attention scores.
        #
        # Formula:
        #
        # Attention = QK^T / sqrt(d)
        #
        # Higher score means tokens are more related.
        attn_scores = queries @ keys.transpose(2,3)

        # Convert mask into boolean values.
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens]

        # Replace future token positions with negative infinity.
        #
        # Softmax(-inf) becomes zero.
        #
        # Therefore future tokens receive no attention.
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Convert scores into probabilities.
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5,
            dim=-1
        )

        # Apply dropout to attention probabilities.
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of value vectors.
        #
        # Shape:
        #
        # (batch, heads, tokens, head_dim)
        context_vec = (
            attn_weights @ values
        ).transpose(1,2)

        # Merge attention heads back together.
        #
        # Before:
        # (batch,tokens,heads,head_dim)
        #
        # After:
        # (batch,tokens,embedding)
        context_vec = context_vec.contiguous().view(
            b,
            num_tokens,
            self.d_out
        )
        # Final projection after combining heads.
        context_vec = self.out_proj(context_vec)


        return context_vec

# Layer Normalization stabilizes the activations inside the Transformer.
#
# During training, values flowing through deep neural networks can become
# unstable. LayerNorm normalizes each token embedding independently.
#
# Input:
# (batch_size, sequence_length, embedding_dimension)
#
# Normalization happens over:
# embedding_dimension
#
# Formula:
#
# normalized_x = (x - mean) / sqrt(variance + epsilon)
#
# Then learnable scale and shift parameters are applied.

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
         # Small constant to prevent division by zero.
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

        # Calculate mean for each token embedding.
        #
        # keepdim=True keeps dimensions:
        #
        # Before:
        # (batch, tokens, embedding)
        #
        # After:
        # (batch, tokens, 1)
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

# GELU activation function.
#
# GPT models use GELU instead of ReLU.
#
# GELU smoothly decides how much information should pass through.
#
# ReLU:
# negative values -> 0
#
# GELU:
# negative values are reduced gradually.

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

# Feed Forward Network inside each Transformer block.
#
# The Transformer block contains:
#
# 1. Multi-head attention
# 2. Feed-forward network
#
# The FFN expands the embedding dimension and compresses it back.
#
# Example:
#
# embedding dimension = 768
#
# First Linear:
# 768 -> 3072
#
# Second Linear:
# 3072 -> 768
#
# Expansion allows the model to learn complex transformations.

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Stack multiple Transformer blocks.
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

# Simple greedy text generation.
#
# The model predicts one token at a time.
#
# Example:
#
# Input:
# "The capital of France is"
#
# Model predicts:
# "Paris"
#
# Then the new token is added back into the input:
#
# "The capital of France is Paris"
#
# The process repeats until max_new_tokens is reached.

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

# Helper function used while loading pretrained weights.
#
# PyTorch parameters are stored as nn.Parameter objects.
#
# This function:
# 1. Checks that shapes match.
# 2. Converts numpy arrays into tensors.
# 3. Wraps them as trainable parameters.
#

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

# Load OpenAI GPT-2 weights into our custom GPT implementation.

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


# Convert text into token IDs.
#
# Example:
#
# Text:
# "Hello"
#
# Token IDs:
# [15496]
#

def text_to_token_ids(text, tokenizer):
    # encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded = tokenizer.encode(text).ids

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor

# Convert token IDs back into text.
#
# Example:
#
# [15496,995]
#
# becomes:
#
# "Hello world"
#

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0).tolist()  # Remove batch dimension and convert to list
    try:
        start_idx = flat.index(2)  # Find the first occurrence of ID 2
        trimmed = flat[start_idx:]  # Slice from that point onward
    except ValueError:
        trimmed = flat  # If 2 not found, decode the whole sequence
    return tokenizer.decode(trimmed, skip_special_tokens=False)


# Calculate loss for one batch.
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

# Calculate average loss over DataLoader
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()