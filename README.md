# GPT-2 Based Question Answering Model

A Transformer-based Question Answering (QA) model built using a GPT-2 architecture.  
The model is trained from scratch using PyTorch with a custom Byte Pair Encoding (BPE) tokenizer and adapted for Gujarati-English language understanding.

The project implements the core components of GPT-2, including:

- Token embeddings
- Positional embeddings
- Multi-head causal self-attention
- Transformer blocks
- Layer normalization
- Feed-forward networks
- Autoregressive text generation
- Temperature and top-k sampling

The model is designed to answer questions by learning contextual relationships between questions and answers from training data.

---

## Project Overview

Large Language Models (LLMs) such as GPT-2 use the Transformer decoder architecture to predict the next token in a sequence.

This project implements a GPT-2 style architecture from scratch and trains it for a Question Answering task.
