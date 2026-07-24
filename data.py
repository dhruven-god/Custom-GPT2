from datasets import load_dataset


dataset = load_dataset("oscar-corpus/OSCAR-2201",
                        language="gu", 
                        streaming=True, # optional
                        split="train") # optional, but the dataset only has a train split

for d in dataset:
    print(d) # prints documents
