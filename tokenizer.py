from transformers import GPT2Tokenizer

# Load the GPT-2 tokenizer
tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
