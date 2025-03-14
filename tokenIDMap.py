import pandas as pd
import pyonmttok

# Load dataset
df = pd.read_csv("masked_token_data_normalized.csv")

# Initialize OpenNMT tokenizer
tokenizer = pyonmttok.Tokenizer("aggressive", joiner_annotate=True)

# Define special tokens
vocab = {"<pad>": 0}  # Predefined special tokens
token_index = 4  # Start numbering normal tokens from 4

# Function to tokenize text and build vocabulary
def build_vocab(text_list):
    """Tokenizes text and updates the vocabulary dictionary with unique token IDs."""
    global token_index
    for text in text_list:
        tokens, _ = tokenizer.tokenize(text)  # Tokenize text
        for token in tokens:
            if token not in vocab:
                vocab[token] = token_index  # Assign unique ID
                token_index += 1

# Build vocabulary from masked sentences and target words
build_vocab(df["masked_sentence"].dropna().astype(str).tolist())
build_vocab(df["target_word"].dropna().astype(str).tolist())

# Save vocabulary to OpenNMT format
with open("vocab.txt", "w", encoding="utf-8") as f:
    for token, token_id in vocab.items():
        f.write(f"{token} {token_id}\n")

print(f"Vocabulary saved with {len(vocab)} tokens.")
