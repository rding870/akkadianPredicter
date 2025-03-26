import pandas as pd
import torch
from datasets import Dataset
import pyonmttok

# ------------------ STEP 1: Load Vocabulary ------------------
def load_vocab(vocab_path):
    """Loads an OpenNMT vocabulary file into a dictionary (token → ID)."""
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:  # Ensure valid format (token ID)
                token, token_id = parts
                vocab[token] = int(token_id)
    return vocab

# Load tokenizer and vocab
tokenizer = pyonmttok.Tokenizer("aggressive", joiner_annotate=False)
vocab = load_vocab("vocab.txt")

# Load dataset
df = pd.read_csv("masked_token_data_normalized.csv")
# ------------------ STEP 2: Define Tokenization Function ------------------
def tokenize_function(examples):
    """Tokenizes input text, converts tokens to IDs, and generates attention masks."""
    input_tokens_batch, _ = tokenizer.tokenize_batch(examples["masked_sentence"])
    label_tokens_batch, _ = tokenizer.tokenize_batch(examples["target_word"])

    input_ids_list = []
    label_ids_list = []

    for input_tokens, label_tokens in zip(input_tokens_batch, label_tokens_batch):
        # Convert tokens to IDs and remove None values
        input_ids = [vocab.get(token, 0) for token in input_tokens]  # Default to 0 (padding token)
        label_ids = [vocab.get(token, -100) for token in label_tokens]  # Default to -100 for loss masking

        input_ids_list.append(input_ids)
        label_ids_list.append(label_ids)

    return {"input_ids": input_ids_list, "labels": label_ids_list}


# ------------------ STEP 3: Apply Tokenization & Save ------------------

# Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df)
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Convert to DataFrame
tokenized_df = pd.DataFrame({
    "input_ids": tokenized_datasets["input_ids"],
    "labels": tokenized_datasets["labels"]
})

max_sentence_length = max(tokenized_df["input_ids"].apply(len))
max_target_length = max(tokenized_df["labels"].apply(len))

print("max sentence length:", max_sentence_length)
print("max_target_length:", max_target_length)

def pad_input(tokens, max_length):
    return tokens + [0] * (max_length - len(tokens))

def pad_label(tokens, max_length):
    return tokens + [-100] * (max_length - len(tokens))

tokenized_df["padded_sentence"] = tokenized_df["input_ids"].apply(lambda x: pad_input(x, max_sentence_length))
tokenized_df["padded_target"] = tokenized_df["labels"].apply(lambda x: pad_label(x, max_target_length))
tokenized_df["attention_mask"] = tokenized_df["padded_sentence"].apply(lambda x: [1 if token != 0 else 0 for token in x])
# Save to CSV
tokenized_df.to_csv("tokenized_data.csv", index=False)

print("Tokenized dataset with attention masks saved as tokenized_data.csv ✅")
