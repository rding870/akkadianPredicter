import pandas as pd
import ast
import torch
import pyonmttok
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer
)

print("Starting...")

# Initialize OpenNMT Tokenizer
tokenizer = pyonmttok.Tokenizer("aggressive", joiner_annotate=True)
print("Loaded tokenizer")

# ✅ Load tokenized dataset from CSV
df = pd.read_csv("tokenized_data.csv")
df = df.iloc[:10000]  # Limit to small subset for debugging
print("Loaded dataframe")

# ✅ Ensure required columns exist
expected_columns = {"padded_sentence", "padded_target", "attention_mask"}
if not expected_columns.issubset(df.columns):
    raise ValueError(f"Missing columns: {expected_columns - set(df.columns)}")

# ✅ Convert string representation of lists into actual lists
df["input_ids"] = df["padded_sentence"].apply(ast.literal_eval)
df["labels"] = df["padded_target"].apply(ast.literal_eval)
df["attention_mask"] = df["attention_mask"].apply(ast.literal_eval)
df = df[["input_ids", "labels", "attention_mask"]]  # Keep only required columns
print("Parsed lists from dataframe")

# ✅ Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df)
print("Converted to Hugging Face dataset")

# ✅ Split into train and test sets
dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
print("Available columns:", dataset["train"].column_names)
print(f"Sample Data:\n{dataset['train'][0]}")

# ✅ Load the pre-trained model
model_name = "praeclarum/cuneiform"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ✅ Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Define Training Arguments
training_args = TrainingArguments(
    output_dir="./cuneiform_finetuned",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    logging_dir="./logs",
    optim="adamw_torch",
    logging_steps=100,                
    save_total_limit=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
    fp16=True,
    label_smoothing_factor=0.1
)

# ✅ Define Correct `collate_fn`
def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    attention_masks = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]
    labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]

    # ✅ Pad sequences correctly
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)  # Mask padding

    # ✅ T5 needs `decoder_input_ids`, so we shift labels
    decoder_input_ids = labels.clone()
    decoder_input_ids[decoder_input_ids == -100] = 0  # Replace padding with 0

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_masks.to(device),
        "labels": labels.to(device),
        "decoder_input_ids": decoder_input_ids.to(device),  # ✅ FIXED: Add decoder input IDs
    }


# ✅ Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=collate_fn  # ✅ No need for lambda function
)

# ✅ Start Training
trainer.train()

# ✅ Save the Fine-Tuned Model
trainer.save_model("./cuneiform_finetuned_model")
print("Training complete! Model saved to './cuneiform_finetuned_model'.")
