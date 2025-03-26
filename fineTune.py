import pandas as pd
import ast
import torch
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer
)

print("🚀 Starting...")

# ✅ Load tokenized dataset from CSV
df = pd.read_csv("mlm_token_data.csv")
df = df.iloc[:10000]  # Optional: use smaller subset for debugging
print("✅ Loaded dataframe")

# ✅ Convert string representation of lists into actual lists
df["input_ids"] = df["input_ids"].apply(ast.literal_eval)
df["labels"] = df["labels"].apply(ast.literal_eval)

# ✅ Compute attention_mask from input_ids (1 for real token, 0 for padding)
df["attention_mask"] = df["input_ids"].apply(lambda ids: [1 if token != 1 else 0 for token in ids])  # 1 = <pad> for XLM-R

# ✅ Keep only required columns
df = df[["input_ids", "labels", "attention_mask"]]
print("✅ Parsed and prepared dataframe")

# ✅ Convert to Hugging Face dataset
dataset = Dataset.from_pandas(df)
print("✅ Converted to Hugging Face dataset")

# ✅ Split into train and test sets
dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
print("✅ Train/Test split complete")
print("📄 Sample Data:", dataset["train"][0])

# ✅ Load the pre-trained XLM-RoBERTa model for MLM
model_name = "xlm-roberta-base"
model = AutoModelForMaskedLM.from_pretrained(model_name)

# ✅ Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("✅ Model loaded and moved to device:", device)

# ✅ Define Training Arguments
training_args = TrainingArguments(
    output_dir="./cuneiform_finetuned",
    evaluation_strategy="epoch",
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
    fp16=torch.cuda.is_available()
)

# ✅ Define correct `collate_fn` for MLM
def collate_fn(batch):
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    attention_masks = [torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch]
    labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]

    # Pad sequences to max length in batch
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=1)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_masks.to(device),
        "labels": labels.to(device)
    }

# ✅ Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=collate_fn
)

# ✅ Start Training
print("🏋️‍♀️ Training starting...")
trainer.train()

# ✅ Save the Fine-Tuned Model
trainer.save_model("./cuneiform_finetuned_model")
print("✅ Training complete! Model saved to './cuneiform_finetuned_model'.")
