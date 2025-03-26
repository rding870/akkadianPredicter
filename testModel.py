import pandas as pd
import torch
import ast
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import Dataset

# ✅ Load and prepare test data
df = pd.read_csv("mlm_token_data.csv").dropna().tail(10)  # last 10 rows for testing
df["input_ids"] = df["input_ids"].apply(ast.literal_eval)
df["labels"] = df["labels"].apply(ast.literal_eval)
df["attention_mask"] = df["input_ids"].apply(lambda ids: [1 if token != 1 else 0 for token in ids])  # 1 = <pad>

dataset = Dataset.from_pandas(df)

# ✅ Load tokenizer and fine-tuned model
model_path = "./cuneiform_finetuned_model"
model = AutoModelForMaskedLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")  # tokenizer wasn't fine-tuned
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ✅ Test MLM predictions
for sample in dataset:
    input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long).to(device)
    attention_mask = torch.tensor([sample["attention_mask"]], dtype=torch.long).to(device)
    labels = sample["labels"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().tolist()

    # ✅ Decode inputs and outputs
    decoded_input = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
    decoded_prediction = tokenizer.decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.decode([t if t != -100 else tokenizer.pad_token_id for t in labels], skip_special_tokens=True)

    print("\n🔹 Input (with mask):", decoded_input)
    print("✅ Ground Truth:", decoded_labels)
    print("📝 Model Prediction:", decoded_prediction)

print("\n✅ MLM Testing Complete!")
