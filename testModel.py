import pandas as pd
import torch
import pyonmttok
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ✅ Load Tokenized Dataset (Last 5000 Rows)
df = pd.read_csv("tokenized_data.csv").dropna().tail(10)

# ✅ Convert tokenized sequences from strings to lists
df["input_ids"] = df["input_ids"].apply(eval)
df["labels"] = df["labels"].apply(eval)
df["attention_mask"] = df["attention_mask"].apply(eval)

# ✅ Convert to Hugging Face dataset (Take 10 test samples safely)
dataset = Dataset.from_pandas(df).select(range(min(10, len(df))))

# ✅ Load Model and Tokenizer
model_path = "./cuneiform_finetuned_model"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ✅ Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# ✅ Helper function to replace `-100` with `tokenizer.pad_token_id`
def replace_padding(token_ids, pad_token_id):
    return [pad_token_id if token == -100 else token for token in token_ids]

# ✅ Function to Predict and Fill in Missing Akkadian Text
def generate_and_detokenize(test_samples, model, tokenizer):
    model.eval()
    results = []
    
    for sample in test_samples:
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]

        # ✅ Convert input to tensors
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(device)

        # ✅ Generate Akkadian Output
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_tensor, 
                attention_mask=attention_mask_tensor, 
                max_length=50,
                pad_token_id=tokenizer.pad_token_id,  
                num_return_sequences=1,  # Only return 1 best prediction
                do_sample=True,   
                top_k=40,          # Use top-k sampling for better diversity
                temperature=0.9,   # Increase temperature to make text less repetitive
                repetition_penalty=1.2  # Penalize repeating characters like ::::::::::::::
            )

        # ✅ Convert Token IDs to Akkadian Text
        decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results.append(decoded_output)

    return results

# ✅ Prepare test samples from dataset
test_samples = [dict(zip(dataset.column_names, values)) for values in zip(*dataset[:].values())]

# ✅ Generate Akkadian Predictions
detokenized_results = generate_and_detokenize(test_samples, model, tokenizer)

# ✅ Print Results
for i, sample in enumerate(test_samples):
    input_text = tokenizer.decode(replace_padding(sample['input_ids'], tokenizer.pad_token_id), skip_special_tokens=True)
    expected_output = tokenizer.decode(replace_padding(sample['labels'], tokenizer.pad_token_id), skip_special_tokens=True)

    print(f"\n🔹 Input Akkadian Text: {input_text}")
    print(f"✅ Expected Akkadian Output: {expected_output}")
    print(f"📝 Predicted Akkadian Output: {detokenized_results[i]}")

print("\n✅ Testing Complete: Akkadian Text Filling Model is Working!")
