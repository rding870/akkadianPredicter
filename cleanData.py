import pandas as pd
import re
import csv
import random

# Character replacements for normalization
akkadian_replacements = str.maketrans({
    "Ā": "a", "Ḫ": "h", "Ī": "i", "Ř": "r",
    "Š": "sh", "Ṣ": "sh", "Ū": "u",
    "ā": "a", "ḫ": "h", "ī": "i", "ř": "r",
    "š": "sh", "ṣ": "sh", "ū": "u"
})

logogram_pattern = re.compile(r"^[A-Z0-9\.{}]+$")

def format_for_model(token):
    """Formats a token for use in model training."""
    if logogram_pattern.match(token):
        return f"_{token.lower()}_"

    match = re.match(r"^(\d+)(disz)$", token)
    if match:
        number, disz = match.groups()
        return f"{number}({disz})"

    token = re.sub(r"[₀₁₂₃₄₅₆₇₈₉]", lambda m: str(ord(m.group(0)) - ord('₀')), token)

    fractions = {"½": "1/2", "⅓": "1/3", "⅔": "2/3"}
    for frac, replacement in fractions.items():
        token = token.replace(frac, replacement)

    return token.lower().translate(akkadian_replacements)

# ✅ Load data
df = pd.read_excel(
    "sentence_filling_train_cleaned_no_ids.xlsx",
    usecols=["fragment_id", "fragment_line_num", "index_in_line", "value"],
    nrows=200000
)

df["value"] = df["value"].astype(str).apply(format_for_model)
df = df.sort_values(by=["fragment_id", "fragment_line_num", "index_in_line"])
lines_per_fragment_line = df.groupby(["fragment_id", "fragment_line_num"])["value"].apply(list).reset_index()

# ✅ Load vocabulary (for token -> ID mapping)
vocab = {}
with open("vocab.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            token, token_id = parts
            vocab[token] = int(token_id)

# Special token IDs
pad_id = vocab.get("<pad>", 0)
mask_id = vocab.get("[MASK]", 2)
unk_id = vocab.get("<unk>", 1)  # ✅ This was missing in your script

# ✅ Prepare MLM samples
mlm_data = []

for _, row in lines_per_fragment_line.iterrows():
    tokens = row["value"]
    if len(tokens) < 3:
        continue

    token_ids = [vocab.get(t, unk_id) for t in tokens]
    input_ids = token_ids.copy()
    labels = [-100] * len(tokens)

    # Mask 15% of tokens
    num_to_mask = max(1, int(len(tokens) * 0.15))
    mask_indices = random.sample(range(len(tokens)), num_to_mask)

    for idx in mask_indices:
        original_id = token_ids[idx]
        prob = random.random()

        if prob < 0.8:
            input_ids[idx] = mask_id
            labels[idx] = original_id
        elif prob < 0.9:
            input_ids[idx] = random.choice(list(vocab.values()))
            labels[idx] = original_id
        # else: leave input unchanged, and labels[idx] remains -100


    mlm_data.append({
        "input_ids": input_ids,
        "labels": labels
    })

# ✅ Save to CSV
mlm_df = pd.DataFrame(mlm_data)
mlm_df.to_csv("mlm_token_data.csv", index=False)

print(f"✅ Generated {len(mlm_data)} masked MLM samples and saved to 'mlm_token_data.csv'.")
