import pandas as pd
import re
import csv

# This attempts to convert to CDLI ATF/clean data
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

# ✅ Load data from Excel
df = pd.read_excel(
    "sentence_filling_train_cleaned_no_ids.xlsx",
    usecols=["fragment_id", "fragment_line_num", "index_in_line", "value"],
    nrows=200000
)

# ✅ Convert values to strings and apply formatting
df["value"] = df["value"].astype(str).apply(format_for_model)

# ✅ Sort by fragment ID, line number, and token index
df = df.sort_values(by=["fragment_id", "fragment_line_num", "index_in_line"])

# ✅ Group by fragment lines to construct sentences
lines_per_fragment_line = df.groupby(["fragment_id", "fragment_line_num"])["value"].apply(list).reset_index()

masked_sentences = []
targets = []

# ✅ Generate masked sentences
for _, row in lines_per_fragment_line.iterrows():
    tokens = row["value"]  # Already formatted by `format_for_model`

    if len(tokens) > 2:
        for i, word in enumerate(tokens):
            masked_line = tokens.copy()
            masked_line[i] = "[MASK]"
            masked_sentence_str = " ".join(masked_line)
            masked_sentences.append(f"Fill in the blank: {masked_sentence_str}")
            targets.append(word)

# ✅ Create shuffled DataFrame
masked_data = pd.DataFrame({"masked_sentence": masked_sentences, "target_word": targets})
masked_data = masked_data.sample(frac=1, random_state=42).reset_index(drop=True)

# ✅ Save as CSV with proper quoting
masked_data.to_csv("masked_token_data_normalized.csv", index=False, quoting=csv.QUOTE_ALL)

print(f"Generated {len(masked_sentences)} masked samples.")
