import json
import random
from datasets import load_dataset

# Load dataset from Hugging Face
ds = load_dataset("ImruQays/Rasaif-Classical-Arabic-English-Parallel-texts")

# Convert only English → Arabic translations
formatted_data = []

# Process each split (train, test, validation if available)
for split in ds.keys():
    for row in ds[split]:  # Iterate over dataset rows
        formatted_data.append({
            "instruction": "Translate to Arabic:",
            "input": row["en"],  # Assuming 'en' column exists
            "output": row["ar"]   # Assuming 'ar' column exists
        })

# Shuffle data to ensure randomness
random.shuffle(formatted_data)

# Split 80% for training, 20% for testing
split_ratio = 0.8
split_index = int(len(formatted_data) * split_ratio)

train_data = formatted_data[:split_index]
test_data = formatted_data[split_index:]

# Save train and test sets separately
with open("data/train_english_to_arabic.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open("data/test_english_to_arabic.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print(f"✅ Conversion complete! Train set: {len(train_data)} samples, Test set: {len(test_data)} samples.")
