import json

# Load JSON file (Replace with actual filename)
with open("train_english_to_arabic.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract all English sentences and combine into a single paragraph
all_sentences = [entry["input"].strip() for entry in data if entry["input"].strip()]
combined_text = " ".join(all_sentences)  # Join all sentences into one paragraph

# Save to a text file
with open("english_sentences.txt", "w", encoding="utf-8") as f:
    f.write(combined_text)

print("✅ Conversion complete! All sentences combined into 'english_sentences.txt'")
