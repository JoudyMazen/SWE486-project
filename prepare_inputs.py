import os
import csv
import random

CSV_FILE = "IMDB Dataset.csv"
TARGET_PER_CATEGORY = 100
random.seed(42)

def classify_length(text):
    word_count = len(text.split())
    if word_count < 50:
        return "short"
    elif word_count <= 120:
        return "medium"
    else:
        return "large"

all_rows = []

with open(CSV_FILE, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        all_rows.append(row)

random.shuffle(all_rows)

short_samples = []
medium_samples = []
large_samples = []

for row in all_rows:
    text = row["review"].strip()
    sentiment = row["sentiment"].strip().lower()

    label = 1 if sentiment == "positive" else 0

    sample = {
        "text": text,
        "label": label,
        "word_count": len(text.split())
    }

    category = classify_length(text)

    if category == "short" and len(short_samples) < TARGET_PER_CATEGORY:
        short_samples.append(sample)

    elif category == "medium" and len(medium_samples) < TARGET_PER_CATEGORY:
        medium_samples.append(sample)

    elif category == "large" and len(large_samples) < TARGET_PER_CATEGORY:
        large_samples.append(sample)

    if (
        len(short_samples) == TARGET_PER_CATEGORY and
        len(medium_samples) == TARGET_PER_CATEGORY and
        len(large_samples) == TARGET_PER_CATEGORY
    ):
        break

os.makedirs("input_data", exist_ok=True)

def save_samples(filename, samples):
    with open(filename, "w", encoding="utf-8") as f:
        for i, sample in enumerate(samples, start=1):
            clean_text = sample["text"].replace("\n", " ").replace("<br />", " ").strip()
            f.write(f"Sample {i}\n")
            f.write(f"Label: {sample['label']}\n")
            f.write(f"Word count: {sample['word_count']}\n")
            f.write(f"Text: {clean_text}\n")
            f.write("-" * 80 + "\n")

save_samples("input_data/short_inputs.txt", short_samples)
save_samples("input_data/medium_inputs.txt", medium_samples)
save_samples("input_data/large_inputs.txt", large_samples)

print("DONE ✅")
print(f"Short: {len(short_samples)}")
print(f"Medium: {len(medium_samples)}")
print(f"Large: {len(large_samples)}")