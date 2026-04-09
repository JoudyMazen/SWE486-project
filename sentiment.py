# Import required libraries for timing and system monitoring
import time
import psutil
from transformers import pipeline

# Load pre-trained DistilBERT model fine-tuned on SST-2 for sentiment classification
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

# Read up to max_samples text entries from an external .txt file
def read_samples(filename, max_samples=3):
    samples = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            # Only extract lines that start with "Text:" prefix
            if line.startswith("Text:"):
                samples.append(line.replace("Text:", "").strip())
                if len(samples) == max_samples:
                    break
    return samples

# Load short, medium, and large input samples from external files
datasets = {
    "short": read_samples("short_inputs.txt", 100),
    "medium": read_samples("medium_inputs.txt", 100),
    "large": read_samples("large_inputs.txt", 100)
}

# Loop through each input size category
for size, texts in datasets.items():
    print(f"\n===== {size.upper()} INPUTS =====")
    times = []

    for i, text in enumerate(texts, start=1):
        # Count words to verify input size
        word_count = len(text.split())

        # Prime CPU measurement before inference
        psutil.cpu_percent(interval=None)

        # Record inference start and end time
        start_time = time.perf_counter()
       # Truncate input to model's maximum token limit (512 tokens)
        result = classifier(text, truncation=True, max_length=512)
        end_time = time.perf_counter()

        # Capture CPU and memory usage during inference window
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_usage = psutil.virtual_memory().percent
        inference_time = round(end_time - start_time, 4)
        times.append(inference_time)

        # Print results for this sample
        print(f"Sample {i} ({word_count} words)")
        print("Result:", result)
        print("Inference time:", inference_time, "seconds")
        print("CPU usage:", cpu_usage, "%")
        print("Memory usage:", memory_usage, "%")
        print("-" * 50)

    # Calculate and print average inference time per category
    avg = round(sum(times) / len(times), 4)
    print(f"Average inference time for {size} inputs: {avg}s")
