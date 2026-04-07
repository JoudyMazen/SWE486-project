import time
import psutil
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

def read_samples(filename, max_samples=3):
    samples = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("Text:"):
                samples.append(line.replace("Text:", "").strip())
                if len(samples) == max_samples:
                    break
    return samples

datasets = {
    "short": read_samples("short_inputs.txt", 3),
    "medium": read_samples("medium_inputs.txt", 3),
    "large": read_samples("large_inputs.txt", 3)
}

for size, texts in datasets.items():
    print(f"\n===== {size.upper()} INPUTS =====")
    times = []

    for i, text in enumerate(texts, start=1):
        word_count = len(text.split())

        psutil.cpu_percent(interval=None)  # prime measurement

        start_time = time.perf_counter()
        result = classifier(text)
        end_time = time.perf_counter()

        cpu_usage = psutil.cpu_percent(interval=None)
        memory_usage = psutil.virtual_memory().percent
        inference_time = round(end_time - start_time, 4)
        times.append(inference_time)

        print(f"Sample {i} ({word_count} words)")
        print("Result:", result)
        print("Inference time:", inference_time, "seconds")
        print("CPU usage:", cpu_usage, "%")
        print("Memory usage:", memory_usage, "%")
        print("-" * 50)

    avg = round(sum(times) / len(times), 4)
    print(f"Average inference time for {size} inputs: {avg}s")