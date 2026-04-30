import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer

ARCHIVE = os.path.join(os.path.dirname(__file__), "..", "archive")
os.makedirs(ARCHIVE, exist_ok=True)

MAX_LEN = 128
MODEL = "distilbert-base-uncased"
CACHE = "./data"

raw = load_dataset("PolyAI/banking77", cache_dir=CACHE, trust_remote_code=True)
split = raw["train"].train_test_split(test_size=0.10, stratify_by_column="label", seed=42)
train_ds = split["train"]
val_ds = split["test"]
test_ds = raw["test"]

label_names = raw["train"].features["label"].names
num_classes = len(label_names)

# class distribution
labels = np.array(train_ds["label"])
counts = np.bincount(labels, minlength=num_classes)
sorted_idx = np.argsort(counts)[::-1]

# token lengths (pre-truncation, no max_length cap)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
lengths = [len(tokenizer(ex["text"], truncation=False)["input_ids"]) for ex in train_ds]
lengths = np.array(lengths)
truncated = int((lengths > MAX_LEN).sum())

lines = [
    "=== Banking77 Stats ===",
    f"Train size: {len(train_ds)}",
    f"Val size: {len(val_ds)}",
    f"Test size: {len(test_ds)}",
    f"Num classes: {num_classes}",
    "",
    "=== Class Distribution (train) ===",
    f"min count: {counts.min()} (intent: {label_names[counts.argmin()]})",
    f"max count: {counts.max()} (intent: {label_names[counts.argmax()]})",
    f"mean: {counts.mean():.1f}",
    f"median: {np.median(counts):.1f}",
    f"imbalance ratio (max/min): {counts.max() / counts.min():.2f}",
    "",
    "=== Token Length (train, pre-truncation) ===",
    f"mean: {lengths.mean():.1f}",
    f"median: {np.median(lengths):.1f}",
    f"p95: {np.percentile(lengths, 95):.1f}",
    f"p99: {np.percentile(lengths, 99):.1f}",
    f"max: {lengths.max()}",
    f"truncation rate at max_len={MAX_LEN} (%): {100 * truncated / len(lengths):.2f}",
]

output = "\n".join(lines)
print(output)

stats_path = os.path.join(ARCHIVE, "data_stats.txt")
with open(stats_path, "w", encoding="utf-8") as f:
    f.write(output + "\n")

# plot 1: class distribution bar chart
fig, ax = plt.subplots(figsize=(18, 5))
ax.bar(range(num_classes), counts[sorted_idx], color="steelblue")
ax.set_xticks(range(num_classes))
ax.set_xticklabels([label_names[i] for i in sorted_idx], rotation=90, fontsize=7)
ax.set_xlabel("Intent")
ax.set_ylabel("Count")
ax.set_title("Banking77 Train Class Distribution (sorted desc)")
plt.tight_layout()
plt.savefig(os.path.join(ARCHIVE, "class_distribution.png"), dpi=100)
plt.close()

# plot 2: token length histogram
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(lengths, bins=40, color="steelblue", edgecolor="white")
ax.axvline(MAX_LEN, color="red", linestyle="--", label=f"max_length={MAX_LEN}")
ax.set_xlabel("Token length")
ax.set_ylabel("Count")
ax.set_title("Banking77 Train Token Lengths (pre-truncation)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(ARCHIVE, "token_length_hist.png"), dpi=100)
plt.close()

print(f"\nSaved: {stats_path}")
print(f"Saved: {os.path.join(ARCHIVE, 'class_distribution.png')}")
print(f"Saved: {os.path.join(ARCHIVE, 'token_length_hist.png')}")
