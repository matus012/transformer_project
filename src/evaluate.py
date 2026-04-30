import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.data import DataConfig, get_label_names, load_banking77, tokenize_dataset
from src.utils import set_seed, setup_logger


def evaluate_run(run_dir: str, split: str = "test") -> dict:
    logger = setup_logger("evaluate")
    set_seed(42)

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, encoding="utf-8") as f:
        cfg_dict = json.load(f)

    model_name = cfg_dict["model_name"]
    max_length = cfg_dict["max_length"]
    val_size = cfg_dict.get("val_size", 0.10)
    seed = cfg_dict.get("seed", 42)

    best_model_dir = os.path.join(run_dir, "best_model")
    logger.info(f"Loading model from {best_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)

    data_cfg = DataConfig(
        model_name=model_name,
        max_length=max_length,
        val_size=val_size,
        seed=seed,
    )
    logger.info("Loading dataset...")
    ds = load_banking77(data_cfg)
    tokenized, _ = tokenize_dataset(ds, data_cfg)
    label_names = get_label_names()

    eval_args = TrainingArguments(
        output_dir=run_dir,
        per_device_eval_batch_size=64,
        report_to="none",
        seed=seed,
    )
    trainer = Trainer(model=model, args=eval_args)

    logger.info(f"Running inference on '{split}' split ({len(tokenized[split])} examples)...")
    predictions = trainer.predict(tokenized[split])
    logits = predictions.predictions
    preds = np.argmax(logits, axis=-1)
    labels = np.array(tokenized[split]["label"])

    accuracy = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    per_class_f1_arr = f1_score(labels, preds, average=None, zero_division=0)
    per_class_f1 = {label_names[i]: round(float(per_class_f1_arr[i]), 4) for i in range(len(label_names))}

    sorted_classes = sorted(per_class_f1.items(), key=lambda x: x[1])
    worst_5 = [{"label": k, "f1": v} for k, v in sorted_classes[:5]]
    best_5 = [{"label": k, "f1": v} for k, v in sorted_classes[-5:]]

    report_str = classification_report(labels, preds, target_names=label_names, digits=4, zero_division=0)

    logger.info(f"accuracy={accuracy:.4f}  macro_f1={macro_f1:.4f}  weighted_f1={weighted_f1:.4f}")
    logger.info(f"Worst 5: {worst_5}")
    logger.info(f"Best  5: {best_5}")

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # save text artifacts
    metrics_dict = {
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "num_examples": len(labels),
        "split": split,
        "per_class_f1": per_class_f1,
        "worst_5_classes": worst_5,
        "best_5_classes": best_5,
    }
    with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)
    with open(os.path.join(run_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_str)

    # confusion matrix
    cm = confusion_matrix(labels, preds)
    annot = np.where(cm >= 5, cm, 0).astype(str)
    annot[annot == "0"] = ""
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(
        cm, ax=ax, cmap="Blues",
        xticklabels=label_names, yticklabels=label_names,
        annot=annot, fmt="s", linewidths=0.3,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"Confusion Matrix — {run_dir} [{split}]", fontsize=13)
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(rotation=0, fontsize=5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=120)
    plt.close()
    logger.info("Saved confusion_matrix.png")

    # per-class F1 horizontal bar chart
    sorted_labels = [k for k, _ in sorted_classes]
    sorted_f1 = [v for _, v in sorted_classes]
    colors = ["red" if v < 0.7 else "orange" if v < 0.85 else "green" for v in sorted_f1]
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.barh(range(len(sorted_labels)), sorted_f1, color=colors)
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=7)
    ax.set_xlabel("F1 Score")
    ax.set_title(f"Per-Class F1 (sorted asc) — {split} set")
    ax.axvline(0.7, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(0.85, color="orange", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "per_class_f1.png"), dpi=120)
    plt.close()
    logger.info("Saved per_class_f1.png")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_examples": int(len(labels)),
        "split": split,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    args = p.parse_args()
    result = evaluate_run(args.run_dir, args.split)
    print(f"[{args.split}] acc={result['accuracy']:.4f}  macro_f1={result['macro_f1']:.4f}  weighted_f1={result['weighted_f1']:.4f}")
