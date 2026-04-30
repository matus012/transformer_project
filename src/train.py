import csv
import os
import subprocess
import time
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)

from src.config import ExperimentConfig
from src.data import DataConfig, load_banking77, tokenize_dataset
from src.utils import set_seed, setup_logger


def _apply_freeze(model, cfg: ExperimentConfig) -> None:
    if cfg.freeze_strategy == "none":
        return
    if cfg.freeze_strategy == "encoder":
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        return
    if cfg.freeze_strategy == "partial":
        # freeze embeddings
        for param in model.distilbert.embeddings.parameters():
            param.requires_grad = False
        # freeze bottom N transformer layers
        n = cfg.partial_freeze_layers
        for layer in model.distilbert.transformer.layer[:n]:
            for param in layer.parameters():
                param.requires_grad = False


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


class MetricsCSVCallback(TrainerCallback):
    def __init__(self, csv_path: str, logger: logging.Logger):
        self.csv_path = csv_path
        self.logger = logger
        self._rows: list[dict] = []

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics is None:
            return
        epoch = int(round(state.epoch or 0))
        # pull train_loss from log_history (last entry with 'loss' key at this epoch)
        train_loss = float("nan")
        for entry in reversed(state.log_history):
            if "loss" in entry and int(round(entry.get("epoch", 0))) == epoch:
                train_loss = entry["loss"]
                break
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(metrics.get("eval_loss", float("nan")), 6),
            "val_acc": round(metrics.get("eval_accuracy", float("nan")), 6),
            "val_macro_f1": round(metrics.get("eval_macro_f1", float("nan")), 6),
        }
        self._rows.append(row)
        self.logger.info(f"Epoch {epoch} | val_loss={row['val_loss']} acc={row['val_acc']} macro_f1={row['val_macro_f1']}")
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1"])
            writer.writeheader()
            writer.writerows(self._rows)


def _save_plots(rows: list[dict], plots_dir: str) -> None:
    os.makedirs(plots_dir, exist_ok=True)
    epochs = [r["epoch"] for r in rows]

    fig, ax = plt.subplots()
    ax.plot(epochs, [r["train_loss"] for r in rows], label="train_loss", marker="o")
    ax.plot(epochs, [r["val_loss"] for r in rows], label="val_loss", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "loss_curve.png"), dpi=100)
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(epochs, [r["val_acc"] for r in rows], label="val_accuracy", marker="o")
    ax.plot(epochs, [r["val_macro_f1"] for r in rows], label="val_macro_f1", marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation Accuracy & Macro-F1")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "accuracy_curve.png"), dpi=100)
    plt.close()


def train(cfg: ExperimentConfig) -> dict:
    os.makedirs(cfg.output_dir, exist_ok=True)

    log_path = os.path.join(cfg.output_dir, "training_log.txt")
    logger = setup_logger("train", log_file=log_path)
    logger.info(f"Starting run: {cfg.run_name}")

    set_seed(cfg.seed)
    cfg.save()
    logger.info(f"Config saved to {cfg.output_dir}/config.json")

    # git commit hash
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_hash = "unknown"
    with open(os.path.join(cfg.output_dir, "git_commit.txt"), "w") as f:
        f.write(git_hash + "\n")
    logger.info(f"Git commit: {git_hash}")

    data_cfg = DataConfig(
        model_name=cfg.model_name,
        max_length=cfg.max_length,
        val_size=cfg.val_size,
        seed=cfg.seed,
    )
    logger.info("Loading dataset...")
    ds = load_banking77(data_cfg)
    tokenized, tokenizer = tokenize_dataset(ds, data_cfg)
    logger.info(f"Train: {len(tokenized['train'])}, Val: {len(tokenized['validation'])}, Test: {len(tokenized['test'])}")

    logger.info(f"Loading model: {cfg.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=cfg.num_labels)
    _apply_freeze(model, cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,}")

    csv_path = os.path.join(cfg.output_dir, "metrics.csv")
    csv_callback = MetricsCSVCallback(csv_path, logger)

    best_model_dir = os.path.join(cfg.output_dir, "best_model")
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        fp16=cfg.fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=_compute_metrics,
        callbacks=[csv_callback],
    )

    logger.info("Starting training...")
    t0 = time.perf_counter()
    trainer.train()
    wall_clock = time.perf_counter() - t0
    logger.info(f"Training done in {wall_clock:.1f}s")

    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    logger.info(f"Best model saved to {best_model_dir}")

    # find best epoch + metric from csv rows
    rows = csv_callback._rows
    best_row = max(rows, key=lambda r: r["val_macro_f1"])

    _save_plots(rows, os.path.join(cfg.output_dir, "plots"))
    logger.info("Plots saved.")

    return {
        "best_val_macro_f1": best_row["val_macro_f1"],
        "best_epoch": best_row["epoch"],
        "wall_clock_sec": round(wall_clock, 1),
    }


if __name__ == "__main__":
    import argparse
    from src.config import baseline_config

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="baseline")
    args = p.parse_args()
    cfg = baseline_config() if args.config == "baseline" else None
    if cfg is None:
        raise ValueError(f"Unknown config: {args.config}")
    result = train(cfg)
    print(f"Best val macro-F1: {result['best_val_macro_f1']:.4f} @ epoch {result['best_epoch']}")
    print(f"Wall-clock: {result['wall_clock_sec']:.1f}s")
