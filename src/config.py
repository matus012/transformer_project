from dataclasses import dataclass, asdict
from typing import Literal
import json
import os


@dataclass
class ExperimentConfig:
    # Identity
    run_name: str

    # Model
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 77

    # Data
    max_length: int = 128
    val_size: float = 0.10

    # Training
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    num_train_epochs: int = 3
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    fp16: bool = True

    # Freezing
    freeze_strategy: Literal["none", "encoder", "partial"] = "none"
    partial_freeze_layers: int = 0

    # Reproducibility
    seed: int = 42

    # Paths
    output_dir: str = ""

    def __post_init__(self):
        if not self.output_dir:
            self.output_dir = f"./experiments/{self.run_name}"

    def save(self, path: str | None = None) -> str:
        path = path or os.path.join(self.output_dir, "config.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        return path


def baseline_config() -> ExperimentConfig:
    return ExperimentConfig(run_name="distilbert_baseline")


def distilbert_lr_low() -> ExperimentConfig:
    return ExperimentConfig(run_name="distilbert_lr_low", learning_rate=2e-5)


def distilbert_lr_high() -> ExperimentConfig:
    return ExperimentConfig(run_name="distilbert_lr_high", learning_rate=1e-4)


def distilbert_batch32() -> ExperimentConfig:
    return ExperimentConfig(run_name="distilbert_batch32", per_device_train_batch_size=32)


def distilbert_epochs5() -> ExperimentConfig:
    return ExperimentConfig(run_name="distilbert_epochs5", num_train_epochs=5)


def distilbert_frozen_encoder() -> ExperimentConfig:
    return ExperimentConfig(run_name="distilbert_frozen_encoder", freeze_strategy="encoder")


def distilbert_frozen_partial() -> ExperimentConfig:
    return ExperimentConfig(
        run_name="distilbert_frozen_partial",
        freeze_strategy="partial",
        partial_freeze_layers=3,
    )


def bert_baseline() -> ExperimentConfig:
    return ExperimentConfig(
        run_name="bert_baseline",
        model_name="bert-base-uncased",
    )


ALL_CONFIGS: dict = {
    "distilbert_baseline": baseline_config,
    "distilbert_lr_low": distilbert_lr_low,
    "distilbert_lr_high": distilbert_lr_high,
    "distilbert_batch32": distilbert_batch32,
    "distilbert_epochs5": distilbert_epochs5,
    "distilbert_frozen_encoder": distilbert_frozen_encoder,
    "distilbert_frozen_partial": distilbert_frozen_partial,
    "bert_baseline": bert_baseline,
}
