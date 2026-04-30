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
