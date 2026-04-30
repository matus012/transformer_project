from dataclasses import dataclass, field
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer


@dataclass
class DataConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    val_size: float = 0.10
    seed: int = 42
    cache_dir: str = "./data"


def load_banking77(cfg: DataConfig) -> DatasetDict:
    raw = load_dataset("PolyAI/banking77", cache_dir=cfg.cache_dir, trust_remote_code=True)
    split = raw["train"].train_test_split(
        test_size=cfg.val_size,
        stratify_by_column="label",
        seed=cfg.seed,
    )
    return DatasetDict({
        "train": split["train"],
        "validation": split["test"],
        "test": raw["test"],
    })


def tokenize_dataset(ds: DatasetDict, cfg: DataConfig) -> tuple[DatasetDict, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length",
        )

    tokenized = ds.map(_tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    return tokenized, tokenizer


def get_label_names() -> list[str]:
    ds = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True)
    return ds.features["label"].names
