from src.data import DataConfig, load_banking77, tokenize_dataset, get_label_names
from src.utils import set_seed, setup_logger

set_seed(42)
log = setup_logger("smoke")
cfg = DataConfig()
ds = load_banking77(cfg)
log.info(f"Splits: {list(ds.keys())}")
log.info(f"Train: {len(ds['train'])}, Val: {len(ds['validation'])}, Test: {len(ds['test'])}")
log.info(f"First train example: {ds['train'][0]}")

tokenized, tok = tokenize_dataset(ds, cfg)
log.info(f"Tokenized columns: {tokenized['train'].column_names}")
log.info(f"First input_ids shape: {tokenized['train'][0]['input_ids'].shape}")
log.info(f"Num labels: {len(get_label_names())}")
