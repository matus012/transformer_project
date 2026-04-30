# Design Decisions

Append-only. Format: YYYY-MM-DD | decision | rationale

2026-04-30 | val split = 90/10 stratified from train, seed=42 | preserves class balance, test set untouched until final eval
2026-04-30 | max_length=128 | Banking77 utterances are short (verify via data_stats.py)
2026-04-30 | padding='max_length' | uniform tensor shapes, simpler than dynamic padding for fp16 training
