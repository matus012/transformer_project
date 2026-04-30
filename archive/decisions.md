# Design Decisions

Append-only. Format: YYYY-MM-DD | decision | rationale

2026-04-30 | val split = 90/10 stratified from train, seed=42 | preserves class balance, test set untouched until final eval
2026-04-30 | max_length=128 | Banking77 utterances are short (verify via data_stats.py)
2026-04-30 | padding='max_length' | uniform tensor shapes, simpler than dynamic padding for fp16 training
2026-04-30 | Trainer with load_best_model_at_end + metric_for_best_model=eval_macro_f1 | macro-F1 is class-balanced metric appropriate for 5.25× imbalance ratio
2026-04-30 | save_total_limit=1 | only keep best checkpoint to manage 8GB VRAM machine disk usage across 8 runs
2026-04-30 | editable install via pyproject.toml + pip install -e . | clean fix for src/ imports, replaces fragile PYTHONPATH workaround
2026-04-30 | per-class F1 plot color-coded by threshold (red <0.7, orange 0.7-0.85, green ≥0.85) | quick visual diagnosis of weak classes for defense talking points
2026-04-30 | confusion matrix annotates only cells with ≥5 counts | reduces visual clutter on 77×77 grid
2026-04-30 | evaluate_run takes run_dir not config | enables eval of any saved run without re-instantiating training config
