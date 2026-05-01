"""Sequential sweep runner for Banking77 experiment runs 2-8.

Usage:
    python scripts/run_all_experiments.py [--only NAME1,NAME2] [--skip-baseline]
"""
import argparse
import json
import os
import sys
import time
import traceback

import torch

# ensure project root on path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ALL_CONFIGS
from src.evaluate import evaluate_run
from src.train import train
from src.utils import set_seed

NEW_RUNS = [
    "distilbert_lr_low",
    "distilbert_lr_high",
    "distilbert_batch32",
    "distilbert_epochs5",
    "distilbert_frozen_encoder",
    "distilbert_frozen_partial",
    "bert_baseline",
]

DEBUG_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "archive", "debug_log.md")


def _append_debug(msg: str) -> None:
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run_one(run_name: str) -> dict:
    """Train + evaluate one run, returning summary dict. Handles BERT OOM."""
    cfg = ALL_CONFIGS[run_name]()
    set_seed(cfg.seed)

    if run_name == "bert_baseline":
        try:
            train_result = train(cfg)
        except torch.cuda.OutOfMemoryError:
            msg = f"2026-05-01 | bert_baseline OOM at batch=16, retrying batch=8 grad_accum=2"
            print(f"  [OOM] {msg}")
            _append_debug(msg)
            # fallback: write note to training_log if dir exists
            log_path = os.path.join(cfg.output_dir, "training_log.txt")
            os.makedirs(cfg.output_dir, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[OOM FALLBACK] {msg}\n")
            torch.cuda.empty_cache()
            cfg2 = ALL_CONFIGS[run_name]()
            cfg2.per_device_train_batch_size = 8
            cfg2.gradient_accumulation_steps = 2
            set_seed(cfg2.seed)
            try:
                train_result = train(cfg2)
            except torch.cuda.OutOfMemoryError:
                # second fallback: batch 4 + grad_accum 4
                msg2 = f"2026-05-01 | bert_baseline OOM at batch=8, retrying batch=4 grad_accum=4"
                print(f"  [OOM2] {msg2}")
                _append_debug(msg2)
                torch.cuda.empty_cache()
                cfg3 = ALL_CONFIGS[run_name]()
                cfg3.per_device_train_batch_size = 4
                cfg3.gradient_accumulation_steps = 4
                set_seed(cfg3.seed)
                train_result = train(cfg3)
    else:
        train_result = train(cfg)

    run_dir = cfg.output_dir
    eval_result = evaluate_run(run_dir, split="test")

    return {
        "run_name": run_name,
        "val_f1": train_result["best_val_macro_f1"],
        "test_f1": eval_result["macro_f1"],
        "wall_clock": train_result["wall_clock_sec"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default="", help="Comma-separated run names to execute")
    parser.add_argument("--skip-baseline", action="store_true", default=True,
                        help="Skip distilbert_baseline (already done). Default: True")
    parser.add_argument("--include-baseline", action="store_true", default=False)
    args = parser.parse_args()

    if args.only:
        target_runs = [r.strip() for r in args.only.split(",")]
    elif args.include_baseline:
        target_runs = list(ALL_CONFIGS.keys())
    else:
        target_runs = NEW_RUNS

    print(f"Runs to execute: {target_runs}\n")

    results: list[dict] = []

    # prepend baseline row from existing artifacts if present
    baseline_dir = ALL_CONFIGS["distilbert_baseline"]().output_dir
    baseline_metrics_path = os.path.join(baseline_dir, "test_metrics.json")
    baseline_csv_path = os.path.join(baseline_dir, "metrics.csv")
    if not args.include_baseline and "distilbert_baseline" not in target_runs:
        if os.path.exists(baseline_metrics_path) and os.path.exists(baseline_csv_path):
            with open(baseline_metrics_path, encoding="utf-8") as f:
                bm = json.load(f)
            import csv
            with open(baseline_csv_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            best_val_f1 = max(float(r["val_macro_f1"]) for r in rows)
            results.append({
                "run_name": "distilbert_baseline",
                "val_f1": best_val_f1,
                "test_f1": bm["macro_f1"],
                "wall_clock": 113.5,
            })

    for run_name in target_runs:
        if run_name not in ALL_CONFIGS:
            print(f"[SKIP] Unknown run: {run_name}")
            continue

        print(f"{'='*60}")
        print(f"Starting: {run_name}")
        print(f"{'='*60}")

        torch.cuda.empty_cache()
        set_seed(42)

        t_start = time.perf_counter()
        try:
            summary = run_one(run_name)
            elapsed = time.perf_counter() - t_start
            summary["wall_clock"] = round(elapsed, 1)
            results.append(summary)
            print(f"[{run_name}] val_f1={summary['val_f1']:.4f} test_f1={summary['test_f1']:.4f} wall_clock={summary['wall_clock']}s")
        except Exception as e:
            elapsed = time.perf_counter() - t_start
            tb = traceback.format_exc()
            print(f"[ERROR] {run_name} failed after {elapsed:.1f}s:\n{tb}")
            entry = (
                f"\n2026-05-01 | run_all_experiments error: {run_name} failed\n"
                f"Error: {e}\nTraceback:\n{tb}\n"
            )
            _append_debug(entry)
            results.append({
                "run_name": run_name,
                "val_f1": float("nan"),
                "test_f1": float("nan"),
                "wall_clock": round(elapsed, 1),
            })

    # final summary table
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'run_name':<30} {'val_f1':>8} {'test_f1':>8} {'wall_clock':>12}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*12}")
    for r in results:
        val = f"{r['val_f1']:.4f}" if r['val_f1'] == r['val_f1'] else "ERROR"
        tst = f"{r['test_f1']:.4f}" if r['test_f1'] == r['test_f1'] else "ERROR"
        print(f"{r['run_name']:<30} {val:>8} {tst:>8} {r['wall_clock']:>11.1f}s")


if __name__ == "__main__":
    main()
