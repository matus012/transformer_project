# Project Plan: Transformer Fine-Tuning for Intent Classification

**Course:** Machine Learning (TUKE FEI, BSc Intelligent Systems, Y2)
**Assignment:** Zadanie 20 — Ladenie neurónovej siete typu Transformer
**Approach:** Fine-tune pretrained DistilBERT on Banking77 dataset for intent classification
**Goal:** Hit bare-minimum rubric requirements with a defendable, simple solution

---

## 1. Assignment & Rubric

Original (SK): "Ladenie neurónovej siete typu Transformer"

Grading (40 points total):
- 10b — correctness of solution (efficiency metrics)
- 10b — explanation of how the program works
- 10b — amount of work and effort (size and number of trained models, visualization)
- 10b — functional application

**Rubric mapping strategy:**
| Points | How we satisfy |
|--------|----------------|
| Correctness | Standard metrics: accuracy, macro-F1, per-class F1, confusion matrix. Clean train/val/test split. |
| Explanation | README + defense prep talking points. Cover transformer basics, tokenization, fine-tuning, hyperparameters. |
| Amount of work | 8 model variants (1 baseline + 6 tuning experiments + 1 different architecture). Rich visualizations. |
| Functional app | Gradio web demo: text input → predicted intent + top-3 confidence scores. |

---

## 2. Scope (locked)

- **Task:** Multi-class intent classification
- **Dataset:** Banking77 (`PolyAI/banking77` on HuggingFace) — 77 banking intents, ~13k examples
- **Primary model:** `distilbert-base-uncased` (66M params)
- **Comparison model:** `bert-base-uncased` (110M params)
- **Training:** Fine-tuning (NOT from scratch — approved by supervisor)
- **Compute:** Local RTX 4060 Laptop 8GB VRAM

**Out of scope (defensive — do NOT expand):**
- Multiple datasets
- Seq2seq / generation tasks
- Custom model architectures
- Knowledge distillation, quantization, LoRA
- Multi-lingual experiments

---

## 3. Techstack

```
Python 3.11.9 (venv, no conda)
torch (CUDA 12.1 build — compatible with CUDA 12.6 driver)
transformers
datasets
evaluate
accelerate
scikit-learn
matplotlib
seaborn
gradio
pandas
numpy
jupyter (exploration only)
```

Pinned versions go in `requirements.txt` during Phase 1 setup.

---

## 4. Repository Structure

```
banking77-transformer-tuning/
├── README.md                      # project overview, results, usage
├── LICENSE                        # MIT
├── requirements.txt               # pinned deps
├── .gitignore                     # ignores data/, models/, __pycache__, .venv
├── context.md                     # symlink/copy of plan_transformer.md
├── status.txt                     # current state
├── archive/
│   ├── debug_log.md              # append-only bug + fix log
│   └── decisions.md              # append-only design decisions
├── src/
│   ├── __init__.py
│   ├── config.py                 # experiment configs (dataclass per run)
│   ├── data.py                   # load + tokenize Banking77
│   ├── train.py                  # training loop (reads config, runs 1 experiment)
│   ├── evaluate.py               # metrics + confusion matrix + per-class F1
│   ├── utils.py                  # seed, logging, device handling
│   └── app.py                    # Gradio demo
├── experiments/                   # one subdir per run
│   └── <run_name>/
│       ├── config.json
│       ├── metrics.csv
│       ├── training_log.txt
│       ├── plots/
│       │   ├── loss_curve.png
│       │   ├── accuracy_curve.png
│       │   └── confusion_matrix.png
│       └── best_model/           # .gitignore'd
├── notebooks/
│   └── analysis.ipynb            # cross-run comparison plots
├── models/                        # .gitignore'd (best model for app)
├── data/                          # .gitignore'd (HF cache)
└── scripts/
    ├── run_all_experiments.py    # runs all 8 configs sequentially
    └── download_data.py          # pre-download dataset
```

**Repo hosting:** Public GitHub repo (portfolio piece). MIT license.

**Defense format (supervisor confirmed):** free-hand verbal defense. No slide deck required. README + working app + experiment folders are the artifacts.

---

## 5. Experiment Matrix (8 runs)

| # | Run Name | Model | LR | Batch | Epochs | Frozen Layers | Purpose |
|---|----------|-------|-----|-------|--------|---------------|---------|
| 1 | `distilbert_baseline` | DistilBERT | 5e-5 | 16 | 3 | none | Baseline |
| 2 | `distilbert_lr_low` | DistilBERT | 2e-5 | 16 | 3 | none | LR sweep (lower) |
| 3 | `distilbert_lr_high` | DistilBERT | 1e-4 | 16 | 3 | none | LR sweep (higher) |
| 4 | `distilbert_batch32` | DistilBERT | 5e-5 | 32 | 3 | none | Batch size effect |
| 5 | `distilbert_epochs5` | DistilBERT | 5e-5 | 16 | 5 | none | More epochs |
| 6 | `distilbert_frozen_encoder` | DistilBERT | 5e-5 | 16 | 3 | all encoder | Head-only training |
| 7 | `distilbert_frozen_partial` | DistilBERT | 5e-5 | 16 | 3 | bottom 3 layers | Partial freeze |
| 8 | `bert_baseline` | BERT-base | 5e-5 | 16 | 3 | none | Architecture comparison |

**Shared settings (all runs):**
- Weight decay: 0.01
- Warmup steps: 500
- Optimizer: AdamW
- Max seq length: 128 (Banking77 utterances are short)
- Seed: 42
- Eval strategy: every epoch
- Save best model by val macro-F1
- Mixed precision: fp16 enabled (`Trainer(fp16=True)`)
- Early stopping: none — fixed epochs, keep best checkpoint by val macro-F1
- OOM fallback: if BERT-base OOMs at batch 16, drop to batch 8 + grad accum 2 (effective 16)
- Logging: CSV (`metrics.csv`) + plain text (`training_log.txt`) only. No W&B / TensorBoard.
- Wall-clock per run: logged for time-vs-accuracy plot
- Determinism: seeds only (`random`, `numpy`, `torch`, `torch.cuda`). No `torch.use_deterministic_algorithms` (kills throughput; documented trade-off).
- Val split: carved from Banking77 train (10003 ex.) — 90/10 stratified, seed=42. Test set (3080 ex.) untouched until final eval.

**Defense narrative arc:**
1. Baseline (run 1) — establish reference
2. Learning rate sweep (runs 1,2,3) — find optimal LR
3. Batch/epoch effects (runs 4,5) — capacity vs compute trade-offs
4. Transfer learning depth (runs 6,7) — show what pretrained layers contribute
5. Architecture comparison (run 8) — DistilBERT vs BERT at same budget

---

## 6. Visualizations Required

**Per-run (8 sets):**
- Training/validation loss curves
- Training/validation accuracy curves
- Confusion matrix on test set
- Per-class F1 bar chart

**Cross-run (comparison):**
- Bar chart: test accuracy per run
- Bar chart: test macro-F1 per run
- Line plot: val loss curves overlaid
- Table: all runs with final metrics
- Scatter: training time vs accuracy

---

## 7. Gradio App (Phase 7)

Minimum viable demo:
- Single text input box ("enter banking query")
- "Predict" button
- Output: top-1 intent label + confidence
- Output: top-3 intents with confidences (bar chart or table)
- Model: best run from experiments (loaded on app startup)
- Launch: `python src/app.py` → localhost:7860

---

## 8. Phases (Execution Plan)

| Phase | Goal | Deliverable | Est. Time |
|-------|------|-------------|-----------|
| 1 | Setup | Repo created, venv, deps installed, GPU check passes | 1h |
| 2 | Data pipeline | Banking77 loaded, tokenized, splits verified, class distribution + token length stats logged | 1-2h |
| 3 | Single training run | Run 1 complete end-to-end with metrics + saved model | 2-3h |
| 4 | Evaluation module | Confusion matrix, per-class F1, classification report working | 1h |
| 5 | Experiment sweep | All 8 runs complete, each logged to own folder | 2-4h (mostly GPU wait) |
| 6 | Analysis notebook | Cross-run comparison plots, final metrics table | 1-2h |
| 7 | Gradio app | Working demo loading best model | 1h |
| 8 | Documentation | README complete, defense prep notes, screenshots | 1-2h |
| 9 | Final polish | Clean repo, final commit, tagged release | 0.5h |

**Total estimate:** 10-16 hours of active work + GPU wait time.

Each phase is gated by user approval before moving to next.

---

## 9. Defense Preparation (CRITICAL)

**Likely teacher questions + talking points:**

1. *What is a Transformer?*
   → Self-attention mechanism, processes sequence in parallel, Q/K/V projections, multi-head attention, residual + LayerNorm, stacked blocks.

2. *Why fine-tuning and not training from scratch?*
   → Compute/data constraints. Pretrained DistilBERT already encodes English. Fine-tuning = swap classifier head + continue training on task data. Approved by supervisor.

3. *What does the [CLS] token do?*
   → Prepended to input, its final hidden state is used as sequence representation for classification.

4. *What is tokenization / WordPiece?*
   → Subword tokenization. Handles OOV by breaking words into known pieces. "fintech" → ["fin", "##tech"].

5. *What do your hyperparameters do?*
   - LR: step size for weight updates. Too high = divergence, too low = slow.
   - Batch size: samples per gradient step. Bigger = more stable gradients, more VRAM.
   - Epochs: passes over dataset. Too many = overfitting.
   - Weight decay: L2 regularization on weights.
   - Warmup: linear LR increase at start, then decay.

6. *Why freeze layers?*
   → Lower layers learn general features, upper layers task-specific. Freezing bottom = preserve general features, faster training. Full freeze = head-only = fastest but weaker.

7. *Why DistilBERT?*
   → Distilled from BERT, 40% smaller, 60% faster, retains ~97% performance. Fits my 8GB VRAM with larger batches.

8. *Why Banking77?*
   → Realistic intent classification scenario, 77 fine-grained classes make it non-trivial, clean HF-available dataset.

9. *How do you evaluate?*
   → Test set (never touched during training). Accuracy + macro-F1 (balanced across classes) + confusion matrix for error analysis.

10. *What's your best model? Why?*
    → [Fill in after experiments]

11. *What didn't work?*
    → [Fill in from debug_log.md during project]

---

## 10. Reproducibility Requirements

- `seed=42` set for: `random`, `numpy`, `torch`, `torch.cuda`
- No `torch.use_deterministic_algorithms` — throughput trade-off documented
- `requirements.txt` with exact versions
- Each experiment saves `config.json` snapshot
- Git commit hash logged per run
- Wall-clock training time logged per run

---

## 11. Deliverable Checklist

- [ ] Public GitHub repo
- [ ] README with project description, setup, usage, results summary
- [ ] 8 experiment folders with logs, metrics, plots
- [ ] Cross-run analysis notebook
- [ ] Working Gradio demo
- [ ] Defense prep notes
- [ ] Trained best-model checkpoint (or HF Hub upload — optional)
- [ ] (Optional) Short write-up if supervisor requests documentation

---

## 12. Workflow (CChat + CCLI Split)

- **This doc (`plan_transformer.md`):** static strategy, fed into every new CChat
- **`status.txt`:** dynamic state, updated per session
- **`coding_plan.md`:** generated by plan-keeper CChat per phase, CCLI reads directly
- **Plan-keeper CChat:** holds `plan_transformer.md` + `status.txt`, generates phase prompts, reviews CCLI output
- **CCLI:** sees `plan_transformer.md` + `coding_plan.md` + `status.txt`, executes one phase at a time, reports back

**Approval gate between phases.** User says "proceed to phase N" → CCLI executes phase N only → reports → review → approve next.

**CCLI phase report format** (CCLI returns this to user, user pastes to plan-keeper CChat):
- Phase N completed: yes/no
- Files created/modified: list
- Acceptance criteria: pass/fail per item
- Deviations from plan: list
- Errors hit + resolutions: list
- Suggested next action

---

## 13. Out-of-Scope / Anti-Goals

Do not add these unless explicitly requested:
- HTML/React/fancy UI beyond Gradio defaults
- LaTeX documentation (unless supervisor demands)
- Multiple languages / multilingual models
- Custom loss functions
- Hyperparameter search frameworks (Optuna, Ray Tune) — manual sweep is fine
- Deployment beyond local Gradio
- Docker / containerization

Keep it boring. Boring = defendable.