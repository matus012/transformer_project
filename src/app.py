"""Gradio demo for Banking77 intent classification.

Run: python src/app.py
URL: http://127.0.0.1:7860
"""
import os
import sys
import time

import torch
import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ensure project root on path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_label_names
from src.utils import get_device

MODEL_DIR = "experiments/distilbert_epochs5/best_model"
TOP_K = 3

# --- module-level model load (once at startup) ---

if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(
        f"Model directory not found: {MODEL_DIR}\n"
        "Re-run Phase 5 sweep (distilbert_epochs5) to generate best_model/."
    )

_device = get_device()
_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
_model.to(_device)
_model.eval()

# use actual intent names (model.config.id2label contains default LABEL_N)
_label_names = get_label_names()


def predict(text: str) -> tuple[str, dict]:
    """Return (top-1 intent label, {label: prob} dict for top-K)."""
    text = text.strip()
    if not text:
        return ("(empty input)", {})

    inputs = _tokenizer(
        text,
        max_length=128,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.inference_mode():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits[0], dim=-1)

    topk_probs, topk_ids = torch.topk(probs, k=TOP_K)
    topk_probs = topk_probs.cpu().tolist()
    topk_ids = topk_ids.cpu().tolist()

    top1 = _label_names[topk_ids[0]]
    confidences = {_label_names[idx]: float(p) for idx, p in zip(topk_ids, topk_probs)}
    return (top1, confidences)


# --- Gradio interface ---

with gr.Blocks(title="Banking77 Intent Classifier") as demo:
    gr.Markdown(
        "## Banking77 Intent Classifier\n"
        "Model: **distilbert_epochs5** · Test macro-F1: **0.9307** · 77 banking intents\n\n"
        "Enter a banking-related query to predict the intent."
    )

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Banking query",
                lines=2,
                placeholder="e.g. I lost my card, what should I do?",
            )
            predict_btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            top1_out = gr.Textbox(label="Top-1 intent")
            topk_out = gr.Label(label="Top-3 confidences", num_top_classes=TOP_K)

    gr.Examples(
        examples=[
            ["I lost my card, what should I do?"],
            ["What's the exchange rate today?"],
            ["My transfer is still pending after 3 days"],
            ["How do I verify my identity?"],
        ],
        inputs=text_input,
    )

    predict_btn.click(fn=predict, inputs=text_input, outputs=[top1_out, topk_out])
    text_input.submit(fn=predict, inputs=text_input, outputs=[top1_out, topk_out])


if __name__ == "__main__":
    print(f"Device: {_device}")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
