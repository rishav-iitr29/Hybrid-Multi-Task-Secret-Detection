import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizerFast
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hybrid_model_architecture.hybrid_model import HybridMultiTaskModel
from hybrid_model_architecture.hybrid_dataset import SpanAwareDataset, collate_fn


# CONFIG
MODEL_CHECKPOINT = "../model_checkpoint/best_model_hybrid.pt"
ADVERSARIAL_CSV = "../adversarial_data/adversarial_eval_with_negatives_alternate.csv"
OUTPUT_JSON = "../results_json/adversarial_results_alternate.json"

MAX_LENGTH = 512
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LOAD MODEL
print("Loading model...")
model = HybridMultiTaskModel()
ckpt = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE)
model.eval()

print("Model loaded")


# LOAD TOKENIZER + DATA
tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")

dataset = SpanAwareDataset(
    ADVERSARIAL_CSV,
    tokenizer,
    max_length=MAX_LENGTH
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

print(f"Loaded {len(dataset)} adversarial samples")


# EVALUATION
all_preds = []
all_labels = []

span_hits = []
entropy_errors = []

by_obfuscation = {}

with torch.no_grad():
    for batch in tqdm(loader, desc="Evaluating adversarial set"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        entropy_gt = batch["entropy"].to(DEVICE)

        outputs = model(input_ids, attention_mask)

        # Classification
        preds = torch.argmax(outputs["classification_logits"], dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Span Localization
        span_available = batch["span_available"].to(DEVICE)
        token_start_gt = batch["token_span_start"].to(DEVICE)
        token_end_gt = batch["token_span_end"].to(DEVICE)

        if span_available.any():
            start_pred = torch.argmax(outputs["span_start_logits"], dim=1)
            end_pred = torch.argmax(outputs["span_end_logits"], dim=1)

            hit = (
                (start_pred == token_start_gt) &
                (end_pred == token_end_gt) &
                span_available
            )

            span_hits.extend(hit.cpu().numpy().tolist())

        # Entropy Regression
        entropy_pred = outputs["entropy_pred"]
        entropy_errors.extend(
            torch.abs(entropy_pred - entropy_gt).cpu().numpy().tolist()
        )


# METRICS
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)

span_accuracy = float(np.mean(span_hits)) if span_hits else None
entropy_mae = float(np.mean(entropy_errors))

results = {
    "classification": {
        "precision": precision,
        "recall": recall,
        "f1": f1
    },
    "span_localization": {
        "exact_match_accuracy": span_accuracy,
        "num_evaluated": len(span_hits)
    },
    "entropy_regression": {
        "mae": entropy_mae
    },
    "num_samples": len(all_labels)
}


# SAVE
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print("\nAdversarial evaluation complete")
print(json.dumps(results, indent=2))