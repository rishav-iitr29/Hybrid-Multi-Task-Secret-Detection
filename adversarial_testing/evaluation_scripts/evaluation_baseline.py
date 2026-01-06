import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import json
import sys
import os


# CONFIG
MODEL_PATH = "../model_checkpoint/best_model_baseline.pt"
ADVERSARIAL_CSV = "../adversarial_data/adversarial_eval_with_negatives_alternate.csv"
OUTPUT_JSON = "../results_json/adversarial_results_baseline_alternate.json"
BATCH_SIZE = 16
THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD MODEL
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from baseline_model.model import BaselineCodeBERTClassifier

print("Loading baseline model...")

model = BaselineCodeBERTClassifier()
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Handle both formats safely
state_dict = checkpoint.get("model_state_dict", checkpoint)
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

print("Model loaded")


# LOAD DATA
df = pd.read_csv(ADVERSARIAL_CSV)

texts = df["code_snippet"].astype(str).tolist()
labels = df["has_secret"].astype(int).tolist()

print(f"Loaded {len(df)} adversarial samples")
print(f"  Positive: {sum(labels)}")
print(f"  Negative: {len(labels) - sum(labels)}")


# EVALUATION
all_preds = []

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Evaluating baseline"):
    batch_texts = texts[i:i + BATCH_SIZE]

    enc = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        probs = torch.softmax(logits, dim=1)
        preds = (probs[:, 1] >= THRESHOLD).long()

    all_preds.extend(preds.cpu().numpy())


# METRICS
precision = precision_score(labels, all_preds, zero_division=0)
recall = recall_score(labels, all_preds, zero_division=0)
f1 = f1_score(labels, all_preds, zero_division=0)

results = {
    "baseline_classifier": {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    },
    "num_samples": len(labels)
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

print("\nBaseline adversarial evaluation complete\n")
print(results)