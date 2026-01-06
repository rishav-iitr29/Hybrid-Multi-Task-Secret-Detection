import csv
import subprocess
import tempfile
import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score
import signal
signal.signal(signal.SIGPIPE, signal.SIG_DFL)

# CONFIG
INPUT_CSV = "../adversarial_data/adversarial_eval_with_negatives_alternate.csv"

GITLEAKS_CMD = ["gitleaks", "detect", "--no-git", "--source"]
TRUFFLEHOG_CMD = ["trufflehog", "filesystem"]

# UTILS
def run_tool(cmd, snippet: str) -> bool:
    """
    Returns True if tool flags the snippet
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(snippet)
        path = f.name

    try:
        result = subprocess.run(
            cmd + [path],
            capture_output=True,
            text=True
        )
        return result.returncode != 0
    finally:
        os.remove(path)

def run_trufflehog(snippet: str) -> bool:
    """
    Returns True if TruffleHog flags the snippet
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.txt")
        with open(file_path, "w") as f:
            f.write(snippet)

        try:
            result = subprocess.run(
                [
                    "trufflehog",
                    "filesystem",
                    tmpdir,
                    "--json",
                    "--no-update",
                    "--only-verified"
                ],
                capture_output=True,
                text=True,
                timeout=20
            )

            # If any JSON output exists → secret found
            return bool(result.stdout.strip())

        except subprocess.TimeoutExpired:
            # Treat timeout as no finding (conservative)
            return False

# ============================================================
# EVALUATION
# ============================================================

def evaluate_tool(tool_name, cmd, rows):
    y_true = []
    y_pred = []

    for r in rows:
        flagged = run_tool(cmd, r["code_snippet"])
        y_pred.append(1 if flagged else 0)
        y_true.append(int(r["has_secret"]))

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ============================================================
# LOAD DATA
# ============================================================

with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Loaded {len(rows)} adversarial samples")

# ============================================================
# RUN EVALUATIONS
# ============================================================

results = {}

print("\nRunning Gitleaks evaluation...")
results["gitleaks"] = evaluate_tool(
    "gitleaks",
    GITLEAKS_CMD,
    rows
)

def evaluate_trufflehog(rows):
    y_true, y_pred = [], []

    for r in rows:
        flagged = run_trufflehog(r["code_snippet"])
        y_pred.append(1 if flagged else 0)
        y_true.append(int(r["has_secret"]))

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }

print("Running TruffleHog evaluation...")
results["trufflehog"] = evaluate_trufflehog(rows)

# ============================================================
# OUTPUT
# ============================================================

print("\n✓ Regex Tool Evaluation Complete\n")
print(json.dumps(results, indent=2))

with open("../results_json/regex_tool_results_alternate.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved results → regex_tool_results.json")