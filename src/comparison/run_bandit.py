#!/usr/bin/env python3
import os
import json
import subprocess
import tempfile
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score)

# --- CONFIG ---
GT_PATH = "datasets/vudenc/prepared/sql/test.jsonl"
RESULT_DIR = "src/comparison/sql_bandit/"
os.makedir(RESULT_DIR, exist_ok=True)
# --- CONFIG ---

def run_bandit_on_code(code: str, idx: int) -> str:
    """Write `code` to a temp file, run Bandit on it, write JSON to RESULT_DIR, return JSON path."""
    # Dump snippet
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        snippet_path = tmp.name

    # Run bandit
    out_json = os.path.join(RESULT_DIR, f"bandit_{idx}.json")
    subprocess.run([
        "bandit", "-q",   
        "-f", "json",                
        "-o", out_json,              
        snippet_path
    ], check=True)

    # Clean up
    os.remove(snippet_path)
    return out_json

def main():
    # Load ground truth
    gt = pd.read_json(GT_PATH, lines=True)
    labels = gt["label"].astype(int).tolist()

    # Run bandit and collect predictions
    preds = []
    for idx, code in enumerate([gt["code"]]):
        json_path = run_bandit_on_code(code, idx)
        data = json.load(open(json_path))
        issues = data.get("results", [])
        # flag = 1 if any MEDIUM/HIGH issue
        flagged = any(
            issue.get("issure_severity") in ("MEDIUM", "HIGH")
            for issue in issues
        )
        preds.append(int(flagged))

    # Compute metrics
    tn, fp, fn_, tp = confusion_matrix(labels, preds).ravel()
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    report_lines = [
        f"TP={tp}  FP={fp}  TN={tn}  FN={fn_}",
        f"Precision={precision:.3f}",
        f"Recall   ={recall:.3f}",
        f"F1       ={f1:.3f}",
    ]

    # Print and save
    print("\n".join(report_lines))
    metrics_path = os.path.join(RESULT_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nMetrics written to {metrics_path}")

if __name__ == "__main__":
    main()