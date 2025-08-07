#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import tempfile
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ── CONFIG ───────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
GT_PATH     = os.path.join(HERE, "command_injection", "command_injection_n7_m128_t5_test.jsonl")
RULE_FILE   = os.path.abspath(os.path.join(HERE, "../../ci_rules/command_injection.yml"))
SEMGREP_OUT = os.path.join(HERE, "command_injection_semgrep", "results")
METRIC_OUT  = os.path.join(HERE, "results")

for d in (SEMGREP_OUT, METRIC_OUT):
    os.makedirs(d, exist_ok=True)

if not os.path.isfile(RULE_FILE):
    print(f"[ERROR] cannot find rule file at {RULE_FILE}", file=sys.stderr)
    sys.exit(1)
# ────────────────────────────────────────────────────────────────

def run_semgrep_on_code(text: str, idx: int) -> dict:
    # Dump snippet to a real .py file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(text)
        snippet_path = tmp.name

    out_json = os.path.join(SEMGREP_OUT, f"semgrep_{idx}.json")
    cmd = [
        "semgrep",         # regex‐only mode
        "--config", RULE_FILE,         
        "--json",
        "--output", out_json,
        snippet_path
    ]
    # Debug: print the exact command
    print(f"[DEBUG] {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode >= 2:
        # real error—print stderr for diagnosis
        print(f"[WARN] semgrep idx={idx} exit={proc.returncode}", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        findings = {"results": []}
    else:
        # exit 0=no matches or 1=matches both produce JSON
        try:
            findings = json.load(open(out_json))
        except Exception:
            findings = {"results": []}

    os.remove(snippet_path)
    return findings

def main():
    df     = pd.read_json(GT_PATH, lines=True)
    labels = df["label"].astype(int).tolist()
    preds  = []

    for idx, code in enumerate(df["code"]):
        text = code if isinstance(code, str) else str(code)
        data = run_semgrep_on_code(text, idx)
        preds.append(int(bool(data.get("results"))))

    tn, fp, fn_, tp = confusion_matrix(labels, preds).ravel()
    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    f1        = f1_score(labels, preds, zero_division=0)

    report = [
        "Dataset: command_injection",
        f"TP={tp}  FP={fp}  TN={tn}  FN={fn_}",
        f"Precision={precision:.3f}",
        f"Recall   ={recall:.3f}",
        f"F1       ={f1:.3f}"
    ]
    print("\n".join(report))

    out_path = os.path.join(METRIC_OUT, "command_injection_semgrep_metrics.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"\nMetrics written to {out_path}")

if __name__ == "__main__":
    main()
