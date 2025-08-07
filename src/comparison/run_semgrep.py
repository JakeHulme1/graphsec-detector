#!/usr/bin/env python3
import os
import ast
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
GT_PATH = "src/comparison/command_injection/command_injection_n7_m128_t5_test.jsonl"
SEMGREP_OUT = "src/comparison/command_injection_semgrep/results"
METRIC_OUT = "src/comparison/results"
os.makedirs(SEMGREP_OUT, exist_ok=True)
os.makedirs(METRIC_OUT, exist_ok=True)
SEMGREP_CONFIG = "p/python.security"
# ────────────────────────────────────────────────────────────────

def make_valid_python(token_str: str) -> tuple[str, bool]:
    """
    Try to parse the raw snippet. If it fails, wrap it in a single quoted
    assignment (via repr) so it’s guaranteed valid Python.
    """
    try:
        ast.parse(token_str)
        return token_str, False
    except SyntaxError:
        wrapped = f"snippet = {token_str!r}\n"
        return wrapped, True

def run_semgrep_on_code(code_str: str, idx: int) -> dict:
    """
    Write `code_str` to a temp .py file, run Semgrep on it, return the parsed JSON
    (or {'results':[]} on any error).
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(code_str)
        snippet_path = tmp.name

    out_json = os.path.join(SEMGREP_OUT, f"semgrep_{idx}.json")
    proc = subprocess.run([
        "semgrep",
        "--config", SEMGREP_CONFIG,
        "--json",
        "--output", out_json,
        snippet_path
    ], capture_output=True, text=True)

    # If Semgrep errors (exit≠0), swallow and treat as no findings
    if proc.returncode != 0:
        print(f"[WARN] semgrep idx={idx} exit={proc.returncode}", file=os.sys.stderr)
        print(proc.stderr, file=os.sys.stderr)
        try:
            os.remove(out_json)
        except OSError:
            pass
        findings = {"results": []}
    else:
        try:
            findings = json.load(open(out_json))
        except Exception:
            findings = {"results": []}

    os.remove(snippet_path)
    return findings

def main():
    # ── Load test split ───────────────────────────────────────────
    df = pd.read_json(GT_PATH, lines=True)
    labels = df["label"].astype(int).tolist()

    preds = []
    wrapped_count = 0

    # ── Run Semgrep per snippet ──────────────────────────────────
    for idx, code in enumerate(df["code"]):
        text = code if isinstance(code, str) else str(code)
        snippet, wrapped = make_valid_python(text)
        if wrapped:
            wrapped_count += 1

        data = run_semgrep_on_code(snippet, idx)
        # Semgrep JSON has a "results" list
        flagged = len(data.get("results", [])) > 0
        preds.append(int(flagged))

    # ── Compute metrics ──────────────────────────────────────────
    tn, fp, fn_, tp = confusion_matrix(labels, preds).ravel()
    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    f1        = f1_score(labels, preds, zero_division=0)

    report = [
        f"Dataset: command_injection",
        f"TP={tp}  FP={fp}  TN={tn}  FN={fn_}",
        f"Precision={precision:.3f}",
        f"Recall   ={recall:.3f}",
        f"F1       ={f1:.3f}",
        f"Wrapped  ={wrapped_count}"
    ]

    # ── Print & save ────────────────────────────────────────────
    print("\n".join(report))
    out_path = os.path.join(METRIC_OUT, "command_injection_semgrep_metrics.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"\nMetrics written to {out_path}")

if __name__ == "__main__":
    main()