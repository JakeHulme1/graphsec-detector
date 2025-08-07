#!/usr/bin/env python3
import os
import re
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
GT_PATH    = "datasets/vudenc/prepared/sql/test.jsonl"
BANDIT_OUT = "src/comparison/sql_bandit/results"
METRIC_OUT = "src/comparison/results"
os.makedirs(BANDIT_OUT, exist_ok=True)
os.makedirs(METRIC_OUT, exist_ok=True)
# ────────────────────────────────────────────────────────────────

def make_valid_python(token_str: str) -> tuple[str, bool]:
    # 1) split out import/from onto their own lines
    code = re.sub(r'\b(import|from)\b', r'\n\1', token_str)
    code = "\n".join(ln.strip() for ln in code.splitlines() if ln.strip())

    # 2) try AST parse
    try:
        ast.parse(code)
        return code, False
    except SyntaxError:
        # 3) fallback: wrap in dummy function
        wrapped = "def _dummy():\n" + "\n".join(f"    {ln}" for ln in code.splitlines())
        return wrapped, True

def run_bandit_on_code(code_str: str, idx: int) -> dict:
    """Write code_str to a temp file, run Bandit, return parsed JSON (or empty on failure)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(code_str)
        snippet_path = tmp.name

    out_json = os.path.join(BANDIT_OUT, f"bandit_{idx}.json")
    res = subprocess.run(
        ["bandit", "-q", "-f", "json", "-o", out_json, snippet_path],
        capture_output=True, text=True
    )

    if res.returncode != 0:
        print(f"[WARN] bandit idx={idx} exit={res.returncode}", file=os.sys.stderr)
        print(res.stderr, file=os.sys.stderr)
        try:
            os.remove(out_json)
        except OSError:
            pass
        findings = {"results": []}
    else:
        with open(out_json) as f:
            findings = json.load(f)

    os.remove(snippet_path)
    return findings

def main():
    gt = pd.read_json(GT_PATH, lines=True)
    labels = gt["label"].astype(int).tolist()
    dataset = os.path.basename(os.path.dirname(GT_PATH))

    preds = []
    wrapper_count = 0

    for idx, row in gt.iterrows():
        raw = row["code"]
        text = raw if isinstance(raw, str) else str(raw)
        snippet, wrapped = make_valid_python(text)
        if wrapped:
            wrapper_count += 1

        data = run_bandit_on_code(snippet, idx)
        issues = data.get("results", [])
        flagged = any(issue.get("issue_severity") in ("MEDIUM", "HIGH") for issue in issues)
        preds.append(int(flagged))

    tn, fp, fn_, tp = confusion_matrix(labels, preds).ravel()
    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    f1        = f1_score(labels, preds, zero_division=0)

    report_lines = [
        f"Dataset: {dataset}",
        f"TP={tp}  FP={fp}  TN={tn}  FN={fn_}",
        f"Precision={precision:.3f}",
        f"Recall   ={recall:.3f}",
        f"F1       ={f1:.3f}",
        f"Wrapped  ={wrapper_count}",
    ]

    print("\n".join(report_lines))
    metrics_file = os.path.join(METRIC_OUT, f"{dataset}_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\nMetrics written to {metrics_file}")

if __name__ == "__main__":
    main()
