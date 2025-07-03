#!/usr/bin/env python3
"""
Summarise grid-search log files and compute the true positive rate.
"""
from pathlib import Path
import argparse, glob, re, pandas as pd

# ─── CLI ──────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--base-dir", default="datasets/vudenc/processed",
                help="Root folder that contains generated JSONLs")
ap.add_argument("--vuln", default=None,
                help="Only summarise logs whose tag starts with this prefix")
args = ap.parse_args()

BASE = Path(args.base_dir)

# ─── regexes for the [STATS] block in each log ────────────────────────
rx_keep = re.compile(r'windows kept\s*:\s*([\d,]+)\s*/\s*([\d,]+)')
rx_sp   = re.compile(r'sparse positives\s*:\s*([\d,]+)')
rx_dup  = re.compile(r'duplicate windows\s*:\s*([\d,]+)')
rx_long = re.compile(r'>\d+ subtoks\s*:\s*([\d,]+)')

def find_jsonl(vuln_full: str, params: str) -> Path | None:
    """
    Flat layout resolver.

    Tries two file names inside the folder:
      1) <first-word>.jsonl           e.g.  path.jsonl
      2) <vuln_full>.jsonl            e.g.  path_disclosure.jsonl
    """
    folder = BASE / f"{vuln_full}_{params}"

    # 1) first word (works for sql, …)
    first = vuln_full.split('_', 1)[0]
    p1 = folder / f"{first}.jsonl"
    if p1.exists():
        return p1

    # 2) full vuln name (needed for path_disclosure, command_injection)
    p2 = folder / f"{vuln_full}.jsonl"
    if p2.exists():
        return p2

    return None

rows = []

for log_path in glob.glob("logs/grid_snippets/*.log"):
    tag = Path(log_path).stem                         # e.g. command_injection_n7_m128_t0
    if args.vuln and not tag.startswith(args.vuln):
        continue                                      # skip others if --vuln set

    text   = Path(log_path).read_text()

    kept, total = map(lambda s: int(s.replace(",", "")),
                      rx_keep.search(text).groups())
    d_sparse = int(rx_sp.search(text).group(1).replace(",", ""))
    d_dup    = int(rx_dup.search(text).group(1).replace(",", ""))
    d_long   = int(rx_long.search(text).group(1).replace(",", ""))

    *vuln_parts, nflag, mflag, tflag = tag.split('_')
    vuln_full = "_".join(vuln_parts)                  # command_injection
    params    = "_".join([nflag, mflag, tflag])       # n7_m128_t0

    jsonl = find_jsonl(vuln_full, params)
    print("DEBUG", jsonl)

    if jsonl is None:
        print(f"[WARN] JSONL not found for {tag}  "
              f"(looked for { (BASE / f'{vuln_full}_{params}' / f'{vuln_full.split('_',1)[0]}.jsonl') })")
        pos_kept = pos_rate = None
    else:
        pos_kept = sum(1 for line in jsonl.open(encoding="utf-8")
                       if '"label": 1' in line)
        pos_rate = round(pos_kept / kept, 6) if kept else 0

    rows.append((tag, total, kept, d_dup, d_sparse, d_long, pos_kept, pos_rate))

# ─── build & print table ───────────────────────────────────────────────
df = pd.DataFrame(rows, columns=[
    "tag","total","kept","d_dupes","d_sparse","d_long","pos_kept","pos_rate"
]).sort_values("kept", ascending=False)

pd.set_option('display.max_rows', None)
print(df.to_string(index=False))