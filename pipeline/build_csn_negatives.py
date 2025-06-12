#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_csn_negatives.py

Downloads the Python subset of CodeSearchNet, filters and labels each
function‐level example as non‐vulnerable, and writes to JSONL.

Output file: csn_negatives.jsonl
"""

import json
import logging
from pathlib import Path
from datasets import load_dataset

# Config 
MIN_TOKENS = 1
MAX_TOKENS = 1865
OUTPUT_PATH = Path("datasets/csn/raw/csn_negatives.jsonl")

# Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)

def main():
    logging.info("Loading CodeSearchNet (python) dataset…")
    ds = load_dataset("Nan-Do/code-search-net-python")

    n_written = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        # iterate over actual records in each split
        for split_name, subset in ds.items():
            logging.info(f"Processing split {split_name} ({len(subset)} examples)")
            for entry in subset:
                tokens = entry["code_tokens"]
                if not (MIN_TOKENS <= len(tokens) <= MAX_TOKENS):
                    continue

                rec = {
                    "repo":          entry.get("repo", ""),
                    "filepath":      entry.get("path", ""),
                    "sourceWithComments": entry["code"],
                    "is_vulnerable": 0
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_written += 1

    logging.info(f"Done. Wrote {n_written} samples to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
