# Script usage
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
patch_code.py

Reads structured VUDENC JSONL data, reconstructs patched (fixed) code by replacing badparts with goodparts, and outputs a new JSONL file including a `new_code` field.

Author: Jake Hulme
Date: 09/06/2025
"""

import json 
from pathlib import Path
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def patch_source_code(source: str, badparts: List[str], goodparts: List[str]) -> str:
    """
    Reconstructs the patched version of the code by replacing badparts with goodparts.

    Args:
        source (str): Original vulnerable source code.
        badparts (List[str]): Lines that were removed in the fix.
        goodparts (List[str]): Lines that replaced them

    Returns:
        str: Patched (new) version of source code 
    """
    lines = source.splitlines()
    patched_lines = []
    replaced = set()

    for line in lines:
        if badparts and line in badparts and line not in replaced:
            index = badparts.index(line)
            if index < len(goodparts):
                patched_lines.append(goodparts[index])
                replaced.add(line)
            else:
                logging.warning("Mismatch: more badparts than goodparts, keeping original line.")
                patched_lines.append(line)

        else:
            patched_lines.append(line)

    return "\n".join(patched_lines)
    
def patch_dataset(input_path: Path, output_path: Path) -> None:
    """
    Loads the raw dataset, applies patching to generate `new_code`, and wries to a new JSONL

    Args:
        input_path (Path): Path to `vudenc_raw.jsonl`
        output_path (Path): Path to `vudenc_patched.jsonl`
    """
    count_total = 0
    count_patched = 0
    count_skipped = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            entry = json.loads(line)
            count_total += 1

            sourceWithComments = entry.get("sourceWithComments", "")
            badparts = entry.get("badparts", "")
            goodparts = entry.get("goodparts", "")

            if not (badparts and goodparts):
                logging.info(f"Skipping patch (no diff): {entry.get('filepath')}, commit {entry.get('commit')}")
                entry["new_code"] = sourceWithComments
                count_skipped += 1
            else:
                new_code = patch_source_code(sourceWithComments, badparts, goodparts)
                entry["new_code"] = new_code
                count_patched += 1

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logging.info(f"Patched dataset writte to {output_path}")
        logging.info(f"Total entries: {count_total}")
        logging.info(f" - Patched: {count_patched}")
        logging.info(f" - Skipped: {count_skipped}")

if __name__ == "__main__":
    input_file = Path("datasets/vudenc/raw/vudenc_raw.jsonl")
    output_file = Path("datasets/vudenc/raw/vudenc_patched.jsonl")

    patch_dataset(input_file, output_file)