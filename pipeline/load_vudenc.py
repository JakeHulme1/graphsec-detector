# Script usage:
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
load_vudenc.py

Reads vulnerability entries from VUDENC dataset `.txt` files, extracts key fields, and saves them to a JSONL file for downstream use.

Author: Jake Hulme
Date: 09/06/2025
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Generator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_vudenc_file(file_path: Path, label: str, cwe_id: str) -> Generator[Dict, None, None]:
    """
    Parses a single VUDENC `.txt` file and creates structured samples.

    Args:
        file_path (Path): Path to the .txt file.
        label (str): Human-readable vulnerability label.
        cwe_id (str): CWE ID.

    Yield:
            Dict: Structured vulnerability entry.
    """
    with file_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # For debugging
    missing_count = 0

    for repo_url, commits in raw_data.items():
        for sha, commit_data in commits.items():
            files = commit_data.get("files", {})
            for filepath, filedata in files.items():
                # Extract the list of changes including "goodparts" and "badparts"
                changes = filedata.get("changes", [])
                first_change = changes[0] if changes else {}
                sample = {
                    "repo": repo_url,
                    "commit": sha,
                    "filepath": filepath,
                    "label": label,
                    "cwe_id": cwe_id,
                    "sourceWithComments": filedata.get("sourceWithComments", ""),
                    "badparts": first_change.get("badparts", []),
                    "goodparts": first_change.get("goodparts", [])
                }
                

                if not first_change.get("badparts"):
                    logging.warning(f"No badparts for {file_path.name} in commit {sha}")
                    missing_count += 1
                

                # Use yield to save memory and add scalability - avoids loading data all at once
                yield sample

    logging.info(f"{file_path.name}: {missing_count} entries missing badparts")


def load_all_vudenc(dir_path: Path, cwe_mapping: Dict[str, str]) -> List[Dict]:
    """
    Loads and parses all VUDENC `.txt` files in a directory.

    Args:
        dir_path (path): Directory containing all VUDENC .txt files
        cwe_mapping (Dict[str, str]): Mapping from labels to CWE IDs

    Returns:
        List[Dict]: List of structured vulnerability entries
    """
    results = []
    for file_path in dir_path.glob("*.txt"):
        # Normalise .txt filename and get the cwe_id
        label = file_path.stem.replace("plain", "").replace("_", " ").strip().lower()
        cwe_id = cwe_mapping.get(label)

        if not cwe_id:
            logging.warning(f"Skipping file {file_path.name} — no CWE mapping for label '{label}'")
            continue

        # Add structured vulnerability entry by running lead_vudenc_file() function
        logging.info(f"Processing {file_path.name} (label: {label}, CWE: {cwe_id})")
        results.extend(load_vudenc_file(file_path, label, cwe_id))

    return results
    
def save_to_jsonl(data: List[Dict], output_path: Path) -> None:
    """
    Saves a list of dictionaries to a JSON Lines (.jsonl) file.

    Args:
        data (List[Dict]): List of dictionaries to write
        output_path (Path): Path to output .jsonl file. 
    """
    with output_path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logging.info(f"Saved {len(data)} samples to {output_path}")

if __name__ == "__main__":
    CWE_MAPPING = {
        "command injection": "CWE-77",
        "open redirect": "CWE-601",
        "path disclosure": "CWE-200",
        "remote code execution": "CWE-94",
        "sql": "CWE-89",
        "xsrf": "CWE-352",
        "xss": "CWE-79"
    }

    dataset_dir = Path("datasets/vudenc/raw")
    output_file = Path("datasets/vudenc/raw/vudenc_raw.jsonl")

    all_samples = load_all_vudenc(dataset_dir, CWE_MAPPING)
    save_to_jsonl(all_samples, output_file)

    logging.info(f"\nDone. Total entries processed: {len(all_samples)}")