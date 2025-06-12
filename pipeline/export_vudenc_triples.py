#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_vudenc_triples.py

Reads VUDENC `.txt` files, then writes to 3 JSONL files:
    - files.jsonl - one entry per file change
    - commits.jsonl - one entry per commit, embedding its files
    - repos.jsonl - one entry per repo, embedding its commits

Usage:
    python pipeline/export_vudenc_triples.py \
        -- input-dir datasets/vudenc/raw \
        -- output-dir datasets/vudenc/raw

Author: Jake Hulme
Date: 11/06/2025
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)

# Helper
def aggregate_file_changes(filepath: str, filedata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given the raw 'files entry for one filepath, merge all hunks into a single summary

    Returns a dict with:
        - filepath
        - add (sum of adds)
        - remove (sum of removes)
        - badparts
        - goodparts
        - source (both commented and non-commented)
    """
    total_add = 0
    total_remove = 0
    all_bad :List[str] = []
    all_good: List[str] = []

    for hunk in filedata.get("changes", []):
        total_add += hunk.get("add", 0)
        total_remove += hunk.get("remove", 0)
        all_bad.extend(hunk.get("badparts", [])) # Use extend as all_bad is of type List which is an iterable
        all_good.extend(hunk.get("goodparts", []))

    return {
        "filepath": filepath.lstrip("/"), # Remove leading '/' as AST requires relative paths
        "add": total_add,
        "remove": total_remove,
        "badparts": all_bad,
        "goodparts": all_good,
        "source": filedata.get("source", ""),
        "sourceWithComments": filedata.get("sourceWithComments", "")
    }


# Loader
def load_vudenc_triples(input_dir: Path, cwe_mapping: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Scan all `.txt` files under input_dir, parse them and return 3 lists:
        - files_list - all file-level records with repo url key
        - commits_list - all commit-level records
        - repos_map: mapping repo_url -> list of its commit records
    """
    files_list: List[Dict[str, Any]] = []
    commits_list: List[Dict[str, Any]] = []
    repos_map: Dict[str, List[Dict[str, Any]]] = {}

    # Create label for JSON, and find corresponding CWE ID
    for txt_path in sorted(input_dir.glob("*.txt")):
        label = txt_path.stem.replace("plain", "").replace("_", " ").strip().lower() # Normalise labels
        cwe_id = cwe_mapping.get(label)
        if not cwe_id:
            logging.warning(f"Skipping {txt_path.name}: no CWE for {cwe_id}")
            continue

        logging.info(f"Loading {txt_path.name} (label='{label}, CWE='{cwe_id}')")
        raw = json.loads(txt_path.read_text(encoding="utf-8"))

        for repo_url, commits in raw.items():
            # ensure repo in map
            repo_commits = repos_map.setdefault(repo_url, [])

            for sha, commit_data in commits.items():
                # build file summaries
                file_summaries = []
                for file_path, file_data in commit_data.get("files", {}).items():
                    summary= aggregate_file_changes(file_path, file_data)
                    # attach the label/cwe/info
                    summary.update({
                        "repo": repo_url,
                        "commit": sha,
                        "label": label,
                        "cwe_id": cwe_id,
                        "is_vulnerable": 1 # mark as vulnerable
                    })
                    files_list.append(summary)
                    file_summaries.append(summary)

                # build commit record
                total_add = sum(f["add"] for f in file_summaries)
                total_remove = sum(f["remove"] for f in file_summaries)

                commit_rec: Dict[str, Any] = {
                    "repo": repo_url,
                    "commit": sha,
                    "label": label,
                    "cwe_id": cwe_id,
                    "keyword": commit_data.get("keyword", ""),
                    "message": commit_data.get("msg", ""),
                    "diff": commit_data.get("diff", ""),
                    "files": [
                        {
                            # this only includes per-file stats, not full source code
                            "file": f["filepath"],
                            "add": f["add"],
                            "remove": f["remove"],
                            "badparts": f["badparts"],
                            "goodparts": f["goodparts"]
                        }
                        for f in file_summaries
                    ],
                    "total_add": total_add,
                    "total_remove": total_remove
                }

                commits_list.append(commit_rec)
                repo_commits.append(commit_rec)

    return {
        "files": files_list,
        "commits": commits_list,
        "repos": repos_map
    }

# JSONL writer
def dump_jsonl(records:List[Dict[str, Any]], path: Path) -> None:
    """
    Write a list of dicts to the given .jsonl file, one JSON object per line
    """
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logging.info(f"Wrote {len(records)} records to {path.name}")

# JSON writer
def dump_repos_jsonl(repos_map: Dict[str, List[Dict[str, Any]]], path: Path) -> None:
    """
    Convert each repo_url -> commits list into a single record
    with aggregates, and write to JSONL.
    """
    out: List[Dict[str, Any]] = []
    for repo_url, commits in repos_map.items():
        total_commits = len(commits)
        total_files = sum(len(c["files"]) for c in commits)
        total_add = sum(c["total_add"] for c in commits)
        total_remove = sum(c["total_remove"] for c in commits)

        out.append({
            "repo": repo_url,
            "total_commits": total_commits,
            "total_files_changed": total_files,
            "total_add": total_add,
            "total_remove": total_remove,
            "commits": commits
        })

    with path.open("w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logging.info(f"Wrote {len(out)} repo records to {path.name}")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Export VUDENC dataset to file/commit/repo JSONL triples"
    )
    p.add_argument(
        "--input-dir", "-i", type=Path, required=True,
        help="Directory containing VUDENC .txt files"
    )
    p.add_argument(
        "--output-dir", "-o", type=Path, required=True,
        help="Directory to write files.jsonl, commits.jsonl, repos.jsonl"
    )
    args = p.parse_args()

    CWE_MAPPING = {
        "command injection": "CWE-77",
        "open redirect":      "CWE-601",
        "path disclosure":    "CWE-200",
        "remote code execution":"CWE-94",
        "sql":                "CWE-89",
        "xsrf":               "CWE-352",
        "xss":                "CWE-79"
    }

    outputs = load_vudenc_triples(args.input_dir, CWE_MAPPING)

    # dump the 3 jsonl files
    dump_jsonl(outputs["files"], args.output_dir/"files.jsonl")
    dump_jsonl(outputs["commits"], args.output_dir / "commits.jsonl")
    dump_repos_jsonl(outputs["repos"], args.output_dir / "repos.jsonl")

logging.info("Finished!")