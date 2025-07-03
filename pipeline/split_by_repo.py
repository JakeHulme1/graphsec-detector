#!/usr/bin/env python3

"""
split_by_repo.py

This script takes a labeled .jsonl file and produces
train/val/test splits with strict repo-exclusivity using
a two-phase assignment method.
"""

import json
import random
import argparse
from collections import defaultdict, Counter
from pathlib import Path


def count_repo_stats(input_path: Path) -> defaultdict:
    """
    Collect per-repo statistics:
      - total examples
      - positive examples (label==1)
    """
    stats = defaultdict(lambda: {"total": 0, "pos": 0})
    with open(input_path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            s = stats[rec["repo"]]
            s["total"] += 1
            s["pos"]   += rec.get("label", 0)
    return stats


def assign_label_balanced(repo_stats: dict,
                          train_frac: float,
                          val_frac: float,
                          seed: int) -> dict:
    """
    Two-phase split:
      A) distribute repos with positives by positive count
      B) distribute repos with zero positives by total size
    """
    random.seed(seed)

    # 1) Compute global totals and numeric targets
    N_all = sum(s["total"] for s in repo_stats.values())
    P_all = sum(s["pos"]   for s in repo_stats.values())
    test_frac = 1.0 - train_frac - val_frac

    N_tgt = {
        "train": train_frac * N_all,
        "val":   val_frac   * N_all,
        "test":  test_frac  * N_all,
    }
    P_tgt = {
        "train": train_frac * P_all,
        "val":   val_frac   * P_all,
        "test":  test_frac  * P_all,
    }

    splits = ["train", "val", "test"]
    repo2split = {}

    # Phase A: assign positive-containing repos
    pos_repos = [r for r, s in repo_stats.items() if s["pos"] > 0]
    pos_repos.sort(key=lambda r: repo_stats[r]["pos"], reverse=True)
    accP = {s: 0.0 for s in splits}
    for r in pos_repos:
        # pick the split most under its positive target
        best = min(splits, key=lambda sp: accP[sp] / (P_tgt[sp] or 1e-8))
        repo2split[r] = best
        accP[best] += repo_stats[r]["pos"]

    # Phase B: assign negative-only repos
    neg_repos = [r for r, s in repo_stats.items() if s["pos"] == 0]
    neg_repos.sort(key=lambda r: repo_stats[r]["total"], reverse=True)
    # initialize total-size accumulators from Phase A
    accN = {s: 0.0 for s in splits}
    for r, sp in repo2split.items():
        accN[sp] += repo_stats[r]["total"]
    for r in neg_repos:
        best = min(splits, key=lambda sp: accN[sp] / (N_tgt[sp] or 1e-8))
        repo2split[r] = best
        accN[best] += repo_stats[r]["total"]

    # Summary
    print("[SPLIT SUMMARY]")
    for sp in splits:
        tot = int(accN[sp])
        pos = int(accP.get(sp, 0))
        neg = tot - pos
        pct = 100 * pos / tot if tot else 0.0
        print(f"  {sp:<5} total={tot:6d}  pos={pos:6d} ({pct:.1f}%)  neg={neg:6d}")

    return repo2split


def split_records(input_path: Path, repo2split: dict, out_dir: Path) -> None:
    """
    Stream input .jsonl and write each record to its assigned split file.
    """
    input_path = Path(input_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vuln = input_path.stem
    writers = {}
    for split in ["train", "val", "test"]:
        fn = out_dir / f"{vuln}_{split}.jsonl"
        print(f"[DEBUG] opening {split} -> {fn}")
        writers[split] = open(fn, 'w')

    with open(input_path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            repo = rec.get("repo")
            split = repo2split.get(repo)
            if split:
                writers[split].write(line)

    for w in writers.values():
        w.close()

# Buffer all lines per split
    buffers = {"train": [], "val": [], "test": []}
    with open(input_path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            sp  = repo2split.get(rec["repo"])
            if sp in buffers:
                buffers[sp].append(line)

    # Shuffle and write out
    for split, lines in buffers.items():
        random.seed(42)        # or args.seed
        random.shuffle(lines)
        fn = out_dir / f"{vuln}_{split}.jsonl"
        print(f"[DEBUG] writing {len(lines)} lines to {fn}")
        with open(fn, 'w') as f_out:
            f_out.writelines(lines)


def main():
    p = argparse.ArgumentParser(
        description="Repo-exclusive, two-phase JSONL splitter"
    )
    p.add_argument("input",    help="path to .jsonl being split")
    p.add_argument("out_dir",  help="directory for train/val/test files")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val-frac",   type=float, default=0.15)
    p.add_argument("--seed",       type=int,   default=42)
    args = p.parse_args()

    repo_stats = count_repo_stats(Path(args.input))
    repo2split  = assign_label_balanced(
        repo_stats,
        args.train_frac,
        args.val_frac,
        args.seed
    )
    print(Counter(repo2split.values()))
    split_records(Path(args.input), repo2split, Path(args.out_dir))
    print(f"Done! Wrote train/val/test to {args.out_dir}")


if __name__ == '__main__':
    main()
