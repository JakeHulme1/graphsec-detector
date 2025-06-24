#!/usr/bin/env pyhton3

"""
split_by_repo.py

This script takes a labeled .jsonl file of vulnerabilities and produces
train/val/test splits with *no* repository leakage and *label-balanced*
fractions (70/15/15 by default).

Author: Jake Hulme
Date: 17/06/2025
"""

import json, random, argparse
from collections import defaultdict, Counter
from pathlib import Path

def count_repo_stats(input_path: Path) -> defaultdict:
    """
    Stream the input file once to collect per-repo statistics:
        - total number of examples in each repo
        - number of positive (label==1) examples in each repo

    Returns:
        repo_stats: { repo_url: {"total": int, "pos": int} }
    """
    stats = defaultdict(lambda: {"total": 0, "pos": 0})
    with open(input_path) as f:
        for line in f:
            rec = json.loads(line)
            s = stats[rec["repo"]]
            # increment total counnt for this repo
            s["total"] += 1
            # increment positive count if label == 1
            s["pos"] += rec["label"]
    return stats

# def assign_label_balanced(repo_stats: defaultdict, train_frac: float, val_frac: float, seed: int) -> set:
#     """
#     Greedy assignment of repos to train/val/test to match total + label proportions.
#     Prioritizes total count matching to avoid under-allocation (e.g., empty train set).

#     Returns:
#         repo2split: {repo_url: "train"|"val"|"test"}
#     """
#     # compute global targets
#     G_total = sum(s["total"] for s in repo_stats.values())
#     G_pos = sum(s["pos"] for s in repo_stats.values())

#     # define fractions for all 3 splits
#     test_frac = 1.0-train_frac-val_frac

#     # compute targets for each split
#     targets = {
#         "train": {"total": train_frac * G_total, "pos": train_frac * G_pos},
#         "val": {"total": val_frac * G_total, "pos": val_frac * G_pos},
#         "test": {"total": test_frac * G_total,"pos": test_frac * G_pos}
#     }

#     # running accumulators for each split
#     acc = {split: {"total": 0, "pos": 0} for split in targets}

#     # shuffle repos
#     items = list(repo_stats.items()) # [(repo_url), {total, pos}), ...]
#     random.Random(seed).shuffle(items)

#     repo2split = {}
#     # assign each repo in turn
#     for repo, s in items:
#         best_split, best_err = None, float("inf")
#         # consider each split for this repo
#         for split, tgt in targets.items():
#             # hypothetical new totals
#             new_tot = acc[split]["total"] + s["total"]
#             new_pos = acc[split]["pos"] + s["pos"]
#             # compute squared error against target
#             dt = new_tot - tgt["total"]
#             dp = new_pos - tgt["pos"]
#             err = dt*dt + dp*dp
#             if err < best_err:
#                 best_err, best_split = err, split
#         repo2split[repo] = best_split

#         # update accumulators
#         acc[best_split]["total"] += s["total"]
#         acc[best_split]["pos"]   += s["pos"]

#     return repo2split

def assign_label_balanced(repo_stats: defaultdict, train_frac: float, val_frac: float, seed: int, input_path: Path) -> dict:
    items = list(repo_stats.items())
    random.Random(seed).shuffle(items)

    total_repos = len(items)
    n_train = int(train_frac * total_repos)
    n_val = int(val_frac * total_repos)

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]

    repo2split = {}
    acc = {"train": {"total": 0, "pos": 0}, "val": {"total": 0, "pos": 0}, "test": {"total": 0, "pos": 0}}

    for repo, stats in train_items:
        repo2split[repo] = "train"
        acc["train"]["total"] += stats["total"]
        acc["train"]["pos"] += stats["pos"]

    for repo, stats in val_items:
        repo2split[repo] = "val"
        acc["val"]["total"] += stats["total"]
        acc["val"]["pos"] += stats["pos"]

    for repo, stats in test_items:
        repo2split[repo] = "test"
        acc["test"]["total"] += stats["total"]
        acc["test"]["pos"] += stats["pos"]

    print(f"[SUMMARY] {input_path.name}:")
    for split in ("train", "val", "test"):
        total = acc[split]["total"]
        pos = acc[split]["pos"]
        neg = total - pos
        pct = 100 * pos / total if total else 0
        print(f"  {split:<5} total={total:6d}  pos={pos:6d} ({pct:.1f}%)  neg={neg:6d}")

    return repo2split



def split_records(input_path: Path, repo2split: dict, out_dir: Path) -> None:
    """
    Stream the input file a second time and dump each record
    to its assigned split file (train/val/test)
    """
    # Force to path (was having errors here)
    input_path = Path(input_path)
    out_dir    = Path(out_dir)
    # make sure out_dir exists
    out_dir.mkdir(parents=True, exist_ok=True) 

    # get vulnerability type for filename
    vuln = input_path.stem
    
    # build the writers
    writers = {}
    missing = set()
    for split in ("train", "val", "test"):

        fn = out_dir / f"{vuln}_{split}.jsonl"
        print(f"[DEBUG] opening split {split} -> {fn}")
        writers[split] = open(fn, "w")

    # stream and write
    with open(input_path, "r") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            repo = rec["repo"]
            split = repo2split.get(repo)
            if split is None:
                print(f"[WARNING] no split for repo={repo!r}, skipping")
                continue

            # write the json in corrext split
            writers[split].write(line)

        # close
        for w in writers.values():
            w.close()

def main():
    p = argparse.ArgumentParser(
        description="Repo-exclusive, label-balanced JSONL splitter"
    )
    p.add_argument("input", help="path to .jsonl being split")
    p.add_argument("out_dir", help="where train/val/test go")
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # gather stats
    repo_stats = count_repo_stats(Path(args.input))

    # compute label balanced split assignment
    repo2split = assign_label_balanced(
        repo_stats,
        args.train_frac,
        args.val_frac,
        args.seed,
        Path(args.input)
    )

    print(Counter(repo2split.values()))

    # dump records into split files
    split_records(Path(args.input), repo2split, Path(args.out_dir))

    print(f"Done! Wrote train/val/test to {args.out_dir}")

    

if __name__ == "__main__":
    main()