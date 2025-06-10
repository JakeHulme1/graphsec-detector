# quick_audit.py
from collections import Counter, defaultdict
import json, pathlib

skip_stats = Counter()
reason_per_file = defaultdict(list)

for line in pathlib.Path("C:/Projects/graphsec-detector/datasets/vudenc/raw/vudenc_raw.jsonl").read_text(encoding="utf-8").splitlines():
    e = json.loads(line)
    if not (e["badparts"] and e["goodparts"]):
        skip_stats[e["label"]] += 1
        reason_per_file[e["label"]].append(e["filepath"])

print("Skipped by class:", skip_stats)
