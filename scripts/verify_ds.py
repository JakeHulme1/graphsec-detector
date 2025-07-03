import json, pathlib
path = pathlib.Path("datasets/vudenc/processed/xsrf_n7_m128_t0.05.ds30.jsonl")
pos = tot = 0
with path.open() as fh:
    for line in fh:
        tot += 1
        if '"label": 1' in line:
            pos += 1
print("pos_rate =", pos / tot)