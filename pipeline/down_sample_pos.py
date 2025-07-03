import json, random, argparse, pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--target", type=float, default=0.30)
args = ap.parse_args()

data = [json.loads(l) for l in open(args.inp)]
pos = [d for d in data if d["label"] == 1]
neg = [d for d in data if d["label"] == 0]
k_pos = int(len(neg)*args.target / (1-args.target))
selpos = random.sample(pos, k_pos)

out = pathlib.Path(args.inp).with_suffix("").with_suffix(".ds30.jsonl")
with out.open("w") as fh:
    for d in selpos+neg: fh.write(json.dumps(d)+"\n")
print("wrote", out)