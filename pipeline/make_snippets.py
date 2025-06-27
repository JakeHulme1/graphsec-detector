"""
Original Author: Laura Wartschinski
Original GitHub repo: https://github.com/LauraWartschinski/VulnerabilityDetection/tree/master

Heavily modified by: Jake Hulme
Date: 13/06/2025
Adds:
  - CLI-tunable window/stride
  - sparse-positive filter
  - GraphCodeBERT sub-token budget guard
  - window-level de-duplication
  - drop-rate statistics
"""

from __future__ import annotations
from hashlib import blake2b
import sys
import json
from datetime import datetime
from pathlib import Path
import argparse
from . import vudencutils as vu
from importlib import reload
reload(vu)

# Helper function
def vuln_token_ratio(snippet: str, vuln_spans):
    toks = snippet.split()
    hit  = sum(1 for i,_ in enumerate(toks)
                 if any(start<=i<end for start,end in vuln_spans))
    return hit / max(1,len(toks))

# CLI
p = argparse.ArgumentParser()
p.add_argument("vuln",                help="vulnerability keyword (sql, xss, xsrf …)")
p.add_argument("--dump-only", action="store_true",
               help="stop after JSONL creation – no Word2Vec etc.")
p.add_argument("--raw-dir",  default="datasets/vudenc/raw",
               help="folder with plain_* files")
p.add_argument("--raw-ext",  default="", help="extension of raw files (.json, .txt)")
p.add_argument("--out-dir",  default="prepared",
               help="directory for JSONL output")
#  ★ NEW FLAGS ★
p.add_argument("--step",        type=int,   default=5,
               help="focus-window stride n (tokens)")
p.add_argument("--context",     type=int,   default=200,
               help="context window length m (tokens)")
p.add_argument("--occ-thresh",  type=float, default=0.05,
               help="min vulnerable-token ratio required for a positive window")
p.add_argument("--max-subtok",  type=int,   default=480,
               help="skip windows longer than this after BPE (GraphCodeBERT budget)")
args = p.parse_args()


# Budget helper (lazy-loaded so --dump-only stays light-weight)
TOKENIZER = None
def subtok_len(snippet: str) -> int:
    global TOKENIZER
    if TOKENIZER is None:
        from transformers import AutoTokenizer
        TOKENIZER = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    return len(TOKENIZER.encode(snippet, add_special_tokens=False))


# Load raw VUDENC
raw_path = Path(args.raw_dir) / f"plain_{args.vuln}{args.raw_ext}"
if not raw_path.exists():
    sys.exit(f"[ERR] Cannot find raw dataset: {raw_path}")
with raw_path.open(encoding="utf-8") as fh:
    data = json.load(fh)
print(f"[INFO] Loaded {raw_path} – {len(data):,} root repos   ({datetime.now():%H:%M})")


step        = args.step
fulllength  = args.context

all_blocks      : list[dict] = []
seen_hashes     : set[bytes] = set()

# Counters for transparency
dropped_sparse  = dropped_long = dropped_dupes = 0

for repo_url, commits in data.items():
    for sha, commit in commits.items():
        for file_path, fmeta in commit.get("files", {}).items():
            src = fmeta.get("source")
            if not src:
                continue

            # collect all vulnerable spans (badparts) as lists of strings
            badparts = [bad for ch in fmeta["changes"] for bad in ch["badparts"]]
            if not badparts:
                continue

            positions = vu.findpositions(badparts, src)

            # split into windows
            for code, label in vu.getblocks(src, positions, step, fulllength):
                # 1) content-hash de-duplication
                h = blake2b(code.encode(), digest_size=16).digest()
                if h in seen_hashes:
                    dropped_dupes += 1
                    continue
                seen_hashes.add(h)

                # 2) sparse-positive filter
                if label == 1:
                    occ = vuln_token_ratio(code, positions)
                    if occ < args.occ_thresh:
                        dropped_sparse += 1
                        continue
                else:
                    occ = 0.0

                # 3) BPE budget guard
                if subtok_len(code) > args.max_subtok:
                    dropped_long += 1
                    continue

                all_blocks.append({
                    "code":  code,
                    "label": int(label),
                    "repo":  repo_url,
                    "occ":   occ                    # keep occupancy for optional re-weighting
                })


# Dump result
out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
out_path = out_root / f"{args.vuln}.jsonl"
with out_path.open("w", encoding="utf-8") as f_out:
    for item in all_blocks:
        f_out.write(json.dumps(item, ensure_ascii=False) + "\n")


total_in  = len(all_blocks) + dropped_dupes + dropped_sparse + dropped_long
print(f"[STATS] windows kept      : {len(all_blocks):>7,} / {total_in:,}")
print(f"[STATS] duplicate windows : {dropped_dupes:>7,}")
print(f"[STATS] sparse positives  : {dropped_sparse:>7,}")
print(f"[STATS] >{args.max_subtok} subtoks : {dropped_long:>7,}")
print(f"[OK] JSONL written to {out_path}")









#mode = "sql"

# # Setup CLI
# parser = argparse.ArgumentParser()
# parser.add_argument("vuln", help="vulnerability keyword (sql, xss, xsrf …)")
# parser.add_argument("--dump-only", action="store_true",
#                     help="Just create labelled snippets and exit")
# parser.add_argument("--out-dir", default="prepared",
#                     help="Where to store the JSONL/NPZ artefacts")
# parser.add_argument("--raw-dir", default="datasets/vudenc/raw",
#                     help="Folder that contains plain_* files")
# parser.add_argument("--raw-ext", default="",
#                     help="Extension for raw files, e.g. .json")
# # Tunabe heuristic CL atguments
# parser.add_argument("--step", type=int, default=5,        # ← expose n
#                     help="focus-window stride (tokens)")
# parser.add_argument("--context", type=int, default=200,   # ← expose m
#                     help="max context length (tokens)")
# parser.add_argument("--occ-thresh", type=float, default=.05,
#                     help="minimum vulnerable-token ratio for label-1 windows")
# parser.add_argument("--max-subtok", type=int, default=480,
#                     help="discard windows whose BPE length exceeds this")
# args = parser.parse_args()

# mode = args.vuln

# progress = 0
# count = 0


# ### paramters for the filtering and creation of samples
# restriction = [20000,5,6,10] #which samples to filter out
# step = args.step #step lenght n in the description
# fulllength = args.context #context length m in the description

# mode2 = str(step)+"_"+str(fulllength) 

# ### hyperparameters for the w2v model
# mincount = 10 #minimum times a word has to appear in the corpus to be in the word2vec model
# iterationen = 100 #training iterations for the word2vec model
# s = 200 #dimensions of the word2vec model
# w = "withString" #word2vec model is not replacing strings but keeping them

# #get word2vec model
# w2v = "word2vec_"+w+str(mincount) + "-" + str(iterationen) +"-" + str(s)
# w2vmodel = "w2v/" + w2v + ".model"

# if not args.dump_only:
#     if not os.path.isfile(w2vmodel):
#         print("word2vec model is missing")
#         sys.exit(1)
#     #w2v_model = Word2Vec.load(w2vmodel)
#     #word_vectors = w2v_model.wv

# #load data
# raw_path = Path(args.raw_dir) / f"plain_{mode}{args.raw_ext}"
# if not raw_path.exists():
#     sys.exit(f"[ERR] Cannot find raw dataset: {raw_path}")
# with raw_path.open("r", encoding="utf-8") as infile:
#   data = json.load(infile)
  
# now = datetime.now() # current date and time
# nowformat = now.strftime("%H:%M")
# print("finished loading. ", nowformat)

# allblocks = []

# for r in data:
#   progress = progress + 1
  
#   for c in data[r]:
    
#     if "files" in data[r][c]:                      
#     #  if len(data[r][c]["files"]) > restriction[3]:
#         #too many files
#     #    continue
      
#       for f in data[r][c]["files"]:
        
#   #      if len(data[r][c]["files"][f]["changes"]) >= restriction[2]:
#           #too many changes in a single file
#    #       continue
        
#         if not "source" in data[r][c]["files"][f]:
#           #no sourcecode
#           continue
        
#         if "source" in data[r][c]["files"][f]:
#           sourcecode = data[r][c]["files"][f]["source"]                          
#      #     if len(sourcecode) > restriction[0]:
#             #sourcecode is too long
#      #       continue
          
#           allbadparts = []
          
#           for change in data[r][c]["files"][f]["changes"]:
            
#                 #get the modified or removed parts from each change that happened in the commit                  
#                 badparts = change["badparts"]
#                 count = count + len(badparts)
                
#            #     if len(badparts) > restriction[1]:
#                   #too many modifications in one change
#            #       break
                
#                 for bad in badparts:
#                   #check if they can be found within the file
#                   pos = vudencutils.findposition(bad,sourcecode)
#                   if not -1 in pos:
#                       allbadparts.append(bad)
                      
#              #   if (len(allbadparts) > restriction[2]):
#                   #too many bad positions in the file
#              #     break
                      
#           if(len(allbadparts) > 0):
#          #   if len(allbadparts) < restriction[2]:
#               #find the positions of all modified parts
#               positions = vudencutils.findpositions(allbadparts,sourcecode)

#               #get the file split up in samples
#               blocks = vudencutils.getblocks(sourcecode, positions, step, fulllength)
              
#               for b in blocks:
#                   #each is a tuple of code and label
#                   code, label = b
#                   allblocks.append({
#                      "code": code,
#                      "label": label,
#                      "repo": r # outer loop's repo id
#                   })


# #  DUMP-ONLY  -->  write JSONL & stop here
# if args.dump_only:
#     out_root = Path(args.out_dir)
#     out_root.mkdir(parents=True, exist_ok=True)

#     out_path = out_root / f"{mode}.jsonl"
#     with out_path.open("w", encoding="utf-8") as f_out:
#       for item in allblocks:
#         f_out.write(json.dumps({
#           "code": item["code"],
#           "label": int(item["label"]),
#           "repo": item["repo"] # keep the repo
#         }) + "\n")