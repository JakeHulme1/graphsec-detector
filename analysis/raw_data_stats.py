import json
from pathlib import Path
from collections import Counter

RAW_DIR      = Path(r"C:\Projects\graphsec-detector\datasets\vudenc\raw")
RAW_PATTERN  = "plain_*.txt"                     # all raw files
PY_SUFFIXES  = {".py", ".py"}                   


repos_seen   = set()           # unique GitHub repos  -> projects
files_seen   = set()           # (repo_url, rel_path) -> unique files
n_commits    = 0


for txt in RAW_DIR.glob(RAW_PATTERN):
    with open(txt, encoding="utf-8") as fh:
        for line in fh:                       # raw file may contain >1 JSON object
            line = line.strip()
            if not line:
                continue
            blob = json.loads(line)           # top-level dict

            # level-1: repository URL 
            for repo_url, commits in blob.items():
                repos_seen.add(repo_url)

                # level-2: commit SHA
                for sha, meta in commits.items():
                    n_commits += 1

                    # level-3: file paths inside commit
                    for rel_path in meta["files"].keys():
                        # optionally ignore non-Python files
                        if PY_SUFFIXES and Path(rel_path).suffix not in PY_SUFFIXES:
                            continue
                        files_seen.add((repo_url, rel_path))


print(f"# projects (repos) : {len(repos_seen):,}")
print(f"# commits          : {n_commits:,}")
print(f"# unique .py files : {len(files_seen):,}")
