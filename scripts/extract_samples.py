import json
import ijson
from pathlib import Path
from pprint import pprint

# This script loads the nested VUDENC dataset (dict of dicts), extracts one commit per repo, and prints the structure of each.

# ------------------------
# DATASET STRUCTURE OVERVIEW:
# ------------------------
# Top-level JSON is a dict where:
# - Keys = repository URLs
# - Values = dict of commits
#
# Each commit is:
#   "<commit_sha>": {
#       "url": "https://...",
#       "html_url": "https://...",
#       "sha": "...",
#       "keyword": "open redirect malicious",
#       "diff": "diff --git ...\n--- old_file\n+++ new_file\n@@ code changes ..."
#   }
#
# - commit_sha: A unique hash ID for that Git commit (e.g. SHA-1)
# - diff: A unified diff showing the vulnerable (old) and fixed (new) code


RAW_PATH = Path("datasets/raw/PyCommitsWithDiffs.json")
SAMPLE_PATH = Path("datasets/raw/sample_by_repo.json")

print(f"Streaming sample from: {RAW_PATH}")

samples = []

try:
    with open(RAW_PATH, "rb") as f:
        # Stream top-level key-value pairs: repo_url -> {commit_sha: {...}, ...}
        for repo_url, commits_dict in ijson.kvitems(f, ""):
            if not isinstance(commits_dict, dict):
                continue

            # Take the first commit in this repo
            for commit_sha, commit_data in commits_dict.items():
                commit_data["repo"] = repo_url
                commit_data["sha"] = commit_sha
                samples.append(commit_data)
                break  # Only one commit per repo

            if len(samples) >= 100:
                break

    # Save the sample for inspection
    with open(SAMPLE_PATH, "w", encoding="utf-8") as out:
        json.dump(samples, out, indent=2)

    print(f"Wrote {len(samples)} samples to {SAMPLE_PATH}")
    print("\nFirst sample entry:")
    pprint(samples[0])

except Exception as e:
    import traceback
    print("Error while processing the dataset:")
    traceback.print_exc()
