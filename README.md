# graphsec-detector

> Transformer-based vulnerability detector for Python code using the VUDENC dataset and GraphCodeBERT.

---

## Table of Contents

- [About](#about)
- [Dataset](#dataset)
  - [Manual Download](#manual-download)
  - [Automated Download](#automated-download)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)

---

## About

`graphsec-detector` leverages the GraphCodeBERT transformer model to detect security vulnerabilities in Python source code. It uses the VUDENC dataset, which contains Python commits labeled with Common Weakness Enumeration (CWE) types.

## Dataset
This project uses the VUDENC dataset which contains Python commits labeled with vulnerability types. 

<details>
  <summary>Click to expand</summary>

This project uses the **VUDENC** dataset which contains Python commits labeled with vulnerability types.

**Reference:**  
*Wartschinski, L., et al. (2022). "Vudenc: Vulnerability detection with deep learning on a natural codebase for python." Inf. Softw. Technol., 144(C).*

- [GitHub Repository](https://github.com/LauraWartschinski/VulnerabilityDetection/tree/master)  
- [Download Dataset on Zenodo](https://zenodo.org/records/3559203)

Note: `datasets/raw/` is `.gitignore`d to avoid committing large datasets. `.gitkeep` is used to preserve folder structure.

</details>

### Manual Download 

If you already have the dataset:

Place `PyCommitsWithDiff.json` in: `graphsec-detector/datasets/raw`

### Automated Download

You can run the dataset fetcher using the provided script `download_dataset.sh` This can be executed by running Git Bash in the project root and executing the following command:

```sh
bash scripts/download_dataset.sh
```

## Data Preprocessing Pipeline
This section outlines the steps required to convert the raw `.json`` dataset into a clean, line-delimited `.jsonl' format and process it into structured code records.

### Step 1. Convert `PyCommitsWithDiffs.json` -> `PyCommitsWithDiffs.jsonl`

The raw dataset is a large (~9GB) JSON array. To make it streamable and compatible with the parser, convert it to `.jsonl` using jq:

```bash
jq -c '.[]' datasets/raw/PyCommitsWithDiffs.json > datasets/raw/PyCommitsWithDiffs.jsonl
```

### Step 2. Run the Preprocessing Script:

Once the .jsonl file is ready, run the commit parser script:

```bash
python scripts/parse_full_and_diff_cleaned.py
```

This script:
- Detects encoding
- Extracts diffs from each commit
- Filters out comment-only changes
- Outputs a clean `.jsonl` file at: `datasets/processed/clean_commits.jsonl`

Each line in the output contains:
```json
{
  "sha": "...",
  "repo": "...",
  "keyword": "...",
  "old_code_full": "...",
  "new_code_full": "...",
  "old_changed_lines": "...",
  "new_changed_lines": "..."
}
```