# graphsec-detector

> Transformer-based vulnerability detector for Python code using the VUDENC dataset and GraphCodeBERT.

---

## Table of Contents

- [About](#about)
- [Dataset](#dataset)
  - [Download](download)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)

---

## About

`graphsec-detector` uses the GraphCodeBERT transformer model to detect security vulnerabilities in Python source code. It uses the VUDENC dataset, which contains Python commits labeled with Common Weakness Enumeration (CWE) types.

## Dataset

This project uses the **VUDENC** dataset which contains Python commits labeled with vulnerability types.

**Reference:**  
*Wartschinski, L., et al. (2022). "Vudenc: Vulnerability detection with deep learning on a natural codebase for python." Inf. Softw. Technol., 144(C).*

- [GitHub Repository](https://github.com/LauraWartschinski/VulnerabilityDetection/tree/master)  
- [Download Dataset on Zenodo](https://zenodo.org/records/3559203)

Dataset doi: 10.5281/zenodo.3559841

Note: `datasets/raw/` is `.gitignore`d to avoid committing large datasets. `.gitkeep` is used to preserve folder structure.


### Download 

Download the `.txt` files in the `Code\data` directory from the [VUDENC GitHub Repository](https://github.com/LauraWartschinski/VulnerabilityDetection/tree/master) which contains the data for the following vulnerabilities:
- Commmand Injection
- Open Redirect
- Path Disclousre
- Remote Code Execution
- SQL Injection
- Cross-Site Request Forgery (XSRF)
- Cross-Site Scripting (XSS)

Alternatively, download the `.txt` files from the author's [Zenodo Page](https://zenodo.org/records/3559841#.XeVaZNVG2Hs). The files needed are all the ones beginning with `plain_...txt`.

**IMPORTANT**: Place these .txt folders in `datasets\vudenc\raw` otherwise the preprocessing scripts will not work. 

## Data Preprocessing Pipeline
This section outlines the steps required to convert the raw `.txt` dataset into a clean, line-delimited `.jsonl` format and process it into structured code records.

### Step 1. Run the parsing script to extract varying granularities of data

This script parses the VUDENC `.txt` files into three JSONL datasets at different granularities:

- **File-Level** (`files.jsonl)
- **Commit-Level** (`commits.jsonl)
- **Repo-Level** (repos.jsonl)

### Usage
```bash
python export_vudenc_triples.py \
  --input-dir  /path/to/datasets/vudenc/raw \
  --output-dir /path/to/datasets/vudenc/raw
```

This will produce `files.jsonl`, `commits.jsonl`, `repos.jsonl` all under the `--output-dir.