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

**IMPORTANT**: Place these .txt folders in `datasets\vudenc\raw` otherwise the preprocessing scripts will not work. 

## Data Preprocessing Pipeline
This section outlines the steps required to convert the raw `.txt` dataset into a clean, line-delimited `.jsonl` format and process it into structured code records.

### Step 1. Run `load_vudenc.py`

Execute the follwoing command in the project root:
```bash
python pipeline/load_vudenc.py
```

This will:
- Parse all .txt files from the VUDENC dataset.
- Extract metadata, raw source code, and vulnerability labels with CWE IDs.
- Retrieve badparts and goodparts from structured diffs for patch reconstruction.
- Normalise and aggregates all entries into a single vudenc_raw.jsonl file.
- Includes built-in logging for warnings and processing progress.

The `.jsonl` entries will bw of the format:

```json
{"repo": "...", 
"commit": "...", 
"filepath": "...", 
"label": "...", 
"cwe_id": "...", 
"source": "...", 
"badparts": ["..."], 
"goodparts": ["..."]
}
```

### Step 2. Run `patch_code.py`

Execute the follwoing command in the project root:
```bash
python pipeline/patch_code.py
```

This will remove all `badparts` from the source code, and replace it with `goodparts`, creating a new entry in the `.jsonl` called new_code which is the full patched code. 

### Step 3. Remove all comments from dataset

Executed the following command in the project root:
```bash
python pipeline/strip_comments.py
```

Please note that this script uses tokenization to detect comments and there is logging to notify of any failed tokenizations. 861 fields contained tokenization errors, likely due to malformed escape sequences inside strings and imcomplete multi-line strings within the dataset. This leaves 1028 fully cleaned `source` and `new_code` pairs. 

The fully cleaned dataset is saved as `vudenc_stripped.jsonl` in `datasets/vudenc/raw.