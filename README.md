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

## Datasets

### VUDENC

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

### Step 1. Label plain_* files  (`make_snippets.py`)

Please note, `make_snippets.py` is a modified version of the `makemodel.py` which is the script in [VulunerabilityDetection/Code](https://github.com/LauraWartschinski/VulnerabilityDetection) that splits the data into three random segments and trains the LSTM.

The modifications made were:
  - avoid heavy imports (e.g., `tensorflow`)as this version does not require to build a model
  - make the script a 'dump-only' script, just producing the labelled data.
  - include the repo url in the output jsonl for data leakage prevention downstream.

`make_snippets.py` walks every plain_* dataset produced by the VUDENC crawler, applies the sliding-window labelling scheme (See section 4 of the [paper](https://arxiv.org/abs/2201.08441)) and emits JSONL files for each vulnerability.


#### Usage
```bash
  poetry run python -m pipeline.make_snippets <vuln>.txt --dump-only --raw-dir datasets/vudenc/raw --out-dir datasets/vudenc/prepared
```

Each line in the output files is:

```json
{
  "code": "<raw Python snippet>",
  "label": 0 | 1}
  "repo": <repo_url>
```

### Step 2. Create train / val / test repo-exclusive split (`split_by_repo.py`)

This script takes in `.jsonl` files from the previous step and:

- creates the splits (by default it is 70/15/15 but this can be changed in command-line arguments)
- uses lable-balanced assignment

#### Usage

```bash
poetry run python split_by_repo.py \
  path/to/vuln.jsonl \
  path/to/output_folder \
  --train-frac 0.80 \
  --val-frac   0.10 \
  --seed       123
  ```

If no arguments are given for train-frac, val-frac and seed, they will resolve to default values: 70, 15 and 42 respectively.

### Step 3. Extract the DFGs.

#### Step 3.1. Fetch and lock to correct version of GraphCodeBERT 

Since GraphCodeBERT DFG extractor is used, the DFG.py must be imported into the project as a submodule. This step ensures you fetch and lock to the correct version.

Run this from the root:

```bash
git submodule update --init --recursive
```

#### Step 3.2. Make the GraphCodeBERT DFG extracter importable without modifying the submodule.

```bash
ln -s extern/CodeBERT/GraphCodeBERT/translation translation
```

#### Step 3.3. Install Python dependencies (if they aren't already)

```bash
poetry env use python3.11
poetry install
```