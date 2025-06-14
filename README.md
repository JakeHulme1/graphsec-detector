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

### Step 1. Slice plain_* files into labelled code-blocks (`make_snippets.py`)

Please note, `make_snippets.py` is a modified version of the `makemodel.py` which is the script in [VulunerabilityDetection/Code](https://github.com/LauraWartschinski/VulnerabilityDetection) that splits the data into three random segments and trains the LSTM.

The modifications made were:
  - avoid heavy imports (e.g., `tensorflow`)as this version does not require to build a model
  - make the script a 'dump-only' script, just producing the labelled data.

`make_snippets.py` walks every plain_* dataset produced by the VUDENC crawler, applies the sliding-window labelling scheme (See section 4 of the [paper](https://arxiv.org/abs/2201.08441)) and emits three ready-to-train JSONL splits for each vulnerability type:

- sql_train.jsonl     # 70 % of blocks
- sql_valid.jsonl     # 15%
- sql_test.jsonl      # 15%

#### Usage
```bash
  python -m pipeline.make_snippets sql \
  --dump-only \
  --raw-dir   datasets/vudenc/raw \ live
  --out-dir   datasets/vudenc/prepared
```

Each line in the output files is:

```json
{
  "code": "<raw Python snippet>",
  "label": 0 | 1}
```
