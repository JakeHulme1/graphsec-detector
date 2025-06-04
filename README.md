# graphsec-detector
---

## Dataset Download: VUDENC (PyCommitsWithDiffs.json)
This project uses the VUDENC dataset which contains Python commits labeled with vulnerability types. 

The full reference for the academic paper which proposed the VUDENC dataset is: *Wartschinski, L., Noller, Y., Vogel, T., Kehrer, T., and Grunske, L. (2022). Vudenc: Vulnerability detection with deep learning on a natural codebase for python. Inf. Softw. Technol., 144(C).*

The associated GitHub repository can be reached here: [LauraWartschinski/VulnerabilityDetection](https://github.com/LauraWartschinski/VulnerabilityDetection/tree/master)

The full dataset can be downloaded on Zenodo through the following URL (https://zenodo.org/records/3559203)

Note: `datasets/raw/` is intentionally excluded from pushes to avoid pushing large files.

### Option 1: Manual Downlaod
If you already have the dataset downloaded, simply place the file:
`PyCommitsWithDiffs.json`
into the follwing path:
`graphsec-detector/datasets/raw/PyCommitsWithDiffs.json`

### Option 2: Automated Download Script
You can also download the dataset using the provided script:
`scripts/download_dataset.sh`
This can be executed by running Git Bash in the project root and executing the following command:
```sh
bash scripts/download_dataset.sh
```
This will create the `datasets/raw` directory (if missing), download the dataset from Zenodo, and save it as `datasets/raw/PyCommitsWithDiffs.json`

---
