#!/bin/bash

# Create directory if it doesn't exist
mkdir -p datasets/raw/

# Download VUDENC from Zenodo
echo "Downloading VUDENC dataset from Zenodo..."
wget -O datasets/raw/PyCommitsWithDiffs.json "https://zenodo.org/record/7011654/files/PyCommitsWithDiffs.json?download=1"

echo "Dataset downloaded to datasets/raw/"