#!bin/bash

set -e # exit on error

# --- config ---
IMAGE_NAME="graphsec-detector"
POETRY_VERSION="2.1.3"
POETRY_EXTRAS="--with gpu"

# clear outputs
rm -rf outputs/*

# --- build Docker image ---
echo "[*] Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .

# --- run training via Hare ---
echo "[*] Running training inside Docker with Hare"

hare run --image "$IMAGE_NAME" --gpu --env TOKENIZERS_PARALLELISM=false <<EOF

# install poetry inside container if missing
pip show poetry >/dev/null 2>&1 || pip install poetry==$POETRY_VERSION

# install dependencies including GPU support
poetry install --no-interaction --no-ansi $POETRY_EXTRAS

# --- run training ---
poetry run python train.py
EOF
