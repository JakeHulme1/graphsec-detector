#!/usr/bin/env bash
set -euo pipefail

# --- usage/help ---
if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  cat <<EOF
Usage: $0 [GPU_DEVICE]

  GPU_DEVICE   index of the GPU to use inside the container
               (defaults to 2)
EOF
  exit 0
fi

# grab the first arg (or default to 2)
GPU_DEVICE="${1:-2}"
echo "[*] Launching on GPU ${GPU_DEVICE}…"

# --- config ---
IMAGE_NAME="joh46/graphsec-detector:gpu"
POETRY_VERSION="2.1.3"
POETRY_EXTRAS="--with gpu"
HOST_DATA="/mnt/faster0/joh46/datasets/vudenc"

# clear outputs (only if dir exists)
if [[ -d outputs ]]; then
  rm -rf outputs/*
fi

# update code
echo "[*] Pulling latest changes from GitHub…"
git pull --ff-only

# build Docker image
echo "[*] Building Docker image: $IMAGE_NAME"
hare build -t "$IMAGE_NAME" -f Dockerfile .
echo "[*] Docker image successfully built!"

# run training via Hare
echo "[*] Running training inside Docker with Hare"
hare run \
   --gpus "device=${GPU_DEVICE}" \
  -v "$(pwd)":/app \
  -v "${HOST_DATA}":/app/datasets/vudenc:ro \
  -v "$HOME/output-graphsec":/app/outputs \
  -p 10006:6006 \
  "$IMAGE_NAME"