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

# --- Clean & recreate outputs dir ---
if [[ -d output-graphsec ]]; then
  echo "[*] Removing old output-graphsec/ …"
  rm -rf output-graphsec
fi
mkdir -p output-graphsec

# update code
echo "[*] Pulling latest changes from GitHub…"
git pull --ff-only

# build Docker image
echo "[*] Building Docker image: $IMAGE_NAME"
hare build -t "$IMAGE_NAME" -f Dockerfile .
echo "[*] Docker image successfully built!"

# Hyperparameter grid
LRS=(5e-5 3e-5 1e-5)
WDS=(0.01 0.001 0.0001)

for LR in "${LRS[@]}"; do
  for WD in "${WDS[@]}"; do
    EXP_NAME="lr-${LR}_wd-${WD}"
    echo "[*] Starting experiment ${EXP_NAME}"

    # Prepare a fresh copy of the training config for this sweep
    EXP_DIR="${OUTDIR}/${EXP_NAME}"
    mkdir -p "$EXP_DIR"
    cp config/train_config.yaml "${EXP_DIR}/train_config.yaml"
    sed -i "s/^learning_rate: .*/learning_rate: ${LR}/"       "${EXP_DIR}/train_config.yaml"
    sed -i "s/^weight_decay: .*/weight_decay: ${WD}/"         "${EXP_DIR}/train_config.yaml"
    # point our script at this config
    export TRAIN_CONFIG_PATH="${EXP_DIR}/train_config.yaml"

    # Launch via Hare; will use the Dockerfile's CMD (tensorboard + train.py)
    hare run \
      --gpus "device=${GPU_DEVICE}" \
      -v "$(pwd)":/app \
      -v "${HOST_DATA}":/app/datasets/vudenc:ro \
      -v "$PWD/$EXP_DIR":/app/outputs \
      -p 10006:6006 \
      "$IMAGE_NAME"

    echo "[*] Experiment ${EXP_NAME} complete"
  done
done