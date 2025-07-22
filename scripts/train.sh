#!/usr/bin/env bash
set -euo pipefail

GPU_DEVICE="${1:-2}"
echo "[*] Launching on GPU $GPU_DEVICE …"

# --- constants ---
IMAGE_NAME="joh46/graphsec-detector:gpu"
POETRY_EXTRAS="--with gpu"
HOST_DATA="/mnt/faster0/joh46/datasets/vudenc"
OUTDIR="output-graphsec"

# --- clean + recreate ---
echo "[*] Cleaning $OUTDIR/"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

# --- pull & build ---
echo "[*] Pulling latest changes…"
git pull --ff-only

echo "[*] Building Docker image $IMAGE_NAME…"
hare build -t "$IMAGE_NAME" -f Dockerfile .
echo "[*] Image built!"

# --- sweep setup ---
LRS=(5e-5 3e-5 1e-5)
WDS=(0.01 0.001 0.0001)

for LR in "${LRS[@]}"; do
  for WD in "${WDS[@]}"; do
    EXP_NAME="lr-${LR}_wd-${WD}"
    EXP_DIR="${OUTDIR}/${EXP_NAME}"
    echo "[*] Starting experiment $EXP_NAME -> $EXP_DIR"
    mkdir -p "$EXP_DIR"

    # copy & patch train_config.yaml
    cp config/train_config.yaml "$EXP_DIR/train_config.yaml"
    sed -i "s|^learning_rate: .*|learning_rate: ${LR}|g"    "$EXP_DIR/train_config.yaml"
    sed -i "s|^weight_decay: .*|weight_decay: ${WD}|g"      "$EXP_DIR/train_config.yaml"

    # run inside container, exporting TRAIN_CONFIG_PATH so train.py picks it up
    hare run \
      --gpus "device=${GPU_DEVICE}" \
      -e TRAIN_CONFIG_PATH="/app/outputs/train_config.yaml" \
      -v "$(pwd)":/app \
      -v "${HOST_DATA}":/app/datasets/vudenc:ro \
      -v "$PWD/$EXP_DIR":/app/outputs \
      -p 10006:6006 \
      "$IMAGE_NAME"

    echo "[*] Experiment $EXP_NAME complete -> $EXP_DIR"
  done
done

echo "[*] All sweeps complete!"