#!/usr/bin/env bash
set -euo pipefail

# which GPU to use (default 2)
GPU_DEVICE="${1:-2}"
echo "[*] Launching on GPU $GPU_DEVICE …"

# constants
IMAGE_NAME="joh46/graphsec-detector:gpu"
POETRY_EXTRAS="--with gpu"
HOST_DATA="/mnt/faster0/joh46/datasets/vudenc"
OUTDIR="output-graphsec"

# ─── clean + recreate ────
echo "[*] Cleaning $OUTDIR/"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

# ─── pull & build ─────
echo "[*] Pulling latest changes…"
git pull --ff-only

echo "[*] Building Docker image $IMAGE_NAME…"
hare build -t "$IMAGE_NAME" -f Dockerfile .
echo "[*] Image built!"

# ─── hyperparameter grid ────
LRS=(5e-5 3e-5 1e-5)
WDS=(0.01 0.001 0.0001)

for LR in "${LRS[@]}"; do
  for WD in "${WDS[@]}"; do
    EXP_NAME="lr-${LR}_wd-${WD}"
    echo "[*] Starting experiment $EXP_NAME"

    # patch the repo's train_config.yaml in‐place
    sed -i "
      s|^learning_rate: .*|learning_rate: ${LR}|;
      s|^weight_decay: .*|weight_decay: ${WD}|;
      s|^output_dir: .*|output_dir: ${OUTDIR}/${EXP_NAME}|;
    " config/train_config.yaml

    # fire off the run (train.py will pick up config/train_config.yaml automatically)
    hare run \
      --gpus "device=${GPU_DEVICE}" \
      -v "$(pwd)":/app \
      -v "${HOST_DATA}":/app/datasets/vudenc:ro \
      -v "$(pwd)/$OUTDIR":/app/outputs \
      -p 10006:6006 \
      "$IMAGE_NAME"

    echo "[*] Experiment $EXP_NAME complete"
  done
done

echo "[*] All training complete!"
