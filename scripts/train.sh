#!/usr/bin/env bash
set -euo pipefail

# which GPU to use (default 2)
GPU_DEVICE="${1:-2}"
echo "[*] Launching on GPU $GPU_DEVICE …"

# constants
IMAGE_NAME="joh46/graphsec-detector:gpu"
HOST_DATA="/mnt/faster0/joh46/datasets/vudenc"
OUTDIR="output-graphsec"
CONFIG="config/train_config.yaml"

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

# ─── sanitize YAML (un-indent keys) as was getting issues here ────
sed -i -E \
  -e 's/^[[:space:]]*learning_rate:/learning_rate:/' \
  -e 's/^[[:space:]]*weight_decay:/weight_decay:/' \
  "$CONFIG"

# ─── hyperparameter grid ────
LRS=(0.008 0.010 0.012)
WDS=(0.0008 0.0010 0.0012)

for LR in "${LRS[@]}"; do
  for WD in "${WDS[@]}"; do
    EXP_NAME="lr-${LR}_wd-${WD}"
    echo "[*] Starting experiment $EXP_NAME"

    # patch the YAML during training (top-level keys, no indent)
    sed -i -E \
      -e "s|^learning_rate: .*|learning_rate: ${LR}|" \
      -e "s|^weight_decay: .*|weight_decay: ${WD}|" \
      -e "s|^output_dir: .*|output_dir: ${OUTDIR}/${EXP_NAME}|" \
      "$CONFIG"

    # run training
    hare run \
      --gpus "device=${GPU_DEVICE}" \
      -v "$(pwd)":/app \
      -v "${HOST_DATA}":/app/datasets/vudenc:ro \
      -v "$(pwd)/$OUTDIR":/app/output-graphsec \
      -p 10006:6006 \
      "$IMAGE_NAME"

    echo "[*] Experiment $EXP_NAME complete"
  done
done

echo "[*] All training complete!"
