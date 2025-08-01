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

echo "[*] Ensuring $OUTDIR exists…"
mkdir -p "$OUTDIR"

# ─── pull & build ─────
echo "[*] Pulling latest changes…"
git pull --ff-only

echo "[*] Building Docker image $IMAGE_NAME…"
hare build -t "$IMAGE_NAME" -f Dockerfile .
echo "[*] Image built!"

# ─── hyperparameter grid ────
LRS=(3e-5 1e-5)
WDS=(0.01 0.001 0.0001)

# Re-run the previously failed one first
echo "[*] Re-running failed experiment lr-3e-5_wd-0.001"
sed -i "
  s|^learning_rate: .*|learning_rate: 3e-5|;
  s|^weight_decay: .*|weight_decay: 0.001|;
  s|^output_dir: .*|output_dir: ${OUTDIR}/lr-3e-5_wd-0.001|;
" config/train_config.yaml

hare run \
  --gpus "device=${GPU_DEVICE}" \
  -v "$(pwd)":/app \
  -v "${HOST_DATA}":/app/datasets/vudenc:ro \
  -v "$(pwd)/$OUTDIR":/app/outputs \
  -p 10006:6006 \
  "$IMAGE_NAME"

echo "[*] Re-run complete for lr-3e-5_wd-0.001"

# Now do remaining sweeps
for LR in "${LRS[@]}"; do
  for WD in "${WDS[@]}"; do
    EXP_NAME="lr-${LR}_wd-${WD}"

    # skip the already-run or just re-run combo
    if [[ "$EXP_NAME" == "lr-3e-5_wd-0.001" ]] || [[ -f "$OUTDIR/$EXP_NAME/best.pt" ]]; then
      echo "[*] Skipping $EXP_NAME — already trained"
      continue
    fi

    echo "[*] Starting experiment $EXP_NAME"

    # patch the config
    sed -i "
      s|^learning_rate: .*|learning_rate: ${LR}|;
      s|^weight_decay: .*|weight_decay: ${WD}|;
      s|^output_dir: .*|output_dir: ${OUTDIR}/${EXP_NAME}|;
    " config/train_config.yaml

    # run training
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
