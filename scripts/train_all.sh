#!/usr/bin/env bash
set -euo pipefail

# Usage: ./train_all.sh [GPU_DEVICE]
GPU_DEVICE="${1:-2}"
IMAGE_NAME="joh46/graphsec-detector:gpu"
HOST_DATA="/mnt/faster0/joh46/datasets/vudenc"
CONFIG="config/train_config.yaml"
OUT_BASE="output-graphsec"

# ─── Pull & build ────────────────────────────────
echo "[*] Pulling latest changes…"
git pull --ff-only

echo "[*] Building Docker image $IMAGE_NAME…"
hare build -t "$IMAGE_NAME" -f Dockerfile .
echo "[*] Image built!"

# ─── Sanitize YAML (unindents keys) ─────────────────────────────────────────
sed -i -E \
    -e 's/^[[:space:]]*learning_rate:/learning_rate:/' \
    -e 's/^[[:space:]]*weight_decay:/weight_decay:/' \
    "$CONFIG"

# ─── Hyperparameter grid ────────────────────────────────────────────────────
LRS=(0.008 0.010 0.012)
WDS=(0.0010 0.0012 0.0014)

# ─── List your 7 dataset folder names under prepared/ ────────────────────────
DATASETS=(command_injection open_redirect path_disclosure remote_code_execution sql xsrf xss)

for DS in "${DATASETS[@]}"; do
  echo "[*] Grid sweep on dataset: $DS"
  for LR in "${LRS[@]}"; do
    for WD in "${WDS[@]}"; do
      EXP_NAME="lr-${LR}_wd-${WD}"
      OUTDIR="${OUT_BASE}/${DS}/${EXP_NAME}"
      
      # ── Skip if we've already got a best.pt here ───────────────────────
      if [ -f "${OUTDIR}/best.pt" ]; then
        echo "   ↩ Skipping ${DS}/${EXP_NAME}, best.pt already exists."
        continue
      fi


      mkdir -p "$OUTDIR"

      # Patch train_config.yaml
      sed -i -E \
        -e "s|^dataset_name: .*|dataset_name: ${DS}|" \
        -e "s|^learning_rate: .*|learning_rate: ${LR}|" \
        -e "s|^weight_decay: .*|weight_decay: ${WD}|" \
        -e "s|^output_dir: .*|output_dir: ${OUTDIR}|" \
        "$CONFIG"

      echo "[*] Starting $DS / $EXP_NAME"
      hare run \
        --gpus "device=${GPU_DEVICE}" \
        -v "$(pwd)":/app \
        -v "${HOST_DATA}":/app/datasets/vudenc:ro \
        -v "$(pwd)/${OUT_BASE}":/app/output-graphsec \
        -p 10006:6006 \
        "$IMAGE_NAME"

      echo "[*] Completed $DS / $EXP_NAME"
    done
  done
  echo "[*] Finished dataset: $DS (9 runs)"
  echo
done

echo "[*] All grid sweeps complete!"
