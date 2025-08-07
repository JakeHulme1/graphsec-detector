#!/usr/bin/env bash
set -euo pipefail

GPU_DEVICE=2
IMAGE_NAME="joh46/graphsec-detector:gpu"
HOST_DATA="/mnt/faster0/joh46/graphsec-detector/datasets/vudenc"
CONFIG="config/train_config.yaml"
OUT_BASE="$(pwd)/output-graphsec"

# 1) SQL: only these five experiments
SQL_EXPS=(
  lr-0.010_wd-0.0012
  lr-0.010_wd-0.0014
  lr-0.012_wd-0.0010
  lr-0.012_wd-0.0012
  lr-0.012_wd-0.0014
)

for EXP in "${SQL_EXPS[@]}"; do
  DS=sql
  OUTDIR="$OUT_BASE/$DS/$EXP"
  LR=${EXP#lr-}; LR=${LR%_*}
  WD=${EXP#*_wd-}

  echo "[*] Running $DS / $EXP"
  rm -rf "$OUTDIR"
  mkdir -p "$OUTDIR"

  sed -i -E \
    -e "s|^dataset_name: .*|dataset_name: ${DS}|" \
    -e "s|^learning_rate: .*|learning_rate: ${LR}|" \
    -e "s|^weight_decay: .*|weight_decay: ${WD}|" \
    -e "s|^output_dir: .*|output_dir: ${OUTDIR}|" \
    "$CONFIG"

  hare run \
    --gpus "device=${GPU_DEVICE}" \
    -v "$(pwd)":/app \
    -v "${HOST_DATA}":/app/datasets/vudenc:ro \
    -v "${OUT_BASE}":/app/output-graphsec \
    -p 10006:6006 \
    "$IMAGE_NAME"
  echo
done

# 2) XSRF and XSS: full grid (3×3 each)
for DS in xsrf xss; do
  for LR in 0.008 0.010 0.012; do
    for WD in 0.0010 0.0012 0.0014; do
      EXP="lr-${LR}_wd-${WD}"
      OUTDIR="$OUT_BASE/$DS/$EXP"

      echo "[*] Running $DS / $EXP"
      rm -rf "$OUTDIR"
      mkdir -p "$OUTDIR"

      sed -i -E \
        -e "s|^dataset_name: .*|dataset_name: ${DS}|" \
        -e "s|^learning_rate: .*|learning_rate: ${LR}|" \
        -e "s|^weight_decay: .*|weight_decay: ${WD}|" \
        -e "s|^output_dir: .*|output_dir: ${OUTDIR}|" \
        "$CONFIG"

      hare run \
        --gpus "device=${GPU_DEVICE}" \
        -v "$(pwd)":/app \
        -v "${HOST_DATA}":/app/datasets/vudenc:ro \
        -v "${OUT_BASE}":/app/output-graphsec \
        -p 10006:6006 \
        "$IMAGE_NAME"
      echo
    done
  done
done

echo "[*] Done with SQL subset, XSRF and XSS full grid."
