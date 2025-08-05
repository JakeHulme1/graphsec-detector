#!/usr/bin/env bash
set -euo pipefail

# Usage: ./train_all.sh [GPU_DEVICE]
GPU_DEVICE="${1:-2}"
IMAGE="joh46/graphsec-detector:gpu"
DATA_ROOT="/mnt/faster0/joh46/datasets/vudenc/prepared"
CONFIG="config/train_config.yaml"
OUT_BASE="output-graphsec"

# Datasets
DATASETS=(command_injection open_redirect path_disclosure remote_code_execution sql xsrf xss)

# Hyperparameter grids 
LRS=(0.008 0.010 0.012)
WDS=(0.0010 0.0012 0.0014)
GAMMAS=(2.0)
DROPOUTS=(0.3)


for DS in "${DATASETS[@]}"; do
  echo "[*] Starting sweeps on dataset: $DS"
  for LR in "${LRS[@]}"; do
    for WD in "${WDS[@]}"; do
      for GAMMA in "${GAMMAS[@]}"; do
        for DP in "${DROPOUTS[@]}"; do

          EXP="lr${LR}_wd${WD}_g${GAMMA}_d${DP}"
          OUTDIR="${OUT_BASE}/${DS}/${EXP}"
          mkdir -p "$OUTDIR"

          # Patch train_config.yaml in-place
          sed -i -E \
            -e "s|^dataset_name: .*|dataset_name: ${DS}|" \
            -e "s|^learning_rate: .*|learning_rate: ${LR}|" \
            -e "s|^weight_decay: .*|weight_decay: ${WD}|" \
            -e "s|^focal_gamma: .*|focal_gamma: ${GAMMA}|" \
            -e "s|^classifier_dropout: .*|classifier_dropout: ${DP}|" \
            -e "s|^output_dir: .*|output_dir: ${OUTDIR}|" \
            "$CONFIG"

          echo "[*] $DS | $EXP"

          hare run \
            --gpus "device=${GPU_DEVICE}" \
            -v "$(pwd)":/app \
            -v "${DATA_ROOT}/${DS}":/app/datasets/vudenc/prepared/"${DS}":ro \
            -v "$(pwd)/${OUT_BASE}":/app/output-graphsec \
            -p 10006:6006 \
            "$IMAGE" \
            python train.py \
              --config $CONFIG \
              training.epochs=10

          echo "[*] Complete: $DS / $EXP"
          echo

        done
      done
    done
  done
  echo "✓ Finished dataset: $DS"
  echo
done

echo "All sweeps finished!"
