#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# Usage: ./scripts/train_specific.sh [GPU_DEVICE]
GPU_DEVICE="${1:-2}"
IMAGE_NAME="joh46/graphsec-detector:gpu"
HOST_DATA="/mnt/faster0/joh46/graphsec-detector/datasets/vudenc"
CONFIG="config/train_config.yaml"
OUT_BASE="output-graphsec"

# Optional envs
SMOOTH="${SMOOTH:-3}"          # smoothing_window for plots (1 disables)
FORCE="${FORCE:-0}"            # FORCE=1 to delete existing OUTDIR
OUT_SUBDIR="${OUT_SUBDIR:-threshold_run}"

echo "[*] Pulling latest changes…"
git pull --ff-only

echo "[*] Building Docker image $IMAGE_NAME…"
hare build -t "$IMAGE_NAME" -f Dockerfile .
echo "[*] Image built!"

# Normalize YAML keys once
sed -i -E \
  -e 's/^[[:space:]]*learning_rate:/learning_rate:/' \
  -e 's/^[[:space:]]*weight_decay:/weight_decay:/' \
  -e 's/^[[:space:]]*dataset_name:/dataset_name:/' \
  -e 's/^[[:space:]]*output_dir:/output_dir:/' \
  -e 's/^[[:space:]]*threshold:/threshold:/' \
  -e 's/^[[:space:]]*smoothing_window:/smoothing_window:/' \
  "$CONFIG"

# Loop over your 7 exact runs
while read -r DS LR WD THR; do
  [[ -z "${DS:-}" ]] && continue

  OUTDIR="${OUT_BASE}/${DS}/${OUT_SUBDIR}"
  echo
  echo "[*] Starting: DS=${DS} LR=${LR} WD=${WD} THR=${THR}"
  echo "[*] Output:   ${OUTDIR}"

  if [[ -d "$OUTDIR" ]]; then
    if [[ "$FORCE" == "1" ]]; then
      echo "[*] FORCE=1 — removing existing $OUTDIR"
      rm -rf "$OUTDIR"
    else
      echo "[*] Skipping ${DS} (exists). Set FORCE=1 to rerun."
      continue
    fi
  fi
  mkdir -p "$OUTDIR"

  # Patch YAML for this run
  sed -i -E \
    -e "s|^dataset_name: .*|dataset_name: ${DS}|" \
    -e "s|^learning_rate: .*|learning_rate: ${LR}|" \
    -e "s|^weight_decay: .*|weight_decay: ${WD}|" \
    -e "s|^output_dir: .*|output_dir: ${OUTDIR}|" \
    "$CONFIG"

  # threshold (replace or append)
  if grep -q '^threshold:' "$CONFIG"; then
    sed -i -E -e "s|^threshold: .*|threshold: ${THR}|" "$CONFIG"
  else
    echo "threshold: ${THR}" >> "$CONFIG"
  fi

  # smoothing_window (replace or append)
  if grep -q '^smoothing_window:' "$CONFIG"; then
    sed -i -E -e "s|^smoothing_window: .*|smoothing_window: ${SMOOTH}|" "$CONFIG"
  else
    echo "smoothing_window: ${SMOOTH}" >> "$CONFIG"
  fi

  # Run containerized training
  hare run \
    --gpus "device=${GPU_DEVICE}" \
    -v "$(pwd)":/app \
    -v "${HOST_DATA}":/app/datasets/vudenc:ro \
    -v "$(pwd)/${OUT_BASE}":/app/output-graphsec \
    -p :6006 \
    "$IMAGE_NAME"

  echo "[*] Completed: ${DS}"
done <<'EOF'
sql 0.008 0.0010 0.5
command_injection 0.010 0.0010 0.5
remote_code_execution 0.008 0.0014 0.2
path_disclosure 0.008 0.0014 0.4
xss 0.008 0.0010 0.3
xsrf 0.010 0.0012 0.5
open_redirect 0.012 0.0010 0.4
EOF

echo
echo "[*] Done: 7 thresholded runs."
