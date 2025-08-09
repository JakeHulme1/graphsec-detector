#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

GPU_DEVICE="${1:-2}"
IMAGE_NAME="joh46/graphsec-detector:gpu"
HOST_DATA="/mnt/faster0/joh46/graphsec-detector/datasets/vudenc"
CONFIG="config/train_config.yaml"
OUT_BASE="output-graphsec"

SMOOTH="${SMOOTH:-3}"
FORCE="${FORCE:-0}"
OUT_SUBDIR="${OUT_SUBDIR:-threshold_run}"

# sanity checks
[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG"; exit 1; }

# normalize YAML keys once
sed -i -E \
  -e 's/^[[:space:]]*learning_rate:/learning_rate:/' \
  -e 's/^[[:space:]]*weight_decay:/weight_decay:/' \
  -e 's/^[[:space:]]*dataset_name:/dataset_name:/' \
  -e 's/^[[:space:]]*output_dir:/output_dir:/' \
  -e 's/^[[:space:]]*threshold:/threshold:/' \
  -e 's/^[[:space:]]*smoothing_window:/smoothing_window:/' \
  "$CONFIG"

while read -r DS LR WD THR; do
  [[ -z "${DS:-}" ]] && continue

  thr_safe="${THR//./p}"; lr_safe="${LR//./p}"; wd_safe="${WD//./p}"
  OUTDIR="${OUT_BASE}/${DS}/${OUT_SUBDIR}"
  DONE="${OUTDIR}/.done_${lr_safe}_${wd_safe}_${thr_safe}"

  echo
  echo "[*] Starting: DS=${DS} LR=${LR} WD=${WD} THR=${THR}"
  echo "[*] Output:   ${OUTDIR}"

  if [[ "${FORCE}" != "1" && -f "${DONE}" ]]; then
    echo "[*] Skipping ${DS} (done for lr=${LR}, wd=${WD}, thr=${THR}). Set FORCE=1 to rerun."
    continue
  fi

  [[ "${FORCE}" == "1" ]] && { echo "[*] FORCE=1 — removing ${OUTDIR}"; rm -rf "${OUTDIR}"; }
  mkdir -p "${OUTDIR}"

  # Patch YAML for this run; note absolute output path inside the container.
  sed -i -E \
    -e "s|^dataset_name: .*|dataset_name: ${DS}|" \
    -e "s|^learning_rate: .*|learning_rate: ${LR}|" \
    -e "s|^weight_decay: .*|weight_decay: ${WD}|" \
    -e "s|^output_dir: .*|output_dir: /app/${OUTDIR}|" \
    "$CONFIG"

  if grep -q '^threshold:' "$CONFIG"; then
    sed -i -E "s|^threshold: .*|threshold: ${THR}|" "$CONFIG"
  else
    echo "threshold: ${THR}" >> "$CONFIG"
  fi

  if grep -q '^smoothing_window:' "$CONFIG"; then
    sed -i -E "s|^smoothing_window: .*|smoothing_window: ${SMOOTH}|" "$CONFIG"
  else
    echo "smoothing_window: ${SMOOTH}" >> "$CONFIG"
  fi

  # Run (keep your exact port mapping)
  set +e
  hare run \
    --gpus "device=${GPU_DEVICE}" \
    -v "$(pwd)":/app \
    -v "${HOST_DATA}":/app/datasets/vudenc:ro \
    -v "$(pwd)/${OUT_BASE}":/app/output-graphsec \
    -p 10008:6006 \
    "$IMAGE_NAME"
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "!! ${DS} failed with exit ${rc} — continuing"
    continue
  fi

  touch "${DONE}"
  echo "[*] Completed: ${DS} (lr=${LR}, wd=${WD}, thr=${THR})"
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
echo "[*] Done: thresholded runs."
