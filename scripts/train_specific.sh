  # --- unique completion marker per LR/WD/THR
  thr_safe="${THR//./p}"; lr_safe="${LR//./p}"; wd_safe="${WD//./p}"
  OUTDIR="${OUT_BASE}/${DS}/${OUT_SUBDIR}"
  DONE="${OUTDIR}/.done_${lr_safe}_${wd_safe}_${thr_safe}"

  echo
  echo "[*] Starting: DS=${DS} LR=${LR} WD=${WD} THR=${THR}"
  echo "[*] Output:   ${OUTDIR}"

  # Only skip if this exact combo is done (not merely because dir exists)
  if [[ "${FORCE}" != "1" && -f "${DONE}" ]]; then
    echo "[*] Skipping ${DS} (done for lr=${LR}, wd=${WD}, thr=${THR}). Set FORCE=1 to rerun."
    continue
  fi

  # If FORCE=1, wipe; otherwise keep existing files and append new results
  if [[ "${FORCE}" == "1" ]]; then
    echo "[*] FORCE=1 — removing existing ${OUTDIR}"
    rm -rf "${OUTDIR}"
  fi
  mkdir -p "${OUTDIR}"

  # Patch YAML for this run (note absolute path so container writes into mount)
  sed -i -E \
    -e "s|^dataset_name: .*|dataset_name: ${DS}|" \
    -e "s|^learning_rate: .*|learning_rate: ${LR}|" \
    -e "s|^weight_decay: .*|weight_decay: ${WD}|" \
    -e "s|^output_dir: .*|output_dir: /app/${OUTDIR}|" \
    "$CONFIG"

  # threshold + smoothing
  if grep -q '^threshold:' "$CONFIG"; then
    sed -i -E -e "s|^threshold: .*|threshold: ${THR}|" "$CONFIG"
  else
    echo "threshold: ${THR}" >> "$CONFIG"
  fi
  if grep -q '^smoothing_window:' "$CONFIG"; then
    sed -i -E -e "s|^smoothing_window: .*|smoothing_window: ${SMOOTH}|" "$CONFIG"
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
