#!/usr/bin/env bash
set -euo pipefail

mkdir -p prepared

VULNS=(
  path_disclosure_n7_m128_t5_ds30
  open_redirect_n7_m128_t5_ds30
  command_injection_n7_m128_t5
  sql_n7_m128_t5
  xsrf_n7_m128_t5_ds30
  xss_n7_m128_t5
  remote_code_execution_n7_m128_t5_ds30
)

for vuln in "${VULNS[@]}"; do

  # one shared folder per vuiln
  VDIR="datasets/vudenc/prepared/${vuln}"
  mkdir -p "$VDIR"

    for split in train val test; do
    echo "-> ${vuln}_${split}"

    poetry run python -m pipeline.extract_dfg \
      "datasets/vudenc/splits/${vuln}_${split}.jsonl" \
      "$VDIR" \
      "${split}.jsonl"
  done
done