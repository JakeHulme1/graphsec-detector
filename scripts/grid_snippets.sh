#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
mkdir -p prepared

VULN=xss        # rerun per vulnerability

for n in 5 7; do
  for m in 128 200; do
    for t in 0 0.05; do
      TAG="n${n}_m${m}_t${t}"
      echo "-> $TAG"

      poetry run python -m pipeline.make_snippets $VULN \
        --step $n --context $m --occ-thresh $t --dump-only \
        --out-dir datasets/vudenc/processed/${VULN}_${TAG} \
        | tee "logs/${VULN}_${TAG}.log"          # capture the [STATS]
    done
  done
done
