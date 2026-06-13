#!/usr/bin/env bash
set -euo pipefail
mkdir -p experiments/predictors

for trace in experiments/traces/*.jsonl; do
  short=$(basename "$trace" .jsonl)
  echo "[$(date +%T)] training $short ..."
  for h in h1 h4 h16 h64; do
    xqp-train --trace "$trace" --horizon "$h" \
              --out "experiments/predictors/${short}_${h}.json" \
              > "experiments/predictors/${short}_${h}.log"
  done
  xqp-train --trace "$trace" --horizon h4 --per-layer \
            --out "experiments/predictors/${short}_h4_perlayer.json"
done

echo "[$(date +%T)] DONE"
