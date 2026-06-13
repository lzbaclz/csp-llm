#!/usr/bin/env bash
# Trace collection on ga100. Runs the upstream SEER hook to record the
# 4 XQP features per (layer, block, step) plus future-step labels.
#
# Usage:
#   ssh ga100
#   cd ~/codes/papers/next1
#   bash scripts/collect_traces.sh
#
# Outputs JSONL traces under ./experiments/traces/ — one file per model.
set -euo pipefail

OUTDIR="experiments/traces"
mkdir -p "$OUTDIR"

MODELS=(
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
)
N_TRACES=200
MAX_CONTEXT=4096

for model in "${MODELS[@]}"; do
  short=$(basename "$model" | tr / _)
  echo "[$(date +%T)] collecting $model ..."
  python -m xqp.trace_collect_cli \
    --model "$model" \
    --workload mooncake-chat \
    --n-traces "$N_TRACES" \
    --max-context "$MAX_CONTEXT" \
    --out "$OUTDIR/${short}.jsonl" \
    --attn-impl eager \
    --dtype fp16
done

echo "[$(date +%T)] DONE. Traces in $OUTDIR/"
