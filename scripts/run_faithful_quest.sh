#!/usr/bin/env bash
# Decisive faithful-query experiment: extract the raw-dot per-token/per-head-MAX
# Quest signal (query_variants) and test whether it warrants the gated cascade.
# Parallel: Llama on GPU0, Qwen2.5 on GPU1. Writes to /public (never home).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PY:-/home/lzq/miniconda3/envs/csp-llm/bin/python}"
ZOO="${MODEL_ZOO:-/public/model_zoo}"
PROBE="${PROBE:-/public/xqp_traces/quest_probe}"
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
LOG="$ROOT/experiments/logs"; mkdir -p "$LOG"
rm -rf "$PROBE"; mkdir -p "$PROBE"
N="${N:-12}"; NEW="${NEW:-64}"
echo "[$(date -Iseconds)] faithful-quest start N=$N NEW=$NEW PROBE=$PROBE" > "$LOG/quest_faithful.log"

CUDA_VISIBLE_DEVICES=0 "$PY" -u experiments/run_quest_baseline.py --device cuda:0 --n "$N" \
  --max-new-tokens "$NEW" --models "$ZOO/Llama-3.1-8B-Instruct" --tmpdir "$PROBE" \
  --out experiments/results/quest_baseline_llama.json > "$LOG/quest_faithful_llama.log" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 "$PY" -u experiments/run_quest_baseline.py --device cuda:0 --n "$N" \
  --max-new-tokens "$NEW" --models "$ZOO/Qwen2.5-7B-Instruct" --tmpdir "$PROBE" \
  --out experiments/results/quest_baseline_qwen.json > "$LOG/quest_faithful_qwen.log" 2>&1 &
P1=$!
wait $P0; rc0=$?
wait $P1; rc1=$?
echo "[$(date -Iseconds)] EXTRACTION DONE rc=($rc0,$rc1)" >> "$LOG/quest_faithful.log"

# Stratified warrant on the SAME blocks, mean vs faithful query
"$PY" -u experiments/run_quest_stratified.py --traces "$PROBE" --query-field f_query \
  --out experiments/results/quest_stratified_mean.json >> "$LOG/quest_faithful.log" 2>&1
"$PY" -u experiments/run_quest_stratified.py --traces "$PROBE" --query-field f_query_dotmax \
  --out experiments/results/quest_stratified_faithful.json >> "$LOG/quest_faithful.log" 2>&1
echo "[$(date -Iseconds)] ALL DONE" >> "$LOG/quest_faithful.log"
