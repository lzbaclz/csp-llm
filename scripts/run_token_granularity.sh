#!/usr/bin/env bash
# Direction 1: does the query view become COMPLEMENTARY (positive marginal over
# within+cross) at finer granularity? Sweep block_size {1,4,8,32}; block_size=1 is
# token granularity. Faithful per-token/head-max query (query_variants).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PY:-/home/lzq/miniconda3/envs/csp-llm/bin/python}"
ZOO="${MODEL_ZOO:-/public/model_zoo}"; M="$ZOO/Llama-3.1-8B-Instruct"
LOG="$ROOT/experiments/logs"; mkdir -p "$LOG"
N="${N:-2}"; NEW="${NEW:-32}"
echo "[$(date -Iseconds)] token-granularity sweep N=$N NEW=$NEW" > "$LOG/tokgran.log"

run () { # $1=gpu $2=block_size
  local gpu="$1" bs="$2" probe="/public/xqp_traces/probe_bs$2"
  rm -rf "$probe"; mkdir -p "$probe"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u experiments/run_quest_baseline.py --device cuda:0 \
    --n "$N" --max-new-tokens "$NEW" --block-size "$bs" --models "$M" \
    --tmpdir "$probe" --out "experiments/results/quest_bs${bs}.json" \
    > "$LOG/tokgran_bs${bs}.log" 2>&1
  echo "[$(date -Iseconds)] block_size=$bs done rc=$?" >> "$LOG/tokgran.log"
}

# GPU0 takes the heavy bs=1; GPU1 takes the lighter bs=4,8,32
( run 0 1 ) &
P0=$!
( run 1 32 ; run 1 8 ; run 1 4 ) &
P1=$!
wait $P0; wait $P1
echo "[$(date -Iseconds)] SWEEP DONE" >> "$LOG/tokgran.log"

# Aggregate the marginal-of-query-over-within+cross at each granularity
"$PY" - <<'PY' >> "$LOG/tokgran.log" 2>&1
import json, glob, os
print("\n=== query marginal over within+cross vs block_size (Llama) ===")
print(f"{'bs':>4} {'within':>7} {'cross':>7} {'q_cos':>7} {'q_max':>7} {'+q_cos':>8} {'+q_max':>8}")
for bs in (1,4,8,32):
    f=f"experiments/results/quest_bs{bs}.json"
    if not os.path.exists(f): continue
    d=json.load(open(f))
    for stem,v in d['per_model'].items():
        sv=v['single_view_auc']; mg=v['marginal']
        print(f"{bs:>4} {sv['f_within']:>7.3f} {sv['f_cross']:>7.3f} {sv['f_query']:>7.3f} "
              f"{sv['f_query_dotmax']:>7.3f} {mg['f_query']['delta']:>+8.4f} {mg['f_query_dotmax']['delta']:>+8.4f}")
PY
