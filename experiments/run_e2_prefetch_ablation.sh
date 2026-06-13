#!/usr/bin/env bash
# E2 (clean): does the query-free cross-layer prefetch hide recall latency?
# A MATCHED prefetch on/off ablation on the SAME GuardKV policy and SAME seed
# (replacing the earlier guardkv-vs-xqp comparison, which confounded the budgeter
# AND the prefetch path). Multi-seed to get a run-to-run TPOT noise band so the
# tail deltas can be reported with a CI instead of a single-seed point.
# Deploy regime: mooncake, 4K, measured-DMA IO (matches experiments/results/deploy).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEER="${SEER_ROOT:-$ROOT/../SEER}"
PY="${PY:-/home/lzq/miniconda3/envs/csp-llm/bin/python}"
MODEL="${MODEL:-/public/model_zoo/Llama-3.1-8B-Instruct}"
SC=$ROOT/experiments/predictors/xqp_closed_2view_h4.json
BG=$ROOT/experiments/predictors/guardkv_budgeter_a10.json
OUT=$ROOT/experiments/results/e2_ablation; mkdir -p "$OUT"
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SEEDS="${SEEDS:-0 1 2}"
N=${N:-16}; NEW=${NEW:-48}; CTX=${CTX:-4096}

run () { # seed budget cond
  local s=$1 b=$2 cond=$3
  local extra=""; [ "$cond" = "off" ] && extra="--no_prefetch"
  local tag="gk_${cond}_b${b}_s${s}"
  $PY -m seer.eval.runner --model "$MODEL" --policy guardkv \
    --scorer-ckpt "$SC" --budgeter-ckpt "$BG" \
    --workload mooncake --context_length "$CTX" --num_requests "$N" \
    --max_new_tokens "$NEW" --hbm_budget "$b" --seed "$s" $extra \
    --slo "P99=50ms" --out "$OUT/${tag}.json" > "$OUT/${tag}.log" 2>&1
  echo "[$(date +%H:%M:%S)] $tag rc=$? $(grep -o 'F1=.*t=.*s' "$OUT/${tag}.log" | tail -1)"
}

for s in $SEEDS; do
  for b in 0.20 0.30; do
    run "$s" "$b" on
    run "$s" "$b" off
  done
done
echo "E2 ABLATION DONE (dev=$CUDA_VISIBLE_DEVICES seeds=$SEEDS) -> $OUT"
