#!/usr/bin/env bash
# CONSISTENT-CODE deploy sweep: the committed deploy/ dir mixes code versions (the
# SEER XQPPolicy adapter landed 2026-06-01, AFTER the May-30 h2o/quest/infinigen
# sweep that tab:tpot cites). Re-run EVERY selector with the CURRENT SEER code in
# one sweep so the oracle-miss (eps) comparison is apples-to-apples. Includes the
# within-only XQP (xqp_closed_within_h4.json) so we can test the paper's claim that
# the deployable 2-view collapses to within-only (eps_2view ~= eps_within).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEER="${SEER_ROOT:-$ROOT/../SEER}"
PY="${PY:-/home/lzq/miniconda3/envs/csp-llm/bin/python}"
MODEL="${MODEL:-/public/model_zoo/Llama-3.1-8B-Instruct}"
CK2=$ROOT/experiments/predictors/xqp_closed_2view_h4.json
CKW=$ROOT/experiments/predictors/xqp_closed_within_h4.json
BG=$ROOT/experiments/predictors/guardkv_budgeter_a10.json
OUT=$ROOT/experiments/results/deploy_consistent; mkdir -p "$OUT"
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
POLS="${POLS:-h2o quest infinigen xqp xqpwithin guardkv}"
N=${N:-16}; NEW=${NEW:-48}; CTX=${CTX:-4096}

run () { # policy budget
  local pol=$1 b=$2 extra="" name=$1
  case $pol in
    xqp)       extra="--xqp-ckpt $CK2" ;;
    xqpwithin) extra="--xqp-ckpt $CKW"; pol=xqp ;;
    guardkv)   extra="--scorer-ckpt $CK2 --budgeter-ckpt $BG" ;;
  esac
  $PY -m seer.eval.runner --model "$MODEL" --policy "$pol" $extra \
    --workload mooncake --context_length "$CTX" --num_requests "$N" \
    --max_new_tokens "$NEW" --hbm_budget "$b" --slo "P99=50ms" \
    --out "$OUT/${name}_b${b}.json" > "$OUT/${name}_b${b}.log" 2>&1
  echo "[$(date +%H:%M:%S)] ${name}_b${b} rc=$? $(grep -o 'F1=.*t=.*s' "$OUT/${name}_b${b}.log" | tail -1)"
}

for b in 0.20 0.30; do
  for pol in $POLS; do run "$pol" "$b"; done
done
echo "CONSISTENT DEPLOY SWEEP DONE (dev=$CUDA_VISIBLE_DEVICES pols=$POLS) -> $OUT"
