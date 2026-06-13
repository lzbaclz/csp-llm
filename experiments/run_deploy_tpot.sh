#!/usr/bin/env bash
# Deployment study: run the SEER masking simulator with each selector at fixed
# HBM budgets, on the real Llama-3.1-8B + mooncake. Reports TPOT percentiles and
# the mean per-step oracle-miss (eps = fraction of truly-attended blocks evicted).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEER="${SEER_ROOT:-$ROOT/../SEER}"
PY="${PY:-/home/lzq/miniconda3/envs/csp-llm/bin/python}"
MODEL_ZOO="${MODEL_ZOO:-/public/model_zoo}"
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL="${MODEL:-$MODEL_ZOO/Llama-3.1-8B-Instruct}"
CKPT=$ROOT/experiments/predictors/xqp_closed_2view_h4.json
BG=$ROOT/experiments/predictors/guardkv_budgeter_a10.json   # GuardKV budgeter (E4)
OUT=$ROOT/experiments/results/deploy
mkdir -p "$OUT"
N=${N:-16}; NEW=${NEW:-48}; CTX=${CTX:-4096}
POLS="${POLS:-h2o quest infinigen xqp guardkv}"   # override to re-run a subset

run () { # $1=policy $2=budget $3=extra
  local pol=$1 b=$2
  echo "[$(date +%H:%M:%S)] policy=$pol budget=$b"
  $PY -m seer.eval.runner --model "$MODEL" --policy "$pol" $3 \
    --workload mooncake --context_length "$CTX" --num_requests "$N" \
    --max_new_tokens "$NEW" --hbm_budget "$b" --slo P99=50ms \
    --out "$OUT/${pol}_b${b}.json" > "$OUT/${pol}_b${b}.log" 2>&1
  echo "  rc=$? $(grep -o '\[runner\] F1=.*t=.*s' "$OUT/${pol}_b${b}.log" | tail -1)"
}

for b in 0.20 0.30; do
  for pol in $POLS; do
    case $pol in
      xqp)     run "$pol" "$b" "--xqp-ckpt $CKPT" ;;
      guardkv) run "$pol" "$b" "--scorer-ckpt $CKPT --budgeter-ckpt $BG" ;;
      *)       run "$pol" "$b" "" ;;
    esac
  done
done
run full 1.00 ""   # quality ceiling reference

echo "ALL DEPLOY RUNS DONE"
