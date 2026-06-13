#!/usr/bin/env bash
# MULTI-SEED consistent-code deploy sweep, to replace the single-seed tab:tpot with
# a seed-band / request-bootstrap CI on the oracle-miss (eps) and TPOT percentiles.
# Same selectors + budgets + current SEER code as run_deploy_consistent.sh, but
# loops --seed so the H2O-best eps ranking can be reported with a CI instead of a
# single draw. Writes per-(pol,budget,seed) JSON; aggregate with agg_deploy_seeds.py.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEER="${SEER_ROOT:-$ROOT/../SEER}"
PY="${PY:-/home/lzq/miniconda3/envs/csp-llm/bin/python}"
MODEL="${MODEL:-/public/model_zoo/Llama-3.1-8B-Instruct}"
CK2=$ROOT/experiments/predictors/xqp_closed_2view_h4.json
CKW=$ROOT/experiments/predictors/xqp_closed_within_h4.json
BG=$ROOT/experiments/predictors/guardkv_budgeter_a10.json
OUT=$ROOT/experiments/results/deploy_seeds; mkdir -p "$OUT"
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
POLS="${POLS:-h2o quest infinigen xqp xqpwithin guardkv}"
N=${N:-16}; NEW=${NEW:-48}; CTX=${CTX:-4096}
SEEDS="${SEEDS:-0 1 2 3 4}"

run () { # policy budget seed
  local pol=$1 b=$2 s=$3 extra="" name=$1
  case $pol in
    xqp)       extra="--xqp-ckpt $CK2" ;;
    xqpwithin) extra="--xqp-ckpt $CKW"; pol=xqp ;;
    guardkv)   extra="--scorer-ckpt $CK2 --budgeter-ckpt $BG" ;;
  esac
  $PY -m seer.eval.runner --model "$MODEL" --policy "$pol" $extra \
    --workload mooncake --context_length "$CTX" --num_requests "$N" \
    --max_new_tokens "$NEW" --hbm_budget "$b" --seed "$s" --slo "P99=50ms" \
    --out "$OUT/${name}_b${b}_s${s}.json" > "$OUT/${name}_b${b}_s${s}.log" 2>&1
  echo "[$(date +%H:%M:%S)] ${name}_b${b}_s${s} rc=$?"
}

for s in $SEEDS; do
  for b in 0.20 0.30; do
    for pol in $POLS; do run "$pol" "$b" "$s"; done
  done
done
echo "MULTI-SEED DEPLOY SWEEP DONE (dev=$CUDA_VISIBLE_DEVICES seeds=$SEEDS pols=$POLS) -> $OUT"
