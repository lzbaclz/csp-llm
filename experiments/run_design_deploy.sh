#!/usr/bin/env bash
# Full design comparison in deployment (mooncake, Llama-3.1-8B, measured-DMA).
# Question: can a calibrated 2-view selector BEAT the strongest heuristic (H2O) in
# deployment by (a) the free native cross-layer indicator and (b) an H2O-style
# accumulated within feature — recovering the offline +0.10 recall@aggressive-budget?
# Policies:
#   h2o            : strongest deployable heuristic (the bar to beat)
#   xqp            : learned 2-view, reconstructed cross + short-EMA within (current)
#   xqpnative      : + native cross indicator
#   xqpfull        : + native cross + H2O-style accumulated within (the design)
#
#   bash experiments/run_design_deploy.sh
set -u
ROOT=/home/lzq/codes/csp-llm; SEER=/home/lzq/codes/SEER
PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
MODEL=/public/model_zoo/Llama-3.1-8B-Instruct
CK2=$ROOT/experiments/predictors/xqp_closed_2view_h4.json
OUT=$ROOT/experiments/results/design_deploy; mkdir -p "$OUT"
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N=16; NEW=48; CTX=4096

run () { # name extra budget dev
  local name=$1 extra=$2 b=$3 dev=$4
  CUDA_VISIBLE_DEVICES=$dev $PY -m seer.eval.runner --model "$MODEL" --policy "${5:-xqp}" $extra \
    --workload mooncake --context_length "$CTX" --num_requests "$N" \
    --max_new_tokens "$NEW" --hbm_budget "$b" --slo "P99=50ms" \
    --out "$OUT/${name}_b${b}.json" > "$OUT/${name}_b${b}.log" 2>&1
  echo "[$(date +%H:%M:%S)] ${name}_b${b} rc=$? $(grep -o 'F1=.*t=.*s' "$OUT/${name}_b${b}.log" | tail -1)"
}

for b in 0.10 0.20 0.30; do
  run h2o       ""                                              "$b" 0 h2o &
  run xqpfull   "--xqp-ckpt $CK2 --native-cross --within-accum" "$b" 1 xqp &
  wait
  run xqp       "--xqp-ckpt $CK2"                               "$b" 0 xqp &
  run xqpnative "--xqp-ckpt $CK2 --native-cross"                "$b" 1 xqp &
  wait
done
echo "DESIGN DEPLOY DONE -> $OUT"
