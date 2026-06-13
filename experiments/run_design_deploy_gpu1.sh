#!/usr/bin/env bash
set -u
ROOT=/home/lzq/codes/csp-llm; SEER=/home/lzq/codes/SEER
PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
MODEL=/public/model_zoo/Llama-3.1-8B-Instruct
CK2=$ROOT/experiments/predictors/xqp_closed_2view_h4.json
OUT=$ROOT/experiments/results/design_deploy; mkdir -p "$OUT"
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
N=16; NEW=48; CTX=4096
run () { local name=$1 pol=$2 extra=$3 b=$4
  $PY -m seer.eval.runner --model "$MODEL" --policy "$pol" $extra \
    --workload mooncake --context_length "$CTX" --num_requests "$N" \
    --max_new_tokens "$NEW" --hbm_budget "$b" --slo "P99=50ms" \
    --out "$OUT/${name}_b${b}.json" > "$OUT/${name}_b${b}.log" 2>&1
  echo "[$(date +%H:%M:%S)] ${name}_b${b} rc=$? $(grep -o 'F1=.*t=.*s' "$OUT/${name}_b${b}.log" | tail -1)"
}
for b in 0.10 0.20 0.30; do
  run h2o       h2o ""                                              "$b"
  run xqp       xqp "--xqp-ckpt $CK2"                               "$b"
  run xqpnative xqp "--xqp-ckpt $CK2 --native-cross"                "$b"
  run xqpfull   xqp "--xqp-ckpt $CK2 --native-cross --within-accum" "$b"
done
echo "DESIGN DEPLOY (GPU1) DONE -> $OUT"
