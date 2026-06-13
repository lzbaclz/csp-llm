#!/usr/bin/env bash
set -u
SEER=/home/lzq/codes/SEER; PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
CK2=/home/lzq/codes/csp-llm/experiments/predictors/xqp_closed_2view_h4.json
NAT=/home/lzq/codes/csp-llm/experiments/predictors/native_serving_scorer.json
OUT=/home/lzq/codes/csp-llm/experiments/results/e2e_quality
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LONGBENCH_PATH=/public/data_zoo/longbench/data/narrativeqa.jsonl
N=20; NEW=48; CTX=4096
run () { # tag policy budget dev extra
  CUDA_VISIBLE_DEVICES=$4 $PY -m seer.eval.runner --model /public/model_zoo/Llama-3.1-8B-Instruct --policy "$2" $5 \
    --workload longbench --context_length $CTX --num_requests $N --max_new_tokens $NEW --hbm_budget "$3" \
    --slo "P99=200ms" --io_mode measured-dma --chat --out "$OUT/$1.json" > "$OUT/$1.log" 2>&1
  echo "[$(date +%H:%M:%S)] $1 rc=$? $(grep -oE 'F1=[0-9.]+' "$OUT/$1.log"|tail -1)"
}
run full_b1.0 full 1.0 0 "" &
run h2o_b0.10 h2o 0.10 1 "" & wait
run xqp_b0.10 xqp 0.10 0 "--xqp-ckpt $CK2" &
run native_b0.10 native 0.10 1 "--xqp-ckpt $NAT" & wait
run h2o_b0.20 h2o 0.20 0 "" &
run xqp_b0.20 xqp 0.20 1 "--xqp-ckpt $CK2" & wait
run native_b0.20 native 0.20 0 "--xqp-ckpt $NAT" &
run h2o_b0.30 h2o 0.30 1 "" & wait
run xqp_b0.30 xqp 0.30 0 "--xqp-ckpt $CK2" & wait
echo "E2E QUALITY DONE -> $OUT"
