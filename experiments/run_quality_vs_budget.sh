#!/usr/bin/env bash
set -u
SEER=/home/lzq/codes/SEER; PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
CK2=/home/lzq/codes/csp-llm/experiments/predictors/xqp_closed_2view_h4.json
OUT=/home/lzq/codes/csp-llm/experiments/results/quality_vs_budget; mkdir -p "$OUT"
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
N=48; NEW=48; CTX=4096
run () { export LONGBENCH_PATH=/public/data_zoo/longbench/data/$1.jsonl
  $PY -m seer.eval.runner --model /public/model_zoo/Llama-3.1-8B-Instruct --policy xqp --xqp-ckpt $CK2 \
    --workload longbench --context_length $CTX --num_requests $N --max_new_tokens $NEW --hbm_budget $2 \
    --slo "P99=200ms" --io_mode measured-dma --chat --out "$OUT/${1}_xqp_b${2}.json" > "$OUT/${1}_xqp_b${2}.log" 2>&1
  echo "[$(date +%H:%M:%S)] ${1}_b${2} rc=$? $(grep -oE 'F1=[0-9.]+' "$OUT/${1}_xqp_b${2}.log"|tail -1)"
}
for ds in narrativeqa qasper; do
  for b in 0.30 0.40 0.50 0.70; do run $ds $b; done
done
echo "QUALITY VS BUDGET DONE -> $OUT"
