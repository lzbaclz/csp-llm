#!/usr/bin/env bash
set -u
SEER=/home/lzq/codes/SEER; PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
CK2=/home/lzq/codes/csp-llm/experiments/predictors/xqp_closed_2view_h4.json
BG=/home/lzq/codes/csp-llm/experiments/predictors/guardkv_budgeter_a10_h4.json
OUT=/home/lzq/codes/csp-llm/experiments/results/reground_f1; mkdir -p "$OUT"
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N=48; NEW=48; CTX=4096; B=0.20
run () { export LONGBENCH_PATH=/public/data_zoo/longbench/data/$2.jsonl
  CUDA_VISIBLE_DEVICES=$4 $PY -m seer.eval.runner --model /public/model_zoo/Llama-3.1-8B-Instruct --policy "$3" $5 \
    --workload longbench --context_length $CTX --num_requests $N --max_new_tokens $NEW --hbm_budget $B \
    --slo "P99=200ms" --io_mode measured-dma --chat --out "$OUT/$1.json" > "$OUT/$1.log" 2>&1
  echo "[$(date +%H:%M:%S)] $1 rc=$? $(grep -oE 'F1=[0-9.]+' "$OUT/$1.log"|tail -1)"
}
for ds in narrativeqa qasper; do
  run ${ds}_quest $ds quest 0 "" & run ${ds}_guardkv $ds guardkv 1 "--scorer-ckpt $CK2 --budgeter-ckpt $BG" & wait
done
echo "REGROUND DONE (full/h2o/xqp already in e2e_confirm; this adds quest+guardkv)"
