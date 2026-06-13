#!/usr/bin/env bash
set -u
SEER=/home/lzq/codes/SEER; PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/home/lzq/codes/csp-llm/experiments/results/ceiling
log () { # tag model dev
  CUDA_VISIBLE_DEVICES=$3 $PY -m seer.eval.runner --model "$2" --policy h2o \
    --workload mooncake --context_length 4096 --num_requests 24 --max_new_tokens 32 \
    --hbm_budget 0.5 --slo "P99=50ms" --io_mode measured-dma \
    --out "$OUT/run_$1.json" --log-calib "$OUT/feat_$1.json" > "$OUT/run_$1.log" 2>&1
  echo "[$(date +%H:%M:%S)] $1 rc=$? rows=$($PY -c "import json;print(len(json.load(open('$OUT/feat_$1.json'))['rows']))" 2>/dev/null||echo NA)"
}
log llama   /public/model_zoo/Llama-3.1-8B-Instruct 0 &
log qwen    /public/model_zoo/Qwen2.5-7B-Instruct   1 &
wait
log mistral /public/model_zoo/Mistral-7B-Instruct-v0.3 0
echo "CEILING LOG DONE -> $OUT"
