#!/usr/bin/env bash
# Decisive test for the cross-layer query-free prefetch MODULE in an IO-BOUND regime
# (simulated CPU/NVMe offload: analytical IO with high ell_bar). Does predicting layer
# l's hot blocks from layer l-1's attention (87.5% recall, query-free, issued a layer
# ahead) HIDE the recall latency vs no prefetch? vLLM/FlexGen don't do cross-layer
# attention-predicted prefetch; query-aware methods can't issue it a layer ahead.
set -u
SEER=/home/lzq/codes/SEER; PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
OUT=/home/lzq/codes/csp-llm/experiments/results/iobound_prefetch; mkdir -p "$OUT"
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/public/model_zoo/Llama-3.1-8B-Instruct
run () { # tag prefetch_args dev
  CUDA_VISIBLE_DEVICES=$3 $PY -m seer.eval.runner --model $M --policy h2o \
    --workload mooncake --context_length 4096 --num_requests 12 --max_new_tokens 32 \
    --hbm_budget 0.10 --slo "P99=200ms" --io_mode analytical --ell_bar_us 6000 $2 \
    --out "$OUT/$1.json" > "$OUT/$1.log" 2>&1
  echo "[$(date +%H:%M:%S)] $1 rc=$? $(grep -oE 'P50=[0-9.]+ms.*P99=[0-9.]+ms' "$OUT/$1.log"|tail -1)"
}
run noprefetch     "--no_prefetch"               0 &
run xlayer_prefetch "--prefetch_source oracle_prev" 1 &
wait
echo "IOBOUND PREFETCH DONE -> $OUT"
