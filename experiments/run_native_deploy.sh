#!/usr/bin/env bash
set -u
SEER=/home/lzq/codes/SEER; PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
CK=/home/lzq/codes/csp-llm/experiments/predictors/native_serving_scorer.json
OUT=/home/lzq/codes/csp-llm/experiments/results/native_deploy
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run () { # tag model policy budget dev extra
  CUDA_VISIBLE_DEVICES=$5 $PY -m seer.eval.runner --model "$2" --policy "$3" $6 \
    --workload mooncake --context_length 4096 --num_requests 24 --max_new_tokens 32 \
    --hbm_budget "$4" --slo "P99=50ms" --io_mode measured-dma \
    --out "$OUT/$1.json" > "$OUT/$1.log" 2>&1
  echo "[$(date +%H:%M:%S)] $1 rc=$?"
}
L=/public/model_zoo/Llama-3.1-8B-Instruct
M=/public/model_zoo/Mistral-7B-Instruct-v0.3
Q=/public/model_zoo/Qwen2.5-7B-Instruct
# Llama (trained-on): native vs h2o at 3 budgets
for b in 0.10 0.20 0.30; do
  run llama_native_b$b $L native $b 0 "--xqp-ckpt $CK" &
  run llama_h2o_b$b    $L h2o    $b 1 "" &
  wait
done
# transfer: Mistral + Qwen at b0.20 (native trained on Llama)
run mistral_native_b0.20 $M native 0.20 0 "--xqp-ckpt $CK" &
run mistral_h2o_b0.20    $M h2o    0.20 1 "" &
wait
run qwen_native_b0.20 $Q native 0.20 0 "--xqp-ckpt $CK" &
run qwen_h2o_b0.20    $Q h2o    0.20 1 "" &
wait
echo "NATIVE DEPLOY DONE -> $OUT"
