#!/usr/bin/env bash
# Decisive deployment test for the native cross-layer indicator.
# Offline, the 2-view (within+cross) beats the reactive within/H2O signal by
# +0.10 recall@10% — but the *deployment* reconstructs the cross view from the
# previous layer's within-EMA top-r, which throws the advantage away (MIN14), so
# H2O wins in deployment. Here we plumb the NATIVE cross indicator (the true
# prev-layer top-r by real attention, free at decode) via `--native-cross` and ask:
# does the learned 2-view now beat H2O, and is the win largest at aggressive budget?
#
#   bash experiments/run_native_cross_deploy.sh
set -u
ROOT=/home/lzq/codes/csp-llm; SEER=/home/lzq/codes/SEER
PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
MODEL=/public/model_zoo/Llama-3.1-8B-Instruct
CK2=$ROOT/experiments/predictors/xqp_closed_2view_h4.json
OUT=$ROOT/experiments/results/native_cross; mkdir -p "$OUT"
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N=16; NEW=48; CTX=4096

run () { # name policy extra budget dev
  local name=$1 pol=$2 extra=$3 b=$4 dev=$5
  CUDA_VISIBLE_DEVICES=$dev $PY -m seer.eval.runner --model "$MODEL" --policy "$pol" $extra \
    --workload mooncake --context_length "$CTX" --num_requests "$N" \
    --max_new_tokens "$NEW" --hbm_budget "$b" --slo "P99=50ms" \
    --out "$OUT/${name}_b${b}.json" > "$OUT/${name}_b${b}.log" 2>&1
  echo "[$(date +%H:%M:%S)] ${name}_b${b} rc=$? $(grep -o 'F1=.*t=.*s' "$OUT/${name}_b${b}.log" | tail -1)"
}

for b in 0.10 0.20; do
  # GPU0: h2o + xqp(reconstructed) ; GPU1: xqp-native — run the pair concurrently
  run h2o       h2o ""                          "$b" 0 &
  run xqpnative xqp "--xqp-ckpt $CK2 --native-cross" "$b" 1 &
  wait
  run xqp       xqp "--xqp-ckpt $CK2"           "$b" 0
done
echo "NATIVE-CROSS DEPLOY DONE -> $OUT"
