#!/usr/bin/env bash
# =============================================================================
# run_ceiling_raisers.sh -- applied-review ceiling-raisers (#2 + #3), self-waiting.
# Polls until a GPU is (sustained) free, claims it, runs all campaigns SEQUENTIALLY
# on that GPU (never collides with a concurrent job). Fully resumable.
#
#   #2  real-eviction HBM : physically prune KV at a budget -> exact KV bytes + peak
#                           HBM (8B + 14B). KV bytes are a function of kept-count only
#                           => selector-independent => matched budget = equal real HBM.
#   #3  >=13B on-policy   : LongBench-F1 H2O-equivalence at 14B + 32B using the
#                           Llama-trained 2-view ckpt FROZEN (transfer-to-larger-scale);
#                           h2o / xqp / adakv at matched 20% budget -> tab:perlayer rows.
#   (Offline attention extraction is skipped: output_attentions OOMs at >=14B/4K, and
#    the on-policy F1 already answers Q3 -- if magnitude-only ties H2O at 14B/32B,
#    magnitude suffices and the equivalence holds at scale.)
#
#   nohup bash experiments/run_ceiling_raisers.sh > experiments/logs/ceiling.log 2>&1 &
# =============================================================================
set -u
CSP=/home/lzq/codes/csp-llm
PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
ZOO=/public/model_zoo
LOG=$CSP/experiments/logs; mkdir -p "$LOG"
THRESH_MIB=${THRESH_MIB:-70000}                 # need a near-free 80GB GPU for 32B
DSLIST="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique triviaqa"

free_mib () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1" 2>/dev/null; }
wait_for_gpu () {
  echo "[wait] need >=${THRESH_MIB}MiB free (sustained)..." >&2
  while true; do
    for g in 0 1; do
      f=$(free_mib "$g"); [ -z "$f" ] && continue
      if [ "$f" -ge "$THRESH_MIB" ]; then
        sleep 25; f2=$(free_mib "$g")
        [ "$f2" -ge "$THRESH_MIB" ] && { echo "$g"; return; }
      fi
    done
    sleep 60
  done
}

DEV=$(wait_for_gpu)
export CUDA_VISIBLE_DEVICES="$DEV"
export PYTHONPATH=/home/lzq/codes/SEER TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$CSP"
echo "[ceiling] claimed physical GPU $DEV at $(date)"

# ---- #2 real-eviction HBM (selector-independent; 8B + 14B) ----
for M in Llama-3.1-8B-Instruct Qwen2.5-14B-Instruct; do
  out="experiments/results/real_eviction_hbm_${M}.json"
  [ -s "$out" ] && { echo "[ceiling] SKIP real-evict $M"; continue; }
  echo "[ceiling] real-eviction HBM: $M $(date)"
  "$PY" experiments/run_real_eviction_hbm.py --model "$M" --device cuda:0 \
     --ctx 4096 --n 4 --new 32 --budgets "1.0 0.5 0.3 0.2 0.1" --out "$out" \
     >> "$LOG/real_eviction_${M}.log" 2>&1 || echo "[ceiling] real-evict $M FAILED rc=$?"
done

# ---- #3 on-policy F1 at 14B + 32B (Llama 2-view ckpt FROZEN; transfer-to-scale) ----
#  (XQP_CKPT unset -> run_budget_generic uses the Llama-trained xqp_closed_2view_h4.json)
run_onpolicy () {  # $1=model  $2=ctx  $3=N
  local M="$1" CTX="$2" N="$3"; local tag; tag=$(echo "$M" | tr 'A-Z.' 'a-z_')
  echo "[ceiling] on-policy F1: $M ctx=$CTX N=$N $(date)"
  MODEL="$ZOO/$M" OUTROOT="experiments/results/largemodel/$tag" \
    DSLIST="$DSLIST" BUDGETS="0.20" SELLIST="h2o xqp adakv" N="$N" CTX="$CTX" \
    bash experiments/run_budget_generic.sh >> "$LOG/onpolicy_${tag}.log" 2>&1 \
    || echo "[ceiling] on-policy $M FAILED rc=$?"
}
run_onpolicy Qwen2.5-14B-Instruct 4096 64
run_onpolicy Qwen2.5-32B-Instruct 2048 32

echo "[ceiling] FINISHED $(date)"
