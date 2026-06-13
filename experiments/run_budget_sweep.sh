#!/usr/bin/env bash
# =============================================================================
# run_budget_sweep.sh -- budget-at-fixed-quality (applied-review must-fix #1).
# Invert tab:perlayer: instead of F1 at a fixed 20% budget, sweep the budget and
# find the budget at which each selector reaches a fixed task-quality target.
# If the learned 2-view reaches the target at a LOWER budget than H2O -> a memory
# win at equal quality (the adoption metric). If equal -> honest parity reframe.
# Llama-3.1-8B, 7 LongBench QA datasets, {h2o, xqp, adakv} at budgets {.10,.30,.50};
# the 0.20 anchor is reused from experiments/results/expand/llama31_8b.
# MAXJOBS=1 (GPU1 only; GPU0 is a colleague's). Resumable.
# =============================================================================
set -u
SEER=/home/lzq/codes/SEER
PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
LCK=/home/lzq/codes/csp-llm/experiments/predictors/xqp_closed_2view_h4.json
LB_DIR=/public/data_zoo/longbench/data
MODEL=/public/model_zoo/Llama-3.1-8B-Instruct
OUT=/home/lzq/codes/csp-llm/experiments/results/budget_sweep
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
DATASETS=(narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique triviaqa)
SELECTORS=(h2o xqp adakv)
BUDGETS=(0.10 0.30 0.50)
N=64; NEW=48; CTX=4096; SLO="P99=200ms"
is_done () { [ -s "$1" ] && "$PY" - "$1" <<'P' >/dev/null 2>&1
import json,sys;sys.exit(0 if len(json.load(open(sys.argv[1])).get("results",[]))>=1 else 1)
P
}
echo "[budget] start $(date)"
for B in "${BUDGETS[@]}"; do
  od="$OUT/b$B"; mkdir -p "$od"
  for ds in "${DATASETS[@]}"; do
    [ -s "$LB_DIR/$ds.jsonl" ] || continue
    for sel in "${SELECTORS[@]}"; do
      out="$od/${ds}_${sel}.json"; log="${out%.json}.log"
      if is_done "$out"; then echo "[budget] SKIP b$B/$ds/$sel"; continue; fi
      extra=""; [ "$sel" = "xqp" ] && extra="--xqp-ckpt $LCK"
      echo "[budget] RUN b$B/$ds/$sel"
      LONGBENCH_PATH="$LB_DIR/$ds.jsonl" "$PY" -m seer.eval.runner \
        --model "$MODEL" --policy "$sel" $extra --workload longbench \
        --context_length "$CTX" --num_requests "$N" --max_new_tokens "$NEW" \
        --hbm_budget "$B" --slo "$SLO" --io_mode measured-dma --chat --seed 0 \
        --out "$out" > "$log" 2>&1
      echo "[budget] DONE b$B/$ds/$sel rc=$? $(grep -oE 'F1=[0-9.]+' "$log"|tail -1)"
    done
  done
done
echo "[budget] FINISHED $(date)"
