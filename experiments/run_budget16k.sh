#!/usr/bin/env bash
# 16K budget sweep: invert the budget axis at LONG context (must-fix #1, context regime).
# h2o vs xqp at budgets {0.10,0.30} (0.20 already in longctx/c16384_n64), 7 LongBench QA,
# N=64, chunked prefill. Pairs with the 4K sweep to make "no memory win" a 4K+16K grid.
set -u
SEER=/home/lzq/codes/SEER; PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
LCK=/home/lzq/codes/csp-llm/experiments/predictors/xqp_closed_2view_h4.json
LB_DIR=/public/data_zoo/longbench/data; MODEL=/public/model_zoo/Llama-3.1-8B-Instruct
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATASETS=(narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique triviaqa)
SELECTORS=(h2o xqp); BUDGETS=(${BUDGETS:-0.10 0.30}); CTX=16384; N=64; NEW=48; SLO="P99=4000ms"
is_done () { [ -s "$1" ] && "$PY" - "$1" <<'P' >/dev/null 2>&1
import json,sys;sys.exit(0 if len(json.load(open(sys.argv[1])).get("results",[]))>=1 else 1)
P
}
echo "[b16k] start $(date) GPU=$CUDA_VISIBLE_DEVICES budgets=${BUDGETS[*]}"
for B in "${BUDGETS[@]}"; do
  OUT=/home/lzq/codes/csp-llm/experiments/results/longctx/budget16k/b${B}; mkdir -p "$OUT"
  for ds in "${DATASETS[@]}"; do
    [ -s "$LB_DIR/$ds.jsonl" ] || continue
    for sel in "${SELECTORS[@]}"; do
      out="$OUT/${ds}_${sel}.json"; log="${out%.json}.log"
      is_done "$out" && { echo "[b16k] SKIP b$B/$ds/$sel"; continue; }
      extra=""; [ "$sel" = "xqp" ] && extra="--xqp-ckpt $LCK"
      LONGBENCH_PATH="$LB_DIR/$ds.jsonl" "$PY" -m seer.eval.runner \
        --model "$MODEL" --policy "$sel" $extra --workload longbench \
        --context_length "$CTX" --num_requests "$N" --max_new_tokens "$NEW" \
        --hbm_budget "$B" --slo "$SLO" --io_mode measured-dma --chat --seed 0 \
        --prefill_chunk 2048 --no_prefill_attn --skip_prewarm --out "$out" > "$log" 2>&1
      echo "[b16k] DONE b$B/$ds/$sel rc=$? $(grep -oE 'F1=[0-9.]+' "$log"|tail -1)"
    done
  done
done
echo "[b16k] FINISHED $(date)"
