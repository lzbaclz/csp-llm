#!/usr/bin/env bash
# Generic budget-at-fixed-quality driver. Env: MODEL, OUTROOT, DSLIST (space-sep),
# BUDGETS (space-sep), CUDA_VISIBLE_DEVICES. {h2o,xqp,adakv}. Resumable.
set -u
SEER=/home/lzq/codes/SEER
PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
LCK="${XQP_CKPT:-/home/lzq/codes/csp-llm/experiments/predictors/xqp_closed_2view_h4.json}"
LB_DIR=/public/data_zoo/longbench/data
MODEL="${MODEL:?}"; OUTROOT="${OUTROOT:?}"
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATASETS=(${DSLIST:?}); BUDGETS=(${BUDGETS:-0.10 0.30 0.50}); SELECTORS=(${SELLIST:-h2o xqp adakv})
N=${N:-64}; NEW=${NEW:-48}; CTX=${CTX:-4096}; SLO="P99=200ms"
is_done () { [ -s "$1" ] && "$PY" - "$1" <<'P' >/dev/null 2>&1
import json,sys;sys.exit(0 if len(json.load(open(sys.argv[1])).get("results",[]))>=1 else 1)
P
}
echo "[bq] start $(date) GPU=$CUDA_VISIBLE_DEVICES ds=${DATASETS[*]}"
for B in "${BUDGETS[@]}"; do
  od="$OUTROOT/b$B"; mkdir -p "$od"
  for ds in "${DATASETS[@]}"; do
    [ -s "$LB_DIR/$ds.jsonl" ] || continue
    for sel in "${SELECTORS[@]}"; do
      out="$od/${ds}_${sel}.json"; log="${out%.json}.log"
      is_done "$out" && { echo "[bq] SKIP b$B/$ds/$sel"; continue; }
      extra=""; [ "$sel" = "xqp" ] && extra="--xqp-ckpt $LCK"
      LONGBENCH_PATH="$LB_DIR/$ds.jsonl" "$PY" -m seer.eval.runner \
        --model "$MODEL" --policy "$sel" $extra --workload longbench \
        --context_length "$CTX" --num_requests "$N" --max_new_tokens "$NEW" \
        --hbm_budget "$B" --slo "$SLO" --io_mode measured-dma --chat --seed 0 \
        --out "$out" > "$log" 2>&1
      echo "[bq] DONE b$B/$ds/$sel rc=$? $(grep -oE 'F1=[0-9.]+' "$log"|tail -1)"
    done
  done
done
echo "[bq] FINISHED $(date)"
