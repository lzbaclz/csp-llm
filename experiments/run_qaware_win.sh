#!/usr/bin/env bash
# Test whether REAL query-aware selection (SnapKV obs-window, Quest) beats H2O at TIGHT
# budget, with prefill attention ENABLED (the long-ctx scripts used --no_prefill_attn,
# which blinds exactly these methods). Works on a LongBench jsonl OR a synthetic NIAH jsonl.
#   DSLIST="hotpotqa 2wikimqa musique" SRC=longbench bash run_qaware_win.sh   (real tasks)
#   DSLIST="mv8" SRC=/abs/path.jsonl  ...                                     (synthetic)
set -u
SEER=/home/lzq/codes/SEER; PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
MODEL=/public/model_zoo/Llama-3.1-8B-Instruct; LB_DIR=/public/data_zoo/longbench/data
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATASETS=(${DSLIST:-hotpotqa 2wikimqa musique narrativeqa qasper})
SELECTORS=(${SELLIST:-full h2o snapkv quest})
BUDGETS=(${BUDGETS:-0.05 0.10 0.20}); CTX=${CTX:-4096}; N=${N:-32}; NEW=${NEW:-16}
OUT=${OUTROOT:-/home/lzq/codes/csp-llm/experiments/results/multikey/win}; mkdir -p "$OUT"
is_done () { [ -s "$1" ] && "$PY" - "$1" <<'P' >/dev/null 2>&1
import json,sys;sys.exit(0 if len(json.load(open(sys.argv[1])).get("results",[]))>=1 else 1)
P
}
echo "[win] start $(date) GPU=$CUDA_VISIBLE_DEVICES ctx=$CTX ds=${DATASETS[*]} budgets=${BUDGETS[*]}"
for ds in "${DATASETS[@]}"; do
  if [ "${SRC:-longbench}" = "longbench" ]; then LBP="$LB_DIR/$ds.jsonl"; else LBP="$SRC"; fi
  [ -s "$LBP" ] || { echo "[win] MISSING $LBP"; continue; }
  for B in "${BUDGETS[@]}"; do
    for sel in "${SELECTORS[@]}"; do
      out="$OUT/${ds}_${sel}_b${B}.json"; log="${out%.json}.log"
      is_done "$out" && { echo "[win] SKIP $ds/$sel/b$B"; continue; }
      if [ "$sel" = "snapkv" ]; then export SEER_SNAPKV_VARIANT=upstream; else unset SEER_SNAPKV_VARIANT; fi
      LONGBENCH_PATH="$LBP" timeout 700 "$PY" -m seer.eval.runner \
        --model "$MODEL" --policy "$sel" --workload longbench --context_length "$CTX" \
        --num_requests "$N" --max_new_tokens "$NEW" --hbm_budget "$B" --slo "P99=4000ms" \
        --io_mode measured-dma --chat --seed 0 --out "$out" > "$log" 2>&1
      s=$("$PY" -c "import json;print('%.3f'%json.load(open('$out'))['substring_mean'])" 2>/dev/null || echo "ERR")
      f=$("$PY" -c "import json;print('%.3f'%json.load(open('$out'))['f1_mean'])" 2>/dev/null || echo "ERR")
      echo "[win] $ds/$sel/b$B substr=$s f1=$f"
    done
  done
done
echo "[win] FINISHED $(date)"
