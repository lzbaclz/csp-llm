#!/usr/bin/env bash
# =============================================================================
# run_transfer_reverse.sh -- the REVERSE cross-architecture transfer the reviewer
# asked for. The forward direction (Llama-trained 2-view scorer applied FROZEN in
# Qwen's masked-decode loop) is in the main sweep; this runs the other direction:
# a QWEN-trained 2-view scorer (experiments/predictors/xqp_closed_2view_h4_qwen.json,
# fit on Qwen2.5 traces) applied FROZEN in LLAMA-3.1-8B's loop, 7 LongBench QA sets,
# matched 20% budget. We then TOST it against Llama's own H2O (the main sweep's
# llama31_8b/*_h2o.json) to test transfer at the task-quality level in both directions.
# Resumable. GPU via CUDA_VISIBLE_DEVICES.
#
#   CUDA_VISIBLE_DEVICES=1 bash experiments/run_transfer_reverse.sh
# =============================================================================
set -u
SEER=/home/lzq/codes/SEER
PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
QWEN_CKPT=/home/lzq/codes/csp-llm/experiments/predictors/xqp_closed_2view_h4_qwen.json
LB_DIR=/public/data_zoo/longbench/data
OUTDIR=/home/lzq/codes/csp-llm/experiments/results/transfer_rev
MODEL=/public/model_zoo/Llama-3.1-8B-Instruct          # reverse: run in LLAMA's loop

export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${CUDA_VISIBLE_DEVICES:=1}"; export CUDA_VISIBLE_DEVICES

DATASETS=(narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique triviaqa)
N=64; NEW=48; CTX=4096; B=0.20; SLO="P99=200ms"
mkdir -p "$OUTDIR"

is_done () { [ -s "$1" ] && "$PY" - "$1" <<'PYEOF' >/dev/null 2>&1
import json,sys
sys.exit(0 if len(json.load(open(sys.argv[1])).get("results",[]))>=1 else 1)
PYEOF
}

echo "[transfer-rev] start $(date) GPU=$CUDA_VISIBLE_DEVICES (Qwen-scorer in Llama loop)"
for ds in "${DATASETS[@]}"; do
  [ -s "$LB_DIR/$ds.jsonl" ] || { echo "[rev] WARN missing $ds"; continue; }
  out="$OUTDIR/${ds}_xqpqwen.json"; log="$OUTDIR/${ds}_xqpqwen.log"
  if is_done "$out"; then echo "[rev] SKIP $ds"; continue; fi
  echo "[rev] RUN  $ds"
  LONGBENCH_PATH="$LB_DIR/$ds.jsonl" "$PY" -m seer.eval.runner \
    --model "$MODEL" --policy xqp --xqp-ckpt "$QWEN_CKPT" --workload longbench \
    --context_length "$CTX" --num_requests "$N" --max_new_tokens "$NEW" \
    --hbm_budget "$B" --slo "$SLO" --io_mode measured-dma --chat \
    --seed 0 --out "$out" > "$log" 2>&1
  rc=$?; f1=$(grep -oE 'F1=[0-9.]+' "$log" | tail -1)
  echo "[rev] DONE $ds rc=$rc $f1"
done
echo "[transfer-rev] FINISHED $(date)"
