#!/usr/bin/env bash
# =============================================================================
# run_longctx_equiv.sh -- does the xqp == H2O task-quality equivalence hold at
# LONG context (16K/32K)? Addresses the universal reviewer scope critique (≤4K).
# Long-doc LongBench QA at context_length {16384, 32768}, Llama-3.1-8B, budget 0.20,
# {full, h2o, xqp}. Chunked prefill + no-prefill-attn keep 16K/32K within memory.
# Split datasets across 2 GPUs via DSLIST + CUDA_VISIBLE_DEVICES. Resumable.
#   GPU0: CUDA_VISIBLE_DEVICES=0 DSLIST="narrativeqa qasper" CTX=16384 bash run_longctx_equiv.sh
#   GPU1: CUDA_VISIBLE_DEVICES=1 DSLIST="2wikimqa hotpotqa" CTX=16384 bash run_longctx_equiv.sh
# =============================================================================
set -u
SEER=/home/lzq/codes/SEER
PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
LCK=/home/lzq/codes/csp-llm/experiments/predictors/xqp_closed_2view_h4.json
LB_DIR=/public/data_zoo/longbench/data
MODEL=/public/model_zoo/Llama-3.1-8B-Instruct
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATASETS=(${DSLIST:-narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa})
CTX=${CTX:-16384}; N=${N:-32}; NEW=48; B=0.20; SLO="P99=4000ms"
OUT=${OUTROOT:-/home/lzq/codes/csp-llm/experiments/results/longctx/c${CTX}}; mkdir -p "$OUT"
SELECTORS=(${SELLIST:-h2o xqp})   # 'full' OOMs at 16K (non-masking full forward); not needed for xqp==h2o equivalence
is_done () { [ -s "$1" ] && "$PY" - "$1" <<'P' >/dev/null 2>&1
import json,sys;sys.exit(0 if len(json.load(open(sys.argv[1])).get("results",[]))>=1 else 1)
P
}
echo "[longctx] start $(date) GPU=$CUDA_VISIBLE_DEVICES ctx=$CTX ds=${DATASETS[*]}"
for ds in "${DATASETS[@]}"; do
  [ -s "$LB_DIR/$ds.jsonl" ] || { echo "[longctx] WARN missing $ds"; continue; }
  for sel in "${SELECTORS[@]}"; do
    out="$OUT/${ds}_${sel}.json"; log="${out%.json}.log"
    is_done "$out" && { echo "[longctx] SKIP $ds/$sel"; continue; }
    extra=""; [ "$sel" = "xqp" ] && extra="--xqp-ckpt $LCK"
    LONGBENCH_PATH="$LB_DIR/$ds.jsonl" "$PY" -m seer.eval.runner \
      --model "$MODEL" --policy "$sel" $extra --workload longbench \
      --context_length "$CTX" --num_requests "$N" --max_new_tokens "$NEW" \
      --hbm_budget "$B" --slo "$SLO" --io_mode measured-dma --chat --seed 0 \
      --prefill_chunk 2048 --no_prefill_attn --skip_prewarm \
      --out "$out" > "$log" 2>&1
    echo "[longctx] DONE $ds/$sel rc=$? $(grep -oE 'F1=[0-9.]+' "$log"|tail -1)"
  done
done
echo "[longctx] FINISHED $(date)"
