#!/usr/bin/env bash
# SOTA question-aware policies at 4K, MATCHING the expand baseline conditions exactly
# (ctx 4096, --chat, measured-dma, no prefill-attn flags) so they are directly
# comparable to experiments/results/expand/llama31_8b/{ds}_{h2o,xqp}.json (n=64).
set -u
SEER=/home/lzq/codes/SEER; PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
LB_DIR=/public/data_zoo/longbench/data; MODEL=/public/model_zoo/Llama-3.1-8B-Instruct
OUT=/home/lzq/codes/csp-llm/experiments/results/qaware_4k; mkdir -p "$OUT"
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATASETS=(${DSLIST:-narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique triviaqa})
SELECTORS=(${SELLIST:-qanchor qanchor_h2o snapkv_fixed})
N=64; NEW=48; CTX=4096; B=0.20; SLO="P99=200ms"
is_done () { [ -s "$1" ] && "$PY" - "$1" <<'P' >/dev/null 2>&1
import json,sys;sys.exit(0 if len(json.load(open(sys.argv[1])).get("results",[]))>=1 else 1)
P
}
echo "[qaware4k] start $(date) GPU=$CUDA_VISIBLE_DEVICES ds=${DATASETS[*]} sel=${SELECTORS[*]}"
for ds in "${DATASETS[@]}"; do
  [ -s "$LB_DIR/$ds.jsonl" ] || continue
  for sel in "${SELECTORS[@]}"; do
    out="$OUT/${ds}_${sel}.json"; log="${out%.json}.log"
    is_done "$out" && { echo "[qaware4k] SKIP $ds/$sel"; continue; }
    LONGBENCH_PATH="$LB_DIR/$ds.jsonl" "$PY" -m seer.eval.runner \
      --model "$MODEL" --policy "$sel" --workload longbench --context_length "$CTX" \
      --num_requests "$N" --max_new_tokens "$NEW" --hbm_budget "$B" --slo "$SLO" \
      --io_mode measured-dma --chat --seed 0 --out "$out" > "$log" 2>&1
    echo "[qaware4k] DONE $ds/$sel rc=$? $(grep -oE 'F1=[0-9.]+' "$log"|tail -1)"
  done
done
echo "[qaware4k] FINISHED $(date)"
