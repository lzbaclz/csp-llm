#!/usr/bin/env bash
# =============================================================================
# run_disjoint_subset.sh -- sample-robustness via a DISJOINT prompt subset.
#
# Literal --seed is a no-op here (greedy decode + deterministic LongBench order =>
# identical F1; verified). The meaningful robustness question is whether the +/-0.02
# equivalence verdict survives a DIFFERENT prompt sample. This re-runs the four
# matched-budget selectors on prompts [64:128] (LONGBENCH_OFFSET=64) -- disjoint from
# the main result's [0:64] -- across the 7 LongBench QA datasets on Llama-3.1-8B.
#
# MAXJOBS=1 (oracle-forward spikes ~43GB). GPU 1 ONLY (GPU 0 is a colleague's).
# Resumable. Output: experiments/results/expand_disjoint/<ds>_<sel>.json
# =============================================================================
set -u
SEER=/home/lzq/codes/SEER
PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
XQP_CKPT=/home/lzq/codes/csp-llm/experiments/predictors/xqp_closed_2view_h4.json
LB_DIR=/public/data_zoo/longbench/data
OUT=/home/lzq/codes/csp-llm/experiments/results/expand_disjoint
MODEL=/public/model_zoo/Llama-3.1-8B-Instruct

export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
export LONGBENCH_OFFSET=64          # <-- prompts [64:128], disjoint from the main [0:64]

DATASETS=(narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique triviaqa)
SELECTORS=(h2o xqp pyramidkv adakv)
N=64; NEW=48; CTX=4096; B=0.20; SLO="P99=200ms"
mkdir -p "$OUT"

is_done () { [ -s "$1" ] && "$PY" - "$1" <<'PYEOF' >/dev/null 2>&1
import json,sys; sys.exit(0 if len(json.load(open(sys.argv[1])).get("results",[]))>=1 else 1)
PYEOF
}

echo "[disjoint] start $(date) OFFSET=$LONGBENCH_OFFSET GPU=$CUDA_VISIBLE_DEVICES"
for ds in "${DATASETS[@]}"; do
  [ -s "$LB_DIR/$ds.jsonl" ] || { echo "[disjoint] WARN missing $ds"; continue; }
  for sel in "${SELECTORS[@]}"; do
    out="$OUT/${ds}_${sel}.json"; log="$OUT/${ds}_${sel}.log"
    if is_done "$out"; then echo "[disjoint] SKIP $ds/$sel"; continue; fi
    extra=""; [ "$sel" = "xqp" ] && extra="--xqp-ckpt $XQP_CKPT"
    echo "[disjoint] RUN $ds/$sel"
    LONGBENCH_PATH="$LB_DIR/$ds.jsonl" "$PY" -m seer.eval.runner \
      --model "$MODEL" --policy "$sel" $extra --workload longbench \
      --context_length "$CTX" --num_requests "$N" --max_new_tokens "$NEW" \
      --hbm_budget "$B" --slo "$SLO" --io_mode measured-dma --chat \
      --seed 0 --out "$out" > "$log" 2>&1
    echo "[disjoint] DONE $ds/$sel rc=$? $(grep -oE 'F1=[0-9.]+' "$log" | tail -1)"
  done
done
echo "[disjoint] FINISHED $(date)"
