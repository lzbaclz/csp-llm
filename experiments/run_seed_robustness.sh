#!/usr/bin/env bash
# =============================================================================
# run_seed_robustness.sh -- multi-seed robustness for the TOST equivalence.
#
# The main expansion sweep (experiments/results/expand/) is at seed 0. This adds
# seeds 1 and 2 on Llama-3.1-8B for the four matched-budget selectors
# {h2o, xqp, pyramidkv, adakv} across the 7 LongBench QA-F1 datasets, so the
# +/-0.02 equivalence verdict can be checked at EACH of 3 independent prompt
# subsets (greedy decode => seed only changes WHICH prompts are drawn).
#
# Llama only (Qwen at seed 0 already supplies the cross-architecture dimension);
# 'full' omitted (not part of the equivalence contrast). MAXJOBS=1 (the masking
# sim's oracle forward spikes ~43GB; 2 concurrent OOMs). GPU 1 ONLY -- GPU 0 is a
# colleague's. Resumable: existing valid JSON is skipped.
#
#   bash experiments/run_seed_robustness.sh
# =============================================================================
set -u
SEER=/home/lzq/codes/SEER
PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
XQP_CKPT=/home/lzq/codes/csp-llm/experiments/predictors/xqp_closed_2view_h4.json
LB_DIR=/public/data_zoo/longbench/data
OUT_ROOT=/home/lzq/codes/csp-llm/experiments/results/expand_seeds
MODEL=/public/model_zoo/Llama-3.1-8B-Instruct

export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

SEEDS=(1 2)
DATASETS=(narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique triviaqa)
SELECTORS=(h2o xqp pyramidkv adakv)
N=64; NEW=48; CTX=4096; B=0.20; SLO="P99=200ms"

is_done () { [ -s "$1" ] && "$PY" - "$1" <<'PYEOF' >/dev/null 2>&1
import json,sys
sys.exit(0 if len(json.load(open(sys.argv[1])).get("results",[]))>=1 else 1)
PYEOF
}

echo "[seed-robust] start $(date) GPU=$CUDA_VISIBLE_DEVICES seeds=${SEEDS[*]}"
planned=0; ran=0; skipped=0
for s in "${SEEDS[@]}"; do
  outdir="$OUT_ROOT/s$s"; mkdir -p "$outdir"
  for ds in "${DATASETS[@]}"; do
    [ -s "$LB_DIR/$ds.jsonl" ] || { echo "[seed-robust] WARN missing $ds"; continue; }
    for sel in "${SELECTORS[@]}"; do
      planned=$((planned+1))
      out="$outdir/${ds}_${sel}.json"; log="$outdir/${ds}_${sel}.log"
      if is_done "$out"; then echo "[seed-robust] SKIP s$s/$ds/$sel"; skipped=$((skipped+1)); continue; fi
      extra=""; [ "$sel" = "xqp" ] && extra="--xqp-ckpt $XQP_CKPT"
      echo "[seed-robust] RUN  s$s/$ds/$sel"
      LONGBENCH_PATH="$LB_DIR/$ds.jsonl" "$PY" -m seer.eval.runner \
        --model "$MODEL" --policy "$sel" $extra --workload longbench \
        --context_length "$CTX" --num_requests "$N" --max_new_tokens "$NEW" \
        --hbm_budget "$B" --slo "$SLO" --io_mode measured-dma --chat \
        --seed "$s" --out "$out" > "$log" 2>&1
      rc=$?; f1=$(grep -oE 'F1=[0-9.]+' "$log" | tail -1)
      echo "[seed-robust] DONE s$s/$ds/$sel rc=$rc $f1"; ran=$((ran+1))
    done
  done
done
echo "[seed-robust] FINISHED $(date) planned=$planned ran=$ran skipped=$skipped"
