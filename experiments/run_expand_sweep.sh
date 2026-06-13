#!/usr/bin/env bash
# =============================================================================
# run_expand_sweep.sh  --  TOST-equivalence expansion sweep (Idea-1 / GuardKV)
#
# Goal: produce enough PAIRED per-item F1 measurements (same prompts, same
# budget, same harness) to power a TOST equivalence test at margin +/-0.02
# across >=7 English QA-F1 LongBench datasets and 2 architectures.
#
# Harness  : SEER  (seer.eval.runner)  env=~/miniconda3/envs/csp-llm/bin/python
# Selection: per-dataset via LONGBENCH_PATH=.../<dataset>.jsonl + --workload longbench
#            (verified mechanism; the loader's "narrativeqa" config is only the
#             default -- the local-jsonl override picks the actual dataset).
# Metric   : the runner ALWAYS computes SQuAD-v1.1 token F1 (seer.eval.metrics
#            .f1_score) per item, for every longbench dataset -> uniform F1.
# Output   : experiments/results/expand/<arch_short>/<dataset>_<selector>.json
#            Per-item pred/ref/id/f1 live in output["results"] for paired TOST.
#
# Concurrency: at most MAXJOBS=3 processes, ALL on CUDA_VISIBLE_DEVICES=1.
#              GPU 0 belongs to a colleague -- NEVER used here.
# Resumable  : any existing, valid (non-empty, JSON-parseable, n>=1) output is
#              skipped, so re-running fills only the gaps.
#
# NOTE ON pyramidkv / adakv
# -------------------------
# pyramidkv and adakv are NOT implemented in the SEER harness: the runner's
# --policy argparse choices reject them, build_policy() has no entry, and
# `kvpress` (their only impl, via the HALO sibling repo) is not installed in
# the csp-llm env. Running them here would emit crash JSONs, NOT paired items.
# They are therefore listed but GUARDED: each is skipped with a logged reason
# unless a SEER adapter is wired in AND ALLOW_PRESS=1 is exported. When that
# day comes, set ALLOW_PRESS=1 and they fold into the same paired sweep with
# zero other edits. Until then this script produces the full/h2o/xqp paired
# triples (the selectors that ARE runnable in this harness today).
#
# This script does NOT launch on source -- run it explicitly:
#     bash experiments/run_expand_sweep.sh
# =============================================================================
set -u

# ----- fixed paths / env ----------------------------------------------------
SEER=/home/lzq/codes/SEER
PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
XQP_CKPT=/home/lzq/codes/csp-llm/experiments/predictors/xqp_closed_2view_h4.json
LB_DIR=/public/data_zoo/longbench/data
OUT_ROOT=/home/lzq/codes/csp-llm/experiments/results/expand

export PYTHONPATH="$SEER"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# All work pinned to GPU 1 (GPU 0 is the colleague's -- never touch it).
export CUDA_VISIBLE_DEVICES=1

# ----- sweep axes -----------------------------------------------------------
# arch_short : /public/model_zoo path
declare -A MODELS=(
  [llama31_8b]=/public/model_zoo/Llama-3.1-8B-Instruct
  [qwen25_7b]=/public/model_zoo/Qwen2.5-7B-Instruct
)
ARCHS=(llama31_8b qwen25_7b)

# 7 English QA-F1 LongBench tasks (all verified loadable + F1-metric):
#   narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique triviaqa
DATASETS=(narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique triviaqa)

# Selectors. full/h2o/xqp run natively in SEER. pyramidkv/adakv are guarded
# (see header) -- skipped unless ALLOW_PRESS=1 and a SEER adapter exists.
SELECTORS=(full h2o xqp pyramidkv adakv)

# ----- fixed run knobs ------------------------------------------------------
N=64               # num_requests (paired item count per cell)
NEW=48             # max_new_tokens
CTX=4096           # context_length
B=0.20             # hbm_budget (fraction; matched across selectors)
SLO="P99=200ms"    # latency SLO (quality sweep -- slack so F1 is the signal)
MAXJOBS="${MAXJOBS:-2}"   # max concurrent runner procs on GPU 1 (env-overridable; ~33GB each)
ALLOW_PRESS="${ALLOW_PRESS:-0}"   # set 1 once pyramidkv/adakv exist in SEER

mkdir -p "$OUT_ROOT"

# ----- helpers --------------------------------------------------------------

# A cell is "done" if its JSON exists, parses, and has >=1 result item.
is_done () {
  local out="$1"
  [ -s "$out" ] || return 1
  "$PY" - "$out" <<'PYEOF' >/dev/null 2>&1
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    sys.exit(0 if len(d.get("results",[]))>=1 else 1)
except Exception:
    sys.exit(1)
PYEOF
}

# Selector is runnable in the SEER harness today?
selector_supported () {
  case "$1" in
    full|h2o|xqp) return 0 ;;
    pyramidkv|adakv)
      [ "$ALLOW_PRESS" = "1" ] && return 0 || return 1 ;;
    *) return 1 ;;
  esac
}

# Extra CLI args per selector (xqp needs its closed-form ckpt).
selector_extra () {
  case "$1" in
    xqp) echo "--xqp-ckpt $XQP_CKPT" ;;
    *)   echo "" ;;
  esac
}

# Throttle: block until fewer than MAXJOBS background jobs are running.
wait_for_slot () {
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do
    wait -n 2>/dev/null || sleep 2
  done
}

# Launch one cell in the background (caller has already reserved a slot).
run_cell () {
  local arch="$1" model="$2" ds="$3" sel="$4" out="$5" log="$6"
  local extra; extra="$(selector_extra "$sel")"
  (
    export LONGBENCH_PATH="$LB_DIR/$ds.jsonl"
    "$PY" -m seer.eval.runner \
      --model "$model" --policy "$sel" $extra \
      --workload longbench --context_length "$CTX" \
      --num_requests "$N" --max_new_tokens "$NEW" \
      --hbm_budget "$B" --slo "$SLO" \
      --io_mode measured-dma --chat \
      --out "$out" > "$log" 2>&1
    rc=$?
    f1=$(grep -oE 'F1=[0-9.]+' "$log" | tail -1)
    echo "[$(date +%H:%M:%S)] DONE  $arch/$ds/$sel  rc=$rc  $f1"
  ) &
}

# ----- main loop ------------------------------------------------------------
echo "[expand-sweep] start $(date)  GPU=$CUDA_VISIBLE_DEVICES  maxjobs=$MAXJOBS"
echo "[expand-sweep] archs=${ARCHS[*]}"
echo "[expand-sweep] datasets=${DATASETS[*]}"
echo "[expand-sweep] selectors=${SELECTORS[*]}  (ALLOW_PRESS=$ALLOW_PRESS)"

planned=0; launched=0; skipped_done=0; skipped_unsup=0
for arch in "${ARCHS[@]}"; do
  model="${MODELS[$arch]}"
  outdir="$OUT_ROOT/$arch"
  mkdir -p "$outdir"
  for ds in "${DATASETS[@]}"; do
    if [ ! -s "$LB_DIR/$ds.jsonl" ]; then
      echo "[expand-sweep] WARN missing dataset file: $LB_DIR/$ds.jsonl -- skipping $ds"
      continue
    fi
    for sel in "${SELECTORS[@]}"; do
      planned=$((planned+1))
      out="$outdir/${ds}_${sel}.json"
      log="$outdir/${ds}_${sel}.log"

      if ! selector_supported "$sel"; then
        echo "[expand-sweep] SKIP  $arch/$ds/$sel  (not runnable in SEER harness; set ALLOW_PRESS=1 after wiring a kvpress adapter)"
        skipped_unsup=$((skipped_unsup+1))
        continue
      fi
      if is_done "$out"; then
        echo "[expand-sweep] SKIP  $arch/$ds/$sel  (already complete: $out)"
        skipped_done=$((skipped_done+1))
        continue
      fi

      wait_for_slot
      echo "[expand-sweep] RUN   $arch/$ds/$sel -> $out"
      run_cell "$arch" "$model" "$ds" "$sel" "$out" "$log"
      launched=$((launched+1))
    done
  done
done

wait   # drain remaining background jobs

echo "[expand-sweep] FINISHED $(date)"
echo "[expand-sweep] planned=$planned launched=$launched skipped_done=$skipped_done skipped_unsupported=$skipped_unsup"
echo "[expand-sweep] outputs under: $OUT_ROOT/<arch>/<dataset>_<selector>.json"
echo "[expand-sweep] paired TOST: feed output[\"results\"][*].{id,f1} pairs to experiments/tost_equivalence.py"
