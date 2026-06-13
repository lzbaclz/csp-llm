#!/usr/bin/env bash
# =============================================================================
# run_prefetch_alloc.sh -- does the RISK-CONTROLLED per-layer prefetch allocator
# beat uniform at matched budget in the IO-bound regime? (the new mechanism's value)
#
# GuardKV policy, 32K synthetic (RULER-style), realizable qk-SVD-rank8 cross-layer
# hint, at NVMe (ell_bar=3000us) and slow (10000us) tiers. Compares prefetch_alloc:
#   off       : no prefetch (IO ceiling baseline)
#   uniform   : top-k per layer (the current Idea-2 realizable)
#   risk      : concentration-weighted per-layer budget (new)
#   conformal : per-layer split-conformal targeting alpha=0.1 (new, the knob)
#   oracleprev: query-free prev-layer ceiling (upper bound)
# Metric = per_step_io_us (lower = fewer recall misses = better prefetch).
# Runs on GPU0 (CUDA_VISIBLE_DEVICES=0); GPU1 runs the budget sweep concurrently.
# =============================================================================
set -u
ROOT=/home/lzq/codes/csp-llm
SEER=/home/lzq/codes/SEER
PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
MODEL=/public/model_zoo/Llama-3.1-8B-Instruct
SC=$ROOT/experiments/predictors/xqp_closed_2view_h4.json
BG=$ROOT/experiments/predictors/guardkv_budgeter_a10.json
OUT=$ROOT/experiments/results/prefetch_alloc; mkdir -p "$OUT"
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N=${N:-3}; NEW=${NEW:-16}; BUD=0.20; CHUNK=2048; CTX=32768

is_done () { [ -s "$1" ] && "$PY" - "$1" <<'P' >/dev/null 2>&1
import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if d.get("results") else 1)
P
}
run () { # ell tag  extra...
  local ell=$1 tag=$2; shift 2
  local out="$OUT/${tag}.json" log="$OUT/${tag}.log"
  if is_done "$out"; then echo "[alloc] SKIP $tag"; return; fi
  "$PY" -m seer.eval.runner --model "$MODEL" --policy guardkv \
    --scorer-ckpt "$SC" --budgeter-ckpt "$BG" \
    --workload synthetic --context_length "$CTX" --num_requests "$N" --max_new_tokens "$NEW" \
    --hbm_budget "$BUD" --io_mode analytical --ell_bar_us "$ell" \
    --no_prefill_attn --prefill_chunk "$CHUNK" --skip_prewarm --slo "P99=4000ms" \
    "$@" --out "$out" > "$log" 2>&1
  echo "[alloc] DONE $tag rc=$? $(grep -c OutOfMemory "$log"|sed 's/^/OOM=/')"
}

echo "[alloc] start $(date) GPU=$CUDA_VISIBLE_DEVICES"
for ell in 3000 10000; do
  t="t${ell}"
  run "$ell" "${t}_off"        --no_prefetch
  run "$ell" "${t}_uniform"    --prefetch_source qk --prefetch_rank 8 --prefetch_sketch svd --prefetch_alloc uniform
  run "$ell" "${t}_risk"       --prefetch_source qk --prefetch_rank 8 --prefetch_sketch svd --prefetch_alloc risk
  run "$ell" "${t}_conformal"  --prefetch_source qk --prefetch_rank 8 --prefetch_sketch svd --prefetch_alloc conformal --prefetch_alpha 0.10
  run "$ell" "${t}_oracleprev" --prefetch_source oracle_prev
done
echo "[alloc] FINISHED $(date)"
