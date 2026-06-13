#!/usr/bin/env bash
# Idea 2 — does query-free cross-layer prefetch buy real TPOT in the IO-DOMINATED
# (long-context + slow-tier) regime? The 4K deployment study was compute-bound
# (IO 0.08% of TPOT), so prefetch shaved only a sliver. Here we sweep context
# length and tier latency (ell_bar) and compare the SAME GuardKV policy with
# cross-layer prefetch ON vs OFF (--no_prefetch, matched kept-set), measuring the
# prefetch TPOT saving = io(off) - io(on). Long context enabled by chunked prefill
# (--prefill_chunk) which bounds eager attention's O(L^2) prefill.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEER="${SEER_ROOT:-$ROOT/../SEER}"
PY="${PY:-/home/lzq/miniconda3/envs/csp-llm/bin/python}"
MODEL="${MODEL:-/public/model_zoo/Llama-3.1-8B-Instruct}"
SC=$ROOT/experiments/predictors/xqp_closed_2view_h4.json
BG=$ROOT/experiments/predictors/guardkv_budgeter_a10.json
OUT=$ROOT/experiments/results/idea2; mkdir -p "$OUT"
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N=${N:-3}; NEW=${NEW:-16}; BUD=${BUD:-0.20}; CHUNK=${CHUNK:-2048}

run () { # ctx ell_bar pf(on|off) tag
  local ctx=$1 ell=$2 pf=$3 tag=$4
  local extra=""; [ "$pf" = "off" ] && extra="--no_prefetch"
  $PY -m seer.eval.runner --model "$MODEL" --policy guardkv \
    --scorer-ckpt "$SC" --budgeter-ckpt "$BG" \
    --workload synthetic --context_length "$ctx" --num_requests "$N" --max_new_tokens "$NEW" \
    --hbm_budget "$BUD" --io_mode analytical --ell_bar_us "$ell" \
    --no_prefill_attn --prefill_chunk "$CHUNK" --skip_prewarm --slo "P99=4000ms" \
    --out "$OUT/${tag}.json" > "$OUT/${tag}.log" 2>&1
  echo "[$(date +%H:%M:%S)] ${tag}: rc=$? $(grep -c OutOfMemory "$OUT/${tag}.log" | sed 's/^/OOM=/')"
}

# A) context sweep at the NVMe-like tier (ell_bar=3000us), prefetch on vs off
for ctx in 4096 16384 32768 65536; do
  run "$ctx" 3000 on  "ctx${ctx}_nvme_on"
  run "$ctx" 3000 off "ctx${ctx}_nvme_off"
done

# B) tier sweep at 32K: ell_bar in {200 DRAM, 1000, 3000 NVMe, 10000 slow}, on vs off
for ell in 200 1000 3000 10000; do
  run 32768 "$ell" on  "tier${ell}_on"
  run 32768 "$ell" off "tier${ell}_off"
done

echo "IDEA2 SWEEP DONE -> $OUT"
