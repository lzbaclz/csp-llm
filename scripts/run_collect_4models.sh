#!/usr/bin/env bash
# ICDM headline-corpus collection — 4 model families, dual-GPU, real attention
# traces (32 mooncake prompts each → the corpus paper_icdm/ is built on).
# For the larger multi-workload corpus use scripts/run_collect_expanded.sh.
#
# All paths are env-overridable; output is routed through $TRACEDIR with a disk
# preflight so a big collection never silently fills a small home volume:
#   TRACEDIR=/public/xqp_traces N_TRACES=128 bash scripts/run_collect_4models.sh
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/lzq/miniconda3/envs/csp-llm/bin/python}"
MODEL_ZOO="${MODEL_ZOO:-/public/model_zoo}"
TRACEDIR="${TRACEDIR:-$ROOT/experiments/traces}"
N_TRACES="${N_TRACES:-32}"
MAX_CONTEXT="${MAX_CONTEXT:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
LOGDIR="$ROOT/experiments/logs"
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$LOGDIR" "$TRACEDIR"

MISTRAL="$MODEL_ZOO/Mistral-7B-Instruct-v0.3"
ALOG="$LOGDIR/collect_all.log"

# --- Disk preflight: refuse to write a big collection to a near-full volume ---
AVAIL=$("$PY" -c "import shutil;print(shutil.disk_usage('$TRACEDIR').free)")
NEED=$(( N_TRACES * 4 * 95 * 1024 * 1024 ))   # ~95 MB / prompt / model, ×4 models
if [[ "$AVAIL" -lt "$NEED" ]]; then
  echo "ABORT: TRACEDIR=$TRACEDIR has $((AVAIL/1024/1024/1024))G free, need ~$((NEED/1024/1024/1024))G."
  echo "       Point TRACEDIR at a bigger volume, e.g. TRACEDIR=/public/xqp_traces"
  exit 1
fi

collect () {  # $1=gpu  $2=model_path  $3=worker_tag
  local gpu="$1" mp="$2" tag="$3"
  echo "[$(date -Iseconds)] START $tag on gpu$gpu ($mp)" >> "$ALOG"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/collect_traces_attn.py \
    --n-traces "$N_TRACES" --prompt-start 0 --prompt-end "$N_TRACES" \
    --max-context "$MAX_CONTEXT" --max-new-tokens "$MAX_NEW_TOKENS" \
    --workload mooncake --device cuda:0 --worker-id "$tag" \
    --out-dir "$TRACEDIR" --models "$mp" > "$LOGDIR/collect_${tag}.log" 2>&1
  local rc=$?
  echo "$rc" > "$LOGDIR/${tag}.rc"
  echo "[$(date -Iseconds)] END   $tag rc=$rc" >> "$ALOG"
}

echo "[$(date -Iseconds)] start N_TRACES=$N_TRACES CTX=$MAX_CONTEXT NEW=$MAX_NEW_TOKENS TRACEDIR=$TRACEDIR" > "$ALOG"
rm -f "$LOGDIR"/*.rc

# --- Wave 1: Llama (gpu0) + Qwen2.5 (gpu1) ---
collect 0 "$MODEL_ZOO/Llama-3.1-8B-Instruct" llama31_gpu0 & P0=$!
collect 1 "$MODEL_ZOO/Qwen2.5-7B-Instruct"   qwen25_gpu1 & P1=$!
wait $P0; wait $P1

# --- Wave 2: Qwen3-8B (gpu0) + Mistral (gpu1, if present) ---
collect 0 "$MODEL_ZOO/Qwen3-8B" qwen3_gpu0 & P0=$!
if [[ -f "$MISTRAL/config.json" ]] && ls "$MISTRAL"/model-*-of-*.safetensors >/dev/null 2>&1; then
  collect 1 "$MISTRAL" mistral_gpu1 & P1=$!
  wait $P1
else
  echo "[$(date -Iseconds)] MISTRAL NOT PRESENT — skipped (download to $MISTRAL)" >> "$ALOG"
fi
wait $P0

# --- Aggregate per-cell exit codes; fail loudly if any cell failed ---
FAILED=0
for rcf in "$LOGDIR"/*.rc; do
  [[ -f "$rcf" ]] || continue
  rc=$(cat "$rcf")
  if [[ "$rc" != "0" ]]; then
    echo "[$(date -Iseconds)] FAILED cell $(basename "$rcf" .rc) rc=$rc" >> "$ALOG"
    FAILED=1
  fi
done
{ echo "=== trace sizes ($TRACEDIR) ==="; ls -la "$TRACEDIR"/*.jsonl 2>&1; } >> "$ALOG"
if [[ "$FAILED" -ne 0 ]]; then
  echo "[$(date -Iseconds)] COLLECTION FINISHED WITH FAILURES — see $ALOG" | tee -a "$ALOG"
  exit 1
fi
echo "[$(date -Iseconds)] ALL COLLECTION DONE (clean)" | tee -a "$ALOG"
