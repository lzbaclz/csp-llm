#!/usr/bin/env bash
# Download models + datasets required by csp-llm / SEER trace collection.
#
# Models  -> /public/model_zoo/<Name>/
# Datasets -> /public/data_zoo/  (HF hub layout + mooncake JSONL)
#
# Usage:
#   bash scripts/download_assets.sh              # skip complete items
#   bash scripts/download_assets.sh --force      # re-download everything
#   bash scripts/download_assets.sh --cleanup    # kill stale hf jobs + remove locks
#
# Tips if download stalls (effective speed = 0):
#   1. Run with --cleanup first (clears zombie hf + .lock/.incomplete)
#   2. Ensure HF_TOKEN is set (see load_hf_token below)
#   3. Script sets HF_HUB_DISABLE_XET=1 to avoid XET CDN through proxy
#   4. Watch: tail -f experiments/logs/download_assets.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/experiments/logs/download_assets.log"
MODEL_ZOO=/public/model_zoo
DATA_ZOO=/public/data_zoo
HF_HUB="$DATA_ZOO/huggingface/hub"
MOONCAKE_DIR="$DATA_ZOO/mooncake"

FORCE=0
CLEANUP=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --cleanup) CLEANUP=1 ;;
  esac
done

mkdir -p "$MODEL_ZOO" "$HF_HUB" "$MOONCAKE_DIR" "$(dirname "$LOG")"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

load_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    log "HF_TOKEN already set"
    return 0
  fi
  for f in "$HOME/.cache/huggingface/token" "$HOME/.huggingface/token"; do
    if [[ -s "$f" ]]; then
      export HF_TOKEN="$(tr -d '[:space:]' < "$f")"
      log "HF_TOKEN loaded from $f"
      return 0
    fi
  done
  log "WARN: HF_TOKEN not found — downloads may be slow or rate-limited"
  log "  Create token: https://huggingface.co/settings/tokens"
  log "  Then: export HF_TOKEN=hf_..."
}

cleanup_stale_downloads() {
  log "CLEANUP: stopping stale hf/download_assets processes (except this shell)"
  local my_pid=$$
  pgrep -af 'hf download|download_assets\.sh' 2>/dev/null | while read -r line; do
    local pid
    pid="$(echo "$line" | awk '{print $1}')"
    [[ "$pid" == "$my_pid" || "$pid" == "$PPID" ]] && continue
    if echo "$line" | grep -q 'download_assets\.sh'; then
      kill "$pid" 2>/dev/null || true
    elif echo "$line" | grep -q 'hf download'; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  sleep 2
  pkill -9 -f 'hf download mistralai/Mistral' 2>/dev/null || true

  local cache="/public/model_zoo/Mistral-7B-Instruct-v0.3/.cache/huggingface/download"
  if [[ -d "$cache" ]]; then
    log "CLEANUP: remove stale locks/incomplete under $cache"
    find "$cache" -maxdepth 1 \( -name '*.lock' -o -name '*.incomplete' \) -delete 2>/dev/null || true
  fi
}

source "${CONDA_EXE:-$HOME/miniconda3/bin/conda}/etc/profile.d/conda.sh" 2>/dev/null \
  || source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate csp-llm

# Proxy-friendly: disable XET (transfer.xethub.hf.co often fails via HTTP proxy)
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME="$DATA_ZOO/huggingface"
export HF_DATASETS_CACHE="$HF_HUB"

load_hf_token

[[ "$CLEANUP" -eq 1 ]] && cleanup_stale_downloads

# --- helpers ---
model_ready() {
  local dir="$1"
  [[ -f "$dir/config.json" ]] || return 1
  if [[ -f "$dir/model.safetensors.index.json" ]]; then
    local nshard have
    nshard=$(grep -o 'model-[0-9]*-of-[0-9]*\.safetensors' "$dir/model.safetensors.index.json" | sort -u | wc -l)
    have=$(ls "$dir"/model-*-of-*.safetensors 2>/dev/null | wc -l)
    [[ "$nshard" -gt 0 && "$have" -ge "$nshard" ]]
  elif [[ -f "$dir/model.safetensors" ]]; then
    [[ $(stat -c%s "$dir/model.safetensors") -gt 1000000000 ]]
  else
    return 1
  fi
}

download_model() {
  local hf_id="$1" local_dir="$2"
  if [[ "$FORCE" -eq 0 ]] && model_ready "$local_dir"; then
    log "SKIP model $local_dir (complete, $(du -sh "$local_dir" | awk '{print $1}'))"
    return 0
  fi
  log "DOWNLOAD model $hf_id -> $local_dir"
  mkdir -p "$local_dir"
  # stdbuf: line-buffered output when piped to tee
  stdbuf -oL -eL hf download "$hf_id" \
    --local-dir "$local_dir" \
    --exclude 'consolidated.safetensors' \
    --exclude '*.bin' \
    ${HF_TOKEN:+--token "$HF_TOKEN"} \
    2>&1 | stdbuf -oL -eL tee -a "$LOG"
  if model_ready "$local_dir"; then
    log "OK model $local_dir ($(du -sh "$local_dir" | awk '{print $1}'))"
  else
    log "FAIL model $local_dir — shards incomplete"
    return 1
  fi
}

download_dataset() {
  local hf_id="$1" cache_name="$2"
  local dest="$HF_HUB/datasets--${cache_name}"
  if [[ "$FORCE" -eq 0 && -d "$dest" ]] && [[ $(du -sb "$dest" | awk '{print $1}') -gt 1048576 ]]; then
    log "SKIP dataset $hf_id ($(du -sh "$dest" | awk '{print $1}'))"
    return 0
  fi
  log "DOWNLOAD dataset $hf_id -> $dest"
  mkdir -p "$dest"
  stdbuf -oL -eL hf download "$hf_id" --repo-type dataset --local-dir "$dest" \
    ${HF_TOKEN:+--token "$HF_TOKEN"} \
    2>&1 | stdbuf -oL -eL tee -a "$LOG"
  log "OK dataset $hf_id ($(du -sh "$dest" | awk '{print $1}'))"
}

download_mooncake() {
  local dest="$MOONCAKE_DIR/trace.jsonl"
  if [[ "$FORCE" -eq 0 && -s "$dest" ]]; then
    log "SKIP mooncake trace ($(du -sh "$dest" | awk '{print $1}'))"
    return 0
  fi
  local urls=(
    "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl"
    "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/Mooncake_Trace.jsonl"
  )
  for url in "${urls[@]}"; do
    log "FETCH mooncake $url"
    if curl -fsSL --connect-timeout 30 --retry 3 -o "$dest" "$url"; then
      [[ -s "$dest" ]] && { log "OK mooncake $(wc -l < "$dest") lines"; return 0; }
    fi
  done
  log "FAIL mooncake trace — manual: set MOONCAKE_TRACE_PATH"
  return 1
}

log "=== csp-llm asset download start ==="
log "model_zoo=$MODEL_ZOO  data_zoo=$DATA_ZOO  HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"

# --- Models (4 families for run_collect_4models.sh / ICDM) ---
download_model "mistralai/Mistral-7B-Instruct-v0.3" "$MODEL_ZOO/Mistral-7B-Instruct-v0.3"

download_model "Qwen/Qwen2.5-7B-Instruct" "$MODEL_ZOO/Qwen2.5-7B-Instruct"
download_model "meta-llama/Llama-3.1-8B-Instruct" "$MODEL_ZOO/Llama-3.1-8B-Instruct"

if model_ready "$MODEL_ZOO/downloads/Qwen3-8B"; then
  if [[ ! -e "$MODEL_ZOO/Qwen3-8B" ]]; then
    ln -sfn "$MODEL_ZOO/downloads/Qwen3-8B" "$MODEL_ZOO/Qwen3-8B"
  fi
  log "OK Qwen3-8B via downloads/ ($(du -sh "$MODEL_ZOO/downloads/Qwen3-8B" | awk '{print $1}'))"
else
  download_model "Qwen/Qwen3-8B" "$MODEL_ZOO/Qwen3-8B"
fi

# --- Datasets ---
download_mooncake || true

download_dataset "RyokoAI/ShareGPT52K" "RyokoAI--ShareGPT52K"
download_dataset "THUDM/LongBench" "THUDM--LongBench"
download_dataset "THUDM/LongBench-v2" "THUDM--LongBench-v2"
download_dataset "xinrongzhang2022/InfiniteBench" "xinrongzhang2022--InfiniteBench"

log "=== done ==="
log "Set for experiments:"
log "  export MOONCAKE_TRACE_PATH=$MOONCAKE_DIR/trace.jsonl"
log "  export HF_HOME=$DATA_ZOO/huggingface"
log "  export HF_DATASETS_CACHE=$HF_HUB"
log "  export SEER_STRICT_WORKLOAD=1   # optional: forbid RULER fallback"
