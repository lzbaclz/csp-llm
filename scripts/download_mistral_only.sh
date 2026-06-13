#!/usr/bin/env bash
# Minimal script: download only Mistral-7B-Instruct-v0.3 (~14G, 3 shards).
# Use when the full download_assets.sh is stuck or you only need this model.
#
# Usage:
#   export HF_TOKEN=hf_...          # recommended
#   bash scripts/download_mistral_only.sh
#   bash scripts/download_mistral_only.sh --cleanup   # kill stale + clear locks first
set -euo pipefail

MODEL_DIR=/public/model_zoo/Mistral-7B-Instruct-v0.3
LOG=/home/lzq/codes/csp-llm/experiments/logs/download_mistral.log
CLEANUP=0
[[ "${1:-}" == "--cleanup" ]] && CLEANUP=1

mkdir -p "$(dirname "$LOG")" "$MODEL_DIR"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

if [[ -z "${HF_TOKEN:-}" && -s "$HOME/.cache/huggingface/token" ]]; then
  export HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.cache/huggingface/token")"
fi

if [[ "$CLEANUP" -eq 1 ]]; then
  log "Killing stale hf download processes..."
  pkill -f 'hf download mistralai/Mistral' 2>/dev/null || true
  sleep 2
  find "$MODEL_DIR/.cache/huggingface/download" -maxdepth 1 \
    \( -name '*.lock' -o -name '*.incomplete' \) -delete 2>/dev/null || true
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate csp-llm

export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HOME=/public/data_zoo/huggingface

log "=== Mistral download start (HF_HUB_DISABLE_XET=1) ==="
log "Target: $MODEL_DIR"
log "Log: $LOG"

stdbuf -oL -eL hf download mistralai/Mistral-7B-Instruct-v0.3 \
  --local-dir "$MODEL_DIR" \
  --exclude 'consolidated.safetensors' \
  --exclude '*.bin' \
  ${HF_TOKEN:+--token "$HF_TOKEN"} \
  2>&1 | stdbuf -oL -eL tee -a "$LOG"

n=$(ls "$MODEL_DIR"/model-*-of-*.safetensors 2>/dev/null | wc -l)
if [[ "$n" -ge 3 ]]; then
  log "OK: $n safetensor shards, $(du -sh "$MODEL_DIR" | awk '{print $1}')"
else
  log "FAIL: only $n/3 shards — check $LOG and xet/proxy issues"
  exit 1
fi
