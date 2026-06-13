#!/usr/bin/env bash
# Migrate /home/lzq HF models -> /public/model_zoo, datasets -> /public/data_zoo
set -euo pipefail

LOG="/home/lzq/codes/csp-llm/experiments/migrate_lzq_report.log"
HF_HUB="/home/lzq/.cache/huggingface/hub"
MODEL_ZOO="/public/model_zoo"
DATA_ZOO="/public/data_zoo/huggingface/hub"
REPORT="/home/lzq/codes/csp-llm/experiments/migrate_lzq_report.md"

mkdir -p "$DATA_ZOO" "$(dirname "$LOG")"
: > "$LOG"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# Export full HF model cache -> flat model_zoo dir (snapshot contents).
export_hf_model() {
  local cache="$1" dest_name="$2"
  local dest="$MODEL_ZOO/$dest_name"
  local snap
  snap=$(find "$cache/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
  if [[ -z "$snap" ]]; then
    log "SKIP model $dest_name: no snapshot in $cache"
    return 1
  fi
  local nweights
  nweights=$(find "$snap" \( -name '*.safetensors' -o -name '*.bin' \) 2>/dev/null | wc -l)
  if [[ "$nweights" -eq 0 ]]; then
    log "SKIP model $dest_name: stub only ($(du -sh "$cache" | awk '{print $1}'))"
    return 2
  fi
  if [[ -d "$dest" && -f "$dest/config.json" ]]; then
    log "SKIP model $dest_name: already exists at $dest"
    rm -rf "$cache"
    log "  removed HF cache $cache (duplicate of model_zoo)"
    return 0
  fi
  log "MOVE model $dest_name: $cache -> $dest ($(du -sh "$cache" | awk '{print $1}'))"
  mkdir -p "$dest"
  # HF snapshots are symlinks into blobs/ — must dereference (-L).
  rsync -aHL --remove-source-files "$snap/" "$dest/"
  rm -rf "$cache"
  log "  done $dest_name ($(du -sh "$dest" | awk '{print $1}'))"
}

move_dataset() {
  local cache="$1"
  local base dest
  base=$(basename "$cache")
  dest="$DATA_ZOO/$base"
  if [[ -d "$dest" ]]; then
    log "SKIP dataset $base: already at $dest"
    rm -rf "$cache"
    return 0
  fi
  log "MOVE dataset $base: $cache -> $dest ($(du -sh "$cache" | awk '{print $1}'))"
  rsync -aH --remove-source-files "$cache/" "$dest/"
  rm -rf "$cache"
  log "  done $base ($(du -sh "$dest" | awk '{print $1}'))"
}

log "=== migration start ==="
log "root free before: $(df -h / | awk 'NR==2{print $4}')"
log "public free before: $(df -h /public | awk 'NR==2{print $4}')"

# --- full models (>500MB with weights) ---
declare -A MODEL_MAP=(
  ["models--mistralai--Mistral-7B-v0.1"]="Mistral-7B-v0.1"
  ["models--Qwen--Qwen2.5-7B"]="Qwen2.5-7B"
  ["models--meta-llama--Llama-2-7b-hf"]="Llama-2-7B-hf"
  ["models--facebook--opt-1.3b"]="opt-1.3b"
  ["models--HuggingFaceTB--SmolLM-135M"]="SmolLM-135M"
  ["models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli"]="DeBERTa-v3-base-mnli-fever-anli"
)

MOVED_MODELS=()
SKIPPED_MODELS=()
REMOVED_STUBS=()

for cache_name in "${!MODEL_MAP[@]}"; do
  dest_name="${MODEL_MAP[$cache_name]}"
  cache="$HF_HUB/$cache_name"
  [[ -d "$cache" ]] || continue
  if export_hf_model "$cache" "$dest_name"; then
    MOVED_MODELS+=("$dest_name")
  else
    rc=$?
    if [[ $rc -eq 2 ]]; then REMOVED_STUBS+=("$cache_name"); rm -rf "$cache"; fi
    SKIPPED_MODELS+=("$dest_name")
  fi
done

# opt_weights (numpy checkpoint, separate from HF opt-1.3b)
if [[ -d /home/lzq/opt_weights/opt-1.3b-np ]]; then
  dest="$MODEL_ZOO/opt-1.3b-np"
  if [[ -d "$dest" ]]; then
    log "SKIP opt-1.3b-np: already at $dest; removing home copy"
    rm -rf /home/lzq/opt_weights
  else
    log "MOVE opt-1.3b-np -> $dest ($(du -sh /home/lzq/opt_weights | awk '{print $1}'))"
    rsync -aH --remove-source-files /home/lzq/opt_weights/opt-1.3b-np/ "$dest/"
    rm -rf /home/lzq/opt_weights
    MOVED_MODELS+=("opt-1.3b-np")
    log "  done opt-1.3b-np"
  fi
fi

# stub / partial models whose full version already in model_zoo
declare -A STUB_TO_ZOO=(
  ["models--meta-llama--Llama-3.1-8B-Instruct"]="Llama-3.1-8B-Instruct"
  ["models--Qwen--Qwen2.5-7B-Instruct"]="Qwen2.5-7B-Instruct"
  ["models--Qwen--Qwen2.5-14B-Instruct"]="Qwen2.5-14B-Instruct"
  ["models--Qwen--Qwen2.5-32B-Instruct"]="Qwen2.5-32B-Instruct"
  ["models--meta-llama--Meta-Llama-3-8B-Instruct"]="Llama-3.1-8B-Instruct"
)

for cache_name in "${!STUB_TO_ZOO[@]}"; do
  cache="$HF_HUB/$cache_name"
  [[ -d "$cache" ]] || continue
  zoo_name="${STUB_TO_ZOO[$cache_name]}"
  if [[ -d "$MODEL_ZOO/$zoo_name" ]]; then
    log "REMOVE stub $cache_name (full version: $MODEL_ZOO/$zoo_name)"
    rm -rf "$cache"
    REMOVED_STUBS+=("$cache_name -> $zoo_name")
  fi
done

# remaining tiny model stubs (no zoo full version)
for cache in "$HF_HUB"/models--*; do
  [[ -d "$cache" ]] || continue
  sz=$(du -sb "$cache" | awk '{print $1}')
  if [[ "$sz" -lt 104857600 ]]; then  # <100MB
    log "REMOVE tiny stub $(basename "$cache") ($(du -sh "$cache" | awk '{print $1}'))"
    rm -rf "$cache"
    REMOVED_STUBS+=("$(basename "$cache")")
  fi
done

# --- datasets ---
MOVED_DATASETS=()
for cache in "$HF_HUB"/datasets--*; do
  [[ -d "$cache" ]] || continue
  move_dataset "$cache"
  MOVED_DATASETS+=("$(basename "$cache")")
done

# cleanup empty hub dirs
rmdir /home/lzq/.cache/huggingface/hub/.locks 2>/dev/null || true
find /home/lzq/.cache/huggingface -type d -empty -delete 2>/dev/null || true

log "=== migration end ==="
log "root free after: $(df -h / | awk 'NR==2{print $4}')"
log "public free after: $(df -h /public | awk 'NR==2{print $4}')"

# write markdown report
{
  echo "# /home/lzq 资产迁移报告"
  echo ""
  echo "执行时间: $(date -Iseconds)"
  echo ""
  echo "## 磁盘变化"
  echo ""
  echo "| 分区 | 迁移前 | 迁移后 |"
  echo "|------|--------|--------|"
  echo "| \`/\` 根分区 | 见日志 | $(df -h / | awk 'NR==2{print $4" 可用 ("$5" 已用)"}') |"
  echo "| \`/public\` | 见日志 | $(df -h /public | awk 'NR==2{print $4" 可用 ("$5" 已用)"}') |"
  echo ""
  echo "## 迁入 /public/model_zoo 的模型"
  echo ""
  for m in "${MOVED_MODELS[@]}"; do
    if [[ -d "$MODEL_ZOO/$m" ]]; then
      echo "- \`$m\` — $(du -sh "$MODEL_ZOO/$m" 2>/dev/null | awk '{print $1}')"
    fi
  done
  echo ""
  echo "## 迁入 /public/data_zoo 的数据集"
  echo ""
  echo "路径前缀: \`/public/data_zoo/huggingface/hub/\`"
  echo ""
  for d in "$DATA_ZOO"/datasets--*; do
    [[ -d "$d" ]] || continue
    echo "- \`$(basename "$d")\` — $(du -sh "$d" | awk '{print $1}')"
  done
  echo ""
  echo "## 已删除的 stub / 重复 HF 缓存"
  echo ""
  for s in "${REMOVED_STUBS[@]}"; do
    echo "- \`$s\`"
  done
  echo ""
  echo "## 使用方式"
  echo ""
  echo '```bash'
  echo "# 模型（本地路径，无需 HF 下载）"
  echo "python ... --model /public/model_zoo/Mistral-7B-v0.1"
  echo ""
  echo "# 数据集（设置 HF 缓存目录）"
  echo "export HF_HOME=/public/data_zoo/huggingface"
  echo "export HF_DATASETS_CACHE=/public/data_zoo/huggingface/hub"
  echo '```'
  echo ""
  echo "详细日志: \`experiments/migrate_lzq_report.log\`"
} > "$REPORT"

log "Report written to $REPORT"
