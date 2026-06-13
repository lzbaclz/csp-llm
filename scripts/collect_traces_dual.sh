#!/usr/bin/env bash
# Dual-GPU trace collection: split prompts 50/50 per model, merge JSONL shards.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

N_TRACES="${N_TRACES:-200}"
MAX_CONTEXT="${MAX_CONTEXT:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
WORKLOAD="${WORKLOAD:-mooncake}"
HALF=$((N_TRACES / 2))

MODELS=(
  "/public/model_zoo/Llama-3.1-8B-Instruct"
  "/public/model_zoo/Qwen2.5-7B-Instruct"
  "/public/model_zoo/Qwen3-8B"
)

LOGDIR="experiments/logs"
mkdir -p "$LOGDIR" experiments/traces

echo "[$(date -Iseconds)] dual-GPU collect N_TRACES=$N_TRACES split=0:$HALF + $HALF:$N_TRACES" \
  | tee "$LOGDIR/phase1_collect.log"

for model_path in "${MODELS[@]}"; do
  stem=$(basename "$model_path")
  echo "[$(date -Iseconds)] model=$stem" | tee -a "$LOGDIR/phase1_collect.log"

  rm -f "experiments/traces/${stem}.gpu0.jsonl" \
         "experiments/traces/${stem}.gpu1.jsonl" \
         "experiments/traces/${stem}.jsonl"

  python "$ROOT/scripts/collect_traces_attn.py" \
    --n-traces "$N_TRACES" --prompt-start 0 --prompt-end "$HALF" \
    --max-context "$MAX_CONTEXT" --max-new-tokens "$MAX_NEW_TOKENS" \
    --workload "$WORKLOAD" --device cuda:0 --worker-id gpu0 \
    --out-suffix gpu0 --models "$model_path" \
    >> "$LOGDIR/phase1_collect.log" 2>&1 &
  pid0=$!

  python "$ROOT/scripts/collect_traces_attn.py" \
    --n-traces "$N_TRACES" --prompt-start "$HALF" --prompt-end "$N_TRACES" \
    --max-context "$MAX_CONTEXT" --max-new-tokens "$MAX_NEW_TOKENS" \
    --workload "$WORKLOAD" --device cuda:1 --worker-id gpu1 \
    --out-suffix gpu1 --models "$model_path" \
    >> "$LOGDIR/phase1_collect.log" 2>&1 &
  pid1=$!

  wait "$pid0" "$pid1"

  cat "experiments/traces/${stem}.gpu0.jsonl" \
      "experiments/traces/${stem}.gpu1.jsonl" \
    > "experiments/traces/${stem}.jsonl"

  rows=$(wc -l < "experiments/traces/${stem}.jsonl")
  echo "[$(date -Iseconds)] merged $stem rows=$rows" | tee -a "$LOGDIR/phase1_collect.log"

  rm -f "experiments/traces/${stem}.gpu0.jsonl" \
        "experiments/traces/${stem}.gpu1.jsonl"
done

python3 - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
p = Path("experiments/logs/status.json")
p.write_text(json.dumps({
    "phase": "collect_done",
    "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "dual_gpu": True,
}, indent=2) + "\n")
PY

echo "[$(date -Iseconds)] dual-GPU collect DONE" | tee -a "$LOGDIR/phase1_collect.log"
