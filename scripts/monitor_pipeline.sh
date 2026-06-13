#!/usr/bin/env bash
# Monitor XQP ga100 pipeline progress (one-shot snapshot).
# Usage:
#   bash scripts/monitor_pipeline.sh          # human-readable snapshot
#   bash scripts/monitor_pipeline.sh --watch  # refresh every 30s
#   bash scripts/monitor_pipeline.sh --json   # machine-readable status only
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$ROOT/experiments/logs"
STATUS="$LOGDIR/status.json"
PIDFILE="$LOGDIR/pipeline.pid"

json_only=false
watch=false
for arg in "$@"; do
  case "$arg" in
    --json) json_only=true ;;
    --watch) watch=true ;;
  esac
done

emit() {
  if $json_only; then
    if [[ -f "$STATUS" ]]; then cat "$STATUS"; else echo '{}'; fi
    return
  fi

  echo "========== XQP pipeline @ $(date -Iseconds) =========="
  if [[ -f "$PIDFILE" ]]; then
    pid=$(cat "$PIDFILE")
    if kill -0 "$pid" 2>/dev/null; then
      echo "PID $pid RUNNING  (elapsed: $(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' '))"
    else
      echo "PID $pid NOT RUNNING (stale pidfile)"
    fi
  else
    echo "No pidfile ($PIDFILE)"
  fi

  echo
  echo "--- status (main + dual-GPU workers) ---"
  python3 - <<'PY' "$LOGDIR"
import json, glob
from pathlib import Path
logdir = Path(__import__('sys').argv[1])
main = logdir / "status.json"
if main.exists():
    d = json.loads(main.read_text())
    print(f"main  phase={d.get('phase','?')}  env={d.get('env','?')}  "
          f"dual_gpu={d.get('dual_gpu', False)}  updated={d.get('updated_at','?')}")
for sf in sorted(logdir.glob("status.gpu*.json")):
    d = json.loads(sf.read_text())
    wid = d.get('worker_id', sf.stem)
    dev = d.get('device', '?')
    for name, m in (d.get('models') or {}).items():
        dp, tp = m.get('done_prompts', 0), m.get('total_prompts', '?')
        off = m.get('prompt_offset', 0)
        pct = f"{100*dp/tp:.1f}%" if isinstance(tp, int) and tp else "?"
        global_i = off + dp
        print(f"  [{wid} {dev}] {name}: {m.get('status','?')}  "
              f"slice {dp}/{tp} ({pct})  global~{global_i}/200  "
              f"rows={m.get('rows',0)}  eta={m.get('eta_s','?')}s")
if main.exists():
    d = json.loads(main.read_text())
    for name, m in (d.get('models') or {}).items():
        if not list(logdir.glob("status.gpu*.json")):
            dp, tp = m.get('done_prompts',0), m.get('total_prompts','?')
            pct = f"{100*dp/tp:.1f}%" if isinstance(tp,int) and tp else "?"
            print(f"  {name}: {m.get('status','?')}  prompts {dp}/{tp} ({pct})")
PY
  if [[ ! -f "$STATUS" ]] && ! compgen -G "$LOGDIR/status.gpu*.json" > /dev/null; then
    echo "(no status yet)"
  fi

  echo
  echo "--- trace files ---"
  if compgen -G "$ROOT/experiments/traces/*.jsonl" > /dev/null; then
    wc -l "$ROOT/experiments/traces/"*.jsonl 2>/dev/null || true
  else
    echo "(none yet)"
  fi

  echo
  echo "--- results ---"
  ls -la "$ROOT/experiments/results/" 2>/dev/null || echo "(none yet)"

  echo
  echo "--- GPU ---"
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"

  echo
  echo "--- tail pipeline.log ---"
  tail -n 8 "$LOGDIR/pipeline.log" 2>/dev/null || echo "(no pipeline.log)"

  echo
  echo "--- tail phase1_collect.log ---"
  tail -n 5 "$LOGDIR/phase1_collect.log" 2>/dev/null || echo "(no phase1 log)"
}

if $watch; then
  while true; do clear; emit; sleep 30; done
else
  emit
fi
