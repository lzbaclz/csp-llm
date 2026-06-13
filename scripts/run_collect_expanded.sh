#!/usr/bin/env bash
# Expanded XQP trace collection — multi-workload x 4 architectures, dual-GPU.
#
# WHY: the current corpus is ~32 prompts/model on a SINGLE workload (mooncake).
# For a benchmark paper that is the #1 reviewer attack (request-level CIs on
# <=32 groups, one workload). This broadens to N_TRACES prompts across several
# REAL workloads and 4 attention families, writing ONE file per (model,workload):
#     $TRACEDIR/<stem>.<workload>.jsonl
# plus a "<file>.done" sentinel (holding N_TRACES) written ONLY on a clean,
# complete, integrity-verified collection.
#
# INTEGRITY (data must never be synthetic-mislabeled as a real workload):
#   - PREFLIGHT each workload by calling seer.trace.datasets.load_prompts DIRECTLY
#     at the REAL n=N_TRACES (exception not swallowed) and FAIL on ANY ruler
#     "secret password" needle for a non-synthetic workload.
#   - POST-RUN, in the parent: grep the worker log for every silent-fallback
#     signature AND run verify_trace_splits.py --expect N. Any failure -> the
#     output is renamed .QUARANTINED and the whole run exits non-zero.
#
# COMPLETENESS / CRASH-SAFETY:
#   - skip is .done-sentinel based, NOT existence based: a crashed partial file
#     (no .done) is RE-COLLECTED, never silently accepted.
#   - each backgrounded worker's exit status is captured and aggregated; the
#     script exits non-zero if any cell failed or was quarantined.
#
# DISK: collection is large. A disk preflight estimates required bytes from the
# existing 32-prompt corpus and ABORTS if $TRACEDIR lacks room. Point TRACEDIR at
# a big volume (e.g. /public has ~900G free vs ~48G on /).
#
# USAGE:
#   TRACEDIR=/public/xqp_traces N_TRACES=256 WORKLOADS="mooncake sharegpt" \
#       bash scripts/run_collect_expanded.sh
#   bash scripts/run_collect_expanded.sh --preflight-only   # probe datasets + disk, collect nothing
#   FORCE=1 ...                                             # recollect even if .done exists
#
# NOTE on request_id namespacing: each (model,workload) file is collected with
# prompt_offset=0, so request_ids are p0..p{N-1} PER FILE. The benchmark protocol
# MUST key groups on (model, workload, request_id), NOT request_id alone.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export ROOT
PY="${PY:-/home/lzq/miniconda3/envs/csp-llm/bin/python}"
[[ -x "$PY" ]] || PY=python3

# Offline + strict so the SEER loader RAISES (instead of degenerating) on a
# missing dataset. The preflight/postflight catch what the collector swallows.
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HOME="${HF_HOME:-/public/data_zoo/huggingface}"
export SEER_STRICT_WORKLOAD="${SEER_STRICT_WORKLOAD:-1}"

N_TRACES="${N_TRACES:-256}"
MAX_CONTEXT="${MAX_CONTEXT:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
WORKLOADS="${WORKLOADS:-mooncake sharegpt}"
FORCE="${FORCE:-0}"
ZOO="${ZOO:-/public/model_zoo}"
TRACEDIR="${TRACEDIR:-experiments/traces}"
DISK_MARGIN="${DISK_MARGIN:-1.20}"   # require est*margin bytes free
PREFLIGHT_ONLY=0
[[ "${1:-}" == "--preflight-only" ]] && PREFLIGHT_ONLY=1

LOGDIR="experiments/logs"
mkdir -p "$LOGDIR" "$TRACEDIR"
MAIN_LOG="$LOGDIR/collect_expanded.log"
VERIFY="$ROOT/scripts/verify_trace_splits.py"

# GPU0 runs Llama then Qwen3; GPU1 runs Qwen2.5 then Mistral.
GPU0_MODELS=("Llama-3.1-8B-Instruct" "Qwen3-8B")
GPU1_MODELS=("Qwen2.5-7B-Instruct" "Mistral-7B-Instruct-v0.3")
ALL_MODELS=("${GPU0_MODELS[@]}" "${GPU1_MODELS[@]}")

# All MAIN_LOG writes happen in the single-threaded parent (never inside a
# backgrounded worker) to avoid interleaved/garbled audit lines.
log() { echo "[$(date -Iseconds)] $*" >>"$MAIN_LOG"; echo "[$(date -Iseconds)] $*"; }

LB_DIR="${LB_DIR:-/public/data_zoo/longbench/data}"
# Map a workload spec to: <loader-workload>|<file-suffix>|<LONGBENCH_PATH or empty>
#   mooncake / sharegpt / longbench / ruler -> passed through unchanged
#   longbench:<task>  -> loader 'longbench', suffix 'longbench_<task>',
#                        reads $LB_DIR/<task>.jsonl (e.g. longbench:hotpotqa).
parse_wl () {
  case "$1" in
    longbench:*) printf 'longbench|longbench_%s|%s/%s.jsonl' "${1#longbench:}" "$LB_DIR" "${1#longbench:}" ;;
    *)           printf '%s|%s|' "$1" "$1" ;;
  esac
}

FAILED=0
QUARANTINED=()
PRODUCED=()

# ---- Preflight: probe each workload via the REAL loader at the REAL n ---------
preflight_workload () {  # $1=workload -> exit 0 if usable (and REAL, unless synthetic-by-design)
  "$PY" - "$1" "$MAX_CONTEXT" "$N_TRACES" <<'PY'
import os, sys, pathlib
root = pathlib.Path(os.environ["ROOT"])
sys.path.insert(0, str(root))
seer = root.parent / "SEER"
if seer.is_dir():
    sys.path.insert(0, str(seer))
wl, ctx, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
SYNTHETIC_OK = {"ruler", "synthetic", "ruler-synthetic"}
# RULER fallback structural needle (stable across the 3 filler texts).
NEEDLE = "secret password"
try:
    from seer.trace.datasets import load_prompts
    ps = load_prompts(wl, [ctx], n, tokenizer=None)
except Exception as e:
    print(f"FAIL  {wl:12s} {type(e).__name__}: {e}")
    sys.exit(2)
got = len(ps); distinct = len(set(ps))
if got < n:
    print(f"FAIL  {wl:12s} loader returned {got}/{n} prompts (would pad/short)")
    sys.exit(3)
needle = sum(NEEDLE in p for p in ps)
sample = (ps[0][:60].replace(chr(10), " ")) if ps else ""
if needle > 0 and wl.lower() not in SYNTHETIC_OK:
    print(f"FAIL  {wl:12s} {needle}/{got} prompts are RULER synthetic -- "
          f"loader silently fell back; dataset for '{wl}' is NOT fully real.")
    sys.exit(4)
tag = " [SYNTHETIC-by-design]" if wl.lower() in SYNTHETIC_OK else ""
flag = "" if distinct == got else f"  (WARNING {distinct}/{got} distinct)"
print(f"OK    {wl:12s} got={got} distinct={distinct}{tag}{flag}  e.g. {sample!r}")
sys.exit(0)
PY
}

log "=== PREFLIGHT (HF_HOME=$HF_HOME strict=$SEER_STRICT_WORKLOAD n=$N_TRACES) ==="
READY=()
for wl in $WORKLOADS; do
  IFS='|' read -r lpwl suffix lbpath <<<"$(parse_wl "$wl")"
  if out=$(LONGBENCH_PATH="$lbpath" preflight_workload "$lpwl"); then
    log "  [$wl] $out"; READY+=("$wl")
  else
    log "  [$wl] $out"
    log "  -> EXCLUDING '$wl' (unavailable/short/synthetic; download via scripts/download_assets.sh)"
  fi
done
[[ ${#READY[@]} -eq 0 ]] && { log "No usable workloads. Aborting."; exit 1; }
log "Ready workloads: ${READY[*]}"

# ---- Disk preflight: estimate bytes and abort if $TRACEDIR is too small -------
# Baseline = bytes of the existing 32-prompt single-workload corpus (4 models).
BASELINE_BYTES=0
for m in "${ALL_MODELS[@]}"; do
  f="$TRACEDIR/${m}.jsonl"
  [[ -f "$f" ]] && BASELINE_BYTES=$(( BASELINE_BYTES + $(stat -c%s "$f") ))
done
# Fallback if legacy files are gone: ~73MB/prompt/model * 4 models = 292MB/prompt.
[[ "$BASELINE_BYTES" -eq 0 ]] && BASELINE_BYTES=$(( 292*1024*1024*32 ))
AVAIL=$("$PY" -c "import os,shutil; print(shutil.disk_usage('$TRACEDIR').free)")
EST=$("$PY" -c "print(int($BASELINE_BYTES/32*$N_TRACES*${#READY[@]}*$DISK_MARGIN))")
hb() { "$PY" -c "print(f'{$1/2**30:.1f} GiB')"; }
log "Disk: TRACEDIR=$TRACEDIR avail=$(hb "$AVAIL") est_need=$(hb "$EST") (N=$N_TRACES x ${#READY[@]} workloads x4 models, margin=$DISK_MARGIN)"
if [[ "$EST" -gt "$AVAIL" ]]; then
  log "!! ABORT: estimated $(hb "$EST") needed but only $(hb "$AVAIL") free on $TRACEDIR."
  log "   Fix: set TRACEDIR to a bigger volume (e.g. /public has ~900G free), lower N_TRACES, or free space."
  exit 1
fi

if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
  log "--preflight-only: datasets + disk OK; stopping before collection."
  exit 0
fi

# ---- Collection --------------------------------------------------------------
# Worker: runs the collector to $wlog and returns its exit code. No MAIN_LOG
# writes here (parent-only logging). Skip is .done-sentinel based.
collect () {  # $1=gpu $2=model $3=workload-spec  -> returns python rc (or 0 if skipped)
  local gpu="$1" name="$2" spec="$3" lpwl suffix lbpath
  IFS='|' read -r lpwl suffix lbpath <<<"$(parse_wl "$spec")"
  local mp="$ZOO/$name"
  local out="$TRACEDIR/${name}.${suffix}.jsonl"
  local wlog="$LOGDIR/collect_${name}_${suffix}.log"
  [[ ! -d "$mp" ]] && { echo "MISSING_MODEL" >"$wlog"; return 0; }
  if [[ "$FORCE" != "1" && -f "$out.done" && "$(cat "$out.done" 2>/dev/null)" == "$N_TRACES" ]]; then
    echo "SKIP_COMPLETE" >"$wlog"; return 0
  fi
  rm -f "$out.done"
  LONGBENCH_PATH="$lbpath" CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/collect_traces_attn.py \
    --n-traces "$N_TRACES" --prompt-start 0 --prompt-end "$N_TRACES" \
    --max-context "$MAX_CONTEXT" --max-new-tokens "$MAX_NEW_TOKENS" \
    --workload "$lpwl" --device cuda:0 --worker-id "${name}_${suffix}_gpu${gpu}" \
    --out-suffix "$suffix" --out-dir "$TRACEDIR" --models "$mp" \
    >"$wlog" 2>&1
  return $?
}

# Parent-side finalize: integrity-gate one (model,workload) cell after wait.
finalize () {  # $1=model $2=workload-spec $3=rc
  local name="$1" spec="$2" rc="$3" lpwl suffix lbpath
  IFS='|' read -r lpwl suffix lbpath <<<"$(parse_wl "$spec")"
  local out="$TRACEDIR/${name}.${suffix}.jsonl"
  local wlog="$LOGDIR/collect_${name}_${suffix}.log"
  if grep -q "MISSING_MODEL" "$wlog" 2>/dev/null; then
    log "!! FAIL $name/$spec: model dir missing under $ZOO"; FAILED=1; return
  fi
  if grep -q "SKIP_COMPLETE" "$wlog" 2>/dev/null; then
    log "SKIP $name/$spec (.done matches N_TRACES=$N_TRACES)"; PRODUCED+=("$out"); return
  fi
  if [[ "$rc" -ne 0 ]]; then
    log "!! FAIL $name/$spec: collector exited rc=$rc (see $wlog)"; FAILED=1; return
  fi
  # Silent-fallback signatures (collector-level AND SEER-loader-level).
  if grep -qE 'SEER prompts failed|falling back to RULER|padding with RULER' "$wlog" 2>/dev/null; then
    log "!! QUARANTINE $name/$spec: silent synthetic fallback detected -> ${out}.QUARANTINED"
    [[ -f "$out" ]] && mv "$out" "${out}.QUARANTINED"
    QUARANTINED+=("$out"); FAILED=1; return
  fi
  # Completeness + leakage gate.
  if "$PY" "$VERIFY" --expect "$N_TRACES" "$out" >>"$MAIN_LOG" 2>&1; then
    echo "$N_TRACES" >"$out.done"
    log "OK   $name/$spec: verified $N_TRACES requests, leakage-safe -> $out"
    PRODUCED+=("$out")
  else
    log "!! QUARANTINE $name/$spec: verify failed (incomplete/leaky) -> ${out}.QUARANTINED"
    [[ -f "$out" ]] && mv "$out" "${out}.QUARANTINED"
    QUARANTINED+=("$out"); FAILED=1
  fi
}

run_wave () {  # $1=gpu0-model $2=gpu1-model $3=workload
  local m0="$1" m1="$2" wl="$3" rc0 rc1
  log "START $m0/$wl on gpu0  |  $m1/$wl on gpu1"
  collect 0 "$m0" "$wl" &  local P0=$!
  collect 1 "$m1" "$wl" &  local P1=$!
  wait $P0; rc0=$?
  wait $P1; rc1=$?
  finalize "$m0" "$wl" "$rc0"
  finalize "$m1" "$wl" "$rc1"
}

for wl in "${READY[@]}"; do
  log "===== workload: $wl  (N_TRACES=$N_TRACES ctx=$MAX_CONTEXT) ====="
  run_wave "${GPU0_MODELS[0]}" "${GPU1_MODELS[0]}" "$wl"
  run_wave "${GPU0_MODELS[1]}" "${GPU1_MODELS[1]}" "$wl"
done

log "=== SUMMARY ==="
log "produced (${#PRODUCED[@]}): ${PRODUCED[*]:-none}"
[[ ${#QUARANTINED[@]} -gt 0 ]] && log "QUARANTINED (${#QUARANTINED[@]}): ${QUARANTINED[*]}"
if [[ "$FAILED" -ne 0 ]]; then
  log "RESULT: FAILED — some cells missing/quarantined. Re-run (FORCE=1 to redo specific cells)."
  exit 1
fi
log "RESULT: OK — verify the full set:"
log "    $PY $VERIFY --expect $N_TRACES ${PRODUCED[*]}"
exit 0
