#!/usr/bin/env bash
# Full A100 pipeline — conda env csp-llm, dual-GPU trace collection.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source "${CONDA_EXE:-$HOME/miniconda3/bin/conda}/etc/profile.d/conda.sh" 2>/dev/null \
  || source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate csp-llm

export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

LOGDIR="experiments/logs"
mkdir -p "$LOGDIR" experiments/traces experiments/predictors experiments/results

echo $$ > "$LOGDIR/pipeline.pid"

N_TRACES="${N_TRACES:-200}"
MAX_CONTEXT="${MAX_CONTEXT:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
SEER_ROOT="${SEER_ROOT:-$ROOT/../SEER}"

_update_phase() {
  python3 - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone
p = Path("$LOGDIR/status.json")
d = json.loads(p.read_text()) if p.exists() else {}
d["phase"] = "$1"
d["env"] = "csp-llm"
d["dual_gpu"] = True
d["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
p.write_text(json.dumps(d, indent=2) + "\n")
PY
}

echo "[$(date -Iseconds)] pipeline start pid=$$ env=csp-llm dual_gpu=1 N_TRACES=$N_TRACES" \
  | tee "$LOGDIR/pipeline.log"
echo '{"phase":"starting","env":"csp-llm","dual_gpu":true}' > "$LOGDIR/status.json"

# --- Phase 1: dual-GPU trace collection ---
echo "[$(date -Iseconds)] phase1 collect (dual GPU)" | tee -a "$LOGDIR/pipeline.log"
_update_phase collect
bash "$ROOT/scripts/collect_traces_dual.sh"

# --- Phase 2: train ---
echo "[$(date -Iseconds)] phase2 train" | tee -a "$LOGDIR/pipeline.log"
_update_phase train
bash "$ROOT/scripts/train_predictor.sh" 2>&1 | tee "$LOGDIR/phase2_train.log"

# --- Phase 3: e1 ---
echo "[$(date -Iseconds)] phase3 e1 eval" | tee -a "$LOGDIR/pipeline.log"
_update_phase e1_eval
python -m xqp.eval --traces experiments/traces \
                   --predictors experiments/predictors \
                   --out experiments/results/e1_auc.json \
  2>&1 | tee "$LOGDIR/phase3_e1.log"

# --- Phase 4: e2 ---
echo "[$(date -Iseconds)] phase4 e2 wcet" | tee -a "$LOGDIR/pipeline.log"
_update_phase e2_wcet
python -m xqp.bench_wcet --predictors experiments/predictors \
                         --out experiments/results/e2_wcet.json \
  2>&1 | tee "$LOGDIR/phase4_e2.log"

# --- Phase 5: ICDM ---
echo "[$(date -Iseconds)] phase5 icdm" | tee -a "$LOGDIR/pipeline.log"
_update_phase icdm
python experiments/run_icdm_analysis.py --traces experiments/traces --json \
  > experiments/results/icdm_analysis.json \
  2>&1 | tee "$LOGDIR/phase5_icdm_analysis.log"
python experiments/run_icdm_baselines.py --traces experiments/traces --json \
  > experiments/results/icdm_baselines.json \
  2>&1 | tee "$LOGDIR/phase5_icdm_baselines.log"

# --- Phase 6: e3 (optional) ---
if [[ -d "$SEER_ROOT" ]]; then
  echo "[$(date -Iseconds)] phase6 e3 tpot attempt" | tee -a "$LOGDIR/pipeline.log"
  _update_phase e3_tpot
  CKPT="$ROOT/experiments/predictors/Llama-3.1-8B-Instruct_h4.json"
  if [[ -f "$CKPT" ]]; then
    pushd "$SEER_ROOT" >/dev/null
    for budget in 0.20 0.30 0.40; do
      python -m seer.eval.runner \
        --model /public/model_zoo/Llama-3.1-8B-Instruct \
        --policy xqp \
        --xqp-ckpt "$CKPT" \
        --hbm-budget "$budget" \
        --workload mooncake \
        --metrics tpot,miss \
        --out "$ROOT/experiments/results/e3_tpot_B${budget}.json" \
        2>&1 | tee -a "$LOGDIR/phase6_e3.log" || true
    done
    popd >/dev/null
  fi
fi

_update_phase done
echo "[$(date -Iseconds)] pipeline DONE" | tee -a "$LOGDIR/pipeline.log"
rm -f "$LOGDIR/pipeline.pid"
