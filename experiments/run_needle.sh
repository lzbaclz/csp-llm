#!/usr/bin/env bash
# Needle-in-haystack (RULER): the password sits mid-haystack, asked at the end --
# H2O's recency/heavy-hitter bias is DOCUMENTED to evict it. This is where
# question-anchored / retention-aware selection should beat H2O. Long context + tight
# budget = H2O's worst case. Metric = substring match on the password (needle recall).
# Policies probe the spectrum: streaming/recency (should FAIL) ... h2o ... question-aware.
set -u
SEER=/home/lzq/codes/SEER; PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
LCK=/home/lzq/codes/csp-llm/experiments/predictors/xqp_closed_2view_h4.json
MODEL=/public/model_zoo/Llama-3.1-8B-Instruct
OUT=/home/lzq/codes/csp-llm/experiments/results/needle; mkdir -p "$OUT"
export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CTX=${CTX:-16384}; N=${N:-64}
BUDGETS=(${BUDGETS:-0.10 0.20})
SELECTORS=(${SELLIST:-streaming recency h2o xqp qanchor qanchor_h2o snapkv_fixed})
is_done () { [ -s "$1" ] && "$PY" - "$1" <<'P' >/dev/null 2>&1
import json,sys;sys.exit(0 if len(json.load(open(sys.argv[1])).get("results",[]))>=1 else 1)
P
}
echo "[needle] start $(date) GPU=$CUDA_VISIBLE_DEVICES ctx=$CTX budgets=${BUDGETS[*]}"
for B in "${BUDGETS[@]}"; do
  od="$OUT/c${CTX}_b${B}"; mkdir -p "$od"
  for sel in "${SELECTORS[@]}"; do
    out="$od/${sel}.json"; log="${out%.json}.log"
    is_done "$out" && { echo "[needle] SKIP b$B/$sel"; continue; }
    extra=""; [ "$sel" = "xqp" ] && extra="--xqp-ckpt $LCK"
    "$PY" -m seer.eval.runner --model "$MODEL" --policy "$sel" $extra \
      --workload ruler --context_length "$CTX" --num_requests "$N" --max_new_tokens 16 \
      --hbm_budget "$B" --slo "P99=4000ms" --io_mode measured-dma --chat --seed 0 \
      --prefill_chunk 2048 --no_prefill_attn --skip_prewarm --out "$out" > "$log" 2>&1
    sub=$("$PY" -c "import json;d=json.load(open('$out'));print('substr=%.3f'%(d.get('substring_mean',-1)))" 2>/dev/null || echo "?")
    echo "[needle] DONE b$B/$sel rc=$? $sub"
  done
done
echo "[needle] FINISHED $(date)"
