#!/usr/bin/env bash
# =============================================================================
# run_panel_strengthen.sh -- two panel-driven strengthening sweeps.
#
# EXP-C(a)  THIRD-architecture transfer pair (MEDIUM a): the architecture-agnostic
#           Llama-trained 2-view scorer applied FROZEN in MISTRAL-7B's masked-decode
#           loop, vs Mistral's own H2O -> upgrades C4 from one pair (Llama<->Qwen) to
#           >=2 pairs (adds {Llama,Mistral}). 7 datasets x {h2o,xqp}.
#
# EXP-C(c)  TRUE-seed robustness (MEDIUM c): the LongBench loader now draws a
#           genuinely RANDOM subset per LONGBENCH_SEED (greedy decode made --seed a
#           no-op; fixed in seer/trace/datasets.py). Re-run the headline TOST AND
#           tab:perlayer contrasts on 3 real seeds. Llama, 7 datasets,
#           {h2o,xqp,pyramidkv,adakv} x seeds {1,2,3}.
#
# MAXJOBS=1 (oracle-forward spikes ~43GB; concurrency OOMs). GPU 1 ONLY (GPU 0 is a
# colleague's). Resumable: existing valid JSON is skipped.
# =============================================================================
set -u
SEER=/home/lzq/codes/SEER
PY=/home/lzq/miniconda3/envs/csp-llm/bin/python
LCK=/home/lzq/codes/csp-llm/experiments/predictors/xqp_closed_2view_h4.json   # Llama-trained
LB_DIR=/public/data_zoo/longbench/data
LLAMA=/public/model_zoo/Llama-3.1-8B-Instruct
MISTRAL=/public/model_zoo/Mistral-7B-Instruct-v0.3

export PYTHONPATH="$SEER" TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
DATASETS=(narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique triviaqa)
N=64; NEW=48; CTX=4096; B=0.20; SLO="P99=200ms"

is_done () { [ -s "$1" ] && "$PY" - "$1" <<'PYEOF' >/dev/null 2>&1
import json,sys; sys.exit(0 if len(json.load(open(sys.argv[1])).get("results",[]))>=1 else 1)
PYEOF
}
runcell () {  # model policy extra outpath seed_env
  local model="$1" pol="$2" extra="$3" out="$4" senv="$5" log="${4%.json}.log"
  if is_done "$out"; then echo "[strengthen] SKIP $(basename $(dirname $out))/$(basename $out)"; return; fi
  echo "[strengthen] RUN  $out  (LONGBENCH_SEED=$senv)"
  env LONGBENCH_PATH="$LB_DIR/$DS.jsonl" ${senv:+LONGBENCH_SEED=$senv} "$PY" -m seer.eval.runner \
    --model "$model" --policy "$pol" $extra --workload longbench \
    --context_length "$CTX" --num_requests "$N" --max_new_tokens "$NEW" \
    --hbm_budget "$B" --slo "$SLO" --io_mode measured-dma --chat --seed 0 \
    --out "$out" > "$log" 2>&1
  echo "[strengthen] DONE $out rc=$? $(grep -oE 'F1=[0-9.]+' "$log" | tail -1)"
}

echo "[strengthen] start $(date)"

# --- EXP-C(a): Mistral transfer (Llama scorer frozen in Mistral loop) ---------
OUT=/home/lzq/codes/csp-llm/experiments/results/transfer_mistral; mkdir -p "$OUT"
for DS in "${DATASETS[@]}"; do
  [ -s "$LB_DIR/$DS.jsonl" ] || continue
  runcell "$MISTRAL" h2o "" "$OUT/${DS}_h2o.json" ""
  runcell "$MISTRAL" xqp "--xqp-ckpt $LCK" "$OUT/${DS}_xqp.json" ""
done

# --- EXP-C(c): true-seed robustness on Llama ----------------------------------
for s in 1 2 3; do
  OUT=/home/lzq/codes/csp-llm/experiments/results/trueseed/s$s; mkdir -p "$OUT"
  for DS in "${DATASETS[@]}"; do
    [ -s "$LB_DIR/$DS.jsonl" ] || continue
    runcell "$LLAMA" h2o       ""                 "$OUT/${DS}_h2o.json"       "$s"
    runcell "$LLAMA" xqp       "--xqp-ckpt $LCK"  "$OUT/${DS}_xqp.json"       "$s"
    runcell "$LLAMA" pyramidkv ""                 "$OUT/${DS}_pyramidkv.json" "$s"
    runcell "$LLAMA" adakv     ""                 "$OUT/${DS}_adakv.json"     "$s"
  done
done
echo "[strengthen] FINISHED $(date)"
