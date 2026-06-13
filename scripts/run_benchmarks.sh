#!/usr/bin/env bash
# End-to-end benchmark on ga100. Reuses upstream SEER eA/eE runners.
set -euo pipefail

mkdir -p experiments/results

# e1: predictability — AUC across {model, horizon, predictor variant}
python -m xqp.eval --traces experiments/traces \
                   --predictors experiments/predictors \
                   --out experiments/results/e1_auc.json

# e2: WCET — TRT+CUDA-Graph latency per predictor variant, batch 4096
python -m xqp.bench_wcet --predictors experiments/predictors \
                         --out experiments/results/e2_wcet.json

# e3: end-to-end TPOT — plug XQP into upstream SEER policy slot
pushd ../Seer
for budget in 0.20 0.30 0.40; do
  python -m seer.eval.runner --policy xqp --xqp-ckpt \
    ../next1/experiments/predictors/Meta-Llama-3-8B-Instruct_h4.json \
    --hbm-budget "$budget" \
    --workload mooncake-chat \
    --metrics tpot,miss \
    --out "../next1/experiments/results/e3_tpot_B${budget}.json"
done
popd

echo "All experiments complete; results in experiments/results/"
