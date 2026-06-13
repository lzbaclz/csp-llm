"""Fit and save the XQP closed-form predictor used by the SEER deployment policy.

Saves the analysis-selected within+cross (2-view) model — query/recency columns
masked to ~0 weight — as a 4-weight ClosedFormXQP JSON, plus the full 4-view
model for reference.

    python experiments/train_deploy_ckpt.py --traces experiments/traces \
        --model Llama-3.1-8B-Instruct --out experiments/predictors
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from run_icdm_full import load_model_trace, request_split, subsample, TRAIN_N
from xqp.predictor import ClosedFormXQP
from xqp.eval import roc_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="experiments/traces")
    ap.add_argument("--model", default="Llama-3.1-8B-Instruct")
    ap.add_argument("--horizon", default="h4")
    ap.add_argument("--out", default="experiments/predictors")
    ap.add_argument("--trace-file", default=None,
                    help="explicit trace path (e.g. a quest_headline .quest.jsonl for a "
                         "model whose trace is not experiments/traces/<model>.jsonl)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = load_model_trace(a.trace_file or os.path.join(a.traces, f"{a.model}.jsonl"))
    tr_idx, te_idx = request_split(d["rid"])
    tr = subsample(tr_idx, TRAIN_N); te = subsample(te_idx, 150_000)
    F, y = d["F"], d["y"][a.horizon].astype(np.float32)

    for tag, cols in [("2view", [0, 1]), ("4view", [0, 1, 2, 3])]:
        mask = np.zeros(4, np.float32); mask[cols] = 1.0
        cf = ClosedFormXQP.from_fit(F[tr] * mask, y[tr])
        auc = roc_auc(y[te], cf.score(F[te] * mask))
        path = os.path.join(a.out, f"xqp_closed_{tag}_{a.horizon}.json")
        cf.save(path)
        print(f"{tag}: weights={np.round(cf.weights,4).tolist()} bias={float(cf.bias):.4f} "
              f"val_AUC={auc:.4f} -> {path}")


if __name__ == "__main__":
    main()
