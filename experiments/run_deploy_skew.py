"""MIN14: quantify the deployment s_cross proxy.

The serving simulator reconstructs the cross-layer view from the prior layer's
within-EMA top-r (s_query is set neutral). This script quantifies (offline, on
held-out traces) how redundant the cross view is with the within view, how much
marginal AUC the *true* cross view adds, and writes a within-only checkpoint so
the simulator run can isolate the reconstructed-cross contribution end-to-end.

    python experiments/run_deploy_skew.py --traces experiments/traces \
        --model Llama-3.1-8B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from run_icdm_full import load_model_trace, request_split, subsample, TRAIN_N, roc_auc
from xqp.features import topk_indicator
from xqp.predictor import ClosedFormXQP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="experiments/traces")
    ap.add_argument("--model", default="Llama-3.1-8B-Instruct")
    ap.add_argument("--horizon", default="h4")
    ap.add_argument("--out", default="experiments/predictors")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    d = load_model_trace(os.path.join(a.traces, f"{a.model}.jsonl"))
    tr_idx, te_idx = request_split(d["rid"])
    te = subsample(te_idx, 150_000)
    F, y = d["F"], d["y"][a.horizon].astype(np.float32)
    fw, fc = F[te, 0], F[te, 1]

    # (1) redundancy of the TRUE cross indicator with the within view.
    # within top-r set per (rid,layer,step), compared to f_cross (prev-layer top-r).
    rid, layer, step = d["rid"][te], d["layer"][te], d["step"][te]
    key = rid.astype(np.int64) * 100000 + layer.astype(np.int64) * 1000 + step.astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks, fws, fcs = key[order], fw[order], fc[order]
    bnd = np.flatnonzero(np.diff(ks)) + 1
    starts = np.concatenate([[0], bnd]); ends = np.concatenate([bnd, [len(ks)]])
    jacc, agree = [], []
    for s, e in zip(starts, ends):
        if e - s < 4:
            continue
        wtop = topk_indicator(fws[s:e], 0.10) > 0.5
        ctop = fcs[s:e] > 0.5
        u = (wtop | ctop).sum()
        if u:
            jacc.append((wtop & ctop).sum() / u)
        if wtop.sum():
            agree.append((wtop & ctop).sum() / wtop.sum())   # P(cross=1 | within-top-r)
    corr = float(np.corrcoef(fw, fc)[0, 1])

    # (2) marginal AUC of cross over within
    auc_within = roc_auc(y[te], fw)
    cf2 = ClosedFormXQP.from_fit(F[subsample(tr_idx, TRAIN_N)] * np.array([1, 1, 0, 0], np.float32),
                                 y[subsample(tr_idx, TRAIN_N)])
    auc_2v = roc_auc(y[te], cf2.score(F[te] * np.array([1, 1, 0, 0], np.float32)))

    # (3) within-only checkpoint for the end-to-end isolation run
    cf1 = ClosedFormXQP.from_fit(F[subsample(tr_idx, TRAIN_N)] * np.array([1, 0, 0, 0], np.float32),
                                 y[subsample(tr_idx, TRAIN_N)])
    p1 = os.path.join(a.out, f"xqp_closed_within_{a.horizon}.json")
    cf1.save(p1)

    print(f"=== s_cross proxy / redundancy ({a.model}, {a.horizon}) ===")
    print(f"pearson corr(f_within, f_cross)         : {corr:.3f}")
    print(f"Jaccard(within-top-r, cross set)        : {np.mean(jacc):.3f}")
    print(f"P(cross=1 | within in top-r)            : {np.mean(agree):.3f}")
    print(f"AUC within-only                         : {auc_within:.4f}")
    print(f"AUC within+cross (offline true cross)   : {auc_2v:.4f}  (marginal +{auc_2v-auc_within:.4f})")
    print(f"within-only ckpt -> {p1}")

    # Persist the offline skew metrics so the deployment.tex / limitations.tex
    # numbers have a backing file (previously stdout-only -- repro provenance gap).
    metrics = {
        "model": a.model, "horizon": a.horizon, "n_test": int(len(te)),
        "corr_within_cross": round(corr, 4),
        "jaccard_within_topr_cross": round(float(np.mean(jacc)), 4),
        "p_cross_given_within_topr": round(float(np.mean(agree)), 4),
        "auc_within_only": round(float(auc_within), 4),
        "auc_within_plus_cross_offline": round(float(auc_2v), 4),
        "marginal_cross_auc": round(float(auc_2v - auc_within), 4),
        "note": ("offline redundancy of the TRUE cross view vs within; the end-to-end "
                 "reconstructed-cross eps (within-only vs 2-view) comes from the SEER "
                 "deploy sim, see results/deploy_consistent/"),
    }
    mp = os.path.join(os.path.dirname(__file__), "results", "deploy_skew.json")
    os.makedirs(os.path.dirname(mp), exist_ok=True)
    json.dump(metrics, open(mp, "w"), indent=2)
    print(f"metrics -> {mp}")


if __name__ == "__main__":
    main()
