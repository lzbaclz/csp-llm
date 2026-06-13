"""Diagnostic: does the learned (within+cross) model's edge over the reactive
H2O/within signal GROW with the prediction horizon?

Motivation: in deployment H2O (reactive accumulated attention) beats the learned
2-view at h~1 (next-step eviction). But H2O is *backward-looking*; the cross-layer
signal is *structural/anticipatory*. If the cross view's contribution — and the
2-view's margin over within-only — grows as the horizon h lengthens, then for the
LOOKAHEAD that prefetching actually needs (keep/fetch blocks that will be hot in h
steps), a predictive model should beat the reactive heavy-hitter. This script tests
that on the real traces, per horizon, with request-level splits.

    python experiments/run_horizon_headtohead.py --traces experiments/traces \
        --out experiments/results/horizon_headtohead.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from run_icdm_full import (load_model_trace, pool_models, request_split, roc_auc,
                           subsample, HORIZONS)
from xqp.predictor import ClosedFormXQP

CAP = 600_000


def recall_at_budget(score, y, budget):
    """Keep the top-`budget` fraction by score; return recall of salient blocks."""
    y = y.astype(bool)
    n = score.shape[0]
    k = max(1, int(round(budget * n)))
    keep = np.zeros(n, bool)
    keep[np.argpartition(-score, kth=min(k - 1, n - 1))[:k]] = True
    return float((keep & y).sum() / max(1, y.sum()))


def fit_mask(F, y, mask):
    """Fit a closed-form logistic on the masked feature subset; return scorer fn."""
    cf = ClosedFormXQP.from_fit(F * mask, y)
    return lambda G: np.asarray(cf.score(G * mask))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="experiments/traces")
    ap.add_argument("--out", default="experiments/results/horizon_headtohead.json")
    ap.add_argument("--budget", type=float, default=0.10)
    a = ap.parse_args()

    files = [f for f in sorted(glob.glob(os.path.join(a.traces, "*.jsonl"))) if ".smoke." not in f]
    d = pool_models({os.path.basename(f)[:-6]: load_model_trace(f) for f in files
                     if load_model_trace(f)})
    tr_i, te_i = request_split(d["rid"], frac=0.5)
    tr = subsample(tr_i, CAP); te = subsample(te_i, CAP)
    F = d["F"]
    m_w = np.array([1, 0, 0, 0], np.float32)   # within only  (≈ H2O reactive)
    m_c = np.array([0, 1, 0, 0], np.float32)   # cross only    (structural)
    m_wc = np.array([1, 1, 0, 0], np.float32)  # within+cross  (learned 2-view)
    print(f"pooled rows={F.shape[0]:,} | train={len(tr):,} test={len(te):,} | budget={a.budget}", flush=True)

    rows = []
    for h in HORIZONS:
        y = d["y"][h].astype(np.float32)
        ytr, yte = y[tr], y[te]
        # single-view AUCs (no fit needed — monotone in the raw feature)
        auc_within = roc_auc(yte, F[te, 0])
        auc_cross = roc_auc(yte, F[te, 1])
        auc_query = roc_auc(yte, F[te, 2])
        auc_rec = roc_auc(yte, F[te, 3])
        # fitted within+cross
        s_wc = fit_mask(F[tr], ytr, m_wc)
        auc_2v = roc_auc(yte, s_wc(F[te]))
        # recall@budget (the eviction-relevant metric) at several budgets
        s2 = s_wc(F[te])
        rec = dict(horizon=h, pos_rate=float(yte.mean()),
                   auc_within=round(auc_within, 4), auc_cross=round(auc_cross, 4),
                   auc_2view=round(auc_2v, 4), auc_query=round(auc_query, 4),
                   auc_recency=round(auc_rec, 4),
                   gap_2view_minus_within=round(auc_2v - auc_within, 4),
                   gap_cross_minus_within=round(auc_cross - auc_within, 4))
        recall_str = []
        for b in (0.10, 0.20, 0.30):
            rw = recall_at_budget(F[te, 0], yte, b)
            r2 = recall_at_budget(s2, yte, b)
            rh2o = recall_at_budget(F[te, 0], yte, b)  # within ≈ H2O reactive proxy
            rec[f"recall_within_b{int(b*100)}"] = round(rw, 4)
            rec[f"recall_2view_b{int(b*100)}"] = round(r2, 4)
            rec[f"recall_gain_b{int(b*100)}"] = round(r2 - rw, 4)
            recall_str.append(f"b{int(b*100)}:{rw:.3f}->{r2:.3f}(+{r2-rw:.3f})")
        rows.append(rec)
        print(f"  {h}: AUC within={auc_within:.3f} 2view={auc_2v:.3f} (Δ{auc_2v-auc_within:+.4f}) "
              f"| R@budget within->2view: " + "  ".join(recall_str), flush=True)

    # verdict
    g1 = rows[0]["gap_2view_minus_within"]; g64 = rows[-1]["gap_2view_minus_within"]
    cg1 = rows[0]["gap_cross_minus_within"]; cg64 = rows[-1]["gap_cross_minus_within"]
    verdict = ("PREDICTIVE EDGE GROWS with horizon (2view margin over reactive within "
               f"goes {g1:+.4f} @h1 -> {g64:+.4f} @h64)" if g64 > g1 + 0.005 else
               "NO growing predictive edge (margin flat/shrinks with horizon)")
    print("\nVERDICT:", verdict)
    out = dict(budget=a.budget, by_horizon=rows,
               gap_2view_h1=g1, gap_2view_h64=g64,
               gap_cross_h1=cg1, gap_cross_h64=cg64, verdict=verdict)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
