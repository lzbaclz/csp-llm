"""Two follow-ups the main driver flagged as worth a dedicated run:

(A) Minimal-sufficient-model metrics. The relevance/ablation analysis says only
    within+cross carry signal (query, recency are near-irrelevant on real
    traces). Report the full metric vector of the 1-view (within) and 2-view
    (within+cross) calibrated logistic models, so the paper can headline the
    analysis-selected minimal model (which matches GBDT on AUC) honestly.

(B) Adaptive-conformal convergence over a LONG decode stream. The main driver's
    drift split left only ~26 late steps, too few for ACI's threshold to
    converge. Here we run the adaptive-conformal layer over the full ordered
    decode stream (128 steps) on held-out requests and report the realized
    miss-rate trajectory + post-burn-in mean, vs a fixed threshold.

    python experiments/run_icdm_extra.py --traces experiments/traces \
        --out experiments/results/icdm_extra.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from run_icdm_full import (load_model_trace, pool_models, request_split,
                           subsample, roc_auc, TRAIN_N, TEST_N, SEED)
from xqp.dm_metrics import (average_precision, precision_at_k, recall_at_k,
                            expected_calibration_error, brier_score)
from xqp.predictor import ClosedFormXQP
from xqp.conformal import AdaptiveConformalSaliency


def minimal_models(pooled, horizon="h4", seed=SEED):
    tr_idx, te_idx = request_split(pooled["rid"], seed=seed)
    tr = subsample(tr_idx, TRAIN_N, seed); te = subsample(te_idx, TEST_N, seed)
    Ftr, ytr = pooled["F"][tr], pooled["y"][horizon][tr].astype(np.float32)
    Fte, yte = pooled["F"][te], pooled["y"][horizon][te].astype(np.float32)
    out = {}
    # view subsets by column index (within=0, cross=1, query=2, pos=3)
    subsets = {"within(1)": [0], "within+cross(2)": [0, 1],
               "within+cross+query(3)": [0, 1, 2], "all(4)": [0, 1, 2, 3]}
    for name, cols in subsets.items():
        mask = np.zeros(4, np.float32); mask[cols] = 1.0
        cf = ClosedFormXQP.from_fit(Ftr * mask, ytr)
        s = cf.score(Fte * mask)
        out[name] = dict(n_views=len(cols), params=len(cols) + 1,
                         auc=roc_auc(yte, s), auprc=average_precision(yte, s),
                         p_at_10=precision_at_k(yte, s, 0.10),
                         r_at_10=recall_at_k(yte, s, 0.10),
                         ece=expected_calibration_error(yte, s),
                         brier=brier_score(yte, s))
    return out


def conformal_convergence(pooled, horizon="h4", cols=(0, 1), alpha=0.10,
                          per_step=30000, seed=SEED):
    """Fit a frozen within+cross scorer on held-in requests; run adaptive vs
    fixed ACI over the full ordered decode stream on held-out requests."""
    tr_idx, te_idx = request_split(pooled["rid"], seed=seed)
    tr = subsample(tr_idx, TRAIN_N, seed)
    mask = np.zeros(4, np.float32); mask[list(cols)] = 1.0
    ytr = pooled["y"][horizon][tr].astype(np.float32)
    base = ClosedFormXQP.from_fit(pooled["F"][tr] * mask, ytr)
    # split-conformal fixed threshold: pick tau on HELD-IN data so the realized
    # miss rate equals alpha (the alpha-quantile of positive-class scores), the
    # fair budget/coverage-matched fixed baseline (not an arbitrary tau0=0.5).
    cal_scores = base.score(pooled["F"][tr] * mask)
    pos_scores = cal_scores[ytr > 0.5]
    tau_split = float(np.quantile(pos_scores, alpha)) if pos_scores.size else 0.5

    # build the test stream ordered by decode step (sampled per step)
    rng = np.random.default_rng(seed)
    step_te = pooled["step"][te_idx]
    F_te = pooled["F"][te_idx] * mask
    y_te = pooled["y"][horizon][te_idx].astype(np.float32)
    steps = np.unique(step_te)
    stream = []
    for t in steps:
        idx = np.where(step_te == t)[0]
        if idx.size > per_step:
            idx = idx[rng.choice(idx.size, per_step, replace=False)]
        if idx.size >= 8 and 0 < y_te[idx].sum() < idx.size:
            stream.append((F_te[idx], y_te[idx], int(t)))

    def run(gamma, tau0=0.5):
        aci = AdaptiveConformalSaliency(scorer=base, alpha=alpha, gamma=gamma, tau=tau0)
        traj = []
        for F, y, t in stream:
            err = aci.observe(F, y)
            traj.append(dict(step=t, miss=err, set_size=aci._sizes[-1], tau=aci.tau))
        n = len(traj)
        half = traj[n // 2:]
        return dict(gamma=gamma, n_steps=n,
                    mean_miss_all=float(np.mean([x["miss"] for x in traj])),
                    mean_miss_2nd_half=float(np.mean([x["miss"] for x in half])),
                    mean_set_size_2nd_half=float(np.mean([x["set_size"] for x in half])),
                    final_tau=aci.tau, trajectory=traj)

    return dict(alpha=alpha, base="within+cross closed (frozen)",
                base_auc=roc_auc(y_te, base.score(F_te)),
                tau_split_conformal=tau_split,
                adaptive_g05=run(0.05), adaptive_g10=run(0.10),
                fixed_tau05=run(0.0, tau0=0.5),
                fixed_split_conformal=run(0.0, tau0=tau_split))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", default="experiments/results/icdm_extra.json")
    a = ap.parse_args()
    files = [f for f in sorted(glob.glob(os.path.join(a.traces, "*.jsonl"))) if ".smoke." not in f]
    models = {}
    for f in files:
        nm = os.path.basename(f)[:-len(".jsonl")]
        print(f"[load] {nm}", flush=True)
        d = load_model_trace(f)
        if d:
            models[nm] = d
    pooled = pool_models(models)
    print("[A] minimal models ...", flush=True)
    mm = minimal_models(pooled)
    for k, v in mm.items():
        print(f"    {k:24s} AUC={v['auc']:.4f} AUPRC={v['auprc']:.4f} P@10={v['p_at_10']:.3f} "
              f"R@10={v['r_at_10']:.3f} ECE={v['ece']:.4f} params={v['params']}", flush=True)
    print("[B] conformal convergence ...", flush=True)
    cc = conformal_convergence(pooled)
    for key in ("adaptive_g05", "adaptive_g10", "fixed_tau05", "fixed_split_conformal"):
        r = cc[key]
        print(f"    {key:13s} g={r['gamma']}: miss(all)={r['mean_miss_all']:.3f} "
              f"miss(2nd-half)={r['mean_miss_2nd_half']:.3f} "
              f"set(2nd-half)={r['mean_set_size_2nd_half']:.3f} final_tau={r['final_tau']:.3f}", flush=True)
    out = dict(models=list(models.keys()), minimal_models=mm, conformal=cc)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
