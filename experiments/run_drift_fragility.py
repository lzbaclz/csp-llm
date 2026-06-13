"""E3 — fragility under drift: the WORST-CASE per-step missed-saliency tail.

DefensiveKV's point: importance shifts abruptly during *certain* intervals,
producing outlier steps where the retained cache captures far less than the
average suggests. We compare three thresholding policies at the same target
alpha=0.1 over the decode-step stream and report the TAIL (p90/p99/max) of the
per-step miss-rate, plus the fraction of steps that "blow out" (miss > 2*alpha):
  - fixed-global tau (one split-conformal threshold, the H2O/SnapKV-style static cut)
  - coverage-driven per-layer tau (E1; static but layer-aware)
  - adaptive-conformal (xqp.conformal; updates tau per step => absorbs drift)

A guarantee that holds on average but not at the tail is no guarantee; this is
where GuardKV's adaptive layer earns its place.

    python experiments/run_drift_fragility.py --traces experiments/traces
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from run_icdm_full import load_model_trace, pool_models, request_split, subsample
from xqp.predictor import ClosedFormXQP
from xqp.gated_predictor import _gather4
from xqp.budgeter import CoverageDrivenBudgeter, _conformal_tau
from xqp.conformal import AdaptiveConformalSaliency

H = "h4"; COLS = (0, 1); ALPHA = 0.10; CAP = 600_000


def tail(x):
    x = np.asarray(x, np.float64)
    return dict(mean=round(float(x.mean()), 4), p90=round(float(np.percentile(x, 90)), 4),
                p99=round(float(np.percentile(x, 99)), 4), max=round(float(x.max()), 4),
                frac_blowout=round(float((x > 2 * ALPHA).mean()), 4))


def per_step_miss(keep, y, step):
    sal = y > 0.5
    out = []
    for s in np.unique(step):
        m = step == s; ms = sal & m
        if ms.sum() >= 4:
            out.append((ms & ~keep).sum() / ms.sum())
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="experiments/traces")
    ap.add_argument("--out", default="experiments/results/drift_fragility.json")
    a = ap.parse_args()
    files = [f for f in sorted(glob.glob(os.path.join(a.traces, "*.jsonl"))) if ".smoke." not in f]
    d = pool_models({k: v for k, v in {os.path.basename(f)[:-6]: load_model_trace(f) for f in files}.items() if v})

    u = np.unique(d["rid"]); rng = np.random.default_rng(0); rng.shuffle(u)
    cut = int(0.5 * len(u))
    fit = np.where(np.isin(d["rid"], list(u[:cut])))[0]
    test = np.where(np.isin(d["rid"], list(u[cut:])))[0]
    fit = subsample(fit, CAP); test = subsample(test, CAP)
    y = d["y"][H].astype(np.float32); lay = d["layer"]; step = d["step"]

    scorer = ClosedFormXQP.from_fit(_gather4(d["F"][fit], COLS), y[fit])
    p_test = np.asarray(scorer.score(_gather4(d["F"][test], COLS)))
    print(f"test rows={len(test):,} layers={int(lay.max()+1)} steps={int(step[test].max()+1)}", flush=True)

    res = {}
    # (1) fixed-global tau: one split-conformal threshold calibrated on the FIT
    # split's salient scores (a clean held-out comparator -- NOT the test set).
    p_fit = np.asarray(scorer.score(_gather4(d["F"][fit], COLS)))
    tau_g = _conformal_tau(p_fit[y[fit] > 0.5], ALPHA)
    keep_fixed = p_test >= tau_g
    res["fixed_global"] = tail(per_step_miss(keep_fixed, y[test], step[test]))

    # (2) coverage-driven per-layer
    bg = CoverageDrivenBudgeter.calibrate(scorer, d["F"][fit], y[fit], lay[fit], cols=COLS, alpha=ALPHA)
    keep_cov, _ = bg.keep_mask(d["F"][test], lay[test], per_layer=True)
    res["coverage_per_layer"] = tail(per_step_miss(keep_cov, y[test], step[test]))

    # (3) adaptive-conformal over the decode-step stream (pooled per step)
    aci = AdaptiveConformalSaliency(scorer=type("S", (), {"score": staticmethod(
        lambda F: scorer.score(_gather4(F, COLS)))})(), alpha=ALPHA, gamma=0.1, tau=tau_g)
    miss_adapt = []
    for s in np.unique(step[test]):
        m = step[test] == s
        if (y[test][m] > 0.5).sum() >= 4:
            miss_adapt.append(aci.observe(d["F"][test][m], y[test][m]))
    res["adaptive_conformal"] = tail(np.asarray(miss_adapt))

    out = dict(target_alpha=ALPHA, policies=res)
    for k, v in res.items():
        print(f"  {k:20s} mean={v['mean']:.3f} p90={v['p90']:.3f} p99={v['p99']:.3f} "
              f"max={v['max']:.3f} frac(miss>2a)={v['frac_blowout']:.3f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
