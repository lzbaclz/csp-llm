"""E1 — coverage-driven budgeter: turn a target miss-rate alpha into the (per-layer)
KV budget, with a split-conformal guarantee. The headline of GuardKV
(experiments/DESIGN_v2_conformal_crosslayer.md).

Shows: (a) validity — realized miss-rate tracks the target alpha; (b) budget is an
OUTPUT — each alpha induces an emergent retention ratio; (c) per-layer calibration
auto-allocates budget (PyramidKV-style) and EQUALIZES miss across layers, where a
single global threshold or a fixed top-ratio leaves some layers far above alpha
(no guarantee). Splits are request-level: fit scorer / calibrate tau / test are
disjoint prompts.

    python experiments/run_coverage_budget.py --traces experiments/traces \
        --out experiments/results/coverage_budget.json
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
from xqp.budgeter import CoverageDrivenBudgeter, fixed_ratio_select

H = "h4"
COLS = (0, 1)        # within + cross (the calibrated minimal scorer)
CAP = 400_000


def three_way(rid, seed=0):
    """request-level 50/25/25 split: fit-scorer / calibrate-tau / test."""
    u = np.unique(rid); rng = np.random.default_rng(seed); rng.shuffle(u)
    a, b = int(0.5 * len(u)), int(0.75 * len(u))
    fit, cal, te = set(u[:a]), set(u[a:b]), set(u[b:])
    sel = lambda S: np.where(np.isin(rid, list(S)))[0]
    return sel(fit), sel(cal), sel(te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="experiments/traces")
    ap.add_argument("--alphas", default="0.05,0.10,0.15,0.20")
    ap.add_argument("--out", default="experiments/results/coverage_budget.json")
    ap.add_argument("--save-dir", default=None,
                    help="if set, persist the fitted scorer + budgeter (at --save-alpha) "
                         "as ckpts for the SEER GuardKV policy")
    ap.add_argument("--save-alpha", type=float, default=0.10)
    a = ap.parse_args()
    alphas = [float(x) for x in a.alphas.split(",")]

    files = [f for f in sorted(glob.glob(os.path.join(a.traces, "*.jsonl"))) if ".smoke." not in f]
    models = {os.path.basename(f)[:-6]: load_model_trace(f) for f in files}
    d = pool_models({k: v for k, v in models.items() if v})

    fit_i, cal_i, te_i = three_way(d["rid"])
    fit = subsample(fit_i, CAP); cal = subsample(cal_i, CAP); te = subsample(te_i, CAP)
    y = d["y"][H].astype(np.float32); lay = d["layer"]

    scorer = ClosedFormXQP.from_fit(_gather4(d["F"][fit], COLS), y[fit])
    p_te = np.asarray(scorer.score(_gather4(d["F"][te], COLS)))   # for fixed-ratio baseline
    print(f"pooled rows={d['F'].shape[0]:,} | fit={len(fit):,} cal={len(cal):,} test={len(te):,} "
          f"| layers={int(lay.max()+1)} | scorer=within+cross", flush=True)

    out = {"cols": "within+cross", "n_test": int(len(te)), "by_alpha": []}
    for al in alphas:
        bg = CoverageDrivenBudgeter.calibrate(scorer, d["F"][cal], y[cal], lay[cal],
                                              cols=COLS, alpha=al)
        if a.save_dir and abs(al - a.save_alpha) < 1e-9:
            os.makedirs(a.save_dir, exist_ok=True)
            sp = os.path.join(a.save_dir, "guardkv_scorer_h4.json")
            bp = os.path.join(a.save_dir, f"guardkv_budgeter_a{int(round(al*100)):02d}_h4.json")
            scorer.save(sp); bg.save(bp)
            print(f"  SAVED scorer->{sp}  budgeter(alpha={al})->{bp}", flush=True)
        ev = bg.evaluate(d["F"][te], y[te], lay[te], per_layer=True)
        ev_global = bg.evaluate(d["F"][te], y[te], lay[te], per_layer=False)
        # fixed top-ratio at the SAME emergent overall budget (Ada-KV/SnapKV style)
        keep_fixed = fixed_ratio_select(p_te, ev["emergent_budget"])
        sal = y[te] > 0.5
        miss_fixed = float((sal & ~keep_fixed).sum() / max(1, sal.sum()))
        # per-layer worst-case miss for fixed-ratio (single global ranking)
        fl_miss = []
        for l in np.unique(lay[te]):
            m = lay[te] == l; ms = sal & m
            if ms.sum() >= 8:
                fl_miss.append(float((ms & ~keep_fixed).sum() / ms.sum()))
        rec = dict(
            target_alpha=al,
            realized_miss=round(ev["realized_miss"], 4),
            emergent_budget=round(ev["emergent_budget"], 4),
            per_layer_miss_std=round(ev["per_layer_miss_std"], 4),
            global_tau_miss=round(ev_global["realized_miss"], 4),
            global_tau_miss_std=round(ev_global["per_layer_miss_std"], 4),
            fixed_ratio_miss_at_same_budget=round(miss_fixed, 4),
            fixed_ratio_worst_layer_miss=round(max(fl_miss), 4) if fl_miss else None,
            emergent_budget_per_layer={k: round(v["budget"], 3) for k, v in ev["per_layer"].items()},
        )
        out["by_alpha"].append(rec)
        print(f"  alpha={al:.2f}: realized_miss={rec['realized_miss']:.3f} "
              f"emergent_budget={rec['emergent_budget']:.3f} "
              f"per-layer miss std={rec['per_layer_miss_std']:.3f} | "
              f"fixed-ratio@same-budget miss={rec['fixed_ratio_miss_at_same_budget']:.3f} "
              f"(worst layer {rec['fixed_ratio_worst_layer_miss']})", flush=True)

    # per-layer budget profile at alpha=0.10 (is it PyramidKV-like?)
    a10 = next(r for r in out["by_alpha"] if abs(r["target_alpha"] - 0.10) < 1e-9)
    prof = a10["emergent_budget_per_layer"]
    ls = sorted(prof)
    print("\nemergent per-layer budget @alpha=0.10 (first/mid/last layers): "
          f"{prof[ls[0]]:.2f} / {prof[ls[len(ls)//2]]:.2f} / {prof[ls[-1]]:.2f}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
