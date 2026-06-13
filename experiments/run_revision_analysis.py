"""Revision analyses that harden the should-fix statistical attacks (A9/A11/A12/A27).

Runs on the EXISTING mooncake headline corpus (experiments/traces/*.jsonl, 43.9M
rows) -- CPU only, no GPU, no re-extraction. Reuses run_icdm_full's request-level
split + request-clustered bootstrap so every number is on the same footing as the
paper's headline table. Four blocks:

  A11  paired clustered-bootstrap dAUC + dAUPRC + TOST (margin 0.005) for the
       analysis-selected 2-view model vs GBDT. Converts the unsupported
       "statistically indistinguishable" into a defensible equivalence claim
       (AUC) and an honest ranking gap (AUPRC).

  A12  GBDT calibration is a CONFIG artifact, not a model-class law: compare
       2-view-closed vs GBDT{balanced, unbalanced, balanced+isotonic} vs Quest.

  A27  ECE is binning-robust: equal-width, equal-mass, and debiased ECE, each
       with a clustered bootstrap CI; Brier with CI.

  A9   uCMI "sizing certificate" is estimator-robust: sweep (n_bins x n_cond),
       a label-permutation null floor, and a clustered bootstrap CI -- showing
       within/cross earn a parameter and query(weak)/recency do not, at every
       binning. (The FAITHFUL-query uCMI is in run_quest_headline.py.)

    python experiments/run_revision_analysis.py --traces experiments/traces \
        --out experiments/results/revision_analysis.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

import run_icdm_full as R  # load_model_trace, pool_models, request_split, subsample, roc_auc, clustered_*
from xqp.features import FEATURE_NAMES
from xqp.predictor import ClosedFormXQP
from xqp.dm_metrics import average_precision, brier_score
from xqp.info_theory import conditional_mi_view, mutual_information

HEADLINE_H = "h4"
TRAIN_N, TEST_N, MI_N = 120_000, 150_000, 300_000
N_BOOT = 400
SEED = 0


# --------------------------------------------------------------------------- #
# calibration metrics (equal-width / equal-mass / debiased)
# --------------------------------------------------------------------------- #
def _ece_from_bins(y, p, edges):
    n = y.shape[0]
    ece = 0.0
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        m = (p > lo) & (p <= hi) if b > 0 else (p >= lo) & (p <= hi)
        c = int(m.sum())
        if c == 0:
            continue
        ece += (c / n) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(ece)


def ece_equal_width(y, p, n_bins=10):
    return _ece_from_bins(y, p, np.linspace(0.0, 1.0, n_bins + 1))


def ece_equal_mass(y, p, n_bins=10):
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = -1e-9, 1.0 + 1e-9
    edges = np.unique(edges)
    return _ece_from_bins(y, p, edges)


def ece_debiased(y, p, n_bins=10):
    """Equal-mass ECE minus per-bin sampling bias of |conf-acc| under perfect
    calibration (acc_b ~ Binomial): subtract E|acc_b - conf_b| ~ sqrt(conf(1-conf)/n_b).
    Clipped at 0. A documented, conservative debiasing of the L1 ECE."""
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = -1e-9, 1.0 + 1e-9
    edges = np.unique(edges)
    n = y.shape[0]
    val = 0.0
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        m = (p > lo) & (p <= hi) if b > 0 else (p >= lo) & (p <= hi)
        c = int(m.sum())
        if c < 2:
            continue
        conf = float(p[m].mean()); acc = float(y[m].mean())
        bias = np.sqrt(max(conf * (1 - conf), 0.0) / c)
        val += (c / n) * max(0.0, abs(conf - acc) - bias)
    return float(val)


def _clustered_metric_ci(metric_fn, y, p, groups, n_boot=N_BOOT, seed=SEED):
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    obs = metric_fn(y, p)
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_g[g] for g in samp])
        v = metric_fn(y[rows], p[rows])
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    return dict(value=float(obs), lo=float(np.percentile(vals, 2.5)),
                hi=float(np.percentile(vals, 97.5)))


# --------------------------------------------------------------------------- #
# GBDT variants
# --------------------------------------------------------------------------- #
def fit_lgbm(F, y, balanced=True, seed=SEED):
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(max_depth=3, n_estimators=150,
                             class_weight=("balanced" if balanced else None),
                             verbosity=-1, random_state=seed).fit(F, y.astype(int))
    return lambda X: clf.predict_proba(np.asarray(X, np.float32))[:, 1].astype(np.float32)


def fit_lgbm_isotonic(F, y, seed=SEED):
    """Balanced GBDT + isotonic recalibration on a held-out slice of train."""
    from sklearn.isotonic import IsotonicRegression
    rng = np.random.default_rng(seed)
    n = F.shape[0]; perm = rng.permutation(n); cut = int(0.8 * n)
    fit_i, cal_i = perm[:cut], perm[cut:]
    base = fit_lgbm(F[fit_i], y[fit_i], balanced=True, seed=seed)
    iso = IsotonicRegression(out_of_bounds="clip").fit(base(F[cal_i]), y[cal_i])
    return lambda X: iso.predict(base(X)).astype(np.float32)


# --------------------------------------------------------------------------- #
def block_A11_A12_A27(d, seed=SEED):
    tr_idx, te_idx = R.request_split(d["rid"], seed=seed)
    tr = R.subsample(tr_idx, TRAIN_N, seed); te = R.subsample(te_idx, TEST_N, seed)
    Ftr, ytr = d["F"][tr], d["y"][HEADLINE_H][tr].astype(np.float32)
    Fte, yte = d["F"][te], d["y"][HEADLINE_H][te].astype(np.float32)
    grp = d["rid"][te]
    mask2 = np.array([1, 1, 0, 0], np.float32)              # within+cross
    cf2 = ClosedFormXQP.from_fit(Ftr * mask2, ytr)
    scorers = {
        "2view-closed": lambda X: cf2.score(X * mask2),
        "GBDT-balanced": fit_lgbm(Ftr, ytr, balanced=True, seed=seed),
        "GBDT-unbalanced": fit_lgbm(Ftr, ytr, balanced=False, seed=seed),
        "GBDT-bal+isotonic": fit_lgbm_isotonic(Ftr, ytr, seed=seed),
        "Quest(raw)": lambda X: X[:, 2],
    }
    S = {k: np.asarray(f(Fte), np.float32) for k, f in scorers.items()}

    # ---- A11: paired dAUC / dAUPRC + TOST(margin 0.005), 2view - GBDT-balanced
    margin = 0.005
    a11 = {}
    for metric_name, mfn in [("auc", R.roc_auc), ("auprc", average_precision)]:
        t = R.clustered_paired_test(mfn, yte, S["2view-closed"], S["GBDT-balanced"],
                                    grp, n_boot=N_BOOT, seed=seed)
        # 90% CI for TOST = inner [5,95]; equivalence iff CI subset [-margin, margin]
        lo90, hi90 = t.get("lo"), t.get("hi")
        a11[metric_name] = dict(delta_2view_minus_gbdt=t["delta"], ci95_lo=lo90, ci95_hi=hi90,
                                tost_margin=margin,
                                tost_equivalent=bool(lo90 is not None and lo90 > -margin and hi90 < margin),
                                p_value=t.get("p_value"))

    # ---- A12 + A27: calibration table (ECE 3 binnings + Brier, all with CIs)
    calib = {}
    for name, s in S.items():
        calib[name] = dict(
            auc=R.roc_auc(yte, s), auprc=average_precision(yte, s),
            ece_equal_width=_clustered_metric_ci(ece_equal_width, yte, s, grp, seed=seed),
            ece_equal_mass=_clustered_metric_ci(ece_equal_mass, yte, s, grp, seed=seed),
            ece_debiased=_clustered_metric_ci(ece_debiased, yte, s, grp, seed=seed),
            brier=_clustered_metric_ci(brier_score, yte, s, grp, seed=seed),
        )
    # calibration multiplier (2view vs GBDT-balanced) at each binning
    mult = {}
    for bk in ["ece_equal_width", "ece_equal_mass", "ece_debiased"]:
        a = calib["GBDT-balanced"][bk]["value"]; b = calib["2view-closed"][bk]["value"]
        mult[bk] = float(a / b) if b > 0 else float("inf")
    return dict(n_test=int(len(te)), n_test_requests=int(len(np.unique(grp))),
                pos_rate_test=float(yte.mean()), A11=a11, calibration=calib,
                calib_multiplier_gbdt_over_2view=mult)


def block_A9_ucmi(d, seed=SEED):
    """uCMI robustness for the 4 views: binning sweep + permutation null + clustered CI."""
    mi = R.subsample(np.arange(d["F"].shape[0]), MI_N, seed)
    F = d["F"][mi]; y = d["y"][HEADLINE_H][mi].astype(np.int64); rid = d["rid"][mi]
    names = list(FEATURE_NAMES)
    rng = np.random.default_rng(seed)

    # (a) binning sweep: point uCMI at each (n_bins, n_cond)
    sweep = {}
    for nb in (8, 12, 16):
        for nc in (3, 5, 8):
            key = f"nbins{nb}_ncond{nc}"
            sweep[key] = {names[i]: conditional_mi_view(F, y, i, n_bins=nb, n_cond=nc)
                          for i in range(4)}

    # (b) permutation-null floor + (c) clustered bootstrap CI at headline (12,3)
    NB, NC = 12, 3
    point = {names[i]: conditional_mi_view(F, y, i, n_bins=NB, n_cond=NC) for i in range(4)}
    relevance = {names[i]: mutual_information(F[:, i], y, n_bins=16) for i in range(4)}

    n_perm, n_bc = 150, 200
    null = {names[i]: [] for i in range(4)}
    for _ in range(n_perm):
        yp = rng.permutation(y)
        for i in range(4):
            null[names[i]].append(conditional_mi_view(F, yp, i, n_bins=NB, n_cond=NC))
    null_stats = {k: dict(mean=float(np.mean(v)), p95=float(np.percentile(v, 95)),
                          p99=float(np.percentile(v, 99))) for k, v in null.items()}

    uniq = np.unique(rid); idx_by_g = {g: np.where(rid == g)[0] for g in uniq}
    boot = {names[i]: [] for i in range(4)}
    for _ in range(n_bc):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_g[g] for g in samp])
        Fb, yb = F[rows], y[rows]
        for i in range(4):
            boot[names[i]].append(conditional_mi_view(Fb, yb, i, n_bins=NB, n_cond=NC))
    ci = {k: dict(lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))
          for k, v in boot.items()}

    return dict(headline_binning=dict(n_bins=NB, n_cond=NC), n_mi_rows=int(len(mi)),
                relevance_mi=relevance, ucmi_point=point, ucmi_sweep=sweep,
                ucmi_perm_null=null_stats, ucmi_clustered_ci=ci,
                n_perm=n_perm, n_boot=n_bc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="experiments/traces")
    ap.add_argument("--glob", default="*.jsonl")
    ap.add_argument("--out", default="experiments/results/revision_analysis.json")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.traces, a.glob)))
    files = [f for f in files if ".smoke." not in f]
    models = {}
    for f in files:
        name = os.path.basename(f)[:-len(".jsonl")]
        print(f"[load] {name} ...", flush=True)
        dd = R.load_model_trace(f)
        if dd is not None:
            models[name] = dd
            print(f"   {dd['F'].shape[0]:,} rows, {dd['n_requests']} requests", flush=True)
    pooled = R.pool_models(models)
    print(f"[pooled] {pooled['F'].shape[0]:,} rows, {pooled['n_requests']} requests", flush=True)

    out = dict(corpus="mooncake (headline)", files=list(models.keys()),
               config=dict(TRAIN_N=TRAIN_N, TEST_N=TEST_N, MI_N=MI_N, N_BOOT=N_BOOT, seed=SEED))
    print("[A11/A12/A27] paired TOST + calibration table ...", flush=True)
    out["A11_A12_A27"] = block_A11_A12_A27(pooled)
    print("[A9] uCMI robustness ...", flush=True)
    out["A9_ucmi"] = block_A9_ucmi(pooled)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)

    # ---- console summary
    a11 = out["A11_A12_A27"]["A11"]
    print("\n=== A11 paired (2view - GBDT) ===")
    for m in ("auc", "auprc"):
        r = a11[m]
        print(f"  d{m.upper():<6} {r['delta_2view_minus_gbdt']:+.4f} "
              f"[{r['ci95_lo']:+.4f},{r['ci95_hi']:+.4f}]  TOST(0.005)="
              f"{'PASS' if r['tost_equivalent'] else 'fail'}")
    print("=== calibration (ECE equal-width / Brier, value[lo,hi]) ===")
    for k, v in out["A11_A12_A27"]["calibration"].items():
        ew = v["ece_equal_width"]; br = v["brier"]
        print(f"  {k:<20} AUC {v['auc']:.3f} | ECE-ew {ew['value']:.4f} "
              f"[{ew['lo']:.4f},{ew['hi']:.4f}] | Brier {br['value']:.4f}")
    print("  multiplier GBDT/2view:", out["A11_A12_A27"]["calib_multiplier_gbdt_over_2view"])
    print("=== A9 uCMI (headline 12,3): point / null-p95 / CI ===")
    u = out["A9_ucmi"]
    for k in u["ucmi_point"]:
        print(f"  {k:<10} uCMI {u['ucmi_point'][k]:.4f} | null-p95 {u['ucmi_perm_null'][k]['p95']:.4f} "
              f"| CI [{u['ucmi_clustered_ci'][k]['lo']:.4f},{u['ucmi_clustered_ci'][k]['hi']:.4f}]")
    print(f"\nWROTE {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
