"""Comprehensive ICDM analysis on REAL attention traces — the single driver that
produces every number in paper_icdm.

Hardening over run_icdm_analysis.py / run_icdm_baselines.py:
  * REQUEST-LEVEL splits (group by request_id) instead of row-level random
    splits — rows from the same prompt are highly correlated, so a row-level
    split leaks and inflates AUC. All within-model train/test and the transfer
    matrix hold out whole requests.
  * Per-model AND pooled ("across model families") metrics.
  * Real (Adam-trained) TinyMLP in the Pareto/headline, not random init.
  * Learned baselines trained on a capped subsample (LightGBM/sklearn) for
    tractability; all methods share the same train/test sample for fairness.
  * Bootstrap CIs + paired-bootstrap significance vs XQP-closed on the pooled
    headline table.
  * Cross-architecture transfer matrix with per-model standardization.
  * Concept drift (temporal step-split): static vs online vs refit-oracle,
    plus adaptive-conformal vs fixed-threshold coverage.
  * Ablations: leave-one-view-out, INT8-quantized query proxy.

Usage:
    python experiments/run_icdm_full.py --traces experiments/traces \
        --out experiments/results/icdm_full.json
"""
from __future__ import annotations

import argparse
import glob
import re
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scipy.stats import rankdata

from xqp.features import FEATURE_NAMES
from xqp.eval import topk_recall


def roc_auc(y_true, y_score) -> float:
    """Vectorized Mann-Whitney AUC (average ranks via scipy.rankdata).

    Equivalent to xqp.eval.roc_auc but without its Python tie-handling loop,
    which is O(N) per call and dominates the bootstrap when baselines have
    binary scores (large tie runs). Used everywhere in this driver, including
    inside the bootstrap, for ~10x speedup.
    """
    y = np.asarray(y_true, np.float64).reshape(-1)
    s = np.asarray(y_score, np.float64).reshape(-1)
    n_pos = float(y.sum()); n_neg = float(y.size) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)  # average ranks, ties handled
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
from xqp.dm_metrics import (
    average_precision, precision_at_k, recall_at_k,
    expected_calibration_error, brier_score, reliability_curve,
)
from xqp.info_theory import redundancy_report, mutual_information
from xqp.predictor import ClosedFormXQP, PairwiseXQP, TinyMLPXQP
from xqp.baselines import single_signal_baselines, all_learned_baselines
from xqp.stats import bootstrap_ci, paired_bootstrap_test
from xqp.sota_iterations.iter5_online import OnlineXQP
from xqp.conformal import run_conformal_stream

HORIZONS = ("h1", "h4", "h16", "h64")
HEADLINE_H = "h4"
TRAIN_N = 120_000     # cap for fitting learned baselines (fair: all methods see same)
TEST_N = 150_000      # eval + bootstrap sample (from held-out requests)
MI_N = 400_000        # sample for MI / redundancy / per-view
N_BOOT = 250
SEED = 0


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_model_trace(path: str) -> dict | None:
    """Stream a JSONL trace into compact typed arrays (low memory).

    NOTE on request ids: the trace harness writes request_id="p0" for every
    prompt (each prompt is extracted in its own call where the local index is
    always 0). We therefore recover true per-prompt boundaries from the
    deterministic emission order (step-major, layer-minor, block-minor): the
    first row of each prompt is uniquely (step==0, layer==0, block_idx==0).
    A request id = cumulative count of those reset markers. This is exactly
    equivalent to a correct per-prompt id and needs no GPU re-run.
    """
    layer, step, blk = [], [], []
    fw, fc, fq, fp = [], [], [], []
    ys = {h: [] for h in HORIZONS}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            layer.append(r["layer"]); step.append(r["step"]); blk.append(r["block_idx"])
            fw.append(r["f_within"]); fc.append(r["f_cross"])
            fq.append(r["f_query"]); fp.append(r["f_pos"])
            for h in HORIZONS:
                ys[h].append(r[f"y_{h}"])
    if not layer:
        return None
    layer = np.asarray(layer, np.int16); step = np.asarray(step, np.int16)
    blk = np.asarray(blk, np.int32)
    boundary = (step == 0) & (layer == 0) & (blk == 0)
    rid = (np.cumsum(boundary) - 1).astype(np.int32)
    F = np.stack([np.asarray(fw, np.float32), np.asarray(fc, np.float32),
                  np.asarray(fq, np.float32), np.asarray(fp, np.float32)], axis=1)
    y = {h: np.asarray(ys[h], np.int8) for h in HORIZONS}
    # Drop non-finite feature rows (fp16 attention overflow occurs in a few
    # last-layer rows of some models; we discard rather than impute). Boundaries
    # are computed on the full sequence first, so request recovery is unaffected.
    finite = np.isfinite(F).all(axis=1)
    n_drop = int((~finite).sum())
    if n_drop:
        F = F[finite]; rid = rid[finite]; layer = layer[finite]; step = step[finite]
        y = {h: v[finite] for h, v in y.items()}
    return dict(
        rid=rid, layer=layer, step=step, F=F, y=y,
        n_requests=int(boundary.sum()), n_dropped_nonfinite=n_drop,
    )


def pool_models(models: dict) -> dict:
    """Concatenate per-model compact dicts; namespace request ids across models."""
    rids, layers, steps, Fs = [], [], [], []
    ys = {h: [] for h in HORIZONS}
    offset = 0
    n_drop = 0
    for name, d in models.items():
        rids.append(d["rid"].astype(np.int64) + offset)
        offset += d["n_requests"]
        n_drop += int(d.get("n_dropped_nonfinite", 0))
        layers.append(d["layer"]); steps.append(d["step"]); Fs.append(d["F"])
        for h in HORIZONS:
            ys[h].append(d["y"][h])
    return dict(
        rid=np.concatenate(rids), layer=np.concatenate(layers),
        step=np.concatenate(steps), F=np.concatenate(Fs),
        y={h: np.concatenate(ys[h]) for h in HORIZONS},
        n_requests=int(offset), n_dropped_nonfinite=int(n_drop),
    )


# --------------------------------------------------------------------------- #
# splits / sampling
# --------------------------------------------------------------------------- #
def request_split(rid: np.ndarray, frac: float = 0.25, seed: int = SEED):
    uniq = np.unique(rid)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_test = max(1, int(frac * len(uniq)))
    test_reqs = set(perm[:n_test].tolist())
    is_test = np.isin(rid, list(test_reqs))
    return np.where(~is_test)[0], np.where(is_test)[0]


def subsample(idx: np.ndarray, n: int, seed: int = SEED) -> np.ndarray:
    if len(idx) <= n:
        return idx
    rng = np.random.default_rng(seed)
    return idx[rng.choice(len(idx), size=n, replace=False)]


def clustered_bootstrap_ci(metric_fn, y, s, groups, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI that resamples whole REQUESTS (groups), not rows, so the
    within-request correlation is respected (row bootstrap understates variance
    on this clustered stream). Returns {mean, lo, hi, n_groups}."""
    y = np.asarray(y); s = np.asarray(s); groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_g[g] for g in samp])
        yy = y[rows]
        if yy.sum() == 0 or yy.sum() == yy.shape[0]:
            continue
        v = metric_fn(yy, s[rows])
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals, np.float64)
    if vals.size == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n_groups=int(len(uniq)))
    return dict(mean=float(vals.mean()), lo=float(np.percentile(vals, 2.5)),
                hi=float(np.percentile(vals, 97.5)), n_groups=int(len(uniq)))


def clustered_paired_test(metric_fn, y, sa, sb, groups, n_boot=N_BOOT, seed=SEED):
    """Paired test of metric(A)-metric(B) resampling whole requests."""
    y = np.asarray(y); sa = np.asarray(sa); sb = np.asarray(sb); groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    obs = metric_fn(y, sa) - metric_fn(y, sb)
    deltas = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_g[g] for g in samp])
        yy = y[rows]
        if yy.sum() == 0 or yy.sum() == yy.shape[0]:
            continue
        da = metric_fn(yy, sa[rows]); db = metric_fn(yy, sb[rows])
        if np.isfinite(da) and np.isfinite(db):
            deltas.append(da - db)
    deltas = np.asarray(deltas, np.float64)
    if deltas.size == 0:
        return dict(delta=float(obs), p_value=float("nan"))
    p = 2.0 * (np.mean(deltas <= 0) if obs >= 0 else np.mean(deltas >= 0))
    return dict(delta=float(obs), lo=float(np.percentile(deltas, 2.5)),
                hi=float(np.percentile(deltas, 97.5)), p_value=float(min(1.0, p)))


# --------------------------------------------------------------------------- #
# analyses
# --------------------------------------------------------------------------- #
def dataset_summary(d: dict) -> dict:
    F = d["F"]
    out = dict(n_rows=int(F.shape[0]), n_requests=int(d["n_requests"]),
               n_dropped_nonfinite=int(d.get("n_dropped_nonfinite", 0)),
               n_layers=int(d["layer"].max() + 1), max_step=int(d["step"].max()),
               pos_rate={h: float(d["y"][h].mean()) for h in HORIZONS},
               feature_mean={FEATURE_NAMES[i]: float(F[:, i].mean()) for i in range(4)},
               feature_std={FEATURE_NAMES[i]: float(F[:, i].std()) for i in range(4)})
    return out


def per_view_auc(F, y) -> dict:
    return {FEATURE_NAMES[i]: dict(auc=roc_auc(y, F[:, i]),
                                   auprc=average_precision(y, F[:, i]),
                                   relevance_mi=mutual_information(F[:, i], y))
            for i in range(4)}


def fit_all_methods(Ftr, ytr, seed=SEED):
    """Return ordered list of (name, scorer-callable, params)."""
    cf = ClosedFormXQP.from_fit(Ftr, ytr)
    mask2 = np.array([1, 1, 0, 0], np.float32)              # within+cross only
    cf2 = ClosedFormXQP.from_fit(Ftr * mask2, ytr)
    pw = PairwiseXQP.from_fit(Ftr, ytr)
    mlp = TinyMLPXQP.from_fit(Ftr, ytr, epochs=300, seed=seed)
    methods = [
        ("XQP-closed", lambda F: cf.score(F), 4),
        ("within+cross(2)", lambda F: cf2.score(F * mask2), 3),
        ("XQP-pairwise", lambda F: pw.score(F), 15),
        ("XQP-MLP", lambda F: mlp.score(F), mlp.n_params()),
    ]
    for b in single_signal_baselines():
        methods.append((b.name, (lambda bb: (lambda F: bb.score(F)))(b), 0))
    for b in all_learned_baselines(Ftr, ytr, seed=seed):
        p = b.meta.get("params") or b.meta.get("leaf_nodes") or 0
        methods.append((b.name, (lambda bb: (lambda F: bb.score(F)))(b), p))
    return methods, cf, pw, mlp


def headline_table(d: dict, horizon=HEADLINE_H, with_ci=True, seed=SEED) -> dict:
    tr_idx, te_idx = request_split(d["rid"], seed=seed)
    tr = subsample(tr_idx, TRAIN_N, seed)
    te = subsample(te_idx, TEST_N, seed)
    Ftr, ytr = d["F"][tr], d["y"][horizon][tr].astype(np.float32)
    Fte, yte = d["F"][te], d["y"][horizon][te].astype(np.float32)
    te_grp = d["rid"][te]                          # request id per test row (for clustering)
    methods, cf, pw, mlp = fit_all_methods(Ftr, ytr, seed=seed)
    ref = cf.score(Fte)
    rows = []
    for name, scorer, params in methods:
        s = np.asarray(scorer(Fte), np.float32)
        row = dict(method=name, params=int(params),
                   auc=roc_auc(yte, s), auprc=average_precision(yte, s),
                   p_at_10=precision_at_k(yte, s, 0.10),
                   r_at_10=recall_at_k(yte, s, 0.10),
                   ece=expected_calibration_error(yte, s),
                   brier=brier_score(yte, s))
        if with_ci:
            # request-CLUSTERED bootstrap (resamples whole prompts): the honest
            # CI on this correlated stream — far wider than a row bootstrap.
            aci = clustered_bootstrap_ci(roc_auc, yte, s, te_grp, n_boot=N_BOOT, seed=seed)
            api = clustered_bootstrap_ci(average_precision, yte, s, te_grp, n_boot=N_BOOT, seed=seed)
            row.update(auc_lo=aci["lo"], auc_hi=aci["hi"], n_test_requests=aci["n_groups"],
                       auprc_lo=api["lo"], auprc_hi=api["hi"])
            if name != "XQP-closed":
                t = clustered_paired_test(roc_auc, yte, ref, s, te_grp, n_boot=N_BOOT, seed=seed)
                row.update(auc_delta_vs_closed=t["delta"], p_vs_closed=t["p_value"])
        rows.append(row)
    gap = clustered_paired_test(roc_auc, yte, pw.score(Fte), cf.score(Fte),
                                te_grp, n_boot=N_BOOT, seed=seed)
    return dict(horizon=horizon, n_train=int(len(tr)), n_test=int(len(te)),
                n_test_requests=int(len(np.unique(te_grp))),
                pos_rate_test=float(yte.mean()), table=rows,
                interaction_gap=dict(delta_auc=gap["delta"], lo=gap.get("lo"),
                                     hi=gap.get("hi"), p_value=gap["p_value"]))


def auc_vs_horizon(d: dict, seed=SEED) -> dict:
    """e1: closed-form + best-view AUC per horizon (request split)."""
    tr_idx, te_idx = request_split(d["rid"], seed=seed)
    tr = subsample(tr_idx, TRAIN_N, seed); te = subsample(te_idx, TEST_N, seed)
    out = {}
    for h in HORIZONS:
        ytr = d["y"][h][tr].astype(np.float32); yte = d["y"][h][te].astype(np.float32)
        cf = ClosedFormXQP.from_fit(d["F"][tr], ytr)
        s = cf.score(d["F"][te])
        pv = per_view_auc(d["F"][te], yte)
        out[h] = dict(closed_auc=roc_auc(yte, s),
                      closed_auprc=average_precision(yte, s),
                      closed_recall_at_10=topk_recall(yte, s, 0.10),
                      best_single_view=max(pv, key=lambda k: pv[k]["auc"]),
                      best_single_auc=max(v["auc"] for v in pv.values()),
                      pos_rate=float(yte.mean()))
    return out


def pareto(d: dict, horizon=HEADLINE_H, seed=SEED) -> dict:
    tr_idx, te_idx = request_split(d["rid"], seed=seed)
    tr = subsample(tr_idx, TRAIN_N, seed); te = subsample(te_idx, TEST_N, seed)
    Ftr, ytr = d["F"][tr], d["y"][horizon][tr].astype(np.float32)
    Fte, yte = d["F"][te], d["y"][horizon][te].astype(np.float32)
    # axis 1: number of views (greedy by single-view AUC on train)
    order = sorted(range(4), key=lambda i: -roc_auc(ytr, Ftr[:, i]))
    by_nviews = []
    for k in range(1, 5):
        active = order[:k]
        mask = np.zeros(4, np.float32); mask[active] = 1.0
        cf = ClosedFormXQP.from_fit(Ftr * mask, ytr)
        by_nviews.append(dict(n_views=k, views=[FEATURE_NAMES[i] for i in active],
                              auc=roc_auc(yte, cf.score(Fte * mask))))
    # axis 2: model complexity
    cf = ClosedFormXQP.from_fit(Ftr, ytr)
    pw = PairwiseXQP.from_fit(Ftr, ytr)
    mlp = TinyMLPXQP.from_fit(Ftr, ytr, epochs=300, seed=seed)
    by_model = [
        dict(model="closed", params=4, auc=roc_auc(yte, cf.score(Fte))),
        dict(model="pairwise", params=15, auc=roc_auc(yte, pw.score(Fte))),
        dict(model="tinymlp", params=int(mlp.n_params()), auc=roc_auc(yte, mlp.score(Fte))),
    ]
    return dict(by_n_views=by_nviews, by_model=by_model)


def view_ablation(d: dict, horizon=HEADLINE_H, seed=SEED) -> dict:
    """Leave-one-view-out: drop in AUC when each view is removed."""
    tr_idx, te_idx = request_split(d["rid"], seed=seed)
    tr = subsample(tr_idx, TRAIN_N, seed); te = subsample(te_idx, TEST_N, seed)
    Ftr, ytr = d["F"][tr], d["y"][horizon][tr].astype(np.float32)
    Fte, yte = d["F"][te], d["y"][horizon][te].astype(np.float32)
    full = roc_auc(yte, ClosedFormXQP.from_fit(Ftr, ytr).score(Fte))
    out = {"full_auc": full, "drop": {}}
    for i in range(4):
        mask = np.ones(4, np.float32); mask[i] = 0.0
        auc = roc_auc(yte, ClosedFormXQP.from_fit(Ftr * mask, ytr).score(Fte * mask))
        out["drop"][FEATURE_NAMES[i]] = dict(auc_without=auc, auc_drop=full - auc)
    return out


def quant_query_ablation(d: dict, horizon=HEADLINE_H, seed=SEED) -> dict:
    """INT8-quantize the query view (256 levels over [0,1]) — cost-saving proxy."""
    tr_idx, te_idx = request_split(d["rid"], seed=seed)
    tr = subsample(tr_idx, TRAIN_N, seed); te = subsample(te_idx, TEST_N, seed)
    ytr = d["y"][horizon][tr].astype(np.float32); yte = d["y"][horizon][te].astype(np.float32)
    Ftr, Fte = d["F"][tr].copy(), d["F"][te].copy()
    full = roc_auc(yte, ClosedFormXQP.from_fit(Ftr, ytr).score(Fte))
    q = lambda F: np.round(np.clip(F, 0, 1) * 255) / 255.0
    Ftr_q, Fte_q = Ftr.copy(), Fte.copy()
    Ftr_q[:, 2] = q(Ftr[:, 2]); Fte_q[:, 2] = q(Fte[:, 2])
    auc_q = roc_auc(yte, ClosedFormXQP.from_fit(Ftr_q, ytr).score(Fte_q))
    return dict(fp32_auc=full, int8_query_auc=auc_q, auc_drop=full - auc_q)


def calibration(d: dict, horizon=HEADLINE_H, seed=SEED) -> dict:
    tr_idx, te_idx = request_split(d["rid"], seed=seed)
    tr = subsample(tr_idx, TRAIN_N, seed); te = subsample(te_idx, TEST_N, seed)
    Ftr, ytr = d["F"][tr], d["y"][horizon][tr].astype(np.float32)
    Fte, yte = d["F"][te], d["y"][horizon][te].astype(np.float32)
    from xqp.baselines import fit_gbdt
    cf = ClosedFormXQP.from_fit(Ftr, ytr)
    gbdt = fit_gbdt(Ftr, ytr, seed=seed)   # LightGBM only (avoid refitting MLP/SGD)
    out = {}
    for name, s in [("XQP-closed", cf.score(Fte)), ("GBDT", gbdt.score(Fte)),
                    ("Quest(raw)", Fte[:, 2])]:
        out[name] = dict(ece=expected_calibration_error(yte, s),
                         brier=brier_score(yte, s),
                         reliability=reliability_curve(yte, s, n_bins=10))
    return out


def drift_study(d: dict, horizon=HEADLINE_H, seed=SEED) -> dict:
    """Temporal step-split: train early decode steps, test late ones."""
    step = d["step"]; order = np.argsort(step, kind="stable")
    F = d["F"][order]; y = d["y"][horizon][order].astype(np.float32); st = step[order]
    n = F.shape[0]
    # cap for speed but keep temporal order
    if n > 1_500_000:
        keep = np.linspace(0, n - 1, 1_500_000).astype(int)
        F, y, st = F[keep], y[keep], st[keep]
        n = F.shape[0]
    a, b = int(0.6 * n), int(0.8 * n)
    Ftr, ytr = F[:a], y[:a]
    Fmid, ymid = F[a:b], y[a:b]
    Fte, yte = F[b:], y[b:]
    res = dict(train_steps=[int(st[0]), int(st[a - 1])],
               test_steps=[int(st[b]), int(st[-1])], n_test=int(Fte.shape[0]))
    if yte.sum() == 0 or yte.sum() == yte.shape[0]:
        res["note"] = "degenerate late labels"; return res
    static = ClosedFormXQP.from_fit(Ftr, ytr)
    res["auc_static"] = roc_auc(yte, static.score(Fte))
    online = OnlineXQP(predictor=ClosedFormXQP.from_fit(Ftr, ytr),
                       update_every=8, buffer_size=512, learning_rate=0.3)
    for i in range(0, Fmid.shape[0], 64):
        online.observe(Fmid[i:i + 64], ymid[i:i + 64])
    res["auc_online"] = roc_auc(yte, online.score(Fte))
    res["online_gain"] = res["auc_online"] - res["auc_static"]
    res["auc_refit_oracle"] = roc_auc(yte, ClosedFormXQP.from_fit(
        np.concatenate([Ftr, Fmid]), np.concatenate([ytr, ymid])).score(Fte))
    # conformal coverage on the late stream (group by step)
    base = static
    late_steps = np.unique(st[b:])
    stream = []
    for s_ in late_steps:
        m = (st[b:] == s_)
        if m.sum() >= 4:
            stream.append((Fte[m], yte[m]))
    if len(stream) >= 8:
        adapt = run_conformal_stream(base, stream, alpha=0.10, gamma=0.05, adaptive=True)
        fixed = run_conformal_stream(base, stream, alpha=0.10, gamma=0.0, adaptive=False, tau0=0.5)
        res["conformal"] = dict(
            target_alpha=0.10, n_steps=len(stream),
            adaptive_miss=adapt["realized_miss_rate"], adaptive_set_size=adapt["avg_set_size"],
            fixed_miss=fixed["realized_miss_rate"], fixed_set_size=fixed["avg_set_size"])
    return res


def transfer_matrix(models: dict, horizon=HEADLINE_H, standardize=True, seed=SEED) -> dict:
    """Train closed-form on model A (held-in requests), test AUC on model B
    (held-out requests). Per-model standardization makes weights transferable."""
    names = list(models.keys())
    # precompute per-model standardizer + train/test samples
    prep = {}
    for nm, d in models.items():
        tr_idx, te_idx = request_split(d["rid"], seed=seed)
        tr = subsample(tr_idx, TRAIN_N, seed); te = subsample(te_idx, TEST_N, seed)
        Ftr, Fte = d["F"][tr].astype(np.float32), d["F"][te].astype(np.float32)
        if standardize:
            mu = Ftr.mean(0); sd = Ftr.std(0) + 1e-6
            Ftr = (Ftr - mu) / sd; Fte = (Fte - mu) / sd
        prep[nm] = dict(Ftr=Ftr, ytr=d["y"][horizon][tr].astype(np.float32),
                        Fte=Fte, yte=d["y"][horizon][te].astype(np.float32))
    fitted = {nm: ClosedFormXQP.from_fit(prep[nm]["Ftr"], prep[nm]["ytr"]) for nm in names}
    mat = {}
    for a in names:
        mat[a] = {}
        for b in names:
            mat[a][b] = roc_auc(prep[b]["yte"], fitted[a].score(prep[b]["Fte"]))
    diag = np.mean([mat[a][a] for a in names])
    off = np.mean([mat[a][b] for a in names for b in names if a != b])
    return dict(standardize=standardize, matrix=mat,
                mean_within=float(diag), mean_cross=float(off),
                mean_transfer_drop=float(diag - off))


# --------------------------------------------------------------------------- #
def main(argv=None):
    global TRAIN_N, TEST_N
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", default="experiments/results/icdm_full.json")
    ap.add_argument("--glob", default="*.jsonl",
                    help="filename glob within --traces (e.g. '*.mooncake.jsonl' "
                         "to analyze one workload of the expanded corpus)")
    ap.add_argument("--train-n", type=int, default=TRAIN_N,
                    help="row cap for fitting (raise to realize the wider-corpus CI)")
    ap.add_argument("--test-n", type=int, default=TEST_N,
                    help="row cap for eval/bootstrap (raise so all held-out requests appear)")
    a = ap.parse_args(argv)
    TRAIN_N, TEST_N = a.train_n, a.test_n

    t0 = time.time()
    files = sorted(glob.glob(os.path.join(a.traces, a.glob)))
    files = [f for f in files if ".smoke." not in f]
    models = {}
    for f in files:
        name = os.path.basename(f)[:-len(".jsonl")]
        print(f"[load] {name} ...", flush=True)
        d = load_model_trace(f)
        if d is not None:
            models[name] = d
            print(f"       {d['F'].shape[0]:,} rows, {d['n_requests']} requests, "
                  f"{int(d['layer'].max()+1)} layers, pos_rate(h4)={d['y']['h4'].mean():.3f}",
                  flush=True)
    if not models:
        print("no traces found", file=sys.stderr); return 1

    # Each file is <model>.<workload>.jsonl. Split on the LAST '.': model names
    # keep their version dots (Qwen2.5, v0.3) because a real workload suffix
    # always starts with a lowercase letter (mooncake / sharegpt / longbench /
    # longbench_multifieldqa_zh). Legacy un-suffixed files -> workload "default".
    def split_mw(nm):
        m, dot, w = nm.rpartition(".")
        if dot and re.match(r"^[a-z][a-z0-9_-]*$", w):
            return m, w
        return nm, "default"
    by_model_files, by_wl_files = {}, {}
    for nm, d in models.items():
        m, w = split_mw(nm)
        by_model_files.setdefault(m, {})[nm] = d
        by_wl_files.setdefault(w, {})[nm] = d
    by_model = {m: pool_models(fs) for m, fs in by_model_files.items()}
    by_wl = {w: pool_models(fs) for w, fs in by_wl_files.items()}
    pooled = pool_models(models)
    print(f"[group] {len(models)} files -> {len(by_model)} models x {len(by_wl)} workloads",
          flush=True)

    out = dict(generated_s=None, trace_dir=a.traces, glob=a.glob,
               files=list(models.keys()),
               models=sorted(by_model_files), workloads=sorted(by_wl_files),
               config=dict(TRAIN_N=TRAIN_N, TEST_N=TEST_N, MI_N=MI_N,
                           N_BOOT=N_BOOT, headline_horizon=HEADLINE_H,
                           split="request-level (group hold-out)"))

    # ---- global pooled headline (all files) ----
    print("[pooled] redundancy + headline table ...", flush=True)
    mi_sample = subsample(np.arange(pooled["F"].shape[0]), MI_N, SEED)
    out["pooled"] = dict(
        summary=dataset_summary(pooled),
        per_view={h: per_view_auc(pooled["F"][mi_sample], pooled["y"][h][mi_sample].astype(np.float32))
                  for h in HORIZONS},
        redundancy=redundancy_report(pooled["F"][mi_sample],
                                     pooled["y"][HEADLINE_H][mi_sample],
                                     feature_names=list(FEATURE_NAMES)),
        headline=headline_table(pooled, with_ci=True),
        auc_vs_horizon=auc_vs_horizon(pooled),
        pareto=pareto(pooled),
        view_ablation=view_ablation(pooled),
        quant_query_ablation=quant_query_ablation(pooled),
        calibration=calibration(pooled),
        drift=drift_study(pooled),
    )
    print("   pooled done", flush=True)

    # ---- per-WORKLOAD (NEW: external validity) — pools all models within a
    #      workload; CI included so "within+cross holds on every workload" is
    #      a claim with error bars, not a point estimate. ----
    out["per_workload"] = {}
    for w, d in by_wl.items():
        print(f"[per-workload] {w} ...", flush=True)
        mi_s = subsample(np.arange(d["F"].shape[0]), MI_N, SEED)
        out["per_workload"][w] = dict(
            summary=dataset_summary(d),
            per_view={hh: per_view_auc(d["F"][mi_s], d["y"][hh][mi_s].astype(np.float32))
                      for hh in HORIZONS},
            redundancy=redundancy_report(d["F"][mi_s], d["y"][HEADLINE_H][mi_s],
                                         feature_names=list(FEATURE_NAMES)),
            headline=headline_table(d, with_ci=True),
            calibration=calibration(d),
            drift=drift_study(d),
        )

    # ---- per-MODEL (architecture; pools workloads within a model) ----
    out["per_model"] = {}
    for m, d in by_model.items():
        print(f"[per-model] {m} ...", flush=True)
        mi_s = subsample(np.arange(d["F"].shape[0]), MI_N, SEED)
        out["per_model"][m] = dict(
            summary=dataset_summary(d),
            redundancy=redundancy_report(d["F"][mi_s], d["y"][HEADLINE_H][mi_s],
                                         feature_names=list(FEATURE_NAMES)),
            headline=headline_table(d, with_ci=False),
            auc_vs_horizon=auc_vs_horizon(d),
            view_ablation=view_ablation(d),
            drift=drift_study(d),
        )

    # ---- transfer matrix over ARCHITECTURES (group by model, pool workloads) ----
    print("[transfer] cross-architecture matrix ...", flush=True)
    out["transfer"] = dict(
        standardized=transfer_matrix(by_model, standardize=True),
        raw=transfer_matrix(by_model, standardize=False),
    )

    out["generated_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWROTE {a.out}  ({out['generated_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
