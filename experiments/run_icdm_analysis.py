"""ICDM gate + DM-metrics driver.

Produces, from a trace directory (real A100 traces) or the synthetic fallback,
everything the ICDM paper's §3/§5 needs AND the single go/no-go gate
(ICDM_PIVOT.md §A): the redundancy table, the pairwise interaction gap, ranking
+ calibration metrics, the accuracy-vs-budget Pareto, and the drift comparison.

    python experiments/run_icdm_analysis.py                 # synthetic (harness check)
    python experiments/run_icdm_analysis.py --traces DIR    # real traces (the real gate)
    python experiments/run_icdm_analysis.py --traces DIR --horizon h4 --json

IMPORTANT: on synthetic data the redundancy numbers only reflect the generator's
hard-coded correlations. The *decision* must be made on real attention traces.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from xqp.features import FEATURE_NAMES
from xqp.eval import synthetic_dataset, roc_auc
from xqp.dm_metrics import (
    average_precision, precision_at_k, recall_at_k,
    expected_calibration_error, brier_score,
)
from xqp.info_theory import redundancy_report
from xqp.predictor import ClosedFormXQP, PairwiseXQP
from xqp.sota_iterations.iter5_online import OnlineXQP


def _load_traces(trace_dir: str, horizon: str):
    """Concatenate all <dir>/*.jsonl into (F, y, step). Returns None if empty."""
    from xqp.trace import load_trace
    files = sorted(glob.glob(os.path.join(trace_dir, "*.jsonl")))
    Fs, ys, steps = [], [], []
    for f in files:
        rows = load_trace(f)
        if not rows or f"y_{horizon}" not in rows:
            continue
        Fs.append(np.stack([rows["f_within"], rows["f_cross"],
                            rows["f_query"], rows["f_pos"]], axis=1).astype(np.float32))
        ys.append(rows[f"y_{horizon}"].astype(np.float32))
        steps.append(rows["step"].astype(np.int64) if "step" in rows
                     else np.arange(len(rows[f"y_{horizon}"])))
    if not Fs:
        return None
    return np.concatenate(Fs), np.concatenate(ys), np.concatenate(steps)


def _split(F, y, frac=0.2, seed=0):
    n = F.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    nv = int(frac * n)
    return F[perm[nv:]], y[perm[nv:]], F[perm[:nv]], y[perm[:nv]]


def per_view(F, y) -> dict:
    """Single-view AUC / AUPRC (raw monotone column — no fit needed)."""
    out = {}
    for i, name in enumerate(FEATURE_NAMES):
        out[name] = dict(auc=roc_auc(y, F[:, i]),
                         auprc=average_precision(y, F[:, i]))
    return out


def fused_metrics(F, y, seed=0) -> dict:
    Ftr, ytr, Fva, yva = _split(F, y, seed=seed)
    cf = ClosedFormXQP.from_fit(Ftr, ytr)
    s = cf.score(Fva)
    pw = PairwiseXQP.from_fit(Ftr, ytr)
    s_pw = pw.score(Fva)
    auc_cf, auc_pw = roc_auc(yva, s), roc_auc(yva, s_pw)
    return dict(
        closed=dict(auc=auc_cf, auprc=average_precision(yva, s),
                    p_at_10=precision_at_k(yva, s, 0.10),
                    r_at_10=recall_at_k(yva, s, 0.10),
                    ece=expected_calibration_error(yva, s),
                    brier=brier_score(yva, s)),
        pairwise=dict(auc=auc_pw, auprc=average_precision(yva, s_pw),
                      ece=expected_calibration_error(yva, s_pw)),
        pairwise_gap=float(auc_pw - auc_cf),
    )


def accuracy_budget_pareto(F, y, seed=0) -> dict:
    """Two budget axes: (1) #views computed (feature subset, masking dropped
    columns to 0 then fitting); (2) model complexity (closed 4 vs pairwise 15).
    #views is the meaningful cost axis — the query cosine is the expensive view."""
    Ftr, ytr, Fva, yva = _split(F, y, seed=seed)
    D = F.shape[1]
    by_nviews = []
    # incrementally add views in descending single-view AUC order
    order = sorted(range(D), key=lambda i: -roc_auc(ytr, Ftr[:, i]))
    for k in range(1, D + 1):
        active = order[:k]
        mask = np.zeros(D, dtype=np.float32)
        mask[active] = 1.0
        cf = ClosedFormXQP.from_fit(Ftr * mask, ytr)
        auc = roc_auc(yva, cf.score(Fva * mask))
        by_nviews.append(dict(n_views=k,
                              views=[FEATURE_NAMES[i] for i in active],
                              auc=float(auc)))
    cf = ClosedFormXQP.from_fit(Ftr, ytr)
    pw = PairwiseXQP.from_fit(Ftr, ytr)
    by_model = [
        dict(model="closed", params=4, auc=roc_auc(yva, cf.score(Fva))),
        dict(model="pairwise", params=15, auc=roc_auc(yva, pw.score(Fva))),
        dict(model="tinymlp", params=148, auc=None, note="needs MLP trainer (P2 TODO)"),
    ]
    return dict(by_n_views=by_nviews, by_model=by_model)


def drift_eval(F, y, step) -> dict:
    """Temporally-ordered: train on early stream, compare static vs online on
    the late stream. Demonstrates the concept-drift / online-adaptation story."""
    order = np.argsort(step, kind="stable")
    F, y = F[order], y[order]
    n = F.shape[0]
    a, b = int(0.6 * n), int(0.8 * n)
    Ftr, ytr = F[:a], y[:a]
    Fmid, ymid = F[a:b], y[a:b]
    Fte, yte = F[b:], y[b:]
    if yte.sum() == 0 or yte.sum() == yte.shape[0] or Ftr.shape[0] < 16:
        return dict(note="insufficient/degenerate late-stream labels")
    static = ClosedFormXQP.from_fit(Ftr, ytr)
    auc_static = roc_auc(yte, static.score(Fte))
    online = OnlineXQP(predictor=ClosedFormXQP.from_fit(Ftr, ytr),
                       update_every=8, buffer_size=512, learning_rate=0.3)
    for i in range(0, Fmid.shape[0], 32):
        online.observe(Fmid[i:i + 32], ymid[i:i + 32])
    auc_online = roc_auc(yte, online.score(Fte))
    return dict(auc_static=float(auc_static), auc_online=float(auc_online),
                online_gain=float(auc_online - auc_static))


def gate_verdict(red: dict, fused: dict) -> dict:
    """Apply ICDM_PIVOT.md §A decision rule."""
    gap = fused["pairwise_gap"]
    n_syn = red["n_synergistic_pairs"]
    # is at least one view complementary (independent) to the others?
    independent_pairs = [p for p in red["pairs"] if p["verdict"] == "independent"]
    small_gap = abs(gap) <= 0.005
    if n_syn == 0 and small_gap and independent_pairs:
        track = "RESEARCH track — C2 holds (log-linear near-optimal in practice)"
    elif small_gap:
        track = "APPLIED track — fusion fine but redundancy story muddy; lead with data source"
    else:
        track = "RECONSIDER — pairwise gap large; interactions matter, stay MLSys/ICML"
    return dict(pairwise_gap=gap, n_synergistic_pairs=n_syn,
                n_independent_pairs=len(independent_pairs),
                recommendation=track)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=None, help="dir of real *.jsonl traces")
    ap.add_argument("--horizon", default="h4")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.traces:
        loaded = _load_traces(a.traces, a.horizon)
        if loaded is None:
            print(f"no usable traces in {a.traces}", file=sys.stderr)
            return 1
        F, y, step = loaded
        source = f"real:{a.traces}"
    else:
        F, y = synthetic_dataset(n_blocks=256, n_steps=64, seed=0)
        step = np.repeat(np.arange(64), 256)
        source = "SYNTHETIC (harness check only — NOT a valid gate)"

    red = redundancy_report(F, y, feature_names=list(FEATURE_NAMES))
    fused = fused_metrics(F, y)
    results = dict(
        source=source, n=int(F.shape[0]), pos_rate=float(y.mean()),
        per_view=per_view(F, y),
        fused=fused,
        redundancy=red,
        accuracy_budget_pareto=accuracy_budget_pareto(F, y),
        drift=drift_eval(F, y, step),
        gate=gate_verdict(red, fused),
    )
    print(json.dumps(results, indent=2))
    if not a.json:
        g = results["gate"]
        print("\n" + "=" * 64, file=sys.stderr)
        print(f"SOURCE: {source}", file=sys.stderr)
        print(f"pairwise ΔAUC = {g['pairwise_gap']:+.4f} | synergistic pairs = "
              f"{g['n_synergistic_pairs']} | independent pairs = {g['n_independent_pairs']}",
              file=sys.stderr)
        print(f"GATE: {g['recommendation']}", file=sys.stderr)
        print("=" * 64, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
