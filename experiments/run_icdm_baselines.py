"""ICDM §5 headline table: XQP vs a competitive baseline suite, with bootstrap
CIs and a paired-bootstrap significance test against XQP-closed.

    python experiments/run_icdm_baselines.py                 # synthetic (harness check)
    python experiments/run_icdm_baselines.py --traces DIR     # real traces
    python experiments/run_icdm_baselines.py --traces DIR --json

The make-or-break read: does the 4-weight calibrated closed form match the
GBDT/MLP on AUC/AUPRC (supporting "a minimal calibrated model suffices") while
winning on calibration (ECE) and cost? On synthetic the answer is yes; the
*decision* must be made on real attention traces.
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
from xqp.dm_metrics import average_precision, precision_at_k, expected_calibration_error
from xqp.predictor import ClosedFormXQP, PairwiseXQP
from xqp.baselines import single_signal_baselines, all_learned_baselines
from xqp.stats import bootstrap_ci, paired_bootstrap_test


def _load_traces(trace_dir, horizon):
    from xqp.trace import load_trace
    Fs, ys = [], []
    for f in sorted(glob.glob(os.path.join(trace_dir, "*.jsonl"))):
        rows = load_trace(f)
        if not rows or f"y_{horizon}" not in rows:
            continue
        Fs.append(np.stack([rows["f_within"], rows["f_cross"],
                            rows["f_query"], rows["f_pos"]], axis=1).astype(np.float32))
        ys.append(rows[f"y_{horizon}"].astype(np.float32))
    if not Fs:
        return None
    return np.concatenate(Fs), np.concatenate(ys)


def build_table(F, y, seed=0, n_boot=500):
    n = F.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    nv = int(0.2 * n)
    tr, te = perm[nv:], perm[:nv]
    Ftr, ytr, Fte, yte = F[tr], y[tr], F[te], y[te]

    methods = []  # (name, scores_on_test, params)
    cf = ClosedFormXQP.from_fit(Ftr, ytr)
    methods.append(("XQP-closed", cf.score(Fte), 4))
    pw = PairwiseXQP.from_fit(Ftr, ytr)
    methods.append(("XQP-pairwise", pw.score(Fte), 15))
    for b in single_signal_baselines():
        methods.append((b.name, b.score(Fte), b.meta["params"]))
    for b in all_learned_baselines(Ftr, ytr, seed=seed):
        methods.append((b.name, b.score(Fte), b.meta.get("params") or b.meta.get("leaf_nodes")))

    ref_scores = methods[0][1]   # XQP-closed is the reference for significance
    rows = []
    for name, s, params in methods:
        auc_ci = bootstrap_ci(roc_auc, yte, s, n_boot=n_boot, seed=seed)
        ap_ci = bootstrap_ci(average_precision, yte, s, n_boot=n_boot, seed=seed)
        row = dict(
            method=name, params=params,
            auc=auc_ci["mean"], auc_lo=auc_ci["lo"], auc_hi=auc_ci["hi"],
            auprc=ap_ci["mean"], auprc_lo=ap_ci["lo"], auprc_hi=ap_ci["hi"],
            p_at_10=precision_at_k(yte, s, 0.10),
            ece=expected_calibration_error(yte, s),
        )
        if name != "XQP-closed":
            t = paired_bootstrap_test(roc_auc, yte, ref_scores, s, n_boot=n_boot, seed=seed)
            row["auc_delta_vs_closed"] = t["delta"]   # >0 means XQP-closed better
            row["p_vs_closed"] = t["p_value"]
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=None)
    ap.add_argument("--horizon", default="h4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.traces:
        loaded = _load_traces(a.traces, a.horizon)
        if loaded is None:
            print(f"no usable traces in {a.traces}", file=sys.stderr)
            return 1
        F, y = loaded
        source = f"real:{a.traces}"
    else:
        F, y = synthetic_dataset(n_blocks=256, n_steps=64, seed=a.seed)
        source = "SYNTHETIC (harness check — NOT a valid result)"

    rows = build_table(F, y, seed=a.seed)
    payload = dict(source=source, n=int(F.shape[0]), pos_rate=float(y.mean()),
                   features=list(FEATURE_NAMES), table=rows)
    if a.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"\nSOURCE: {source}   (n={F.shape[0]}, pos={y.mean():.3f})")
    print(f"{'method':22s} {'AUC [95% CI]':>22s} {'AUPRC':>7s} {'P@10':>6s} {'ECE':>6s} {'params':>8s} {'p vs closed':>12s}")
    for r in rows:
        ci = f"{r['auc']:.3f}[{r['auc_lo']:.3f},{r['auc_hi']:.3f}]"
        pv = "" if r["method"] == "XQP-closed" else f"{r.get('p_vs_closed', float('nan')):.3f}"
        print(f"{r['method']:22s} {ci:>22s} {r['auprc']:7.3f} {r['p_at_10']:6.3f} "
              f"{r['ece']:6.3f} {str(r['params']):>8s} {pv:>12s}")
    print("\nReading: XQP-closed wins if it (a) is statistically indistinguishable "
          "from GBDT/MLP on AUC/AUPRC (p>0.05) at far fewer params, and (b) has "
          "the lowest ECE. Confirm on REAL traces before claiming it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
