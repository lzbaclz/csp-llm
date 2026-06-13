"""Synthetic-trace pilot — the sandbox-runnable (CPU, no-GPU) experiment driver.

This reproduces every *synthetic* number quoted in experiments/e1_*.md,
e2_*.md, e3_*.md. The real headline numbers in the paper come from the ga100
run (scripts/run_benchmarks.sh); this pilot validates that the predictor fits
are well-conditioned, that the signal-drop ablation has the expected sign, and
that the closed form is launch-bound rather than compute-bound.

Usage:
    python experiments/run_pilot.py            # pretty-print
    python experiments/run_pilot.py --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Make `xqp` importable when run as a bare script (python experiments/run_pilot.py)
# without an editable install, so the e1/e2/e3 repro commands work as documented.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from xqp.eval import synthetic_dataset, roc_auc, measure_wcet_cpu
from xqp.predictor import ClosedFormXQP, PairwiseXQP, TinyMLPXQP
from xqp.profile_breakdown import cpu_stage_breakdown
from xqp.sota_iterations.iter4_per_head import per_head_features, PerHeadClosedFormXQP


def _split(F, y, frac=0.2, seed=0):
    n = F.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    nv = int(frac * n)
    return F[perm[nv:]], y[perm[nv:]], F[perm[:nv]], y[perm[:nv]]


def e1_auc_vs_signals(seeds=(0, 1, 2, 3, 4)) -> dict:
    """Closed-form / pairwise AUC + single-signal AUCs, averaged over seeds."""
    names = ["recency_only", "within_only", "cross_only", "query_only",
             "closed_form", "pairwise"]
    # Single-signal AUC = AUC of the raw (monotone) feature column; no fit needed
    # since AUC is invariant to any monotone transform of a 1-D score.
    cols = {"recency_only": 3, "within_only": 0, "cross_only": 1,
            "query_only": 2}
    acc = {n: [] for n in names}
    for s in seeds:
        F, y = synthetic_dataset(n_blocks=256, n_steps=64, seed=s)
        Ftr, ytr, Fva, yva = _split(F, y, seed=s)
        for n, c in cols.items():
            acc[n].append(roc_auc(yva, Fva[:, c]))
        cf = ClosedFormXQP.from_fit(Ftr, ytr)
        acc["closed_form"].append(roc_auc(yva, cf.score(Fva)))
        pw = PairwiseXQP.from_fit(Ftr, ytr)
        acc["pairwise"].append(roc_auc(yva, pw.score(Fva)))
    return {n: dict(mean=float(np.mean(v)), std=float(np.std(v))) for n, v in acc.items()}


def e2_perlayer_and_extensions(seed=0) -> dict:
    """Shared vs per-layer; indicator vs continuous-cross; per-head (iter4)."""
    out = {}
    # shared vs per-layer
    F, y = synthetic_dataset(n_blocks=128, n_steps=64, seed=seed)
    layer_ids = (np.arange(F.shape[0]) // 256) % 8
    Ftr, ytr, Fva, yva = _split(F, y, seed=seed)
    lids_tr = (np.arange(Ftr.shape[0]) // 256) % 8
    shared = ClosedFormXQP.from_fit(Ftr, ytr)
    perlayer = ClosedFormXQP.from_fit(Ftr, ytr, layer_ids=lids_tr, per_layer=True)
    lids_va = (np.arange(Fva.shape[0]) // 256) % 8
    s_shared = shared.score(Fva)
    s_pl = np.zeros(Fva.shape[0], dtype=np.float32)
    for l in np.unique(lids_va):
        m = lids_va == l
        s_pl[m] = perlayer.score(Fva[m], layer=int(l))
    out["shared_auc"] = float(roc_auc(yva, s_shared))
    out["per_layer_auc"] = float(roc_auc(yva, s_pl))

    # per-head (iter4): heads carry complementary query signal
    rng = np.random.default_rng(seed)
    B, H, d, T = 128, 4, 16, 64
    F_ph, ys = [], []
    importance = rng.gamma(0.5, 1.0, size=B).astype(np.float32)
    last_used = np.zeros(B, dtype=np.float32)
    for t in range(T):
        importance = 0.85 * importance + 0.15 * rng.gamma(0.5, 1.0, size=B)
        ema_within = np.stack([importance + 0.1 * rng.normal(size=B) for _ in range(H)], axis=1)
        ema_within = np.clip(ema_within, 0, None).astype(np.float32)
        K = rng.normal(size=(B, H, d)).astype(np.float32)
        # one "retrieval head" aligns with importance
        q = rng.normal(size=(H, d)).astype(np.float32)
        feats = per_head_features(ema_within=ema_within, ema_prev_layer=importance,
                                  K_layer=K, q_heads=q, step=t, last_used=last_used)
        nxt = np.clip(0.85 * importance + 0.15 * rng.gamma(0.5, 1.0, size=B), 0, None)
        k = max(1, int(0.10 * B))
        y = np.zeros(B, dtype=np.float32)
        y[np.argpartition(-nxt, k - 1)[:k]] = 1.0
        F_ph.append(feats); ys.append(y)
    F_ph = np.concatenate(F_ph, axis=0); ys = np.concatenate(ys, axis=0)
    ntr = int(0.8 * F_ph.shape[0])
    ph = PerHeadClosedFormXQP.from_fit(F_ph[:ntr], ys[:ntr])
    out["per_head_max_auc"] = float(roc_auc(ys[ntr:], ph.score_blocks(F_ph[ntr:], reduce="max")))
    out["per_head_mean_auc"] = float(roc_auc(ys[ntr:], ph.score_blocks(F_ph[ntr:], reduce="mean")))
    # block-level baseline: mean-over-heads features, single 4-weight fit
    Fb = F_ph.mean(axis=1)
    cf = ClosedFormXQP.from_fit(Fb[:ntr], ys[:ntr])
    out["block_level_auc"] = float(roc_auc(ys[ntr:], cf.score(Fb[ntr:])))
    return out


def e3_wcet(batch=4096) -> dict:
    F, y = synthetic_dataset(n_blocks=512, n_steps=16, seed=0)
    Fb = F[:batch].astype(np.float32)
    cf = ClosedFormXQP.from_fit(F, y)
    pw = PairwiseXQP.from_fit(F, y)
    mlp = TinyMLPXQP.random_init(seed=0)
    return {
        "batch": batch,
        "closed_form": measure_wcet_cpu(cf, Fb, n_replays=3000),
        "pairwise_note": "pairwise uses augmented features; CPU-only sanity",
        "tinymlp": measure_wcet_cpu(mlp, Fb, n_replays=3000),
        "closed_form_stage_breakdown": cpu_stage_breakdown(cf, Fb, n=3000),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    results = {
        "e1_auc_vs_signals": e1_auc_vs_signals(),
        "e2_perlayer_and_extensions": e2_perlayer_and_extensions(),
        "e3_wcet": e3_wcet(),
    }
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
