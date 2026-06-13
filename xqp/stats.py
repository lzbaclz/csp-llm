"""Statistical-rigor harness for the ICDM evaluation.

ICDM reviewers want error bars and significance on the headline deltas, not
point estimates. This provides metric-agnostic bootstrap confidence intervals
and a *paired* bootstrap test for the difference between two scorers on the same
test set (the right test when comparing XQP-closed against each baseline on
identical items).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(metric_fn: Callable, y_true: np.ndarray, y_score: np.ndarray,
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """(mean, lo, hi) of `metric_fn(y, score)` over `n_boot` resamples.

    Resamples that become single-class (so the metric is undefined) are skipped.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    n = y_true.shape[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = y_true[idx].sum()
        if s == 0 or s == n:
            continue
        v = metric_fn(y_true[idx], y_score[idx])
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    return dict(mean=float(vals.mean()),
                lo=float(np.percentile(vals, 100 * alpha / 2)),
                hi=float(np.percentile(vals, 100 * (1 - alpha / 2))),
                n=int(vals.size))


def paired_bootstrap_test(metric_fn: Callable, y_true: np.ndarray,
                          score_a: np.ndarray, score_b: np.ndarray,
                          n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap test of H0: metric(A) == metric(B) on the same items.

    Returns the observed delta = metric(A) - metric(B), a CI on the delta, and a
    two-sided p-value (fraction of resampled deltas on the opposite side of 0,
    doubled). Pairing (same resample indices for A and B) controls the
    correlation between the two scorers and is far more powerful than comparing
    independent CIs.
    """
    y_true = np.asarray(y_true).reshape(-1)
    score_a = np.asarray(score_a).reshape(-1)
    score_b = np.asarray(score_b).reshape(-1)
    n = y_true.shape[0]
    rng = np.random.default_rng(seed)
    obs = metric_fn(y_true, score_a) - metric_fn(y_true, score_b)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = y_true[idx].sum()
        if s == 0 or s == n:
            continue
        da = metric_fn(y_true[idx], score_a[idx])
        db = metric_fn(y_true[idx], score_b[idx])
        if np.isfinite(da) and np.isfinite(db):
            deltas.append(da - db)
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size == 0:
        return dict(delta=float(obs), p_value=float("nan"), lo=float("nan"), hi=float("nan"))
    # two-sided p: proportion of resamples on the opposite side of 0 from obs
    if obs >= 0:
        p = 2.0 * float(np.mean(deltas <= 0))
    else:
        p = 2.0 * float(np.mean(deltas >= 0))
    p = min(1.0, p)
    return dict(delta=float(obs),
                lo=float(np.percentile(deltas, 2.5)),
                hi=float(np.percentile(deltas, 97.5)),
                p_value=p, n=int(deltas.size))
