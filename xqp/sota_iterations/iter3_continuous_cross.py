"""Iteration 3 — Continuous cross-layer signal (InfiniGen-strong).

XQP's original `s_cross` is a 1-bit top-r indicator of the previous layer's
hot set. InfiniGen (OSDI'24) showed the *continuous* prev-layer attention
magnitude is a stronger predictor of the next layer's top-k than the
indicator (reported AUC 0.89 vs 0.87). The continuous variant already
exists in `features.extract_features(cross_signal="continuous")`; this
module makes it a first-class, self-contained transform and supplies the
truncated-Gaussian natural-parameter coefficient that backs the updated
Bayes-optimality argument in paper §3.2.

Why a separate module (vs. flipping the default): the saved closed-form
weights were fit against the indicator feature, so flipping the global
default would silently invalidate them. New fits should pass
`cross_signal="continuous"`; this module is the recommended entry point.
"""
from __future__ import annotations

import numpy as np

from ..features import extract_features


def continuous_cross_signal(ema_prev_layer: np.ndarray) -> np.ndarray:
    """Normalized continuous prev-layer EMA in [0, 1].

    This replaces the 1-bit `topk_indicator` with the full magnitude, keeping
    the tail information InfiniGen relies on. First-layer callers (no
    predecessor) should pass the current-layer EMA, matching the
    `extract_features` convention.
    """
    a = np.asarray(ema_prev_layer, dtype=np.float32).reshape(-1)
    if a.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    return (a / (a.max() + 1e-9)).astype(np.float32)


def make_continuous_features(
    *,
    ema_within: np.ndarray,
    ema_prev_layer: np.ndarray | None,
    K_layer: np.ndarray,
    q_prev: np.ndarray,
    step: int,
    last_used: np.ndarray,
    r_cross: float = 0.10,
    w_recency: float = 64.0,
) -> np.ndarray:
    """(B, 4) feature matrix with the continuous cross-layer signal.

    Thin wrapper over `extract_features(cross_signal="continuous")` so callers
    that want the InfiniGen-strong variant have a single, obvious entry point.
    """
    return extract_features(
        ema_within=ema_within,
        ema_prev_layer=ema_prev_layer,
        K_layer=K_layer,
        q_prev=q_prev,
        step=step,
        last_used=last_used,
        r_cross=r_cross,
        w_recency=w_recency,
        cross_signal="continuous",
    )


def truncated_gaussian_coefficient(f: np.ndarray, y: np.ndarray) -> dict:
    """Linear-discriminant coefficient for a continuous feature in [0, 1].

    Backs the paper §3.2 claim that, when a feature is modelled as a
    (truncated) Gaussian whose mean depends on the latent importance class
    z ∈ {0, 1} with a shared variance, the log-posterior is *linear* in the
    feature with slope (μ₁ − μ₀)/σ². This is the natural-exponential-family
    statement: for a continuous feature the closed-form logistic weight is
    Bayes-optimal exactly when an indicator feature would be under Bernoulli.

    Returns dict(w, b, mu0, mu1, var) where `w` is the LDA slope and `b` the
    intercept of the log-likelihood ratio log p(f|z=1)/p(f|z=0).
    """
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if f.shape[0] != y.shape[0]:
        raise ValueError(f"shape mismatch f={f.shape}, y={y.shape}")
    pos = y > 0.5
    neg = ~pos
    if pos.sum() < 2 or neg.sum() < 2:
        raise ValueError("need >=2 samples per class to estimate Gaussian params")
    mu1 = float(f[pos].mean())
    mu0 = float(f[neg].mean())
    # pooled (shared) variance — the LDA assumption
    var = float((((f[pos] - mu1) ** 2).sum() + ((f[neg] - mu0) ** 2).sum())
                / (f.shape[0] - 2))
    var = max(var, 1e-9)
    w = (mu1 - mu0) / var
    b = -(mu1 ** 2 - mu0 ** 2) / (2 * var)
    return dict(w=float(w), b=float(b), mu0=mu0, mu1=mu1, var=var)
