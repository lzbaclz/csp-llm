"""Feature extraction for XQP.

Per (layer, block, decode-step) we extract a 4-dim vector:
  f0 = s_within  : current-layer attention EMA, normalized to [0, 1]
  f1 = s_cross   : previous-layer top-r indicator (in {0, 1}) — InfiniGen-style
  f2 = s_query   : cosine(q_{t-1}, K_b) — Quest-style proxy
  f3 = s_pos     : sigmoid((t - p_b) / w_recency)

All extraction is *vectorized over blocks* and stateless w.r.t. the predictor;
the EMA itself is updated externally by the policy.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = ("s_within", "s_cross", "s_query", "s_pos")
FEATURE_DIM = 4


def _safe_l2(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    return np.sqrt((x * x).sum(axis=axis, keepdims=True) + eps)


def cosine_query_key(q: np.ndarray, K: np.ndarray, *, reduce: str = "max") -> np.ndarray:
    """Cosine query–key similarity.

    Supports two input shapes (addresses round-1 review m6 multi-head batching):
      - q: (d,)        K: (B, d)        → returns (B,) cosine similarity.
      - q: (H, d)      K: (B, H, d)     → returns (B,) head-aggregated similarity,
        with `reduce ∈ {"mean", "max"}` controlling per-block aggregation.

    Implementation note: we use cosine rather than raw dot product so the
    scale is bounded and weights are comparable across layers with different
    norm conventions. The Quest paper uses raw dot products; we report both
    in §experiments to address the round-1 review concern.
    """
    q = np.asarray(q, dtype=np.float32)
    K = np.asarray(K, dtype=np.float32)
    if q.ndim == 1:
        if K.ndim != 2 or K.shape[1] != q.shape[0]:
            raise ValueError(f"shape mismatch: q={q.shape}, K={K.shape}")
        qn = float(np.sqrt((q * q).sum()) + 1e-9)
        Kn = _safe_l2(K, axis=1).reshape(-1)
        return (K @ q) / (qn * Kn)
    if q.ndim == 2:  # multi-head
        if K.ndim != 3 or K.shape[1] != q.shape[0] or K.shape[2] != q.shape[1]:
            raise ValueError(f"shape mismatch (multi-head): q={q.shape}, K={K.shape}")
        # per-head cosine: (B, H)
        qn = _safe_l2(q, axis=1).reshape(1, -1)            # (1, H)
        Kn = _safe_l2(K, axis=2).reshape(K.shape[0], K.shape[1])  # (B, H)
        # einsum: (B, H, d) · (H, d) → (B, H)
        num = np.einsum("bhd,hd->bh", K, q)
        per_head = num / (qn * Kn + 1e-9)
        if reduce == "mean":
            return per_head.mean(axis=1)
        if reduce == "max":
            return per_head.max(axis=1)
        raise ValueError(f"reduce must be mean or max, got {reduce!r}")
    raise ValueError(f"q must be 1D or 2D, got shape {q.shape}")


def dot_query_key(q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Raw dot product q·K (Quest convention). Use this when keys/queries are
    quantized in the same scale; otherwise cosine is more stable."""
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    K = np.asarray(K, dtype=np.float32)
    if K.ndim != 2 or K.shape[1] != q.shape[0]:
        raise ValueError(f"shape mismatch: q={q.shape}, K={K.shape}")
    return K @ q


def topk_indicator(scores: np.ndarray, r: float) -> np.ndarray:
    """Return a 0/1 vector of length len(scores) with 1s on the top-r fraction."""
    n = scores.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    k = max(1, int(np.ceil(r * n)))
    idx = np.argpartition(-scores, kth=min(k - 1, n - 1))[:k]
    out = np.zeros(n, dtype=np.float32)
    out[idx] = 1.0
    return out


def recency(step: int, last_used: np.ndarray, window: float = 64.0) -> np.ndarray:
    """Exponential-decay recency: fresh → 1, old → 0.

    BUGFIX (audit): previously `1/(1+exp(delta/window))` which maps
    fresh→0.5 (not 1) and old→0 — asymmetric. Now uses `exp(-delta/window)`
    which gives fresh (delta=0) → 1 exactly, old → 0 monotonically.
    """
    delta = np.asarray(step - last_used, dtype=np.float32)
    delta = np.clip(delta, 0, None)   # guard: never look "future"
    return np.exp(-delta / window).astype(np.float32)


def query_proxy_mean(q_history: np.ndarray, k: int = 1) -> np.ndarray:
    """Round-2 MR3: replace single-token q_{t-1} with mean of last k tokens.

    q_history: (T, d) recent decoded queries; we take the last k and mean.
    """
    q_history = np.asarray(q_history, dtype=np.float32)
    if q_history.ndim == 1:
        return q_history
    k = min(k, q_history.shape[0])
    return q_history[-k:].mean(axis=0)


def extract_features(
    *,
    ema_within: np.ndarray,
    ema_prev_layer: np.ndarray | None,
    K_layer: np.ndarray,
    q_prev: np.ndarray,
    step: int,
    last_used: np.ndarray,
    r_cross: float = 0.10,
    w_recency: float = 64.0,
    cross_signal: str = "indicator",
) -> np.ndarray:
    """Return (B, 4) feature matrix.

    Args:
        ema_within: (B,) attention EMA for the current layer.
        ema_prev_layer: (B,) attention EMA for layer l-1, or None for the first layer
            (then s_cross is set to ema_within itself, matching the InfiniGen
            convention that the first layer has no predecessor).
        K_layer: (B, d_head) key vectors for this layer.
        q_prev: (d_head,) the previous-step decoded token's query vector.
        step: current decode step.
        last_used: (B,) step index of last access.
        r_cross: top-fraction for s_cross indicator.
        w_recency: sigmoid window in steps.

    Returns:
        (B, 4) float32 matrix in column order FEATURE_NAMES.
    """
    ema_within = np.asarray(ema_within, dtype=np.float32).reshape(-1)
    B = ema_within.shape[0]
    if B == 0:
        # No blocks (e.g. an empty request, or the very first decode step):
        # return a well-formed (0, 4) matrix instead of crashing on .max().
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    if ema_prev_layer is None:
        ema_prev_layer = ema_within
    s_within = ema_within / (ema_within.max() + 1e-9)
    # Round-1 M2: also support the continuous variant of the cross-layer signal.
    # `indicator` is the original InfiniGen-style 0/1 top-r flag; `continuous`
    # uses the normalized magnitude. The default stays `indicator` for
    # back-compat with the saved predictor weights.
    ema_prev_arr = np.asarray(ema_prev_layer, dtype=np.float32)
    if cross_signal == "indicator":
        s_cross = topk_indicator(ema_prev_arr, r_cross)
    elif cross_signal == "continuous":
        s_cross = ema_prev_arr / (ema_prev_arr.max() + 1e-9)
    else:
        raise ValueError(
            f"cross_signal must be 'indicator' or 'continuous', got {cross_signal!r}"
        )
    s_query = cosine_query_key(q_prev, K_layer)
    # rescale cosine from [-1, 1] to [0, 1] so all features share scale
    s_query = 0.5 * (s_query + 1.0)
    s_pos = recency(step, np.asarray(last_used, dtype=np.float32), window=w_recency)
    F = np.stack([s_within, s_cross, s_query, s_pos], axis=1).astype(np.float32)
    assert F.shape == (B, FEATURE_DIM), f"unexpected feature shape {F.shape}"
    return F


def feature_summary(F: np.ndarray) -> dict:
    """Useful when sanity-checking trace dumps."""
    return {
        name: dict(mean=float(F[:, i].mean()), std=float(F[:, i].std()),
                   min=float(F[:, i].min()), max=float(F[:, i].max()))
        for i, name in enumerate(FEATURE_NAMES)
    }
