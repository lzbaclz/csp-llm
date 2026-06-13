"""Iteration 1 — Quest-style per-block max cosine.

Quest (ICML'24) uses page-level max(q·K[j]) over a block to identify
top-k candidates. XQP's original `s_query` is cos(q, K_block_mean),
which loses the tail; adding `s_query_max` captures Quest's signal.

Both features are kept; the closed-form predictor learns the optimal
linear combination on training traces.
"""
from __future__ import annotations

import numpy as np

from ..features import _safe_l2


def cosine_query_key_per_token(q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Per-token cosine; returns (B,) max-cosine if K is (B, d), else
    (B, n_tok) if K is (B, n_tok, d).

    For block-granularity Quest emulation, K should be (B, n_tok, d) where
    n_tok is the block size (typically 16 or 32).
    """
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    K = np.asarray(K, dtype=np.float32)
    qn = float(np.sqrt((q * q).sum()) + 1e-9)

    if K.ndim == 2:  # (B, d)
        Kn = _safe_l2(K, axis=1).reshape(-1)
        return ((K @ q) / (qn * Kn)).astype(np.float32)
    if K.ndim == 3:  # (B, n_tok, d)
        # (B, n_tok, d) · (d,) → (B, n_tok)
        num = K @ q                                # (B, n_tok)
        Kn = _safe_l2(K, axis=2).reshape(K.shape[0], K.shape[1])
        return (num / (qn * Kn)).astype(np.float32)
    raise ValueError(f"K must be 2D or 3D, got {K.shape}")


def query_max_min_per_block(q: np.ndarray, K_block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (max_cosine_per_block, min_cosine_per_block) of shape (B,).

    Quest uses both the min and max to bound the attention score range.
    Both features are useful: max captures "best-case attention", min
    captures worst-case.
    """
    per_token = cosine_query_key_per_token(q, K_block)
    if per_token.ndim == 1:
        # K_block was (B, d), no within-block variation
        return per_token, per_token
    return per_token.max(axis=1), per_token.min(axis=1)


def quest_bound_features(
    q: np.ndarray, K_block: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute Quest-style bound features for the closed-form predictor.

    Returns a dict with keys: s_query_max, s_query_min, s_query_mean.
    All rescaled from [-1, 1] to [0, 1] for compatibility with the
    rest of the XQP feature scale.
    """
    qmax, qmin = query_max_min_per_block(q, K_block)
    if qmax.ndim == 1 and qmax.shape == qmin.shape and (qmax == qmin).all():
        qmean = qmax
    else:
        qmean = (qmax + qmin) / 2.0
    rescale = lambda x: 0.5 * (x + 1.0)
    return dict(
        s_query_max=rescale(qmax).astype(np.float32),
        s_query_min=rescale(qmin).astype(np.float32),
        s_query_mean=rescale(qmean).astype(np.float32),
    )
