"""Iteration 4 — Per-head feature aggregation (DoubleSparse / SqueezeAttention).

DoubleSparse (2024) and SqueezeAttention (2024) show that per-head attention
sparsity is more predictive than a per-block mean: different heads specialize,
so a single block-level query-cosine throws away the "which head" signal.

This module extends the within-layer and query features to per-(block, head)
granularity. The cross-layer signal and recency stay per-block (heads agree at
the block level for those), broadcast across the head axis. The closed-form
weights become (H, 4) per layer — total still <1 KB because H is 4–32.

Per-head scores are aggregated (max or mean) before block selection, matching
the `policy.aggregate` step described in ITERATIONS.md §4.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..features import recency, topk_indicator, _safe_l2
from ..predictor import ClosedFormXQP, _sigmoid
from .iter1_quest_bound import cosine_query_key_per_token


def per_head_features(
    *,
    ema_within: np.ndarray,      # (B, H) per-head per-block within-layer EMA
    ema_prev_layer: np.ndarray | None,  # (B,) block-level prev-layer EMA, or None
    K_layer: np.ndarray,         # (B, H, d) per-head key vectors
    q_heads: np.ndarray,         # (H, d) per-head query proxy
    step: int,
    last_used: np.ndarray,       # (B,) last-access step per block
    r_cross: float = 0.10,
    w_recency: float = 64.0,
    cross_signal: str = "indicator",
) -> np.ndarray:
    """Return a (B, H, 4) per-head feature tensor in FEATURE_NAMES order.

    s_within (per-head) and s_query (per-head) vary across the head axis;
    s_cross and s_pos are block-level signals broadcast across heads.
    """
    ema_within = np.asarray(ema_within, dtype=np.float32)
    if ema_within.ndim != 2:
        raise ValueError(f"ema_within must be (B, H), got {ema_within.shape}")
    B, H = ema_within.shape
    if B == 0:
        return np.zeros((0, H, 4), dtype=np.float32)
    K_layer = np.asarray(K_layer, dtype=np.float32)
    q_heads = np.asarray(q_heads, dtype=np.float32)
    if K_layer.shape[:2] != (B, H):
        raise ValueError(f"K_layer must be (B, H, d), got {K_layer.shape}")
    if q_heads.shape[0] != H:
        raise ValueError(f"q_heads must be (H, d), got {q_heads.shape}")

    # s_within per head: normalize within each head over blocks
    s_within = ema_within / (ema_within.max(axis=0, keepdims=True) + 1e-9)  # (B, H)

    # s_cross block-level, broadcast to heads
    if ema_prev_layer is None:
        # first layer: fall back to the per-head within EMA reduced to block level
        prev = s_within.mean(axis=1)
    else:
        prev = np.asarray(ema_prev_layer, dtype=np.float32).reshape(-1)
    if cross_signal == "indicator":
        s_cross_b = topk_indicator(prev, r_cross)
    elif cross_signal == "continuous":
        s_cross_b = prev / (prev.max() + 1e-9)
    else:
        raise ValueError(f"cross_signal must be 'indicator' or 'continuous', got {cross_signal!r}")
    s_cross = np.repeat(s_cross_b.reshape(B, 1), H, axis=1)  # (B, H)

    # s_query per head: cosine(q_h, K[:, h, :]) rescaled to [0, 1]
    s_query = np.empty((B, H), dtype=np.float32)
    for h in range(H):
        c = cosine_query_key_per_token(q_heads[h], K_layer[:, h, :])  # (B,)
        s_query[:, h] = 0.5 * (c + 1.0)

    # s_pos block-level, broadcast
    s_pos_b = recency(step, np.asarray(last_used, dtype=np.float32), window=w_recency)
    s_pos = np.repeat(s_pos_b.reshape(B, 1), H, axis=1)  # (B, H)

    F = np.stack([s_within, s_cross, s_query, s_pos], axis=2).astype(np.float32)  # (B, H, 4)
    assert F.shape == (B, H, 4), f"unexpected per-head feature shape {F.shape}"
    return F


@dataclass
class PerHeadClosedFormXQP:
    """One 4-weight logistic regression per head; weights are (H, 4)."""
    weights: np.ndarray = field(default_factory=lambda: np.zeros((1, 4), dtype=np.float32))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=np.float32)
        self.bias = np.asarray(self.bias, dtype=np.float32)

    @property
    def n_heads(self) -> int:
        return self.weights.shape[0]

    @classmethod
    def from_fit(cls, F: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> "PerHeadClosedFormXQP":
        """Fit per-head weights.

        Args:
            F: (N, H, 4) per-head features.
            y: (N,) shared block-level labels, or (N, H) per-head labels.
        """
        F = np.asarray(F, dtype=np.float32)
        if F.ndim != 3 or F.shape[2] != 4:
            raise ValueError(f"F must be (N, H, 4), got {F.shape}")
        N, H, _ = F.shape
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = np.repeat(y.reshape(N, 1), H, axis=1)
        if y.shape != (N, H):
            raise ValueError(f"y must be (N,) or (N, H), got {y.shape}")
        W = np.zeros((H, 4), dtype=np.float32)
        Bz = np.zeros((H,), dtype=np.float32)
        for h in range(H):
            w, b = ClosedFormXQP._fit_single(F[:, h, :], y[:, h], l2=l2)
            W[h] = w
            Bz[h] = b
        return cls(weights=W, bias=Bz)

    def score(self, F: np.ndarray) -> np.ndarray:
        """(B, H) per-head probabilities for a (B, H, 4) feature tensor."""
        F = np.asarray(F, dtype=np.float32)
        if F.ndim != 3 or F.shape[1:] != (self.n_heads, 4):
            raise ValueError(
                f"F must be (B, {self.n_heads}, 4), got {F.shape}"
            )
        # z[b, h] = F[b, h, :] · W[h] + bias[h]
        z = np.einsum("bhd,hd->bh", F, self.weights) + self.bias.reshape(1, -1)
        return _sigmoid(z.astype(np.float32))

    def score_blocks(self, F: np.ndarray, reduce: str = "max") -> np.ndarray:
        """Aggregate per-head scores into one (B,) block score (max or mean)."""
        ph = self.score(F)  # (B, H)
        if reduce == "max":
            return ph.max(axis=1)
        if reduce == "mean":
            return ph.mean(axis=1)
        raise ValueError(f"reduce must be 'max' or 'mean', got {reduce!r}")

    def n_params(self) -> int:
        return int(self.weights.size + self.bias.size)
