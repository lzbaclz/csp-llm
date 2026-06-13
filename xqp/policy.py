"""XQP policy.

Implements the KVPolicy.select_to_keep(block_stats, budget, step) contract
that the upstream SEER repo already exposes (seer/policy/base.py).

The policy:
  1. Builds the 4-dim feature matrix via xqp.features.extract_features.
  2. Scores each block with the chosen predictor (closed-form or MLP).
  3. Forces sink (first S blocks) + sliding-window (last W blocks).
  4. Greedy-fills the remaining budget by descending score.
  5. Returns the set of indices to keep on HBM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .features import extract_features
from .predictor import ClosedFormXQP, TinyMLPXQP


@dataclass
class BlockStats:
    """Snapshot of all blocks at a given decode step.

    Field shapes:
        ema_within     (B,)        attention EMA, current layer
        ema_prev_layer (B,) | None  attention EMA, previous layer (None at l=0)
        K_layer        (B, d_head)  key vectors for current layer
        q_prev         (d_head,)    last decoded token's query vector
        last_used      (B,)         step index of last access
        layer          int          current layer index (for per-layer scoring)
    """
    ema_within: np.ndarray
    K_layer: np.ndarray
    q_prev: np.ndarray
    last_used: np.ndarray
    ema_prev_layer: Optional[np.ndarray] = None
    layer: int = 0


@dataclass
class XQPPolicy:
    predictor: object  # ClosedFormXQP or TinyMLPXQP
    n_sink: int = 4
    n_window: int = 4
    r_cross: float = 0.10
    w_recency: float = 64.0
    # multi-horizon: for TinyMLP, which horizon head to use (1, 4, 16, 64)
    horizon_idx: int = 1
    # safe-fallback threshold; if rolling max(score) < this, revert to recency-only.
    # NOTE (round-1 review m3): the window is **per-layer**, so the effective
    # rolling window in step count is fallback_window_per_layer. With L=80 and
    # window=100, this means the fallback reacts within roughly 100 decode
    # steps of layer 0's view, not 100/80 ≈ 1 step.
    fallback_p_threshold: float = 0.4
    fallback_window_per_layer: int = 100
    # round-2 MR2: recovery_tau controls how fast the policy *exits* fallback.
    # The decision statistic is an EMA of max(p) with smoothing 2/(tau+1);
    # a small tau (default 16) recovers within ~tau layer-steps instead of
    # waiting a full fallback_window_per_layer. Set recovery_tau ==
    # fallback_window_per_layer to recover the original slow hard-window behavior.
    recovery_tau: int = 16
    _per_layer_history: dict = field(default_factory=dict)
    _per_layer_ema: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.predictor, (ClosedFormXQP, TinyMLPXQP)):
            raise TypeError("predictor must be ClosedFormXQP or TinyMLPXQP")

    def select_to_keep(self, stats: BlockStats, budget: int, step: int) -> set[int]:
        """Return the set of block indices to keep on HBM."""
        B = stats.ema_within.shape[0]
        if budget >= B:
            return set(range(B))
        if budget <= self.n_sink + self.n_window:
            # Edge case: budget too small to even fit sink+window; honor sink first.
            return set(range(min(self.n_sink, budget)))

        F = extract_features(
            ema_within=stats.ema_within,
            ema_prev_layer=stats.ema_prev_layer,
            K_layer=stats.K_layer,
            q_prev=stats.q_prev,
            step=step,
            last_used=stats.last_used,
            r_cross=self.r_cross,
            w_recency=self.w_recency,
        )

        # Score
        if isinstance(self.predictor, ClosedFormXQP):
            scores = self.predictor.score(
                F, layer=(stats.layer if self.predictor.per_layer else None)
            )
        else:
            scores = self.predictor.score(F, horizon_idx=self.horizon_idx)

        # Fallback (round-1 m3 + round-2 MR2): per-layer rolling statistic of
        # max score. We keep a hard window for diagnostics (back-compat) and an
        # EMA for the *decision*, so recovery is governed by recovery_tau rather
        # than the full window length.
        layer_key = int(stats.layer)
        cur_max = float(scores.max())
        hist = self._per_layer_history.setdefault(layer_key, [])
        hist.append(cur_max)
        if len(hist) > self.fallback_window_per_layer:
            hist.pop(0)
        # EMA of max(p) with timescale recovery_tau (round-2 MR2 fast recovery).
        tau = max(1, int(self.recovery_tau))
        alpha = 2.0 / (tau + 1.0)
        ema_prev = self._per_layer_ema.get(layer_key)
        ema_max = cur_max if ema_prev is None else (alpha * cur_max + (1 - alpha) * ema_prev)
        self._per_layer_ema[layer_key] = ema_max
        # Arm only after a full window of warmup (unchanged from round-1); the
        # decision then tracks the EMA, which both enters and exits fallback
        # within ~recovery_tau steps instead of the full window.
        if len(hist) >= self.fallback_window_per_layer and ema_max < self.fallback_p_threshold:
            scores = F[:, 3]  # fallback to recency-only

        # Force-keep sink + sliding window
        keep = set(range(min(self.n_sink, B)))
        win_lo = max(0, B - self.n_window)
        keep |= set(range(win_lo, B))

        # Greedy-fill remaining budget
        remaining = budget - len(keep)
        if remaining <= 0:
            return keep
        eligible = [i for i in range(B) if i not in keep]
        eligible.sort(key=lambda i: -float(scores[i]))
        keep |= set(eligible[:remaining])
        return keep

    def score_only(self, stats: BlockStats, step: int) -> np.ndarray:
        """Just return scores — useful for AUC eval without committing to a budget."""
        F = extract_features(
            ema_within=stats.ema_within,
            ema_prev_layer=stats.ema_prev_layer,
            K_layer=stats.K_layer,
            q_prev=stats.q_prev,
            step=step,
            last_used=stats.last_used,
            r_cross=self.r_cross,
            w_recency=self.w_recency,
        )
        if isinstance(self.predictor, ClosedFormXQP):
            return self.predictor.score(
                F, layer=(stats.layer if self.predictor.per_layer else None)
            )
        return self.predictor.score(F, horizon_idx=self.horizon_idx)
