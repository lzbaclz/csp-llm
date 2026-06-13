"""HuggingFace transformers integration — addresses 100-round R72.

A thin wrapper that subclasses `transformers.cache_utils.Cache` and
delegates the per-step hot-set decision to XQPPolicy.

This is a *scaffold* — the actual HF attention path injection requires
a forward-hook on each transformer block. Tests exercise the metadata
path; full HF integration requires `pip install transformers` and a real
GPU.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .policy import BlockStats, XQPPolicy


class HFXQPCacheAdapter:
    """Adapter object that the model-side hook calls per layer per step."""

    def __init__(self, policy: XQPPolicy, n_layers: int, n_kv_heads: int,
                 head_dim: int, block_size: int = 32, budget_per_layer: int = 1024):
        self.policy = policy
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.budget = budget_per_layer
        # per (layer, block) hotness EMA — populated by the model hook
        self._ema: dict = {}

    def update_ema(self, layer: int, attn_weights: np.ndarray) -> None:
        """Called per attention call with (B,) per-block weights."""
        prev = self._ema.get(layer)
        a = np.asarray(attn_weights, dtype=np.float32)
        if prev is None or prev.shape != a.shape:
            self._ema[layer] = a
        else:
            self._ema[layer] = 0.7 * prev + 0.3 * a

    def step(self, layer: int, K_layer: np.ndarray, q_prev: np.ndarray,
             last_used: np.ndarray, step_id: int) -> set:
        """Return set of block indices to keep on HBM at this (layer, step)."""
        ema = self._ema.get(layer)
        if ema is None:
            return set(range(K_layer.shape[0]))
        ema_prev = self._ema.get(layer - 1) if layer > 0 else None
        stats = BlockStats(
            ema_within=ema,
            ema_prev_layer=ema_prev,
            K_layer=K_layer,
            q_prev=q_prev,
            last_used=last_used,
            layer=layer,
        )
        return self.policy.select_to_keep(stats, budget=self.budget, step=step_id)
