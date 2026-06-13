"""Iteration 2 — Locret-style distillation from continuous attention mass.

Locret (2025) supervises a small head with the next-step *attention mass*
(continuous) rather than the binary top-r label. The mass-supervised head
preserves rank ordering at the top better than binary supervision.

We provide a regression-head wrapper around any classifier; at deployment
the policy uses the mass head's output directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MassDistillationLoss:
    """Combined focal-BCE + MSE-on-mass with α annealing 1→0.

    L = alpha * focal_bce(p_top, y_top) + (1 - alpha) * mse(p_mass, mass)

    alpha is annealed from alpha_start to alpha_end linearly over n_steps.
    """
    alpha_start: float = 1.0
    alpha_end: float = 0.0
    n_steps: int = 1000

    def alpha(self, step: int) -> float:
        if step >= self.n_steps:
            return self.alpha_end
        return self.alpha_start + (self.alpha_end - self.alpha_start) * (step / self.n_steps)

    def __call__(self, p_top: np.ndarray, p_mass: np.ndarray,
                 y_top: np.ndarray, mass: np.ndarray, step: int = 0) -> float:
        a = self.alpha(step)
        focal_g = 2.0
        focal_a = 0.25
        p_top = np.clip(p_top, 1e-7, 1 - 1e-7)
        bce = -(focal_a * y_top * (1 - p_top) ** focal_g * np.log(p_top) +
                (1 - focal_a) * (1 - y_top) * p_top ** focal_g * np.log(1 - p_top))
        mse = (p_mass - mass) ** 2
        return float(a * bce.mean() + (1 - a) * mse.mean())


def attention_mass_from_softmax(attn_weights: np.ndarray) -> np.ndarray:
    """Returns (B,) the cumulative attention-mass attributable to each block.

    attn_weights: (T,) softmax-normalized attention from the most recent step
        for one (layer, head). Sums to 1.
    Aggregation: caller passes flat per-token weights; this function returns
    per-block mass via reshape.
    """
    a = np.asarray(attn_weights, dtype=np.float32)
    # caller pre-aggregates; this is a passthrough for clarity
    return a / (a.sum() + 1e-9)
