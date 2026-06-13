"""Iteration 5 — Online distillation against live attention.

A small ring buffer of recent (features, observed top-r label) pairs.
Every N steps we run a single Newton (IRLS) step on the closed-form
weights, adapting to distribution shift.

Cost analysis: 1 IRLS step on N=1024 samples of 4 features = ~30µs CPU
(matrix-mat-mul 4×1024 plus 4×4 solve), well below the per-token budget.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..predictor import ClosedFormXQP, _sigmoid


@dataclass
class OnlineXQP:
    """Wraps ClosedFormXQP with an online update buffer."""
    predictor: ClosedFormXQP
    buffer_size: int = 1024
    update_every: int = 32
    learning_rate: float = 0.5   # damping on the IRLS step
    l2: float = 1e-3
    _buf_F: deque = field(default=None)
    _buf_y: deque = field(default=None)
    _step: int = 0

    def __post_init__(self):
        if self._buf_F is None:
            self._buf_F = deque(maxlen=self.buffer_size)
        if self._buf_y is None:
            self._buf_y = deque(maxlen=self.buffer_size)

    def observe(self, F_batch: np.ndarray, y_batch: np.ndarray) -> None:
        """Buffer one batch of (feature, label) pairs."""
        F_batch = np.asarray(F_batch, dtype=np.float32)
        y_batch = np.asarray(y_batch, dtype=np.float32).reshape(-1)
        for i in range(F_batch.shape[0]):
            self._buf_F.append(F_batch[i].copy())
            self._buf_y.append(float(y_batch[i]))
        self._step += 1
        if self._step % self.update_every == 0 and len(self._buf_F) >= 32:
            self._update()

    def _update(self) -> None:
        """One damped Newton step (no full IRLS convergence)."""
        if self.predictor.per_layer:
            return  # online update for per-layer model is more complex; future work
        F = np.stack(list(self._buf_F), axis=0).astype(np.float32)
        y = np.asarray(list(self._buf_y), dtype=np.float32)
        N = F.shape[0]
        w = self.predictor.weights
        b = float(self.predictor.bias)
        z = F @ w + b
        p = _sigmoid(z)
        # gradient
        Fa = np.concatenate([F, np.ones((N, 1), dtype=np.float32)], axis=1)
        wa = np.concatenate([w, [b]])
        grad = Fa.T @ (p - y) / N + self.l2 * wa / N
        # Newton step with damping
        W = p * (1 - p)
        H = (Fa.T * W) @ Fa / N + self.l2 * np.eye(Fa.shape[1]) / N
        try:
            step = np.linalg.solve(H + 1e-6 * np.eye(Fa.shape[1]), grad)
        except np.linalg.LinAlgError:
            step = grad
        wa_new = wa - self.learning_rate * step
        self.predictor.weights = wa_new[:-1].astype(np.float32)
        self.predictor.bias = np.float32(wa_new[-1])

    def score(self, F: np.ndarray) -> np.ndarray:
        return self.predictor.score(F)

    def n_buffered(self) -> int:
        return len(self._buf_F)
