"""Bayesian posterior over closed-form weights — addresses 100-round R52.

For the closed-form predictor with 4 weights + bias, the Laplace
approximation gives a Gaussian posterior centered at the MAP estimate
with covariance = inverse Hessian. We expose this to enable Bayesian
credible intervals on predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .predictor import ClosedFormXQP, _sigmoid


@dataclass
class BayesianClosedFormXQP:
    """Wraps ClosedFormXQP with Laplace-approximation posterior."""
    predictor: ClosedFormXQP
    posterior_mean: np.ndarray = field(default=None)
    posterior_cov: np.ndarray = field(default=None)

    @classmethod
    def from_fit(cls, F: np.ndarray, y: np.ndarray, l2: float = 1e-3
                 ) -> "BayesianClosedFormXQP":
        pred = ClosedFormXQP.from_fit(F, y, l2=l2)
        F = np.asarray(F, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        N, D = F.shape
        Fa = np.concatenate([F, np.ones((N, 1), dtype=np.float32)], axis=1)
        # MAP
        wa = np.concatenate([pred.weights, [float(pred.bias)]])
        z = Fa @ wa
        p = _sigmoid(z)
        # Hessian (with L2)
        W = p * (1 - p)
        I = np.eye(D + 1, dtype=np.float32) * l2
        I[D, D] = 0.0
        H = (Fa.T * W) @ Fa / N + I / N
        cov = np.linalg.inv(H + 1e-6 * np.eye(D + 1))
        return cls(predictor=pred,
                   posterior_mean=wa.astype(np.float32),
                   posterior_cov=cov.astype(np.float32))

    def predictive_distribution(self, F: np.ndarray, n_samples: int = 100
                                 ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean_score, std_score) under the posterior."""
        rng = np.random.default_rng(0)
        L = np.linalg.cholesky(self.posterior_cov + 1e-6 * np.eye(self.posterior_cov.shape[0]))
        samples = self.posterior_mean[None, :] + (L @ rng.normal(size=(L.shape[0], n_samples))).T
        Fa = np.concatenate([F, np.ones((F.shape[0], 1), dtype=np.float32)], axis=1)
        scores = _sigmoid(Fa @ samples.T.astype(np.float32))
        return scores.mean(axis=1), scores.std(axis=1)
