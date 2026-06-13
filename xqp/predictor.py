"""XQP predictors.

Two implementations:
- ClosedFormXQP: logistic regression with 4 weights (+ bias). The single
  weight vector is shared across layers; an optional per-layer variant
  has 4 weights per layer (still <1 KB total for L=80).
- TinyMLPXQP: 4 → 16 (GELU) → 4 logits (multi-horizon).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from .features import FEATURE_DIM


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _gelu(x: np.ndarray) -> np.ndarray:
    # tanh-approx GELU; ONNX-friendly and TRT-friendly
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    """d/dx of the tanh-approx GELU (for the TinyMLP backward pass)."""
    c = math.sqrt(2.0 / math.pi)
    u = c * (x + 0.044715 * x ** 3)
    th = np.tanh(u)
    du = c * (1.0 + 3 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th ** 2) * du


@dataclass
class ClosedFormXQP:
    """4-weight logistic regression; optional per-layer.

    weights: (4,) shared, or (L, 4) per-layer.
    bias:    scalar shared, or (L,) per-layer.
    """
    weights: np.ndarray = field(default_factory=lambda: np.zeros(FEATURE_DIM, dtype=np.float32))
    bias: np.ndarray = field(default_factory=lambda: np.zeros((), dtype=np.float32))
    per_layer: bool = False

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=np.float32)
        self.bias = np.asarray(self.bias, dtype=np.float32)

    @classmethod
    def from_fit(cls, F: np.ndarray, y: np.ndarray, l2: float = 1e-3,
                 layer_ids: Optional[np.ndarray] = None,
                 per_layer: bool = False) -> "ClosedFormXQP":
        """Fit weights via IRLS / Newton with L2 regularization.

        Args:
            F: (N, 4) feature matrix
            y: (N,) {0, 1} labels (= block in top-r at next step)
            l2: ridge strength
            layer_ids: (N,) layer index for per_layer fits
            per_layer: whether to fit per-layer weights
        """
        F = np.asarray(F, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if F.shape[0] != y.shape[0] or F.shape[1] != FEATURE_DIM:
            raise ValueError(f"bad shapes F={F.shape}, y={y.shape}")
        if per_layer:
            if layer_ids is None:
                raise ValueError("layer_ids required for per_layer=True")
            layer_ids = np.asarray(layer_ids, dtype=np.int64)
            L = int(layer_ids.max()) + 1
            W = np.zeros((L, FEATURE_DIM), dtype=np.float32)
            B = np.zeros((L,), dtype=np.float32)
            for l in range(L):
                mask = layer_ids == l
                if mask.sum() < 8:
                    continue
                w, b = cls._fit_single(F[mask], y[mask], l2=l2)
                W[l] = w
                B[l] = b
            return cls(weights=W, bias=B, per_layer=True)
        w, b = cls._fit_single(F, y, l2=l2)
        return cls(weights=w.astype(np.float32),
                   bias=np.float32(b), per_layer=False)

    @staticmethod
    def _fit_single(F: np.ndarray, y: np.ndarray, l2: float = 1e-3,
                    max_iter: int = 40, tol: float = 1e-6,
                    class_weight: str | None = None):
        """IRLS / Newton-Raphson with L2 ridge.

        100-round R41 (Meta FAIR): added `class_weight="balanced"` so the
        ~90% negative class doesn't dominate when labels are top-10%.
        """
        N, D = F.shape
        if N == 0:
            raise ValueError("cannot fit XQP on an empty dataset (N=0)")
        if class_weight == "balanced":
            n_pos = max(1.0, float(y.sum()))
            n_neg = max(1.0, N - n_pos)
            sample_w = np.where(y > 0.5, N / (2 * n_pos), N / (2 * n_neg)).astype(np.float32)
        else:
            sample_w = np.ones(N, dtype=np.float32)
        # Augment with bias column
        Fa = np.concatenate([F, np.ones((N, 1), dtype=np.float32)], axis=1)  # (N, D+1)
        w = np.zeros(D + 1, dtype=np.float32)
        I = np.eye(D + 1, dtype=np.float32) * l2
        I[D, D] = 0.0  # don't regularize bias
        prev_loss = float("inf")
        for it in range(max_iter):
            z = Fa @ w
            p = _sigmoid(z)
            # weighted negative log-likelihood
            loss = -float(np.mean(sample_w *
                                   (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))))
            loss += 0.5 * l2 * float(w[:D] @ w[:D]) / N
            grad = Fa.T @ (sample_w * (p - y)) / N + (I @ w) / N
            # Hessian
            W = sample_w * p * (1 - p)
            H = (Fa.T * W) @ Fa / N + I / N
            try:
                step = np.linalg.solve(H + 1e-6 * np.eye(D + 1), grad)
            except np.linalg.LinAlgError:
                step = grad
            w_new = w - step
            if abs(prev_loss - loss) < tol:
                w = w_new
                break
            prev_loss = loss
            w = w_new
        return w[:D].astype(np.float32), float(w[D])

    def score(self, F: np.ndarray, layer: Optional[int] = None) -> np.ndarray:
        """Return (B,) probabilities."""
        if self.per_layer:
            if layer is None:
                raise ValueError("layer required when per_layer=True")
            z = F @ self.weights[layer] + self.bias[layer]
        else:
            z = F @ self.weights + self.bias
        return _sigmoid(z.astype(np.float32))

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps({
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist() if self.bias.ndim else float(self.bias),
            "per_layer": self.per_layer,
        }))

    @classmethod
    def load(cls, path: str | Path) -> "ClosedFormXQP":
        obj = json.loads(Path(path).read_text())
        return cls(
            weights=np.asarray(obj["weights"], dtype=np.float32),
            bias=np.asarray(obj["bias"], dtype=np.float32),
            per_layer=bool(obj["per_layer"]),
        )


@dataclass
class PairwiseXQP:
    """Round-2 MR1: closed-form with pairwise interactions.

    Score = sigmoid(w·f + f·M·f + b), where M is a 4×4 symmetric matrix.
    Free parameters: 4 + 10 + 1 = 15. Still well under 1 KB; deployment
    envelope unchanged. Fit by IRLS in the augmented feature space
    [f_i, f_i*f_j for i<=j].
    """
    w: np.ndarray = field(default_factory=lambda: np.zeros(FEATURE_DIM, dtype=np.float32))
    M: np.ndarray = field(default_factory=lambda: np.zeros((FEATURE_DIM, FEATURE_DIM), dtype=np.float32))
    bias: float = 0.0

    @staticmethod
    def _augment(F: np.ndarray) -> np.ndarray:
        """Concatenate [F, F_i*F_j for i<=j] → (N, 4 + 10) features."""
        N, D = F.shape
        pairs = []
        for i in range(D):
            for j in range(i, D):
                pairs.append(F[:, i] * F[:, j])
        aug = np.concatenate([F, np.stack(pairs, axis=1)], axis=1)
        return aug.astype(np.float32)

    @classmethod
    def from_fit(cls, F: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> "PairwiseXQP":
        """Fit via IRLS in the augmented feature space."""
        F = np.asarray(F, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        aug = cls._augment(F)
        # reuse ClosedFormXQP._fit_single
        w_aug, b = ClosedFormXQP._fit_single(aug, y, l2=l2)
        # split back
        D = F.shape[1]
        w_lin = w_aug[:D].astype(np.float32)
        w_int = w_aug[D:].astype(np.float32)
        M = np.zeros((D, D), dtype=np.float32)
        k = 0
        for i in range(D):
            for j in range(i, D):
                if i == j:
                    M[i, j] = w_int[k]
                else:
                    M[i, j] = M[j, i] = 0.5 * w_int[k]
                k += 1
        return cls(w=w_lin, M=M, bias=float(b))

    def score(self, F: np.ndarray) -> np.ndarray:
        F = np.asarray(F, dtype=np.float32)
        # quadratic term: f·M·f computed via (FM)·f sum
        FM = F @ self.M
        quad = (FM * F).sum(axis=1)
        z = F @ self.w + quad + self.bias
        return _sigmoid(z)

    def n_params(self) -> int:
        # 4 linear + 10 unique entries in symmetric 4x4 + 1 bias = 15
        return 4 + 10 + 1

    def project_psd(self, jitter: float = 1e-6) -> None:
        """100-round R05 (Princeton): project M onto the PSD cone so that
        f·M·f >= 0, keeping the score bounded below. Spectral
        decomposition + non-negative-clip + symmetrize."""
        eigvals, eigvecs = np.linalg.eigh(self.M.astype(np.float64))
        eigvals = np.maximum(eigvals, jitter)
        M_psd = (eigvecs * eigvals) @ eigvecs.T
        M_psd = 0.5 * (M_psd + M_psd.T)
        self.M = M_psd.astype(np.float32)


@dataclass
class TinyMLPXQP:
    """4 → 16 (GELU) → H heads logits.

    Kept tiny so TRT export + CUDA-Graph capture lands in 30 µs P99.9 envelope.
    Multi-horizon: H=4 heads correspond to horizons {1, 4, 16, 64}.
    Total params: 4*16 + 16 + 16*4 + 4 = 148.
    """
    W1: np.ndarray = field(default_factory=lambda: np.zeros((FEATURE_DIM, 16), dtype=np.float32))
    b1: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=np.float32))
    W2: np.ndarray = field(default_factory=lambda: np.zeros((16, 4), dtype=np.float32))
    b2: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    def score(self, F: np.ndarray, horizon_idx: int = 0) -> np.ndarray:
        h = _gelu(F @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        return _sigmoid(z[:, horizon_idx])

    @classmethod
    def random_init(cls, seed: int = 0) -> "TinyMLPXQP":
        rng = np.random.default_rng(seed)
        return cls(
            W1=(rng.normal(0, 0.5, (FEATURE_DIM, 16)).astype(np.float32)),
            b1=np.zeros(16, dtype=np.float32),
            W2=(rng.normal(0, 0.5, (16, 4)).astype(np.float32)),
            b2=np.zeros(4, dtype=np.float32),
        )

    def n_params(self) -> int:
        return sum(int(np.prod(x.shape)) for x in (self.W1, self.b1, self.W2, self.b2))

    @classmethod
    def from_fit(cls, F: np.ndarray, y: np.ndarray, *, epochs: int = 300,
                 lr: float = 0.05, l2: float = 1e-4, seed: int = 0,
                 balanced: bool = True) -> "TinyMLPXQP":
        """Train the 4->16->4 net by full-batch Adam on BCE.

        Makes the deployment MLP variant a *real* trained model (it was
        random-init only). `y` may be (N,) shared across the 4 horizon heads or
        (N, 4) per-horizon. Class imbalance is handled with balanced per-column
        sample weights.
        """
        F = np.asarray(F, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        N = F.shape[0]
        if N == 0:
            raise ValueError("cannot fit TinyMLPXQP on an empty dataset (N=0)")
        Y = np.repeat(y.reshape(N, 1), 4, axis=1) if y.ndim == 1 else y
        if Y.shape != (N, 4):
            raise ValueError(f"y must be (N,) or (N, 4), got {y.shape}")
        # balanced per-column sample weights
        if balanced:
            sw = np.ones((N, 4), dtype=np.float32)
            for c in range(4):
                npos = max(1.0, float((Y[:, c] > 0.5).sum()))
                nneg = max(1.0, N - npos)
                sw[:, c] = np.where(Y[:, c] > 0.5, N / (2 * npos), N / (2 * nneg))
        else:
            sw = np.ones((N, 4), dtype=np.float32)
        m = cls.random_init(seed)
        params = {"W1": m.W1, "b1": m.b1, "W2": m.W2, "b2": m.b2}
        adam = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in params.items()}
        b1_, b2_, eps = 0.9, 0.999, 1e-8
        for t in range(1, epochs + 1):
            z1 = F @ params["W1"] + params["b1"]
            h = _gelu(z1)
            z2 = h @ params["W2"] + params["b2"]
            p = _sigmoid(z2)
            dz2 = (sw * (p - Y)) / N                       # (N, 4)
            grads = {
                "W2": h.T @ dz2 + l2 * params["W2"],
                "b2": dz2.sum(0),
                "W1": F.T @ ((dz2 @ params["W2"].T) * _gelu_grad(z1)) + l2 * params["W1"],
                "b1": ((dz2 @ params["W2"].T) * _gelu_grad(z1)).sum(0),
            }
            for k in params:
                mt, vt = adam[k]
                mt = b1_ * mt + (1 - b1_) * grads[k]
                vt = b2_ * vt + (1 - b2_) * grads[k] ** 2
                adam[k] = (mt, vt)
                mhat = mt / (1 - b1_ ** t)
                vhat = vt / (1 - b2_ ** t)
                params[k] = params[k] - lr * mhat / (np.sqrt(vhat) + eps)
        return cls(W1=params["W1"].astype(np.float32), b1=params["b1"].astype(np.float32),
                   W2=params["W2"].astype(np.float32), b2=params["b2"].astype(np.float32))
