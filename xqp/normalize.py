"""Cross-model feature normalization — addresses 100-round R11 / R20.

Different model families produce different attention-score scales (e.g.
GQA vs MHA, varying head counts). To make the closed-form weights
transferable, we normalize features to a canonical scale before fitting.

Approach: compute z-score-then-clip per feature on a calibration set,
store the normalizer, apply at inference.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class FeatureNormalizer:
    """Per-feature z-score + clip to [0, 1] range."""
    mean: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    std: np.ndarray = field(default_factory=lambda: np.ones(4, dtype=np.float32))
    clip_z: float = 3.0

    @classmethod
    def from_calibration(cls, F_calib: np.ndarray, clip_z: float = 3.0) -> "FeatureNormalizer":
        F = np.asarray(F_calib, dtype=np.float32)
        if F.ndim != 2 or F.shape[0] == 0:
            raise ValueError(
                f"calibration set must be a non-empty (N, D) array, got shape {F.shape}"
            )
        return cls(
            mean=F.mean(axis=0).astype(np.float32),
            std=(F.std(axis=0) + 1e-6).astype(np.float32),
            clip_z=clip_z,
        )

    def transform(self, F: np.ndarray) -> np.ndarray:
        z = (F - self.mean) / self.std
        z = np.clip(z, -self.clip_z, self.clip_z)
        # rescale to [0, 1] for compatibility with closed-form Bayes setup
        return ((z + self.clip_z) / (2 * self.clip_z)).astype(np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "clip_z": float(self.clip_z),
        }))

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        obj = json.loads(Path(path).read_text())
        return cls(
            mean=np.asarray(obj["mean"], dtype=np.float32),
            std=np.asarray(obj["std"], dtype=np.float32),
            clip_z=float(obj["clip_z"]),
        )
