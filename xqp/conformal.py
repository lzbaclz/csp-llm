"""Adaptive conformal saliency sets under drift — the methodological novelty.

Instead of emitting a point score per block, we emit a *set* of blocks with a
distribution-free guarantee on the salient-block MISS RATE that holds under
*arbitrary* concept drift, via Adaptive Conformal Inference (ACI; Gibbs &
Candès, 2021). The downstream KV system cares about exactly this quantity:
bound the probability of dropping a truly-salient block, even as the attention
distribution drifts during decoding.

Mechanism. Maintain an inclusion threshold tau_t. The retained set is
S_t = {b : score(b) >= tau_t}. After the true labels for step t are revealed,
the realized *miss rate* err_t = (salient blocks excluded) / (salient blocks)
drives an online update
    tau_{t+1} = clip(tau_t + gamma * (alpha - err_t), 0, 1),
which lowers the threshold (keeps more) when we miss too much and raises it
(keeps fewer, more efficient) when we over-cover. Under this update the long-run
average miss rate converges to the target alpha regardless of how the
score<->label relationship drifts — no distributional assumption. We report the
realized miss rate (should track alpha) and the average set size |S_t| (the
efficiency; smaller = more eviction headroom).

Any object with a ``score(F) -> [0,1]`` method works as the base scorer
(ClosedFormXQP, PairwiseXQP, a baseline, ...), so the guarantee composes with
the low-redundancy fusion of the rest of the package.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AdaptiveConformalSaliency:
    """ACI wrapper turning a per-block scorer into a drift-robust salient set."""
    scorer: object             # must expose .score(F) -> probabilities in [0,1]
    alpha: float = 0.10        # target miss rate (drop <= alpha of salient blocks)
    gamma: float = 0.05        # adaptation rate
    tau: float = 0.5           # current inclusion threshold
    _errs: list = field(default_factory=list)
    _sizes: list = field(default_factory=list)

    def _scores(self, F: np.ndarray) -> np.ndarray:
        return np.asarray(self.scorer.score(np.asarray(F, dtype=np.float32))).reshape(-1)

    def select(self, F: np.ndarray):
        """Return (keep_mask, scores) for the current threshold."""
        s = self._scores(F)
        return (s >= self.tau), s

    def observe(self, F: np.ndarray, y_true: np.ndarray) -> float:
        """Reveal labels for one step: record miss rate + set size, update tau.

        Returns this step's realized miss rate.
        """
        s = self._scores(F)
        y = np.asarray(y_true).reshape(-1)
        keep = s >= self.tau
        self._sizes.append(float(keep.mean()))
        pos = y > 0.5
        npos = int(pos.sum())
        err = 0.0 if npos == 0 else float((pos & (~keep)).sum()) / npos
        self._errs.append(err)
        # ACI update: keep more when missing (err>alpha), fewer when over-covering
        self.tau = float(np.clip(self.tau + self.gamma * (self.alpha - err), 0.0, 1.0))
        return err

    def realized_miss_rate(self) -> float:
        return float(np.mean(self._errs)) if self._errs else float("nan")

    def avg_set_size(self) -> float:
        return float(np.mean(self._sizes)) if self._sizes else float("nan")


def run_conformal_stream(scorer, stream, *, alpha=0.10, gamma=0.05,
                         adaptive=True, tau0=0.5) -> dict:
    """Process a temporally ordered stream of (F_t, y_t) steps.

    `adaptive=False` freezes tau at tau0 (the fixed-threshold baseline that
    loses coverage under drift). Returns realized miss rate, target, avg set
    size, and the per-step miss trajectory.
    """
    aci = AdaptiveConformalSaliency(scorer=scorer, alpha=alpha,
                                    gamma=(gamma if adaptive else 0.0), tau=tau0)
    traj = [aci.observe(F, y) for (F, y) in stream]
    return dict(adaptive=adaptive, alpha=alpha,
                realized_miss_rate=aci.realized_miss_rate(),
                avg_set_size=aci.avg_set_size(),
                final_tau=aci.tau, n_steps=len(traj),
                miss_trajectory=traj)
