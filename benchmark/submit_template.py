"""KVSalienceBench submission template.

Copy this file, implement `score`, and evaluate with:
    python benchmark/run_leaderboard.py --submission your_method.py \
        --traces '/public/xqp_traces/*.jsonl'

CONTRACT
--------
`score(F)` receives an (N, 4) float32 feature matrix whose columns are, in order,
benchmark.protocol.FEATURE_COLUMNS = ("s_within", "s_cross", "s_query", "s_pos"):
  s_within : current-layer attention-magnitude EMA, max-normalized to [0,1]
  s_cross  : previous-layer top-r membership indicator in {0,1}
  s_query  : Quest-style query-key affinity proxy (mean-pooled cosine)
  s_pos    : recency, exp(-(t - last_used)/w)
and must return an (N,) array of probabilities in [0,1] that each block is in the
top-10% most-attended set 4 decode steps later (the headline horizon h4).

You may train on held-out-request TRAIN prompts; the protocol evaluates you on
disjoint held-out-request TEST prompts (benchmark.protocol.request_split). The
reference baseline to beat is the 3-parameter within+cross calibrated logistic
model (benchmark/reference_model.json) — it ties gradient-boosted trees on
ranking on long-context workloads while being far better calibrated. Beating it
materially on AUPRC AND calibration (ECE), especially on short-context workloads,
is the open challenge.
"""
from __future__ import annotations
import numpy as np


def score(F: np.ndarray) -> np.ndarray:
    """REPLACE THIS. The trivial example below just returns the within-EMA view
    (a strong single-signal baseline). A real submission fits/loads a model."""
    F = np.asarray(F, np.float32)
    s_within = F[:, 0]
    return s_within  # probabilities need not be calibrated for ranking, but ECE counts
