"""SOTA-driven iterations for XQP.

Each module targets a specific weakness of the base predictor relative to a
named prior method, and is kept *self-contained* (it does not mutate the core
`features`/`predictor`/`policy` modules) so the base 4-weight closed form and
its saved weights stay stable. See ITERATIONS.md for the problem/architecture/
expected-gain writeup behind each one.

  iter1 — Quest page-min/max bounds         (Quest, ICML'24)
  iter2 — Locret-style mass distillation     (Locret, ICLR'25)
  iter3 — Continuous cross-layer signal      (InfiniGen, OSDI'24)
  iter4 — Per-head feature aggregation       (DoubleSparse / SqueezeAttention)
  iter5 — Online distillation under shift    (offline baselines: Locret, DoubleSparse)
"""
from .iter1_quest_bound import (
    cosine_query_key_per_token,
    query_max_min_per_block,
    quest_bound_features,
)
from .iter2_distill import (
    MassDistillationLoss,
    attention_mass_from_softmax,
)
from .iter3_continuous_cross import (
    continuous_cross_signal,
    make_continuous_features,
    truncated_gaussian_coefficient,
)
from .iter4_per_head import (
    per_head_features,
    PerHeadClosedFormXQP,
)
from .iter5_online import OnlineXQP

__all__ = [
    # iter1
    "cosine_query_key_per_token",
    "query_max_min_per_block",
    "quest_bound_features",
    # iter2
    "MassDistillationLoss",
    "attention_mass_from_softmax",
    # iter3
    "continuous_cross_signal",
    "make_continuous_features",
    "truncated_gaussian_coefficient",
    # iter4
    "per_head_features",
    "PerHeadClosedFormXQP",
    # iter5
    "OnlineXQP",
]
