"""Tests for the regime-gated selective cascade and the unique-information measure."""
import numpy as np
import pytest

from xqp.gated_predictor import SelectiveCascadeXQP, _gather4
from xqp.predictor import ClosedFormXQP
from xqp.info_theory import conditional_mi_view, unique_information_report


def _synthetic(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    within = rng.random(n).astype(np.float32)
    cross = (rng.random(n) < 0.3).astype(np.float32)
    query = rng.random(n).astype(np.float32)
    recency = rng.random(n).astype(np.float32)
    # label driven by within + cross (magnitude) AND query only when within is low
    logit = 3 * within + 1.5 * cross + 2.0 * query * (within < 0.3)
    y = (rng.random(n) < 1 / (1 + np.exp(-(logit - 2)))).astype(np.float32)
    F = np.stack([within, cross, query, recency], 1)
    return F, y


def test_gather4_layout():
    F = np.arange(12, dtype=np.float32).reshape(3, 4)
    g = _gather4(F, (0, 1, 3))
    assert np.allclose(g[:, 0], F[:, 0]) and np.allclose(g[:, 1], F[:, 1])
    assert np.allclose(g[:, 2], F[:, 3])      # third source col lands in slot 2
    assert np.allclose(g[:, 3], 0.0)


def test_budget_endpoints_and_monotonic_cost():
    F, y = _synthetic()
    casc = SelectiveCascadeXQP.from_fit(F, y, base_cols=(0, 1), expert_cols=(0, 1, 2), query_col=2)
    # budget 0 == base only (no query computed)
    p0, defer0, frac0 = casc.predict_with_cost(F, budget=0.0)
    assert frac0 == 0.0
    assert np.allclose(p0, casc.base.score(_gather4(F, (0, 1))))
    # budget 1 == expert everywhere
    p1, defer1, frac1 = casc.predict_with_cost(F, budget=1.0)
    assert frac1 == 1.0
    assert np.allclose(p1, casc.expert.score(_gather4(F, (0, 1, 2))))
    # intermediate budget routes ~that fraction
    _, _, frac = casc.predict_with_cost(F, budget=0.2)
    assert 0.19 <= frac <= 0.21


def test_cold_rule_defers_low_within():
    F, y = _synthetic()
    casc = SelectiveCascadeXQP.from_fit(F, y)
    _, defer, _ = casc.predict_with_cost(F, budget=0.2, rule="cold")
    # deferred blocks have lower within-EMA than non-deferred
    assert F[defer, 0].mean() < F[~defer, 0].mean()


def test_conditional_mi_redundant_vs_unique():
    rng = np.random.default_rng(1)
    n = 40000
    x0 = rng.random(n).astype(np.float32)
    x1 = x0.copy()                      # redundant with x0
    x2 = rng.random(n).astype(np.float32)  # independent
    y = (rng.random(n) < 1 / (1 + np.exp(-(3 * x0 + 3 * x2 - 3)))).astype(np.int64)
    F = np.stack([x0, x1, x2], 1)
    u_redundant = conditional_mi_view(F, y, 1)   # x1 given {x0,x2}: ~0
    u_unique = conditional_mi_view(F, y, 2)       # x2 given {x0,x1}: >0
    assert u_unique > u_redundant + 0.01
    assert u_redundant < 0.01


def test_unique_information_report_keys():
    F, y = _synthetic()
    rep = unique_information_report(F, y, feature_names=["within", "cross", "query", "recency"])
    assert set(rep) == {"within", "cross", "query", "recency"}
    for v in rep.values():
        assert "relevance_mi" in v and "unique_cmi" in v
