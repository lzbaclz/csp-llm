"""Tests for the coverage-driven budgeter (split-conformal validity)."""
import numpy as np
import pytest

from xqp.budgeter import CoverageDrivenBudgeter, fixed_ratio_select, _conformal_tau


class _IdentityScorer:
    """score(F) = F[:,0] (already a probability-like column)."""
    def score(self, F):
        return np.asarray(F)[:, 0]


def _data(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.random(n).astype(np.float32)          # 'score' in col 0
    layer = rng.integers(0, 4, n)
    y = (rng.random(n) < p).astype(np.float32)    # salient prob increases with score
    F = np.stack([p, np.zeros(n), np.zeros(n), np.zeros(n)], 1)
    return F, y, layer


def test_conformal_tau_monotone():
    s = np.linspace(0, 1, 1000)
    assert _conformal_tau(s, 0.05) < _conformal_tau(s, 0.20) < _conformal_tau(s, 0.50)


def test_split_conformal_validity_holds_alpha():
    Fc, yc, lc = _data(60000, 0)
    Ft, yt, lt = _data(60000, 1)                  # exchangeable test set
    for al in (0.05, 0.10, 0.20):
        bg = CoverageDrivenBudgeter.calibrate(_IdentityScorer(), Fc, yc, lc, cols=(0, 1), alpha=al)
        ev = bg.evaluate(Ft, yt, lt, per_layer=True)
        # realized miss must be at/below the target (with finite-sample slack)
        assert ev["realized_miss"] <= al + 0.02, (al, ev["realized_miss"])
        # smaller alpha => keep more (bigger budget)
    b05 = CoverageDrivenBudgeter.calibrate(_IdentityScorer(), Fc, yc, lc, alpha=0.05).evaluate(Ft, yt, lt)
    b20 = CoverageDrivenBudgeter.calibrate(_IdentityScorer(), Fc, yc, lc, alpha=0.20).evaluate(Ft, yt, lt)
    assert b05["emergent_budget"] > b20["emergent_budget"]


def test_per_layer_equalizes_miss_vs_global():
    # make layers have different score scales so a single global tau is unfair
    rng = np.random.default_rng(2)
    n = 80000
    layer = rng.integers(0, 4, n)
    p = (rng.random(n) * (0.4 + 0.2 * layer)).astype(np.float32)   # layer-dependent scale
    y = (rng.random(n) < p / (0.4 + 0.2 * layer)).astype(np.float32)
    F = np.stack([p, np.zeros(n), np.zeros(n), np.zeros(n)], 1)
    bg = CoverageDrivenBudgeter.calibrate(_IdentityScorer(), F, y, layer, alpha=0.10)
    per = bg.evaluate(F, y, layer, per_layer=True)
    glob = bg.evaluate(F, y, layer, per_layer=False)
    # per-layer thresholds equalize miss across layers better than one global tau
    assert per["per_layer_miss_std"] <= glob["per_layer_miss_std"] + 1e-6


def test_fixed_ratio_select_budget():
    s = np.random.default_rng(0).random(1000)
    m = fixed_ratio_select(s, 0.2)
    assert 195 <= m.sum() <= 205
    assert s[m].min() >= s[~m].max() - 1e-6      # kept are the top-scoring


def test_save_and_norm_curve_roundtrip(tmp_path):
    import json
    Fc, yc, lc = _data(40000, 3)
    bg = CoverageDrivenBudgeter.calibrate(_IdentityScorer(), Fc, yc, lc, alpha=0.10)
    nc = bg.norm_curve()
    # normalized positions span [0,1], thresholds are valid probabilities
    assert nc[0][0] == 0.0 and abs(nc[-1][0] - 1.0) < 1e-6
    assert all(0.0 <= t <= 1.0 for _, t in nc)
    p = tmp_path / "bg.json"
    bg.save(str(p))
    obj = json.loads(p.read_text())
    assert obj["alpha"] == pytest.approx(0.10)
    assert len(obj["norm_curve"]) == len(bg.tau_by_layer)
    assert "tau_global" in obj
