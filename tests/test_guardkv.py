"""Tests for the GuardKV controller (select / budget cap / sink+window / adapt / prefetch)."""
import numpy as np

from xqp.guardkv import GuardKV


class _Scorer:
    def score(self, F):
        return np.asarray(F)[:, 0]      # score = within (col 0)


def _g(tau=0.5, gamma=0.0):
    return GuardKV(scorer=_Scorer(), tau_by_layer={0: tau}, cols=(0, 1),
                   alpha=0.10, gamma=gamma, sink=2, window=2, tau_global=tau)


def test_select_threshold_and_floor():
    n = 50
    F = np.zeros((n, 4), np.float32); F[:, 0] = np.linspace(0, 1, n)
    g = _g(tau=0.8)
    keep, p = g.select(F, layer=0, block_ids=np.arange(n))
    # all blocks with within>=0.8 kept
    assert keep[F[:, 0] >= 0.8].all()
    # sink (first 2) + window (last 2) force-kept even if low score
    assert keep[0] and keep[1] and keep[-1] and keep[-2]


def test_budget_cap_respected():
    n = 100
    F = np.zeros((n, 4), np.float32); F[:, 0] = np.random.default_rng(0).random(n)
    g = _g(tau=0.0)                      # tau=0 would keep all
    keep, _ = g.select(F, layer=0, budget=20, block_ids=np.arange(n))
    assert keep.sum() <= 20
    # the highest-within blocks (besides forced sink/window) are kept
    assert keep[np.argmax(F[:, 0])]


def test_floor_exceeds_budget_capped():
    # A1 regression: when the sink+window FLOOR (10) exceeds the budget (4), the
    # KVPolicy contract (return <= budget) must still hold -- the floor is
    # truncated to its highest-scoring members, never returned whole.
    n = 20
    F = np.zeros((n, 4), np.float32); F[:, 0] = np.linspace(0, 1, n)
    g = GuardKV(scorer=_Scorer(), tau_by_layer={0: 0.0}, cols=(0, 1),
                alpha=0.10, gamma=0.0, sink=5, window=5, tau_global=0.0)
    keep, _ = g.select(F, layer=0, budget=4, block_ids=np.arange(n))
    assert keep.sum() == 4            # exactly the budget, NOT the 10-block floor
    assert keep.sum() <= 4            # the contract


def test_observe_miss_and_adapt():
    n = 100
    F = np.zeros((n, 4), np.float32); F[:, 0] = np.linspace(0, 1, n)
    y = (F[:, 0] > 0.9).astype(float)    # top 10% salient
    g = _g(tau=0.95, gamma=0.1)
    keep, _ = g.select(F, layer=0, block_ids=np.arange(n))
    tau0 = g.tau_by_layer[0]
    err = g.observe_miss(keep, y, layer=0)
    assert 0.0 <= err <= 1.0
    # missing salient (err>alpha) should LOWER tau (keep more next time)
    if err > g.alpha:
        assert g.tau_by_layer[0] < tau0


def test_prefetch_hint_topr():
    within = np.array([0.1, 0.9, 0.2, 0.8, 0.05, 0.95, 0.3, 0.7, 0.15, 0.6], np.float32)
    g = _g()
    hint = g.prefetch_hint(within, r=0.3)   # top 30% = 3 blocks
    assert len(hint) == 3
    assert 5 in hint and 1 in hint          # the two highest
