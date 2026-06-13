"""Unit tests for XQPPolicy."""
import numpy as np

from xqp.policy import BlockStats, XQPPolicy
from xqp.predictor import ClosedFormXQP
from xqp.eval import synthetic_dataset


def _make_stats(B=64, d=32, layer=0, seed=0):
    rng = np.random.default_rng(seed)
    return BlockStats(
        ema_within=rng.random(B).astype(np.float32),
        ema_prev_layer=rng.random(B).astype(np.float32),
        K_layer=rng.normal(size=(B, d)).astype(np.float32),
        q_prev=rng.normal(size=d).astype(np.float32),
        last_used=rng.integers(0, 100, size=B).astype(np.float32),
        layer=layer,
    )


def test_select_to_keep_returns_set_of_size_budget():
    F, y = synthetic_dataset(n_blocks=64, n_steps=32)
    pred = ClosedFormXQP.from_fit(F, y)
    policy = XQPPolicy(predictor=pred)
    stats = _make_stats(B=64)
    keep = policy.select_to_keep(stats, budget=20, step=10)
    assert isinstance(keep, set)
    assert len(keep) == 20


def test_sink_and_window_forced():
    F, y = synthetic_dataset(n_blocks=64, n_steps=32)
    pred = ClosedFormXQP.from_fit(F, y)
    policy = XQPPolicy(predictor=pred, n_sink=4, n_window=4)
    stats = _make_stats(B=128)
    keep = policy.select_to_keep(stats, budget=20, step=50)
    # sink should be in
    assert {0, 1, 2, 3}.issubset(keep)
    # window should be in
    assert {124, 125, 126, 127}.issubset(keep)


def test_budget_larger_than_blocks_keeps_all():
    F, y = synthetic_dataset(n_blocks=64, n_steps=32)
    pred = ClosedFormXQP.from_fit(F, y)
    policy = XQPPolicy(predictor=pred)
    stats = _make_stats(B=20)
    keep = policy.select_to_keep(stats, budget=100, step=0)
    assert keep == set(range(20))


def _const_predictor(bias):
    """Predictor whose score is sigmoid(bias) for every block (weights=0)."""
    return ClosedFormXQP(weights=np.zeros(4, dtype=np.float32), bias=np.float32(bias))


def test_recovery_tau_default_is_16():
    assert XQPPolicy(predictor=_const_predictor(0.0)).recovery_tau == 16


def test_fast_recovery_tau_exits_fallback_quicker():
    """round-2 MR2: a small recovery_tau exits fallback within ~tau steps;
    a large one (timescale >> window) stays in fallback far longer, given the
    same constant max-score stream into both policies."""
    B, budget, window = 64, 20, 4
    stats = _make_stats(B=B)

    def run(tau):
        pol = XQPPolicy(predictor=_const_predictor(-5.0),   # max score ~0.0067
                        fallback_window_per_layer=window,
                        fallback_p_threshold=0.4,
                        recovery_tau=tau)
        # Phase A: low scores, warm past the window so fallback is armed+engaged
        for t in range(window + 3):
            pol.select_to_keep(stats, budget=budget, step=t)
        ema_low = pol._per_layer_ema[0]
        # Phase B: switch to high scores (~0.993); count steps to cross threshold
        pol.predictor = _const_predictor(5.0)
        steps_to_recover = None
        for k in range(1, 50):
            pol.select_to_keep(stats, budget=budget, step=window + 3 + k)
            if pol._per_layer_ema[0] >= 0.4:
                steps_to_recover = k
                break
        return ema_low, steps_to_recover

    ema_low_fast, fast = run(tau=2)
    ema_low_slow, slow = run(tau=window * 50)   # very slow EMA
    # both entered fallback during the low phase
    assert ema_low_fast < 0.4 and ema_low_slow < 0.4
    # small tau recovers in a couple of steps; large tau takes strictly longer
    assert fast is not None and fast <= 2
    assert slow is None or slow > fast
