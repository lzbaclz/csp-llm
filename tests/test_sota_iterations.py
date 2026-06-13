"""Tests for the 5 SOTA-driven iteration modules."""
import numpy as np

from xqp.eval import roc_auc, synthetic_dataset
from xqp.predictor import ClosedFormXQP
from xqp.sota_iterations.iter1_quest_bound import (
    cosine_query_key_per_token,
    query_max_min_per_block,
    quest_bound_features,
)
from xqp.sota_iterations.iter2_distill import (
    MassDistillationLoss,
    attention_mass_from_softmax,
)
from xqp.sota_iterations.iter3_continuous_cross import (
    continuous_cross_signal,
    make_continuous_features,
    truncated_gaussian_coefficient,
)
from xqp.sota_iterations.iter4_per_head import (
    per_head_features,
    PerHeadClosedFormXQP,
)
from xqp.sota_iterations.iter5_online import OnlineXQP


# Iter 1 — Quest bounds
def test_per_token_cosine_3d():
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    K = np.array([
        [[1.0, 0, 0], [0.5, 0.5, 0]],
        [[-1, 0, 0], [0, 1, 0]],
    ], dtype=np.float32)
    c = cosine_query_key_per_token(q, K)
    assert c.shape == (2, 2)
    assert c[0, 0] == 1.0
    assert c[1, 0] == -1.0


def test_query_max_min_per_block():
    q = np.ones(4, dtype=np.float32)
    K = np.random.default_rng(0).normal(size=(16, 8, 4)).astype(np.float32)
    qmax, qmin = query_max_min_per_block(q, K)
    assert qmax.shape == (16,) and qmin.shape == (16,)
    assert (qmax >= qmin).all()


def test_quest_bound_features_in_unit_interval():
    q = np.ones(4, dtype=np.float32)
    K = np.random.default_rng(0).normal(size=(16, 8, 4)).astype(np.float32)
    feats = quest_bound_features(q, K)
    for k, v in feats.items():
        assert (v >= 0).all() and (v <= 1).all(), f"{k} out of [0,1]"


# Iter 2 — Distillation loss
def test_mass_distillation_loss_decreasing_with_perfect_pred():
    loss = MassDistillationLoss(alpha_start=0.5, alpha_end=0.5, n_steps=1)
    y_top = np.array([0, 1, 1, 0, 1], dtype=np.float32)
    mass = np.array([0.05, 0.30, 0.40, 0.05, 0.20], dtype=np.float32)
    perfect = loss(p_top=y_top, p_mass=mass, y_top=y_top, mass=mass)
    noisy = loss(p_top=np.full(5, 0.5), p_mass=np.full(5, 0.2),
                 y_top=y_top, mass=mass)
    assert perfect < noisy


def test_alpha_annealing():
    loss = MassDistillationLoss(alpha_start=1.0, alpha_end=0.0, n_steps=100)
    assert loss.alpha(0) == 1.0
    assert loss.alpha(50) == 0.5
    assert loss.alpha(150) == 0.0


def test_attention_mass_sums_to_one():
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    m = attention_mass_from_softmax(a)
    assert abs(m.sum() - 1.0) < 1e-5


# Iter 5 — Online
def test_online_xqp_observes_and_updates():
    F, y = synthetic_dataset(n_blocks=64, n_steps=8)
    pred = ClosedFormXQP.from_fit(F[:200], y[:200])
    online = OnlineXQP(predictor=pred, update_every=4, buffer_size=64)
    w_before = pred.weights.copy()
    # feed enough batches to trigger several updates
    for i in range(20):
        online.observe(F[i*32:(i+1)*32], y[i*32:(i+1)*32])
    w_after = pred.weights.copy()
    # weights should have moved
    assert not np.allclose(w_before, w_after)
    assert online.n_buffered() > 0


def test_online_does_not_diverge():
    F, y = synthetic_dataset(n_blocks=128, n_steps=8)
    pred = ClosedFormXQP.from_fit(F[:300], y[:300])
    online = OnlineXQP(predictor=pred, update_every=2,
                       buffer_size=128, learning_rate=0.3)
    for i in range(50):
        online.observe(F[i*32:(i+1)*32], y[i*32:(i+1)*32])
    # weights should remain finite
    assert np.isfinite(pred.weights).all()
    assert abs(float(pred.bias)) < 100


# Iter 3 — Continuous cross-layer
def test_continuous_cross_signal_unit_interval():
    rng = np.random.default_rng(0)
    prev = rng.random(64).astype(np.float32)
    s = continuous_cross_signal(prev)
    assert s.shape == (64,)
    assert s.min() >= 0.0 and s.max() <= 1.0
    assert ((s > 0) & (s < 1)).any()        # genuinely continuous, not 0/1
    assert s.max() == 1.0                     # normalized by its own max


def test_continuous_cross_signal_empty():
    assert continuous_cross_signal(np.zeros(0, dtype=np.float32)).shape == (0,)


def test_make_continuous_features_cross_column_fractional():
    rng = np.random.default_rng(1)
    B, d = 32, 16
    F = make_continuous_features(
        ema_within=rng.random(B).astype(np.float32),
        ema_prev_layer=rng.random(B).astype(np.float32),
        K_layer=rng.normal(size=(B, d)).astype(np.float32),
        q_prev=rng.normal(size=d).astype(np.float32),
        step=0, last_used=np.zeros(B, dtype=np.float32),
    )
    assert F.shape == (B, 4)
    # cross column is continuous (not all 0/1)
    assert ((F[:, 1] > 0) & (F[:, 1] < 1)).any()


def test_truncated_gaussian_coefficient_sign():
    """Positive class with a higher feature mean => positive LDA slope."""
    rng = np.random.default_rng(2)
    n = 2000
    y = (rng.random(n) < 0.3).astype(np.float32)
    f = rng.normal(0.0, 1.0, n).astype(np.float32) + 0.8 * y  # pos shifted up
    coef = truncated_gaussian_coefficient(f, y)
    assert coef["mu1"] > coef["mu0"]
    assert coef["w"] > 0
    assert coef["var"] > 0


def test_truncated_gaussian_coefficient_needs_both_classes():
    import pytest
    with pytest.raises(ValueError):
        truncated_gaussian_coefficient(np.ones(10, np.float32), np.ones(10, np.float32))


# Iter 4 — Per-head aggregation
def _per_head_inputs(B=16, H=4, d=8, seed=0):
    rng = np.random.default_rng(seed)
    return dict(
        ema_within=rng.random((B, H)).astype(np.float32),
        ema_prev_layer=rng.random(B).astype(np.float32),
        K_layer=rng.normal(size=(B, H, d)).astype(np.float32),
        q_heads=rng.normal(size=(H, d)).astype(np.float32),
        step=5,
        last_used=rng.integers(0, 5, size=B).astype(np.float32),
    )


def test_per_head_features_shape_and_range():
    F = per_head_features(**_per_head_inputs())
    assert F.shape == (16, 4, 4)
    assert F.min() >= 0.0 and F.max() <= 1.0 + 1e-5


def test_per_head_features_first_layer_no_prev():
    kw = _per_head_inputs()
    kw["ema_prev_layer"] = None
    F = per_head_features(**kw)
    assert F.shape == (16, 4, 4)
    assert np.all(np.isfinite(F))


def test_per_head_features_empty_blocks():
    F = per_head_features(
        ema_within=np.zeros((0, 4), dtype=np.float32),
        ema_prev_layer=None,
        K_layer=np.zeros((0, 4, 8), dtype=np.float32),
        q_heads=np.zeros((4, 8), dtype=np.float32),
        step=0, last_used=np.zeros(0, dtype=np.float32),
    )
    assert F.shape == (0, 4, 4)


def test_per_head_fit_score_and_aggregate():
    rng = np.random.default_rng(3)
    N, H = 400, 4
    F = rng.random((N, H, 4)).astype(np.float32)
    y = (rng.random(N) < 0.3).astype(np.float32)
    ph = PerHeadClosedFormXQP.from_fit(F, y)
    assert ph.weights.shape == (H, 4)
    assert ph.bias.shape == (H,)
    assert ph.n_params() == H * 4 + H
    Fb = rng.random((16, H, 4)).astype(np.float32)
    per_head = ph.score(Fb)
    assert per_head.shape == (16, H)
    assert (per_head >= 0).all() and (per_head <= 1).all()
    s_max = ph.score_blocks(Fb, reduce="max")
    s_mean = ph.score_blocks(Fb, reduce="mean")
    assert s_max.shape == (16,) and s_mean.shape == (16,)
    assert (s_max >= s_mean - 1e-6).all()


def test_per_head_fit_per_head_labels():
    rng = np.random.default_rng(4)
    N, H = 300, 4
    F = rng.random((N, H, 4)).astype(np.float32)
    y = (rng.random((N, H)) < 0.3).astype(np.float32)   # per-head labels
    ph = PerHeadClosedFormXQP.from_fit(F, y)
    assert ph.weights.shape == (H, 4)
