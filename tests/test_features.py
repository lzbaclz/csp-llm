"""Unit tests for xqp.features — runnable in the sandbox (no GPU)."""
import numpy as np
import pytest

from xqp.features import (
    FEATURE_DIM,
    cosine_query_key,
    extract_features,
    recency,
    topk_indicator,
)


def test_feature_dim():
    assert FEATURE_DIM == 4


def test_cosine_query_key_basic():
    q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    K = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [-1.0, 0.0, 0.0]], dtype=np.float32)
    c = cosine_query_key(q, K)
    np.testing.assert_allclose(c, [1.0, 0.0, -1.0], atol=1e-5)


def test_cosine_query_key_shape_mismatch():
    q = np.zeros(3, dtype=np.float32)
    K = np.zeros((5, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        cosine_query_key(q, K)


def test_topk_indicator_top10pct():
    scores = np.arange(20, dtype=np.float32)
    ind = topk_indicator(scores, r=0.10)
    # top-10% of 20 = 2 elements, should be the top-2 (indices 18, 19)
    assert ind.sum() == 2
    assert ind[19] == 1.0 and ind[18] == 1.0
    assert ind[0] == 0.0


def test_recency_decay():
    """Post-audit fix: recency is exp(-delta/window); fresh = 1, old → 0."""
    last_used = np.array([0.0, 50.0, 100.0], dtype=np.float32)
    r = recency(step=100, last_used=last_used, window=64.0)
    assert r[2] > r[1] > r[0]
    assert r[2] == pytest.approx(1.0, abs=1e-6)  # fresh (delta=0)
    assert r[0] < 0.3                              # old (delta=100, window=64)


def test_extract_features_shape_and_range():
    B = 32
    d = 64
    rng = np.random.default_rng(0)
    F = extract_features(
        ema_within=rng.random(B).astype(np.float32),
        ema_prev_layer=rng.random(B).astype(np.float32),
        K_layer=rng.normal(size=(B, d)).astype(np.float32),
        q_prev=rng.normal(size=d).astype(np.float32),
        step=10,
        last_used=rng.integers(0, 10, size=B).astype(np.float32),
    )
    assert F.shape == (B, 4)
    # All features should now be in [0, 1] after normalization
    assert F.min() >= 0.0 and F.max() <= 1.0 + 1e-5


def test_extract_features_first_layer_no_prev():
    B = 16
    d = 32
    rng = np.random.default_rng(1)
    F = extract_features(
        ema_within=rng.random(B).astype(np.float32),
        ema_prev_layer=None,  # first layer, no predecessor
        K_layer=rng.normal(size=(B, d)).astype(np.float32),
        q_prev=rng.normal(size=d).astype(np.float32),
        step=0,
        last_used=np.zeros(B, dtype=np.float32),
    )
    assert F.shape == (B, 4)


def test_topk_indicator_edge_cases():
    # empty
    assert topk_indicator(np.array([], dtype=np.float32), r=0.1).shape == (0,)
    # single element
    out = topk_indicator(np.array([5.0]), r=0.1)
    assert out.sum() == 1
