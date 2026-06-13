"""Edge-case / robustness tests for XQP (degenerate inputs).

These cover the boundaries that previously raised raw NumPy errors: empty block
sets, empty training/calibration data, and single-class label vectors.
"""
import numpy as np
import pytest

from xqp.eval import roc_auc, topk_recall
from xqp.features import FEATURE_DIM, extract_features
from xqp.normalize import FeatureNormalizer
from xqp.policy import BlockStats, XQPPolicy
from xqp.predictor import ClosedFormXQP, PairwiseXQP


def _empty_block_kwargs(d=8):
    return dict(
        ema_within=np.zeros(0, dtype=np.float32),
        ema_prev_layer=None,
        K_layer=np.zeros((0, d), dtype=np.float32),
        q_prev=np.zeros(d, dtype=np.float32),
        step=0,
        last_used=np.zeros(0, dtype=np.float32),
    )


def test_extract_features_empty_blocks():
    F = extract_features(**_empty_block_kwargs())
    assert F.shape == (0, FEATURE_DIM)
    assert F.dtype == np.float32


def test_extract_features_empty_blocks_continuous_cross():
    F = extract_features(cross_signal="continuous", **_empty_block_kwargs())
    assert F.shape == (0, FEATURE_DIM)


def test_extract_features_single_block():
    d = 8
    F = extract_features(
        ema_within=np.array([0.7], dtype=np.float32),
        ema_prev_layer=np.array([0.3], dtype=np.float32),
        K_layer=np.ones((1, d), dtype=np.float32),
        q_prev=np.ones(d, dtype=np.float32),
        step=3,
        last_used=np.array([1.0], dtype=np.float32),
    )
    assert F.shape == (1, FEATURE_DIM)
    assert np.all(np.isfinite(F))


def test_closedform_fit_empty_raises():
    with pytest.raises(ValueError):
        ClosedFormXQP.from_fit(np.zeros((0, FEATURE_DIM), np.float32),
                               np.zeros(0, np.float32))


def test_pairwise_fit_empty_raises():
    with pytest.raises(ValueError):
        PairwiseXQP.from_fit(np.zeros((0, FEATURE_DIM), np.float32),
                             np.zeros(0, np.float32))


def test_normalizer_empty_raises():
    with pytest.raises(ValueError):
        FeatureNormalizer.from_calibration(np.zeros((0, FEATURE_DIM), np.float32))


def test_normalizer_roundtrip_in_unit_range():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(256, FEATURE_DIM)).astype(np.float32)
    norm = FeatureNormalizer.from_calibration(F)
    out = norm.transform(F)
    assert out.min() >= 0.0 and out.max() <= 1.0 + 1e-6


def test_roc_auc_single_class_is_nan():
    y = np.ones(10, dtype=np.float32)          # only positives
    s = np.linspace(0, 1, 10).astype(np.float32)
    assert np.isnan(roc_auc(y, s))


def test_topk_recall_empty_is_nan():
    assert np.isnan(topk_recall(np.zeros(0, np.float32), np.zeros(0, np.float32)))


def test_policy_handles_zero_blocks():
    """A policy step with no blocks must return an empty keep-set, not crash."""
    pred = ClosedFormXQP(weights=np.ones(FEATURE_DIM, np.float32), bias=np.float32(0.0))
    pol = XQPPolicy(predictor=pred)
    stats = BlockStats(
        ema_within=np.zeros(0, np.float32),
        K_layer=np.zeros((0, 8), np.float32),
        q_prev=np.zeros(8, np.float32),
        last_used=np.zeros(0, np.float32),
    )
    assert pol.select_to_keep(stats, budget=4, step=1) == set()
