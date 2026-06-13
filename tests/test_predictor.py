"""Unit tests for ClosedFormXQP fit/score on synthetic data."""
import numpy as np
import pytest

from xqp.predictor import ClosedFormXQP, TinyMLPXQP
from xqp.eval import synthetic_dataset, roc_auc, topk_recall


def test_closedform_synthetic_auc():
    F, y = synthetic_dataset(n_blocks=256, n_steps=64, seed=0)
    # split 80/20
    n = F.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    F_tr, y_tr = F[perm[:n_train]], y[perm[:n_train]]
    F_va, y_va = F[perm[n_train:]], y[perm[n_train:]]
    pred = ClosedFormXQP.from_fit(F_tr, y_tr)
    s = pred.score(F_va)
    auc = roc_auc(y_va, s)
    assert auc > 0.80, f"AUC too low: {auc:.3f} — predictor or synthetic data broken"


def test_closedform_per_layer():
    F, y = synthetic_dataset(n_blocks=128, n_steps=32, seed=1)
    # synthesize layer ids
    layer_ids = (np.arange(F.shape[0]) // 256) % 8
    pred = ClosedFormXQP.from_fit(F, y, layer_ids=layer_ids, per_layer=True)
    assert pred.weights.shape == (8, 4)
    assert pred.bias.shape == (8,)
    s = pred.score(F[:32], layer=0)
    assert s.shape == (32,)


def test_save_load(tmp_path):
    F, y = synthetic_dataset(n_blocks=64, n_steps=16, seed=2)
    pred = ClosedFormXQP.from_fit(F, y)
    out = tmp_path / "pred.json"
    pred.save(out)
    loaded = ClosedFormXQP.load(out)
    np.testing.assert_allclose(loaded.weights, pred.weights)
    np.testing.assert_allclose(loaded.score(F[:16]), pred.score(F[:16]))


def test_tinymlp_n_params():
    m = TinyMLPXQP.random_init(seed=0)
    # 4*16 + 16 + 16*4 + 4 = 148
    assert m.n_params() == 148


def test_tinymlp_score_shape():
    F = np.random.default_rng(0).normal(size=(32, 4)).astype(np.float32)
    m = TinyMLPXQP.random_init(seed=0)
    s = m.score(F, horizon_idx=2)
    assert s.shape == (32,)
    assert (s >= 0).all() and (s <= 1).all()


def test_tinymlp_trainer_learns():
    """Trained TinyMLP should beat random init and reach a sane AUC."""
    F, y = synthetic_dataset(n_blocks=256, n_steps=48, seed=0)
    ntr = int(0.8 * len(F))
    rand_auc = roc_auc(y[ntr:], TinyMLPXQP.random_init(0).score(F[ntr:], horizon_idx=1))
    m = TinyMLPXQP.from_fit(F[:ntr], y[:ntr], epochs=200)
    trained_auc = roc_auc(y[ntr:], m.score(F[ntr:], horizon_idx=1))
    assert m.n_params() == 148
    assert trained_auc > 0.80
    assert trained_auc > rand_auc + 0.05
    assert all(np.isfinite(w).all() for w in (m.W1, m.b1, m.W2, m.b2))


def test_tinymlp_trainer_empty_raises():
    import pytest
    with pytest.raises(ValueError):
        TinyMLPXQP.from_fit(np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
