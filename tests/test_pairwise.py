"""Tests for PairwiseXQP (round-2 MR1 response)."""
import numpy as np

from xqp.eval import roc_auc, synthetic_dataset
from xqp.predictor import ClosedFormXQP, PairwiseXQP
from xqp.features import query_proxy_mean


def test_pairwise_n_params():
    p = PairwiseXQP.from_fit(*synthetic_dataset(n_blocks=64, n_steps=16))
    # 4 linear + 10 pairwise + 1 bias = 15
    assert p.n_params() == 15


def test_pairwise_score_shape():
    F, y = synthetic_dataset(n_blocks=128, n_steps=16)
    p = PairwiseXQP.from_fit(F, y)
    s = p.score(F[:32])
    assert s.shape == (32,)
    assert (s >= 0).all() and (s <= 1).all()


def test_pairwise_ge_closedform_auc():
    """Pairwise should be >= closed-form on average since it strictly
    extends the hypothesis class."""
    F, y = synthetic_dataset(n_blocks=256, n_steps=64, seed=7)
    ntr = int(0.8 * len(F))
    cf = ClosedFormXQP.from_fit(F[:ntr], y[:ntr])
    pw = PairwiseXQP.from_fit(F[:ntr], y[:ntr])
    auc_cf = roc_auc(y[ntr:], cf.score(F[ntr:]))
    auc_pw = roc_auc(y[ntr:], pw.score(F[ntr:]))
    # allow tiny regression from L2 / numerical noise
    assert auc_pw >= auc_cf - 0.01, f"pairwise {auc_pw:.4f} << closed {auc_cf:.4f}"


def test_query_proxy_mean():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(10, 32)).astype(np.float32)
    q1 = query_proxy_mean(H, k=1)
    q4 = query_proxy_mean(H, k=4)
    np.testing.assert_allclose(q1, H[-1])
    np.testing.assert_allclose(q4, H[-4:].mean(axis=0))
