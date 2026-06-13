"""Tests for robustness analysis."""
import json
import numpy as np
import pytest

from xqp.predictor import ClosedFormXQP
from xqp.eval import synthetic_dataset, roc_auc
from xqp.features import cosine_query_key


def test_multi_head_cosine():
    H, d = 4, 32
    B = 16
    rng = np.random.default_rng(0)
    q = rng.normal(size=(H, d)).astype(np.float32)
    K = rng.normal(size=(B, H, d)).astype(np.float32)
    out_max = cosine_query_key(q, K, reduce="max")
    out_mean = cosine_query_key(q, K, reduce="mean")
    assert out_max.shape == (B,)
    assert out_mean.shape == (B,)
    assert (out_max >= out_mean - 1e-5).all()


def test_continuous_cross_signal():
    from xqp.features import extract_features
    B = 32
    d = 32
    rng = np.random.default_rng(0)
    F_ind = extract_features(
        ema_within=rng.random(B).astype(np.float32),
        ema_prev_layer=rng.random(B).astype(np.float32),
        K_layer=rng.normal(size=(B, d)).astype(np.float32),
        q_prev=rng.normal(size=d).astype(np.float32),
        step=0,
        last_used=np.zeros(B, dtype=np.float32),
        cross_signal="indicator",
    )
    F_cont = extract_features(
        ema_within=F_ind[:, 0],  # reuse normalized within to ensure same other features
        ema_prev_layer=rng.random(B).astype(np.float32),
        K_layer=rng.normal(size=(B, d)).astype(np.float32),
        q_prev=rng.normal(size=d).astype(np.float32),
        step=0,
        last_used=np.zeros(B, dtype=np.float32),
        cross_signal="continuous",
    )
    # indicator column should be 0/1
    assert set(np.unique(F_ind[:, 1])).issubset({0.0, 1.0})
    # continuous column should be in [0, 1] but not strictly 0/1
    assert F_cont[:, 1].min() >= 0
    assert F_cont[:, 1].max() <= 1
    # at least one fractional value
    assert ((F_cont[:, 1] > 0) & (F_cont[:, 1] < 1)).any()


def test_cross_model_transfer_no_crash(tmp_path):
    """Synthetic A→B transfer (just smoke-test the plumbing)."""
    from xqp.robustness import cross_model_auc
    # write two synthetic JSONL traces
    import json
    F1, y1 = synthetic_dataset(seed=0, n_blocks=128, n_steps=32)
    F2, y2 = synthetic_dataset(seed=42, n_blocks=128, n_steps=32)
    def dump(F, y, p):
        with open(p, "w") as fh:
            for i in range(F.shape[0]):
                fh.write(json.dumps({
                    "request_id": "x", "layer": int(i % 8), "step": int(i),
                    "block_idx": int(i),
                    "f_within": float(F[i,0]), "f_cross": float(F[i,1]),
                    "f_query": float(F[i,2]), "f_pos": float(F[i,3]),
                    "y_h1": int(y[i]), "y_h4": int(y[i]),
                    "y_h16": int(y[i]), "y_h64": int(y[i]),
                }) + "\n")
    p1 = tmp_path / "a.jsonl"; dump(F1, y1, p1)
    p2 = tmp_path / "b.jsonl"; dump(F2, y2, p2)
    out = cross_model_auc(str(p1), {"b": str(p2)})
    assert "b" in out
    assert 0.0 <= out["b"] <= 1.0
