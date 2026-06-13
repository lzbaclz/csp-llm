"""Unit tests for the ICDM analysis drivers (CPU, no GPU, no real traces).

Covers the load/recovery/split machinery the headline numbers depend on, so a
regression in request-id recovery, NaN handling, the vectorized AUC, or the
leak-free split would be caught.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

run_icdm_full = pytest.importorskip("run_icdm_full")
from xqp.eval import roc_auc as ref_auc  # noqa: E402


def _write_synthetic_trace(path, n_prompts=4, n_layers=3, n_steps=5, n_blocks=6, seed=0):
    """Emit a JSONL in the extractor's row order (step-major, layer-minor,
    block-minor), with request_id deliberately constant 'p0' to exercise the
    boundary-recovery path. One NaN row is injected."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_prompts):
        for t in range(n_steps):
            for l in range(n_layers):
                for b in range(n_blocks):
                    rows.append(dict(
                        request_id="p0", layer=l, step=t, block_idx=b,
                        f_within=float(rng.random()), f_cross=float(rng.integers(0, 2)),
                        f_query=float(rng.random()), f_pos=float(rng.random()),
                        y_h1=int(rng.integers(0, 2)), y_h4=int(rng.integers(0, 2)),
                        y_h16=int(rng.integers(0, 2)), y_h64=int(rng.integers(0, 2))))
    rows[10]["f_within"] = float("nan")  # one non-finite row to be dropped
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return n_prompts, len(rows)


def test_request_boundary_recovery_and_nan_drop(tmp_path):
    p = tmp_path / "M.jsonl"
    n_prompts, n_rows = _write_synthetic_trace(str(p))
    d = run_icdm_full.load_model_trace(str(p))
    # every row had request_id="p0"; boundaries must recover the true prompt count
    assert d["n_requests"] == n_prompts
    assert set(np.unique(d["rid"]).tolist()) == set(range(n_prompts))
    # the single NaN row was dropped
    assert d["n_dropped_nonfinite"] == 1
    assert d["F"].shape[0] == n_rows - 1
    assert np.isfinite(d["F"]).all()


def test_request_split_is_leak_free(tmp_path):
    p = tmp_path / "M.jsonl"
    _write_synthetic_trace(str(p), n_prompts=8)
    d = run_icdm_full.load_model_trace(str(p))
    tr, te = run_icdm_full.request_split(d["rid"], frac=0.25, seed=0)
    tr_reqs = set(d["rid"][tr].tolist())
    te_reqs = set(d["rid"][te].tolist())
    assert tr_reqs and te_reqs
    assert tr_reqs.isdisjoint(te_reqs)            # no prompt in both splits
    assert len(tr) + len(te) == d["F"].shape[0]   # partition is complete


def test_fast_auc_matches_reference():
    rng = np.random.default_rng(1)
    for _ in range(8):
        y = (rng.random(4000) < 0.12).astype(float)
        s = rng.random(4000)
        s = np.where(rng.random(4000) < 0.4, np.round(s * 3) / 3, s)  # heavy ties
        # driver AUC is float64; xqp.eval.roc_auc downcasts to float32, so they
        # agree only to ~float32 precision (the driver version is the accurate one)
        assert abs(run_icdm_full.roc_auc(y, s) - ref_auc(y, s)) < 5e-5


def test_pool_namespaces_requests(tmp_path):
    pa = tmp_path / "A.jsonl"; pb = tmp_path / "B.jsonl"
    _write_synthetic_trace(str(pa), n_prompts=3, seed=1)
    _write_synthetic_trace(str(pb), n_prompts=4, seed=2)
    da = run_icdm_full.load_model_trace(str(pa))
    db = run_icdm_full.load_model_trace(str(pb))
    pooled = run_icdm_full.pool_models({"A": da, "B": db})
    # request ids must not collide across models
    assert pooled["n_requests"] == da["n_requests"] + db["n_requests"]
    assert len(np.unique(pooled["rid"])) == pooled["n_requests"]
