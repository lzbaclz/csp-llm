"""Guards the request-boundary reconstruction in run_icdm_full.load_model_trace.

The trace harness writes request_id="p0" for every prompt, so true per-prompt
boundaries are recovered from the deterministic step-major/layer-minor/block-minor
emission order: the first row of each prompt is uniquely (step==0, layer==0,
block_idx==0). The paper's request-clustered CIs and request-level splits all rest
on this reconstruction, so we assert it recovers exactly the right number of
prompts and a correct, contiguous, equal-size partition.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import numpy as np
import pytest

import run_icdm_full as R

HORIZONS = ("h1", "h4", "h16", "h64")


def _write_synthetic_trace(path, n_prompts, n_layers, n_steps, n_blocks):
    """Emit rows in the real harness order: prompt, step, layer, block.

    Exactly one row per prompt has (step,layer,block)==(0,0,0); the first row of
    the file is prompt 0's, so the cumulative-boundary id is correct from row 0.
    """
    rng = np.random.default_rng(0)
    rows_per_prompt = n_layers * n_steps * n_blocks
    with open(path, "w") as fh:
        for _p in range(n_prompts):
            for step in range(n_steps):
                for layer in range(n_layers):
                    for blk in range(n_blocks):
                        r = dict(
                            request_id="p0",          # harness stamps p0 for every prompt
                            layer=layer, step=step, block_idx=blk,
                            f_within=float(rng.random()), f_cross=float(rng.random()),
                            f_query=float(rng.random()), f_pos=float(rng.random()),
                        )
                        for h in HORIZONS:
                            r[f"y_{h}"] = int(rng.random() < 0.1)
                        fh.write(json.dumps(r) + "\n")
    return rows_per_prompt


def test_request_reconstruction_recovers_exact_prompt_count(tmp_path):
    n_prompts, n_layers, n_steps, n_blocks = 5, 3, 4, 6
    path = str(tmp_path / "Synthetic-Model.jsonl")
    rows_per_prompt = _write_synthetic_trace(path, n_prompts, n_layers, n_steps, n_blocks)

    d = R.load_model_trace(path)

    # exact prompt count recovered from emission order
    assert d["n_requests"] == n_prompts
    assert d["F"].shape[0] == n_prompts * rows_per_prompt
    assert int(d["layer"].max() + 1) == n_layers

    # rid is a contiguous 0..K-1 partition, each prompt equal size
    rids = d["rid"]
    assert set(np.unique(rids).tolist()) == set(range(n_prompts))
    counts = np.bincount(rids)
    assert np.all(counts == rows_per_prompt)
    # ids are non-decreasing in emission order (no interleaving)
    assert np.all(np.diff(rids) >= 0)


def test_request_split_holds_out_whole_prompts(tmp_path):
    """The request-level split must never put rows of one prompt on both sides."""
    path = str(tmp_path / "Synthetic-Model.jsonl")
    _write_synthetic_trace(path, n_prompts=8, n_layers=2, n_steps=3, n_blocks=4)
    d = R.load_model_trace(path)
    tr, te = R.request_split(d["rid"], frac=0.25, seed=0)
    tr_reqs = set(d["rid"][tr].tolist())
    te_reqs = set(d["rid"][te].tolist())
    assert tr_reqs.isdisjoint(te_reqs)               # no prompt leaks across the split
    assert tr_reqs | te_reqs == set(range(8))


def test_nonfinite_rows_dropped_but_boundaries_intact(tmp_path):
    """fp16-NaN rows are dropped, but request recovery is computed on the full
    sequence first, so the prompt count is unaffected."""
    path = str(tmp_path / "Synthetic-Model.jsonl")
    _write_synthetic_trace(path, n_prompts=4, n_layers=2, n_steps=3, n_blocks=4)
    # corrupt a few feature cells to non-finite
    lines = open(path).read().splitlines()
    for i in (10, 25, 40):
        r = json.loads(lines[i]); r["f_within"] = float("nan"); lines[i] = json.dumps(r)
    open(path, "w").write("\n".join(lines) + "\n")
    d = R.load_model_trace(path)
    assert d["n_requests"] == 4                      # boundaries intact despite drops
    assert d["n_dropped_nonfinite"] == 3
