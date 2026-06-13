"""End-to-end CLI tests: synthetic collect -> train -> eval -> bench (all CPU)."""
import json

import pytest

from xqp import eval as xqp_eval
from xqp import bench_wcet
from xqp import trace_collect_cli
from xqp import train_cli
from xqp.trace import load_trace


def test_trace_collect_synthetic_writes_valid_trace(tmp_path):
    out = tmp_path / "model.jsonl"
    rc = trace_collect_cli.main(
        ["--synthetic", "--out", str(out),
         "--synthetic-blocks", "64", "--synthetic-steps", "16", "--seed", "0"]
    )
    assert rc == 0
    rows = load_trace(str(out))
    assert "y_h4" in rows and "f_within" in rows and "layer" in rows
    assert rows["f_within"].shape[0] == 64 * 16


def test_full_cli_pipeline(tmp_path):
    traces = tmp_path / "traces"
    preds = tmp_path / "predictors"
    traces.mkdir(); preds.mkdir()

    # 1) collect synthetic trace
    trace_path = traces / "model.jsonl"
    assert trace_collect_cli.main(
        ["--synthetic", "--out", str(trace_path),
         "--synthetic-blocks", "64", "--synthetic-steps", "16"]
    ) == 0

    # 2) train a predictor (filename convention from scripts/train_predictor.sh)
    pred_path = preds / "model_h4.json"
    assert train_cli.main(
        ["--trace", str(trace_path), "--horizon", "h4", "--out", str(pred_path)]
    ) == 0
    assert pred_path.exists()

    # 3) eval (the e1 step of run_benchmarks.sh)
    eval_out = tmp_path / "e1.json"
    assert xqp_eval.main(
        ["--traces", str(traces), "--predictors", str(preds), "--out", str(eval_out)]
    ) == 0
    res = json.loads(eval_out.read_text())
    assert "model_h4" in res
    auc = res["model_h4"]["auc"]
    assert 0.0 <= auc <= 1.0

    # 4) wcet bench (the e2 step) — CPU sanity
    wcet_out = tmp_path / "e2.json"
    assert bench_wcet.main(
        ["--predictors", str(preds), "--out", str(wcet_out),
         "--batch", "256", "--replays", "100"]
    ) == 0
    w = json.loads(wcet_out.read_text())
    assert "model_h4" in w and "p50" in w["model_h4"]


def test_eval_missing_trace_reports_error(tmp_path):
    preds = tmp_path / "p"; preds.mkdir()
    # a predictor with no matching trace
    from xqp.predictor import ClosedFormXQP
    import numpy as np
    ClosedFormXQP(weights=np.ones(4, np.float32), bias=np.float32(0.0)).save(preds / "ghost_h4.json")
    out = tmp_path / "r.json"
    assert xqp_eval.main(["--traces", str(tmp_path), "--predictors", str(preds), "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert "error" in res["ghost_h4"]


def test_trace_collect_real_path_guarded(tmp_path):
    """Without torch+transformers+CUDA the real path must exit non-zero, not crash."""
    if trace_collect_cli._real_path_available():
        pytest.skip("real GPU path available; guard not exercised")
    rc = trace_collect_cli.main(["--out", str(tmp_path / "x.jsonl")])
    assert rc == 2
