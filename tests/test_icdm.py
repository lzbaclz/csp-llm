"""Tests for the ICDM-pivot additions: DM metrics, redundancy analysis,
and the CPU-testable core of the A100 attention-trace extractor."""
import numpy as np
import pytest

from xqp.dm_metrics import (
    average_precision, precision_at_k, recall_at_k,
    expected_calibration_error, reliability_curve, brier_score,
)
from xqp.info_theory import (
    mutual_information_xx, conditional_mutual_information,
    interaction_information, redundancy_report,
)
from xqp.attn_trace_extract import (
    blockify, update_ema, labels_for_horizons, extract_attention_traces,
    _gpu_available,
)


# ---- DM metrics ----
def test_average_precision_separable_is_one():
    y = np.array([0, 0, 1, 1], dtype=np.float32)
    assert average_precision(y, y) == pytest.approx(1.0, abs=1e-9)


def test_average_precision_no_positives_is_nan():
    assert np.isnan(average_precision(np.zeros(8, np.float32), np.linspace(0, 1, 8)))


def test_average_precision_between_zero_and_one():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.2).astype(np.float32)
    s = rng.random(500).astype(np.float32)
    ap = average_precision(y, s)
    assert 0.0 <= ap <= 1.0


def test_precision_and_recall_at_k():
    # 100 items, top-10 by score are exactly the 10 positives
    score = np.arange(100, dtype=np.float32)
    y = (np.arange(100) >= 90).astype(np.float32)  # top-10 are positive
    assert precision_at_k(y, score, 0.10) == pytest.approx(1.0)
    assert recall_at_k(y, score, 0.10) == pytest.approx(1.0)


def test_brier_and_ece_perfectly_calibrated_low():
    rng = np.random.default_rng(1)
    s = rng.random(20000).astype(np.float32)
    y = (rng.random(20000) < s).astype(np.float32)   # P(y=1|s)=s exactly
    assert expected_calibration_error(y, s, n_bins=10) < 0.05
    assert brier_score(y, s) < 0.30


def test_reliability_curve_shape():
    rng = np.random.default_rng(2)
    s = rng.random(1000).astype(np.float32)
    y = (rng.random(1000) < s).astype(np.float32)
    rc = reliability_curve(y, s, n_bins=10)
    assert len(rc["confidence"]) == 10 and len(rc["accuracy"]) == 10
    assert sum(rc["count"]) == 1000


# ---- redundancy analysis ----
def test_conditional_mi_nonnegative():
    rng = np.random.default_rng(3)
    z = (rng.random(3000) < 0.5).astype(np.int64)
    x = z + 0.3 * rng.normal(size=3000)
    y = z + 0.3 * rng.normal(size=3000)
    assert conditional_mutual_information(x, y, z) >= -1e-6


def test_interaction_information_redundant_is_negative():
    """x,y both ~ z (shared) => redundant => II = I(x;y|z) - I(x;y) < 0."""
    rng = np.random.default_rng(4)
    z = (rng.random(4000) < 0.5).astype(np.int64)
    x = z + 0.2 * rng.normal(size=4000)
    y = z + 0.2 * rng.normal(size=4000)
    assert interaction_information(x, y, z) < 0


def test_interaction_information_synergy_is_positive():
    """XOR: z = a^b, x=a, y=b => given z, x and y are coupled => II > 0."""
    rng = np.random.default_rng(5)
    a = (rng.random(4000) < 0.5).astype(np.int64)
    b = (rng.random(4000) < 0.5).astype(np.int64)
    z = (a ^ b)
    x = a + 0.05 * rng.normal(size=4000)
    y = b + 0.05 * rng.normal(size=4000)
    assert interaction_information(x, y, z) > 0


def test_redundancy_report_structure():
    rng = np.random.default_rng(6)
    F = rng.random((1000, 4)).astype(np.float32)
    y = (rng.random(1000) < 0.3).astype(np.float32)
    rep = redundancy_report(F, y, feature_names=["a", "b", "c", "d"])
    assert set(rep["per_feature_mi"].keys()) == {"a", "b", "c", "d"}
    assert len(rep["pairs"]) == 6  # C(4,2)
    assert "fusion_near_optimal_signal" in rep
    for p in rep["pairs"]:
        assert p["verdict"] in {"redundant", "synergistic", "independent"}


# ---- attention-trace extractor core (CPU) ----
def test_blockify_1d_and_2d():
    x = np.arange(10, dtype=np.float32)
    b = blockify(x, block_size=4)               # blocks: [0..3],[4..7],[8,9]
    assert b.shape == (3,)
    assert b[0] == pytest.approx(1.5) and b[2] == pytest.approx(8.5)
    K = np.ones((10, 5), dtype=np.float32)
    bk = blockify(K, block_size=4)
    assert bk.shape == (3, 5)


def test_update_ema_growing_block_count():
    e1 = update_ema(None, np.array([1.0, 1.0], np.float32))
    e2 = update_ema(e1, np.array([0.0, 0.0, 5.0], np.float32), decay=0.5)
    assert e2.shape == (3,)
    assert e2[0] == pytest.approx(0.5)   # 0.5*1 + 0.5*0
    assert e2[2] == pytest.approx(5.0)   # new block seeds at its value


def test_labels_for_horizons_prefix():
    seq = [np.array([0.1, 0.9]), np.array([0.9, 0.1, 0.5]), np.array([0.2, 0.2, 0.9, 0.1])]
    labels = labels_for_horizons(seq, t=0, n_blocks_t=2, r_label=0.5, horizons=(1, 1, 1, 1))
    # at t+1, restricted to first 2 blocks [0.9,0.1]; top-50% -> block 0
    assert labels[1].shape == (2,)
    assert labels[1][0] == 1 and labels[1][1] == 0


def test_extract_attention_traces_cuda_guard():
    """device='cuda' with no CUDA must raise, not silently run on CPU."""
    import torch  # present in this env
    if torch.cuda.is_available():
        pytest.skip("CUDA available; guard not exercised")
    with pytest.raises(RuntimeError):
        extract_attention_traces("any/model", ["hi"], "/tmp/should_not_write.jsonl",
                                 device="cuda")


def test_extract_attention_traces_cpu_tiny_model(tmp_path):
    """End-to-end extraction on a tiny offline Llama (de-risks the A100 run)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import warnings
    warnings.filterwarnings("ignore")
    from transformers import LlamaConfig, LlamaForCausalLM
    from xqp.trace import load_trace
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=128, attn_implementation="eager")
    model = LlamaForCausalLM(cfg)
    out = tmp_path / "t.jsonl"
    n = extract_attention_traces(model=model, input_ids=[torch.randint(0, 64, (1, 16))],
                                 out_path=str(out), device="cpu", block_size=4,
                                 max_new_tokens=4)
    rows = load_trace(str(out))
    assert n > 0
    assert "y_h4" in rows and "f_query" in rows and "f_within" in rows
    assert set(np.unique(rows["f_cross"]).tolist()).issubset({0.0, 1.0})
    assert rows["f_within"].max() <= 1.01 and rows["f_query"].max() <= 1.01


# ---- baselines + stats ----
def test_baselines_produce_valid_probabilities():
    from xqp.baselines import single_signal_baselines, all_learned_baselines
    from xqp.eval import synthetic_dataset
    F, y = synthetic_dataset(n_blocks=128, n_steps=24, seed=0)
    ntr = int(0.8 * len(F))
    # learned baselines output proper probabilities in [0, 1]
    for b in all_learned_baselines(F[:ntr], y[:ntr]):
        s = b.score(F[ntr:])
        assert s.shape == (len(F) - ntr,)
        assert (s >= -1e-6).all() and (s <= 1 + 1e-6).all(), b.name
    # single-signal heuristics are *ranking* scores (finite); on real traces
    # they lie in [0,1], but the synthetic generator can nudge f_query slightly
    # out of range (AUDIT.md #4), so we only require finiteness here.
    for b in single_signal_baselines():
        s = b.score(F[ntr:])
        assert s.shape == (len(F) - ntr,) and np.all(np.isfinite(s)), b.name


def test_bootstrap_ci_brackets_mean():
    from xqp.stats import bootstrap_ci
    from xqp.eval import synthetic_dataset, roc_auc
    F, y = synthetic_dataset(n_blocks=128, n_steps=24, seed=0)
    ci = bootstrap_ci(roc_auc, y, F[:, 0], n_boot=300)
    assert ci["lo"] <= ci["mean"] <= ci["hi"] and ci["n"] > 0


def test_paired_bootstrap_identical_is_insignificant():
    from xqp.stats import paired_bootstrap_test
    from xqp.eval import synthetic_dataset, roc_auc
    F, y = synthetic_dataset(n_blocks=128, n_steps=24, seed=0)
    t = paired_bootstrap_test(roc_auc, y, F[:, 0], F[:, 0], n_boot=300)
    assert abs(t["delta"]) < 1e-9 and t["p_value"] > 0.5


def test_paired_bootstrap_detects_difference():
    from xqp.stats import paired_bootstrap_test
    from xqp.eval import synthetic_dataset, roc_auc
    F, y = synthetic_dataset(n_blocks=256, n_steps=48, seed=0)
    t = paired_bootstrap_test(roc_auc, y, F[:, 0], F[:, 3], n_boot=500)  # within vs recency
    assert t["delta"] > 0.1 and t["p_value"] < 0.05


# ---- conformal saliency sets under drift (methodological novelty) ----
def test_conformal_holds_coverage_under_drift():
    """ACI tracks the target miss rate under drift; a fixed threshold does not."""
    from xqp.conformal import run_conformal_stream

    class Col0:
        def score(self, F):
            return np.clip(np.asarray(F, np.float32)[:, 0], 0, 1)

    rng = np.random.default_rng(0)
    T, n = 300, 200
    stream = []
    for t in range(T):
        mu = 0.85 - 0.55 * (t / T)                 # positives drift to harder scores
        y = (rng.random(n) < 0.10).astype(np.float32)
        s = np.where(y > 0.5, rng.normal(mu, 0.12, n), rng.normal(0.20, 0.12, n))
        stream.append((np.clip(s, 0, 1).reshape(-1, 1), y))
    ad = run_conformal_stream(Col0(), stream, alpha=0.10, gamma=0.05, adaptive=True, tau0=0.5)
    fx = run_conformal_stream(Col0(), stream, alpha=0.10, gamma=0.0, adaptive=False, tau0=0.5)
    assert abs(ad["realized_miss_rate"] - 0.10) < 0.05          # ACI tracks target
    assert fx["realized_miss_rate"] > ad["realized_miss_rate"] + 0.15  # fixed drifts away
    assert 0.0 <= ad["avg_set_size"] <= 1.0
