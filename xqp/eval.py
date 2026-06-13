"""Evaluation utilities — AUC, top-k recall, and WCET measurement scaffolding."""
from __future__ import annotations

import time
from typing import Iterable

import numpy as np

from .predictor import ClosedFormXQP, TinyMLPXQP


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Standard AUC; O(N log N). No sklearn dependency.

    Implementation: Mann-Whitney U via ranks, where the *largest* score gets
    the *largest* rank. AUC = (sum_of_pos_ranks - n_pos(n_pos+1)/2) / (n_pos*n_neg).
    Ties are handled with average ranks.
    """
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float32).reshape(-1)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_score.shape}")
    n_pos = float(y_true.sum())
    n_neg = float(y_true.shape[0]) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average ranks: where ties get the mean of the ranks they span
    order = np.argsort(y_score)             # ascending: lowest score first
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = y_score[order]
    i = 0
    n = y_score.shape[0]
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        # ranks i..j get average (i+1 + j+1)/2 (1-indexed)
        avg = (i + 1 + j + 1) / 2.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    pos_rank_sum = float(ranks[y_true == 1].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def topk_recall(y_true: np.ndarray, y_score: np.ndarray, k_frac: float = 0.10) -> float:
    """Recall of true top-r set in predicted top-r set."""
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    n = y_true.shape[0]
    if n == 0:
        return float("nan")
    k = max(1, int(np.ceil(k_frac * n)))
    pred_topk = set(np.argpartition(-y_score, kth=min(k - 1, n - 1))[:k].tolist())
    true_pos = set(np.where(y_true > 0.5)[0].tolist())
    if not true_pos:
        return float("nan")
    return len(pred_topk & true_pos) / len(true_pos)


def measure_wcet_cpu(predictor, F: np.ndarray, n_replays: int = 10_000) -> dict:
    """Measure predictor latency on CPU as a sanity check; the *real* WCET
    number must come from the TRT+CUDA-Graph path on ga100."""
    timings_us = []
    # warmup
    for _ in range(50):
        predictor.score(F) if isinstance(predictor, ClosedFormXQP) else predictor.score(F, horizon_idx=0)
    for _ in range(n_replays):
        t0 = time.perf_counter_ns()
        if isinstance(predictor, ClosedFormXQP):
            _ = predictor.score(F)
        else:
            _ = predictor.score(F, horizon_idx=0)
        timings_us.append((time.perf_counter_ns() - t0) / 1e3)
    arr = np.asarray(timings_us)
    return dict(
        p50=float(np.percentile(arr, 50)),
        p99=float(np.percentile(arr, 99)),
        p999=float(np.percentile(arr, 99.9)),
        mean=float(arr.mean()),
        std=float(arr.std()),
        n=int(arr.shape[0]),
    )


def synthetic_dataset(n_blocks: int = 512, n_steps: int = 256, seed: int = 0,
                      stability: float = 0.85) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic (features, labels) dataset that resembles the
    HALO/SEER trace statistics: top-10% concentration, Jaccard ~0.7.

    Returns (F, y) with N = n_blocks * n_steps.
    """
    rng = np.random.default_rng(seed)
    Fs = []
    ys = []
    # latent "true importance" with persistence
    importance = rng.gamma(0.5, 1.0, size=n_blocks).astype(np.float32)
    last_used = np.zeros(n_blocks, dtype=np.float32)
    for t in range(n_steps):
        # walk importance to inject stability ~0.7-0.85 Jaccard
        importance = stability * importance + (1 - stability) * rng.gamma(0.5, 1.0, size=n_blocks)
        # within-layer EMA ~ importance + noise
        within = importance + 0.1 * rng.normal(size=n_blocks)
        within = np.clip(within, 0, None)
        # previous layer is a noisy version
        prev = 0.6 * within + 0.4 * rng.gamma(0.5, 1.0, size=n_blocks)
        # K-q product: importance correlates with cosine 0.3 ± noise
        cos = 0.3 * (importance - importance.mean()) / (importance.std() + 1e-6) + 0.5 * rng.normal(size=n_blocks)
        # recency: random subset accessed recently
        accessed = rng.random(n_blocks) < 0.2
        last_used[accessed] = t

        # Build raw features (we don't call extract_features because we don't have q,K here)
        # Instead, hand-construct the same 4 columns matching the feature schema
        from .features import topk_indicator, recency
        s_within = within / (within.max() + 1e-9)
        s_cross = topk_indicator(prev, 0.10)
        s_query = 0.5 * (cos + 1.0)
        s_pos = recency(t, last_used, window=64.0)
        F = np.stack([s_within, s_cross, s_query, s_pos], axis=1).astype(np.float32)
        # Label: was the block in top-10% at the *next* step?
        next_within = stability * within + (1 - stability) * rng.gamma(0.5, 1.0, size=n_blocks)
        next_within = np.clip(next_within, 0, None)
        y = topk_indicator(next_within, 0.10).astype(np.int64)
        Fs.append(F)
        ys.append(y)
    return np.concatenate(Fs, axis=0), np.concatenate(ys, axis=0)


def _eval_predictor_on_trace(pred: ClosedFormXQP, rows: dict, horizon: str) -> dict:
    """AUC + top-10% recall of one loaded predictor on one loaded trace."""
    F = np.stack([rows["f_within"], rows["f_cross"], rows["f_query"], rows["f_pos"]],
                 axis=1).astype(np.float32)
    y = rows[f"y_{horizon}"].astype(np.float32)
    if getattr(pred, "per_layer", False):
        layer_ids = rows["layer"].astype(np.int64)
        s = np.zeros(F.shape[0], dtype=np.float32)
        for l in np.unique(layer_ids):
            m = layer_ids == l
            s[m] = pred.score(F[m], layer=int(l))
    else:
        s = pred.score(F)
    return dict(auc=roc_auc(y, s), top10_recall=topk_recall(y, s, 0.10),
                n=int(F.shape[0]))


def main(argv=None):
    """xqp-eval — score trained predictors against JSONL traces (CPU).

    Usage: xqp-eval --traces DIR --predictors DIR --out FILE
    Pairs each `<model>_<horizon>[...].json` predictor with the trace
    `<model>.jsonl` and reports AUC + top-10% recall. This is the `e1` block
    of scripts/run_benchmarks.sh; the WCET (`e2`) and TPOT (`e3`) blocks need
    ga100. Predictor filename convention matches scripts/train_predictor.sh.
    """
    import argparse
    import glob
    import json
    import os
    import re
    from pathlib import Path

    from .trace import load_trace

    p = argparse.ArgumentParser(prog="xqp-eval")
    p.add_argument("--traces", required=True, help="dir of <model>.jsonl traces")
    p.add_argument("--predictors", required=True, help="dir of predictor JSONs")
    p.add_argument("--out", required=True, help="output JSON path")
    a = p.parse_args(argv)

    results = {}
    for pf in sorted(glob.glob(os.path.join(a.predictors, "*.json"))):
        name = Path(pf).stem  # e.g. Meta-Llama-3-8B-Instruct_h4[_perlayer]
        m = re.search(r"_(h\d+)", name)
        horizon = m.group(1) if m else "h4"
        short = name[:m.start()] if m else name
        trace_path = os.path.join(a.traces, short + ".jsonl")
        if not os.path.exists(trace_path):
            results[name] = {"error": f"trace not found: {trace_path}"}
            continue
        rows = load_trace(trace_path)
        if not rows or f"y_{horizon}" not in rows:
            results[name] = {"error": "empty trace or missing horizon column"}
            continue
        results[name] = _eval_predictor_on_trace(ClosedFormXQP.load(pf), rows, horizon)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
