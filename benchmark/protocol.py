"""KVSalienceBench — frozen evaluation protocol.

The benchmark task: predict, per KV *block* (contiguous group of token positions,
default 32), the probability that the block will be in the top-r most-attended set
h steps in the future, from cheap per-block features. This module freezes the
canonical protocol so every submission is scored identically.

Design decisions that are LOAD-BEARING (do not change without versioning):
  * REQUEST-LEVEL (group) hold-out split: whole prompts are held out, never rows.
    Rows from one prompt are highly correlated; a row-level split leaks and
    inflates AUC. Request ids are recovered from the deterministic emission order
    (first row of each prompt is uniquely step==0 & layer==0 & block_idx==0).
  * FINITE-FILTER: fp16 attention can overflow to NaN in a few last-layer rows;
    those rows are dropped (not imputed).
  * Headline horizon h4; top-r = 0.10 (10% of blocks are "salient").
  * Metrics: AUC (rank), AUPRC (imbalanced rank), P@k / R@k at k=r (selection),
    ECE + Brier (calibration). Calibration is first-class: a selector that drives
    a memory budget must emit trustworthy probabilities, not just a good ranking.
  * Confidence intervals: request-CLUSTERED bootstrap (resample whole prompts).

A submission is any object/callable mapping a feature matrix to probabilities in
[0,1]; see benchmark/run_leaderboard.py and benchmark/submit_template.py.
"""
from __future__ import annotations

import glob as _glob
import json as _json
import os
import sys

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xqp.features import FEATURE_NAMES  # ("s_within","s_cross","s_query","s_pos")
from xqp.dm_metrics import (
    average_precision, precision_at_k, recall_at_k,
    expected_calibration_error, brier_score,
)

# ---- frozen constants -------------------------------------------------------
HORIZONS = ("h1", "h4", "h16", "h64")
HEADLINE_H = "h4"
TOP_R = 0.10                 # label: block in top-10% attended at t+h
BLOCK_SIZE = 32
TEST_FRAC = 0.25             # request-level hold-out fraction
SEED = 0
N_BOOT = 250
PROTOCOL_VERSION = "kvsaliencebench-1.0"


def roc_auc(y_true, y_score) -> float:
    y = np.asarray(y_true, np.float64).reshape(-1)
    s = np.asarray(y_score, np.float64).reshape(-1)
    npos = float(y.sum()); nneg = float(y.size) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


# ---- loading ----------------------------------------------------------------
def load_trace(path: str, horizon: str = HEADLINE_H, cap: int | None = None) -> dict:
    """Stream one JSONL trace into finite-filtered arrays + recovered request ids.

    `cap` stops after that many rows (still respecting prompt boundaries for the
    request-id recovery); used for fast fitting of the tiny reference model."""
    fw, fc, fq, fp, y, lay, stp, blk = [], [], [], [], [], [], [], []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = _json.loads(line)
            fw.append(r["f_within"]); fc.append(r["f_cross"])
            fq.append(r["f_query"]); fp.append(r["f_pos"])
            y.append(r[f"y_{horizon}"])
            lay.append(r["layer"]); stp.append(r["step"]); blk.append(r["block_idx"])
            if cap and len(fw) >= cap:
                break
    F = np.stack([np.asarray(fw, np.float32), np.asarray(fc, np.float32),
                  np.asarray(fq, np.float32), np.asarray(fp, np.float32)], 1)
    lay = np.asarray(lay, np.int32); stp = np.asarray(stp, np.int32); blk = np.asarray(blk, np.int32)
    boundary = (stp == 0) & (lay == 0) & (blk == 0)
    rid = (np.cumsum(boundary) - 1).astype(np.int32)
    y = np.asarray(y, np.int8)
    finite = np.isfinite(F).all(1)
    return dict(F=F[finite], y=y[finite], rid=rid[finite],
                n_requests=int(boundary.sum()), n_dropped=int((~finite).sum()))


def load_corpus(traces_glob: str, horizon: str = HEADLINE_H, cap: int | None = None) -> dict:
    """Load + pool a glob of trace files; request ids namespaced across files."""
    Fs, ys, rids = [], [], []
    off = 0
    files = sorted(f for f in _glob.glob(traces_glob) if ".smoke." not in f)
    for f in files:
        d = load_trace(f, horizon, cap=cap)
        Fs.append(d["F"]); ys.append(d["y"]); rids.append(d["rid"] + off)
        off += d["n_requests"]
    if not Fs:
        raise FileNotFoundError(f"no traces matched {traces_glob}")
    return dict(F=np.concatenate(Fs), y=np.concatenate(ys),
                rid=np.concatenate(rids), n_requests=off, files=files)


# ---- split / metrics --------------------------------------------------------
def request_split(rid: np.ndarray, frac: float = TEST_FRAC, seed: int = SEED):
    uniq = np.unique(rid)
    rng = np.random.default_rng(seed)
    test = set(rng.permutation(uniq)[: max(1, int(frac * len(uniq)))].tolist())
    is_te = np.isin(rid, list(test))
    return np.where(~is_te)[0], np.where(is_te)[0]


def metrics(y, p) -> dict:
    """The canonical metric set on (labels, probabilities)."""
    y = np.asarray(y); p = np.asarray(p)
    return dict(
        auc=roc_auc(y, p),
        auprc=average_precision(y, p),
        p_at_k=precision_at_k(y, p, TOP_R),
        r_at_k=recall_at_k(y, p, TOP_R),
        ece=expected_calibration_error(y, p),
        brier=brier_score(y, p),
    )


def clustered_bootstrap_ci(y, p, groups, fn=roc_auc, n_boot=N_BOOT, seed=SEED):
    """95% CI by resampling whole REQUESTS (groups), not rows."""
    y = np.asarray(y); p = np.asarray(p); groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_g[g] for g in samp])
        vals.append(fn(y[rows], p[rows]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(np.nanmean(vals)), float(lo), float(hi)


def evaluate(score_fn, corpus: dict, with_ci: bool = True, seed: int = SEED) -> dict:
    """Score a submission on the held-out request split. `score_fn` maps a
    (N,4) feature matrix (column order FEATURE_NAMES) to probabilities in [0,1]."""
    _, te = request_split(corpus["rid"], seed=seed)
    Fte, yte, gte = corpus["F"][te], corpus["y"][te], corpus["rid"][te]
    p = np.asarray(score_fn(Fte), np.float64).reshape(-1)
    if p.shape[0] != yte.shape[0]:
        raise ValueError(f"score_fn returned {p.shape[0]} probs for {yte.shape[0]} rows")
    out = dict(protocol=PROTOCOL_VERSION, horizon=HEADLINE_H,
               n_test=int(yte.size), n_test_requests=int(np.unique(gte).size),
               pos_rate=float(yte.mean()), **metrics(yte, p))
    if with_ci:
        m, lo, hi = clustered_bootstrap_ci(yte, p, gte)
        out["auc_ci95"] = [lo, hi]
    return out


FEATURE_COLUMNS = list(FEATURE_NAMES)
