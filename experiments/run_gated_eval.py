"""Evaluate the regime-gated selective cascade (xqp.gated_predictor).

Produces the design's headline evidence:
  * Budget Pareto: query-compute budget beta -> AUC / recall@k, with beta=0 the
    cheap within+cross model and beta=1 the full query-everywhere expert. The
    knee is the selective win (most of the query benefit at a fraction of cost).
  * Per-regime value: the query view's AUC contribution inside the DEFERRED
    (uncertain) set vs the confident set — the design's claim that query earns
    its place only where magnitude is blind.
  * Unique information: I(Xi;Y|X_{-i}) for within/cross/query — certifies each
    retained parameter adds information no combination of the others does.
  * Coverage: the cascade composed with adaptive conformal holds the target
    missed-saliency rate.

Run on current traces (mean query, col 2) to validate the machinery; rerun with
faithful query (extractor query_variants -> extra column) by passing
``--query-col`` to see whether the selective lift materializes.

    python experiments/run_gated_eval.py --traces experiments/traces \
        --out experiments/results/gated_eval.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from run_icdm_full import load_model_trace, pool_models, request_split, subsample, roc_auc
from xqp.dm_metrics import average_precision, recall_at_k, expected_calibration_error
from xqp.predictor import ClosedFormXQP
from xqp.gated_predictor import SelectiveCascadeXQP, _gather4
from xqp.info_theory import unique_information_report
from xqp.conformal import run_conformal_stream

H = "h4"
TRAIN_N = 300_000
TEST_N = 300_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="experiments/traces")
    ap.add_argument("--query-col", type=int, default=2,
                    help="feature column of the (expensive) query view; 2=mean "
                         "cosine in the current 4-col traces")
    ap.add_argument("--rule", default="confidence", choices=["confidence", "cold"])
    ap.add_argument("--out", default="experiments/results/gated_eval.json")
    a = ap.parse_args()

    files = [f for f in sorted(glob.glob(os.path.join(a.traces, "*.jsonl"))) if ".smoke." not in f]
    models = {os.path.basename(f)[:-6]: load_model_trace(f) for f in files}
    models = {k: v for k, v in models.items() if v}
    d = pool_models(models)
    qc = a.query_col
    base_cols, expert_cols = (0, 1), (0, 1, qc)

    tr_idx, te_idx = request_split(d["rid"])
    tr = subsample(tr_idx, TRAIN_N); te = subsample(te_idx, TEST_N)
    Ftr, ytr = d["F"][tr], d["y"][H][tr].astype(np.float32)
    Fte, yte = d["F"][te], d["y"][H][te].astype(np.float32)

    casc = SelectiveCascadeXQP.from_fit(Ftr, ytr, base_cols=base_cols,
                                        expert_cols=expert_cols, query_col=qc)
    out = {"query_col": qc, "rule": a.rule, "n_test": int(len(te))}

    # ---- Budget Pareto -------------------------------------------------------
    out["pareto"] = []
    for beta in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
        p, defer, frac = casc.predict_with_cost(Fte, budget=beta, rule=a.rule)
        out["pareto"].append(dict(
            budget=beta, query_frac=round(frac, 4),
            auc=roc_auc(yte, p), auprc=average_precision(yte, p),
            recall_at_10=recall_at_k(yte, p, 0.10),
            ece=expected_calibration_error(yte, p)))
        print(f"  beta={beta:<4} query_frac={frac:.3f}  AUC={out['pareto'][-1]['auc']:.4f} "
              f"recall@10={out['pareto'][-1]['recall_at_10']:.3f}", flush=True)

    # ---- Per-regime value of the query view ---------------------------------
    p1 = casc.base.score(_gather4(Fte, base_cols))
    conf = np.abs(2 * p1 - 1.0)
    k = int(0.2 * len(te))
    defer = np.zeros(len(te), bool)
    defer[np.argpartition(conf, k)[:k]] = True
    m_wc = np.array([1, 1, 0, 0], np.float32)
    sc_wc = ClosedFormXQP.from_fit(_gather4(Ftr, base_cols), ytr)
    sc_wcq = casc.expert
    for nm, msk in [("deferred(uncertain)", defer), ("confident", ~defer)]:
        ywc = roc_auc(yte[msk], sc_wc.score(_gather4(Fte[msk], base_cols)))
        ywcq = roc_auc(yte[msk], sc_wcq.score(_gather4(Fte[msk], expert_cols)))
        out.setdefault("regime_query_value", {})[nm] = dict(
            within_cross=ywc, plus_query=ywcq, delta=ywcq - ywc, n=int(msk.sum()))
        print(f"  query value in {nm:20s}: within+cross={ywc:.4f} +query={ywcq:.4f} "
              f"delta={ywcq-ywc:+.4f}", flush=True)

    # ---- Unique information certificate -------------------------------------
    mi_s = subsample(np.arange(d["F"].shape[0]), 300_000)
    names = ["within", "cross", "query", "recency"]
    ui = unique_information_report(d["F"][mi_s][:, :4], d["y"][H][mi_s], feature_names=names)
    out["unique_information"] = ui
    print("  unique info I(Xi;Y|rest):", {k: round(v["unique_cmi"], 4) for k, v in ui.items()}, flush=True)

    # ---- Coverage of the cascade under adaptive conformal -------------------
    class _Wrap:
        def score(self, F):
            return casc.score(F, budget=0.2, rule=a.rule)
    step = d["step"][te]
    stream = []
    for s_ in np.unique(step):
        m = step == s_
        if m.sum() >= 8 and 0 < yte[m].sum() < m.sum():
            stream.append((Fte[m], yte[m]))
    if len(stream) >= 8:
        ad = run_conformal_stream(_Wrap(), stream, alpha=0.10, gamma=0.1, adaptive=True)
        out["conformal"] = dict(target=0.10, miss=ad["realized_miss_rate"],
                                set_size=ad["avg_set_size"])
        print(f"  conformal(cascade,beta=.2): miss={ad['realized_miss_rate']:.3f} "
              f"set={ad['avg_set_size']:.3f}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
