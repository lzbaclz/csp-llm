"""KVSalienceBench leaderboard runner.

Scores the reference baseline and (optionally) a submission under the frozen
protocol (benchmark/protocol.py) on a trace corpus, and prints a leaderboard row
per scorer. A submission is a .py file exposing `score(F)` mapping an (N,4)
feature matrix (column order benchmark.protocol.FEATURE_COLUMNS) to probabilities
in [0,1]. See benchmark/submit_template.py.

    python benchmark/run_leaderboard.py --traces '/public/xqp_traces/*.jsonl'
    python benchmark/run_leaderboard.py --submission my_method.py
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from benchmark import protocol as P
from xqp.predictor import ClosedFormXQP
from xqp.baselines import single_signal_baselines, all_learned_baselines


def reference_scorer(path="benchmark/reference_model.json"):
    d = json.load(open(os.path.join(ROOT, path)))
    w = np.array([d["weights"][c] for c in P.FEATURE_COLUMNS], np.float32)
    m = ClosedFormXQP(weights=w, bias=np.float32(d["bias"]))
    return d["name"], (lambda F: m.score(np.asarray(F, np.float32)))


def load_submission(pyfile):
    spec = importlib.util.spec_from_file_location("submission", pyfile)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "score"):
        raise AttributeError(f"{pyfile} must define score(F) -> probabilities in [0,1]")
    return os.path.basename(pyfile), mod.score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="/public/xqp_traces/*.jsonl")
    ap.add_argument("--cap", type=int, default=400_000, help="rows/file cap (None=all)")
    ap.add_argument("--submission", default=None, help="path to a .py exposing score(F)")
    ap.add_argument("--seeded-baselines", action="store_true",
                    help="also score the built-in single-signal + learned baselines")
    a = ap.parse_args()

    corpus = P.load_corpus(a.traces, cap=(a.cap or None))
    print(f"corpus: {corpus['n_requests']} requests over {len(corpus['files'])} files\n")

    rows = []
    name, fn = reference_scorer()
    rows.append(("REFERENCE " + name.split("(")[0].strip(), P.evaluate(fn, corpus)))

    if a.seeded_baselines:
        tr, _ = P.request_split(corpus["rid"])
        F, y = corpus["F"], corpus["y"].astype(np.float32)
        for b in single_signal_baselines():
            rows.append((b.name, P.evaluate(b.score, corpus)))
        for b in all_learned_baselines(F[tr], y[tr], seed=P.SEED):
            rows.append((b.name, P.evaluate(b.score, corpus)))

    if a.submission:
        sname, sfn = load_submission(a.submission)
        rows.append(("SUBMISSION " + sname, P.evaluate(sfn, corpus)))

    print(f"{'scorer':<42}{'AUC':>8}{'AUPRC':>8}{'P@10':>7}{'R@10':>7}{'ECE':>8}")
    print("-" * 80)
    for nm, r in sorted(rows, key=lambda kv: -kv[1]["auc"]):
        print(f"{nm[:42]:<42}{r['auc']:>8.4f}{r['auprc']:>8.4f}"
              f"{r['p_at_k']:>7.3f}{r['r_at_k']:>7.3f}{r['ece']:>8.4f}")


if __name__ == "__main__":
    main()
