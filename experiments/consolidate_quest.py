"""Consolidate the faithful-query GPU probe into one decisive 4-model verdict.

Reads the per-model probe traces written by run_quest_baseline.py (which contain
the faithful per-head-max Quest column f_query_dotmax), FINITE-FILTERS rows
(run_quest_baseline.load_rows did not, so fp16-NaN f_within contaminated Qwen2.5),
and reports, per model and pooled:
  * single-view AUC (within / cross / query-cosine / dotmean / dotmax)
  * pooled marginal AUC of each query variant OVER within+cross
  * within-EMA tercile marginal of faithful dotmax (the decisive "magnitude-blind
    regime" test: does query help where within is weak?)
  * unique-information certificate I(X_i;Y|X_{-i}) for [within, cross, dotmax]
    (query "earns its parameter" iff its conditional MI is materially > 0)

    python experiments/consolidate_quest.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from xqp.predictor import ClosedFormXQP
from xqp.info_theory import unique_information_report

PROBE_GLOBS = ["/public/xqp_traces/quest_probe/*.quest.jsonl",
               "/public/xqp_traces/quest_probe_g1/*.quest.jsonl"]
FEATS = ["f_within", "f_cross", "f_query", "f_query_dotmean", "f_query_dotmax"]


def auc(y, s):
    y = np.asarray(y, np.float64); s = np.asarray(s, np.float64)
    npos = y.sum(); nneg = y.size - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def load(path):
    cols = {c: [] for c in FEATS + ["y_h4", "request_id"]}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            for c in cols:
                cols[c].append(r.get(c, 0.0))
    d = {c: np.asarray(cols[c], np.float32) for c in FEATS + ["y_h4"]}
    finite = np.isfinite(np.stack([d[c] for c in FEATS], 1)).all(1)
    drop = int((~finite).sum())
    for c in FEATS + ["y_h4"]:
        d[c] = d[c][finite]
    rid_map = {}
    rids = [rid_map.setdefault(v, len(rid_map)) for v in np.asarray(cols["request_id"])[finite]]
    d["rid"] = np.asarray(rids, np.int32)
    d["_dropped"] = drop
    d["_n"] = int(finite.sum())
    return d


def split(rid, seed=0):
    uniq = np.unique(rid); rng = np.random.default_rng(seed)
    te = set(rng.permutation(uniq)[: max(1, len(uniq) // 4)].tolist())
    is_te = np.isin(rid, list(te))
    return ~is_te, is_te


def fit_base_aug(d, col):
    tr, ev = split(d["rid"])
    y = d["y_h4"].astype(np.float32)
    z = np.zeros_like(d["f_within"])
    base = np.stack([d["f_within"], d["f_cross"], z, z], 1)
    aug = np.stack([d["f_within"], d["f_cross"], zscore(d[col]), z], 1)
    bm = ClosedFormXQP.from_fit(base[tr], y[tr])
    am = ClosedFormXQP.from_fit(aug[tr], y[tr])
    return y, ev, base, aug, bm, am


def analyze(d):
    y_all = d["y_h4"]
    sv = {c: auc(y_all, d[c]) for c in FEATS}
    marg = {}
    for col in ["f_query", "f_query_dotmean", "f_query_dotmax"]:
        y, ev, base, aug, bm, am = fit_base_aug(d, col)
        b, g = auc(y[ev], bm.score(base[ev])), auc(y[ev], am.score(aug[ev]))
        marg[col] = dict(base=b, aug=g, delta=g - b)
    # within-EMA tercile marginal of faithful dotmax on eval
    y, ev, base, aug, bm, am = fit_base_aug(d, "f_query_dotmax")
    w_ev = d["f_within"][ev]; yev = y[ev]
    bs = bm.score(base[ev]); gs = am.score(aug[ev])
    q1, q2 = np.quantile(w_ev, [1 / 3, 2 / 3])
    terc = {}
    for name, m in [("low", w_ev <= q1), ("mid", (w_ev > q1) & (w_ev <= q2)), ("high", w_ev > q2)]:
        terc[name] = dict(base=auc(yev[m], bs[m]), aug=auc(yev[m], gs[m]),
                          delta=auc(yev[m], gs[m]) - auc(yev[m], bs[m]))
    # unique-information certificate on [within, cross, dotmax]
    F3 = np.stack([d["f_within"], d["f_cross"], d["f_query_dotmax"]], 1)
    ui = unique_information_report(F3, y_all, feature_names=["within", "cross", "query_dotmax"])
    return dict(n=d["_n"], dropped_nonfinite=d["_dropped"], pos_rate=float(y_all.mean()),
                single_view_auc=sv, marginal=marg, within_tercile_dotmax=terc, unique_info=ui)


def main():
    files = []
    for g in PROBE_GLOBS:
        files += sorted(glob.glob(g))
    if not files:
        print("no probe traces found", file=sys.stderr); return 1
    per_model, loaded = {}, {}
    for f in files:
        stem = os.path.basename(f)[:-len(".quest.jsonl")]
        print(f"[load] {stem} ...", flush=True)
        d = load(f); loaded[stem] = d
        per_model[stem] = analyze(d)
    # pooled (namespace rids)
    off = 0; pool = {c: [] for c in FEATS + ["y_h4"]}; rids = []
    for stem, d in loaded.items():
        for c in FEATS + ["y_h4"]:
            pool[c].append(d[c])
        rids.append(d["rid"] + off); off += int(d["rid"].max() + 1)
    pd = {c: np.concatenate(pool[c]) for c in FEATS + ["y_h4"]}
    pd["rid"] = np.concatenate(rids); pd["_n"] = int(pd["y_h4"].size); pd["_dropped"] = 0
    pooled = analyze(pd)

    out = dict(per_model=per_model, pooled=pooled)
    os.makedirs("experiments/results", exist_ok=True)
    json.dump(out, open("experiments/results/quest_consolidated.json", "w"), indent=2)

    print("\n=== FAITHFUL-QUERY DECISIVE VERDICT (4 models + pooled, finite-filtered) ===")
    print(f"{'model':<24}{'within':>7}{'cross':>7}{'dotmax':>7}{'  marg+dotmax':>13}"
          f"{'  low-w Δ':>9}{'  uCMI(query)':>13}")
    for k in list(per_model) + ["POOLED"]:
        r = per_model.get(k, pooled)
        sv = r["single_view_auc"]; mg = r["marginal"]["f_query_dotmax"]["delta"]
        lowd = r["within_tercile_dotmax"]["low"]["delta"]
        uq = r["unique_info"]["query_dotmax"]["unique_cmi"]
        print(f"{k:<24}{sv['f_within']:>7.3f}{sv['f_cross']:>7.3f}{sv['f_query_dotmax']:>7.3f}"
              f"{mg:>+13.4f}{lowd:>+9.4f}{uq:>13.4f}")
    uw = pooled["unique_info"]
    print(f"\nunique-info (pooled): within uCMI={uw['within']['unique_cmi']:.4f}  "
          f"cross uCMI={uw['cross']['unique_cmi']:.4f}  query_dotmax uCMI={uw['query_dotmax']['unique_cmi']:.4f}")
    print("VERDICT: gated design warranted iff (marg+dotmax > 0 in low-within) AND (query uCMI >> 0).")
    print("WROTE experiments/results/quest_consolidated.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
