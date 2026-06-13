#!/usr/bin/env python3
"""Analyze the query-aware-vs-H2O comparison (multi-value NIAH + real multi-hop QA).

Per dataset and pooled: substring-recall and F1 for full/h2o/snapkv/quest at each
budget, plus paired (by request id) snapkv-minus-h2o and quest-minus-h2o differences
with a request-clustered bootstrap 95% CI. Reports the headline win.
"""
import json, os, glob, sys, numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results/multikey/realqa"
BUDGETS = ["0.05", "0.10", "0.20"]
METRIC = "substring"  # or "f1"


def load(ds, sel, b):
    p = f"{ROOT}/{ds}_{sel}_b{b}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return {r["id"]: r[METRIC] for r in d["results"]}, d.get(f"{METRIC}_mean")


def datasets():
    ds = set()
    for f in glob.glob(f"{ROOT}/*_h2o_b{BUDGETS[0]}.json"):
        ds.add(os.path.basename(f).rsplit("_h2o_b", 1)[0])
    return sorted(ds)


def clustered_ci(diffs_by_ds, nboot=10000, seed=0):
    rng = np.random.RandomState(seed)
    uds = list(diffs_by_ds.keys())
    flat = np.concatenate([diffs_by_ds[d] for d in uds]) if uds else np.array([])
    if len(flat) == 0:
        return (float("nan"),) * 3
    boot = []
    for _ in range(nboot):
        ch = rng.choice(len(uds), len(uds), True)
        v = []
        for ci in ch:
            ix = diffs_by_ds[uds[ci]]
            v.extend(ix[rng.choice(len(ix), len(ix), True)])
        boot.append(np.mean(v))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(flat.mean()), float(lo), float(hi)


def main():
    DS = datasets()
    print(f"=== {ROOT}  metric={METRIC}  datasets={DS} ===\n")
    for b in BUDGETS:
        print(f"--- budget {b} ---")
        print(f"{'dataset':<16}{'full':>7}{'h2o':>7}{'snapkv':>8}{'quest':>7}"
              f"{'snap-h2o':>10}{'quest-h2o':>11}")
        snap_by, quest_by = {}, {}
        agg = {s: [] for s in ["full", "h2o", "snapkv", "quest"]}
        for ds in DS:
            row = {}
            for s in ["full", "h2o", "snapkv", "quest"]:
                r = load(ds, s, b)
                row[s] = r
            if row["h2o"] is None or row["snapkv"] is None:
                continue
            ids = sorted(set(row["h2o"][0]) & set(row["snapkv"][0]))
            sd = np.array([row["snapkv"][0][i] - row["h2o"][0][i] for i in ids])
            snap_by[ds] = sd
            if row["quest"] is not None:
                qids = sorted(set(row["h2o"][0]) & set(row["quest"][0]))
                quest_by[ds] = np.array([row["quest"][0][i] - row["h2o"][0][i] for i in qids])
            means = {s: (row[s][1] if row[s] else float("nan")) for s in agg}
            for s in agg:
                if row[s]:
                    agg[s].append(means[s])
            print(f"{ds:<16}{means['full']:>7.3f}{means['h2o']:>7.3f}"
                  f"{means['snapkv']:>8.3f}{means['quest']:>7.3f}"
                  f"{sd.mean():>+10.3f}{(quest_by.get(ds, np.array([np.nan])).mean()):>+11.3f}")
        sm, slo, shi = clustered_ci(snap_by)
        qm, qlo, qhi = clustered_ci(quest_by)
        pooled = {s: np.mean(agg[s]) if agg[s] else float("nan") for s in agg}
        sig_s = "SIG" if slo > 0 else "ns"
        sig_q = "SIG" if qlo > 0 else "ns"
        print(f"{'POOLED':<16}{pooled['full']:>7.3f}{pooled['h2o']:>7.3f}"
              f"{pooled['snapkv']:>8.3f}{pooled['quest']:>7.3f}"
              f"{sm:>+10.3f}{qm:>+11.3f}")
        print(f"  snapkv-h2o 95%CI [{slo:+.3f},{shi:+.3f}] {sig_s}   "
              f"quest-h2o 95%CI [{qlo:+.3f},{qhi:+.3f}] {sig_q}\n")


if __name__ == "__main__":
    main()
