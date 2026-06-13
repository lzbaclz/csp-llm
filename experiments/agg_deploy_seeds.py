"""Aggregate the multi-seed consistent-code deploy sweep (run_deploy_seeds.sh) into
a seed-band + request-clustered bootstrap CI on the served-oracle miss (eps) and a
seed band on TPOT P99, so tab:tpot can report a CI instead of a single-seed point.

eps is defined exactly as deploy_consistent/SUMMARY.json: the mean of
per_step_eps_measured over all decode steps and requests. The request-clustered
bootstrap resamples whole requests (pooled across seeds), matching the paper's
offline CI methodology.
"""
import json, glob, os, re
import numpy as np

OUT = "experiments/results/deploy_seeds"
POLS = ["h2o", "infinigen", "xqp", "quest", "xqpwithin", "guardkv"]
BUDGETS = ["0.20", "0.30"]
LABEL = {"h2o": "H2O", "infinigen": "InfiniGen", "xqp": "XQP(2-view)",
         "quest": "Quest", "xqpwithin": "XQP(within)", "guardkv": "GuardKV"}


def per_request_eps(path):
    """Return list of per-request eps (mean of that request's per-step eps)."""
    d = json.load(open(path))
    out = []
    for req in d["results"]:
        v = [x for x in req.get("per_step_eps_measured", []) if x is not None]
        if v:
            out.append(float(np.mean(v)))
    return out, float(d["tpot_p99_us"]) / 1000.0, float(d["tpot_p999_us"]) / 1000.0


def boot_ci(vals, n=4000, seed=0):
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.array(vals)
    means = arr[rng.integers(0, len(arr), size=(n, len(arr)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


summary = {"experiment": "multi-seed consistent-code deploy sweep (eps + TPOT CIs)",
           "setup": "Llama-3.1-8B, mooncake, ctx 4096, N=16, measured-dma",
           "eps_def": "mean per_step_eps_measured over all steps x requests (== tab:tpot)",
           "ci": "request-clustered bootstrap (resample whole requests, pooled across seeds)",
           "rows": {}}

found_seeds = set()
for f in glob.glob(f"{OUT}/*_s*.json"):
    m = re.search(r"_s(\d+)\.json$", f)
    if m:
        found_seeds.add(int(m.group(1)))
print(f"seeds present: {sorted(found_seeds)}")

for b in BUDGETS:
    print(f"\n=== budget {b} ===")
    print(f"{'selector':14} {'eps mean':>9} {'eps std':>8} {'eps 95% CI (req-boot)':>24} "
          f"{'P99 ms (mean[min,max])':>24} {'seeds':>6}")
    rank = []
    for pol in POLS:
        files = sorted(glob.glob(f"{OUT}/{pol}_b{b}_s*.json"))
        if not files:
            continue
        seed_eps, p99s, p999s, all_req_eps = [], [], [], []
        for f in files:
            preq, p99, p999 = per_request_eps(f)
            if not preq:
                continue
            seed_eps.append(float(np.mean(preq)))
            p99s.append(p99); p999s.append(p999)
            all_req_eps += preq
        if not seed_eps:
            continue
        eps_mean = float(np.mean(seed_eps)); eps_std = float(np.std(seed_eps))
        lo, hi = boot_ci(all_req_eps)
        p99_mean = float(np.mean(p99s))
        row = {"eps_mean": round(eps_mean, 4), "eps_std_across_seeds": round(eps_std, 4),
               "eps_ci95_reqboot": [round(lo, 4), round(hi, 4)],
               "tpot_p99_ms_mean": round(p99_mean, 2),
               "tpot_p99_ms_range": [round(min(p99s), 2), round(max(p99s), 2)],
               "tpot_p999_ms_mean": round(float(np.mean(p999s)), 2),
               "n_seeds": len(seed_eps), "n_requests_pooled": len(all_req_eps)}
        summary["rows"][f"{pol}_b{b}"] = row
        rank.append((eps_mean, pol))
        print(f"{LABEL[pol]:14} {eps_mean:9.4f} {eps_std:8.4f} "
              f"   [{lo:.4f}, {hi:.4f}]   "
              f"{p99_mean:6.2f} [{min(p99s):.1f},{max(p99s):.1f}]      {len(seed_eps):>3}")
    if rank:
        rank.sort()
        best = rank[0]
        print(f"  best eps: {LABEL[best[1]]} ({best[0]:.4f}); "
              f"order: {' < '.join(LABEL[p] for _, p in rank)}")
        summary["rows"].setdefault("_meta", {})[f"order_b{b}"] = [LABEL[p] for _, p in rank]

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/SUMMARY.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nwrote {OUT}/SUMMARY.json")
