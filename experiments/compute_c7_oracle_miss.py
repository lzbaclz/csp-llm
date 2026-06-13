#!/usr/bin/env python
"""C7: Per-task served-oracle miss with paired bootstrap CIs.

For each LongBench task and policy (h2o, xqp):
  per-request eps = mean(per_step_eps_measured)
  dataset mean eps = mean over its requests of per-request eps
Paired diff = xqp_eps - h2o_eps, paired bootstrap (resample requests, cluster=request).
Pooled across the four datasets analogously.
"""
import json
import os
import numpy as np

DATASETS = ["narrativeqa", "qasper", "hotpotqa", "multifieldqa_en"]
POLICIES = ["h2o", "xqp"]
ROOT = "experiments/results/e2e_confirm"
OUT = "experiments/results/served_oracle_ci/c7_oracle_miss_pertask.json"
N_BOOT = 10000
SEED = 1234
CI = 0.95


def load_per_request_eps(ds, pol):
    """Return dict {request_id: mean(per_step_eps_measured)}."""
    d = json.load(open(os.path.join(ROOT, f"{ds}_{pol}.json")))
    out = {}
    for r in d["results"]:
        eps = r.get("per_step_eps_measured")
        assert eps is not None and len(eps) > 0, f"{ds}_{pol} req {r['id']} empty eps"
        out[r["id"]] = float(np.mean(eps))
    return out, float(d["hbm_budget"]), int(d["context_length"])


def boot_paired_ci(h2o_vals, xqp_vals, rng, n_boot=N_BOOT, ci=CI):
    """Paired bootstrap over requests. Returns CIs for h2o mean, xqp mean, diff(xqp-h2o)."""
    h2o_vals = np.asarray(h2o_vals)
    xqp_vals = np.asarray(xqp_vals)
    n = len(h2o_vals)
    assert len(xqp_vals) == n
    boot_h, boot_x, boot_d = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # resample request indices (cluster unit)
        bh = h2o_vals[idx].mean()
        bx = xqp_vals[idx].mean()
        boot_h.append(bh)
        boot_x.append(bx)
        boot_d.append(bx - bh)
    lo = (1 - ci) / 2 * 100
    hi = (1 + ci) / 2 * 100

    def pct(a):
        return [float(np.percentile(a, lo)), float(np.percentile(a, hi))]

    return pct(boot_h), pct(boot_x), pct(boot_d)


def main():
    rng = np.random.default_rng(SEED)

    # Load all per-request eps, aligned by request id.
    per_ds = {}
    budgets = set()
    ctxs = set()
    for ds in DATASETS:
        h2o_map, b_h, c_h = load_per_request_eps(ds, "h2o")
        xqp_map, b_x, c_x = load_per_request_eps(ds, "xqp")
        budgets.update([b_h, b_x])
        ctxs.update([c_h, c_x])
        common_ids = sorted(set(h2o_map) & set(xqp_map))
        assert common_ids, f"no common ids for {ds}"
        h2o_vals = [h2o_map[i] for i in common_ids]
        xqp_vals = [xqp_map[i] for i in common_ids]
        per_ds[ds] = {
            "ids": common_ids,
            "n_requests": len(common_ids),
            "h2o_vals": h2o_vals,
            "xqp_vals": xqp_vals,
        }

    assert budgets == {0.2}, f"unexpected budgets {budgets}"  # 20% retention

    # Per-dataset stats
    per_dataset = {}
    for ds in DATASETS:
        h2o_vals = per_ds[ds]["h2o_vals"]
        xqp_vals = per_ds[ds]["xqp_vals"]
        ci_h, ci_x, ci_d = boot_paired_ci(h2o_vals, xqp_vals, rng)
        h2o_mean = float(np.mean(h2o_vals))
        xqp_mean = float(np.mean(xqp_vals))
        diff = xqp_mean - h2o_mean  # >0 means xqp has higher miss (h2o better)
        # paired per-request diffs for sign-consistency
        pair_d = np.asarray(xqp_vals) - np.asarray(h2o_vals)
        per_dataset[ds] = {
            "n_requests": per_ds[ds]["n_requests"],
            "h2o_eps": h2o_mean,
            "xqp_eps": xqp_mean,
            "diff_xqp_minus_h2o": diff,
            "h2o_eps_ci95": ci_h,
            "xqp_eps_ci95": ci_x,
            "diff_ci95": ci_d,
            "h2o_lower_miss": h2o_mean < xqp_mean,
            "diff_ci_excludes_zero": (ci_d[0] > 0) or (ci_d[1] < 0),
            "frac_requests_xqp_higher": float(np.mean(pair_d > 0)),
            "frac_requests_tie": float(np.mean(pair_d == 0)),
        }

    # Pooled across datasets: pool all requests (each request is a unit).
    pool_h2o = []
    pool_xqp = []
    pool_keys = []  # (ds, id) so bootstrap resamples request-units across pool
    for ds in DATASETS:
        for i, rid in enumerate(per_ds[ds]["ids"]):
            pool_h2o.append(per_ds[ds]["h2o_vals"][i])
            pool_xqp.append(per_ds[ds]["xqp_vals"][i])
            pool_keys.append((ds, rid))
    ci_h, ci_x, ci_d = boot_paired_ci(pool_h2o, pool_xqp, rng)
    pool_h2o = np.asarray(pool_h2o)
    pool_xqp = np.asarray(pool_xqp)
    pool_diff = float(pool_xqp.mean() - pool_h2o.mean())
    pooled = {
        "n_requests": int(len(pool_h2o)),
        "h2o_eps": float(pool_h2o.mean()),
        "xqp_eps": float(pool_xqp.mean()),
        "diff_xqp_minus_h2o": pool_diff,
        "h2o_eps_ci95": ci_h,
        "xqp_eps_ci95": ci_x,
        "diff_ci95": ci_d,
        "h2o_lower_miss": float(pool_h2o.mean()) < float(pool_xqp.mean()),
        "diff_ci_excludes_zero": (ci_d[0] > 0) or (ci_d[1] < 0),
        "frac_requests_xqp_higher": float(np.mean(pool_xqp - pool_h2o > 0)),
    }

    # Macro mean across the four dataset means (the paper's "mean over four tasks")
    macro_h2o = float(np.mean([per_dataset[ds]["h2o_eps"] for ds in DATASETS]))
    macro_xqp = float(np.mean([per_dataset[ds]["xqp_eps"] for ds in DATASETS]))

    consistent = all(per_dataset[ds]["h2o_lower_miss"] for ds in DATASETS)
    all_ci_sig = all(per_dataset[ds]["diff_ci_excludes_zero"] for ds in DATASETS)

    result = {
        "task": "C7 served-oracle miss per-task with paired bootstrap CIs",
        "source_glob": f"{ROOT}/<dataset>_<policy>.json",
        "datasets": DATASETS,
        "policies": {"h2o": "trivial accumulator (lower=better)", "xqp": "learned 2-view scorer"},
        "retention_budget": 0.2,
        "context_length": sorted(ctxs),
        "metric": "per-request mean(per_step_eps_measured) = served-oracle miss",
        "pairing_unit": "request id (same prompt under h2o vs xqp)",
        "bootstrap": {
            "n_boot": N_BOOT,
            "seed": SEED,
            "ci_level": CI,
            "scheme": "paired percentile bootstrap, cluster unit = request",
            "rng": "numpy default_rng",
        },
        "per_dataset": per_dataset,
        "pooled": pooled,
        "macro_mean_over_four_tasks": {
            "h2o_eps": macro_h2o,
            "xqp_eps": macro_xqp,
            "diff_xqp_minus_h2o": macro_xqp - macro_h2o,
            "rounds_to_paper_claim_0p57_vs_0p62": [round(macro_h2o, 2), round(macro_xqp, 2)],
        },
        "claim_check": {
            "paper_claim": "0.57 (h2o) vs 0.62 (xqp) at 20% retention, mean over four LongBench tasks; H2O lower at every budget/dataset",
            "h2o_lower_miss_all_four_datasets": consistent,
            "diff_ci_excludes_zero_all_four": all_ci_sig,
        },
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)

    # Console table
    print("dataset             n   h2o_eps  xqp_eps  diff(xqp-h2o)  diff_CI95              h2o_lower  CI!=0")
    for ds in DATASETS:
        s = per_dataset[ds]
        print(f"{ds:18s} {s['n_requests']:3d}  {s['h2o_eps']:.4f}   {s['xqp_eps']:.4f}   "
              f"{s['diff_xqp_minus_h2o']:+.4f}        [{s['diff_ci95'][0]:+.4f},{s['diff_ci95'][1]:+.4f}]   "
              f"{str(s['h2o_lower_miss']):5s}     {s['diff_ci_excludes_zero']}")
    p = pooled
    print(f"{'POOLED':18s} {p['n_requests']:3d}  {p['h2o_eps']:.4f}   {p['xqp_eps']:.4f}   "
          f"{p['diff_xqp_minus_h2o']:+.4f}        [{p['diff_ci95'][0]:+.4f},{p['diff_ci95'][1]:+.4f}]   "
          f"{str(p['h2o_lower_miss']):5s}     {p['diff_ci_excludes_zero']}")
    print()
    print(f"macro mean over 4 tasks: h2o={macro_h2o:.4f} (~{macro_h2o:.2f}), xqp={macro_xqp:.4f} (~{macro_xqp:.2f})")
    print(f"H2O lower on ALL four datasets: {consistent}; all four diff-CI exclude 0: {all_ci_sig}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
