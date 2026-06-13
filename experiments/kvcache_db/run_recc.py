"""RECC experiments: cost-vs-cache-size x workload x policy + Belady-with-recompute oracle.

Reports, per (workload, cache-size): each policy's total recovery cost (s), % saved vs
no-cache, the OPT-RC oracle, and RECC's win vs LRU/GDSF + the LRFU ablation (RECC minus
the recompute-cost term, to isolate recompute-awareness).
"""
import json, os, statistics
from gen_traces import gen_trace, trace_stats
from costmodel import CostModel, BYTES_PER_TOKEN
from sim import run_policy, run_oracle

WORKLOADS = {
    "balanced":   (("system",.30),("rag",.30),("conv",.25),("oneshot",.15)),
    "recency":    (("system",.15),("rag",.15),("conv",.55),("oneshot",.15)),  # conv bursts -> recency strong
    "popularity": (("system",.45),("rag",.40),("conv",.05),("oneshot",.10)),  # Zipf, recency weak
    "cost_heavy": (("system",.10),("rag",.55),("conv",.25),("oneshot",.10)),  # long prefixes dominate
}
SIZES = [0.05, 0.10, 0.20, 0.40]          # total cache as fraction of working-set bytes
SPLIT = {"GPU": 0.10, "DRAM": 0.30, "SSD": 0.60}   # within the total budget
POLICIES = ["lru", "lfu", "gdsf", "marconi", "lrfu", "recc"]


def wss_bytes(trace):
    u = {}
    for _, k, l in trace:
        u[k] = l
    return sum(u.values()) * BYTES_PER_TOKEN


def main(n_req=60_000, seeds=(0, 1), out="results/recc/main.json"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    allres = {}
    for wl, mix in WORKLOADS.items():
        allres[wl] = {}
        # average over seeds
        for size in SIZES:
            agg = {p: [] for p in POLICIES + ["OPT-RC"]}
            saved = {p: [] for p in POLICIES + ["OPT-RC"]}
            p99 = {p: [] for p in POLICIES + ["OPT-RC"]}
            stats0 = None
            for seed in seeds:
                t = gen_trace(n_req, seed=seed, mix=mix)
                if stats0 is None:
                    stats0 = trace_stats(t)
                wss = wss_bytes(t)
                cm = CostModel(capacity={tier: SPLIT[tier] * size * wss for tier in ("GPU", "DRAM", "SSD")})
                nocache = sum(cm.recompute_cost_s(l) for _, _, l in t)
                p99 = {p: [] for p in POLICIES + ["OPT-RC"]}
                for p in POLICIES:
                    m = run_policy(t, cm, p)
                    agg[p].append(m["total_cost_s"]); saved[p].append(100 * (1 - m["total_cost_s"] / nocache))
                    p99[p].append(m["p99_cost_ms"])
                o = run_oracle(t, cm)
                agg["OPT-RC"].append(o["total_cost_s"]); saved["OPT-RC"].append(100 * (1 - o["total_cost_s"] / nocache))
                p99["OPT-RC"].append(o["p99_cost_ms"])
            allres[wl][size] = {p: {"cost_s": statistics.mean(agg[p]), "saved_pct": statistics.mean(saved[p]),
                                    "p99_ms": statistics.mean(p99[p])}
                                for p in agg}
            allres[wl]["_stats"] = {k: round(v, 3) if isinstance(v, float) else v for k, v in stats0.items()}
    json.dump(allres, open(out, "w"), indent=2)

    # ---- print summary ----
    for wl in WORKLOADS:
        print(f"\n===== workload={wl}  {allres[wl]['_stats']} =====")
        print(f"{'size':>6} " + " ".join(f"{p:>8}" for p in POLICIES) + f" {'OPT-RC':>8}  {'RECCvsLRU':>9} {'RECCvsGDSF':>10} {'LRUgap':>8}")
        for size in SIZES:
            r = allres[wl][size]
            row = " ".join(f"{r[p]['saved_pct']:>7.1f}%" for p in POLICIES)
            recc, lru, gdsf, opt = r["recc"]["cost_s"], r["lru"]["cost_s"], r["gdsf"]["cost_s"], r["OPT-RC"]["cost_s"]
            rvl = 100 * (1 - recc / lru); rvg = 100 * (1 - recc / gdsf)
            lru_gap = 100 * (lru / opt - 1)   # how far LRU (the trivial baseline) is above oracle
            best = max(POLICIES, key=lambda p: r[p]["saved_pct"])
            print(f"{size:>6} {row} {r['OPT-RC']['saved_pct']:>7.1f}%  {rvl:>+8.1f}% {rvg:>+9.1f}% {lru_gap:>+9.1f}%  best={best}")
    print(f"\nWROTE {out}")
    print("(saved% higher=better; RECCvsLRU/GDSF positive=RECC cheaper; oracle_gap=RECC above oracle, lower=better)")


if __name__ == "__main__":
    main()
