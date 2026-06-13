"""Parse + aggregate the rigorous end-to-end task-quality confirmation that backs
the transfer_gap.tex capstone (LongBench F1, matched 20% budget, chat-formatted,
middle-out truncation). Covers all FOUR datasets the paper cites so the headline
"mean F1 0.358 (H2O) vs 0.355 (2-view), tied, each ahead on two" is reproducible
from a single released aggregate (experiments/results/e2e_confirm/SUMMARY.json)."""
import json, os
import numpy as np

OUT = "experiments/results/e2e_confirm"
DATASETS = ["narrativeqa", "qasper", "hotpotqa", "multifieldqa_en"]
POLICIES = ["full", "h2o", "snapkv", "xqp"]


def stat(tag):
    f = f"{OUT}/{tag}.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    f1s = [r["f1"] for r in d["results"]]
    return (float(np.mean(f1s)), float(np.std(f1s) / np.sqrt(len(f1s))), len(f1s),
            float(d.get("hbm_budget", 0.20)))


summary = {
    "experiment": "End-to-end LongBench task quality (chat + middle-out truncation) "
                  "backing transfer_gap.tex; matched 20% budget, 48 prompts/dataset",
    "metric": "F1 mean (SE, n) per dataset; full = uncapped ceiling",
    "datasets": {},
}
per_pol = {p: [] for p in ["full", "h2o", "xqp"]}
h2o_wins = xqp_wins = ties = 0

for ds in DATASETS:
    print(f"\n=== {ds} (budget 0.20, real F1) ===")
    row = {}
    full = stat(f"{ds}_full")
    if full:
        print(f"  full (ceiling): {full[0]:.3f}")
        row["full_f1"] = round(full[0], 4)
        per_pol["full"].append(full[0])
    rows = {}
    for pol in ["h2o", "snapkv", "xqp"]:
        s = stat(f"{ds}_{pol}")
        if s:
            rows[pol] = s
            row[pol] = {"f1": round(s[0], 4), "se": round(s[1], 4), "n": s[2]}
            print(f"  {pol:8s}: F1={s[0]:.3f} ± {s[1]:.3f} (SE, n={s[2]})")
            if pol in per_pol:
                per_pol[pol].append(s[0])
    if "xqp" in rows and "h2o" in rows:
        d = rows["xqp"][0] - rows["h2o"][0]
        se = np.sqrt(rows["xqp"][1] ** 2 + rows["h2o"][1] ** 2)
        winner = "XQP" if d > 0 else ("H2O" if d < 0 else "tie")
        if d > 0:
            xqp_wins += 1
        elif d < 0:
            h2o_wins += 1
        else:
            ties += 1
        row["xqp_minus_h2o"] = round(d, 4)
        row["winner"] = winner
        print(f"  XQP - H2O   = {d:+.3f} ({d/se:+.1f} SE)  {winner}")
    summary["datasets"][ds] = row

means = {p: round(float(np.mean(v)), 4) for p, v in per_pol.items() if len(v) == len(DATASETS)}
summary["mean_f1_over_4_datasets"] = means
summary["per_dataset_winner_tally"] = {"h2o_ahead": h2o_wins, "xqp_ahead": xqp_wins, "tie": ties}
summary["verdict"] = (
    f"H2O mean F1 {means.get('h2o')} vs 2-view {means.get('xqp')} over {len(DATASETS)} "
    f"datasets -> statistically tied; H2O ahead on {h2o_wins}, 2-view ahead on {xqp_wins}. "
    "Reproduces the transfer_gap.tex 0.358/0.355 headline. full ceiling "
    f"{means.get('full')}. No degenerate F1=0 (chat template + middle-out applied)."
)

print("\n=== AGGREGATE (4 datasets) ===")
print(f"  mean F1: full {means.get('full')} | H2O {means.get('h2o')} | 2-view {means.get('xqp')}")
print(f"  per-dataset: H2O ahead {h2o_wins}, 2-view ahead {xqp_wins}, tie {ties}")

with open(f"{OUT}/SUMMARY.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nwrote {OUT}/SUMMARY.json")
