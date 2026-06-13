"""Parse the design-deploy sweep: eps (oracle FN) per policy x budget, vs H2O."""
import json, glob, os
import numpy as np

OUT = "experiments/results/design_deploy"
POLS = ["h2o", "xqp", "xqpnative", "xqpfull"]
BUDGETS = ["0.10", "0.20", "0.30"]


def agg(f):
    d = json.load(open(f)); e = []; bc = []
    for r in d["results"]:
        e += r.get("per_step_eps_measured", []); bc += r.get("per_step_block_count", [])
    return (float(np.mean(e)) if e else float("nan"),
            float(np.mean(bc)) if bc else float("nan"),
            d["tpot_p99_us"] / 1e3)


rows = {}
for pol in POLS:
    for b in BUDGETS:
        f = f"{OUT}/{pol}_b{b}.json"
        if os.path.exists(f):
            rows[(pol, b)] = agg(f)

print(f"{'policy':12s} " + " ".join(f"b{b}".ljust(10) for b in BUDGETS) + "   (eps = oracle false-neg, lower=better)")
for pol in POLS:
    cells = []
    for b in BUDGETS:
        if (pol, b) in rows:
            e, _, _ = rows[(pol, b)]; cells.append(f"{e:.3f}")
        else:
            cells.append("--")
    print(f"{pol:12s} " + " ".join(c.ljust(10) for c in cells))

print("\nΔ vs H2O (negative = beats H2O):")
for pol in ["xqp", "xqpnative", "xqpfull"]:
    cells = []
    for b in BUDGETS:
        if (pol, b) in rows and ("h2o", b) in rows:
            cells.append(f"{rows[(pol,b)][0]-rows[('h2o',b)][0]:+.3f}")
        else:
            cells.append("--")
    print(f"{pol:12s} " + " ".join(c.ljust(10) for c in cells))
