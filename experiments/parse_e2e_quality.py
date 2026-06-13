"""Parse the real end-to-end LongBench task-quality sweep (F1 vs full at matched budget)."""
import json, os
import numpy as np

OUT = "experiments/results/e2e_quality"


def f1(tag):
    f = f"{OUT}/{tag}.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    return d.get("f1_mean"), d.get("em_mean"), len(d.get("results", []))


full = f1("full_b1.0")
print(f"FULL CACHE (upper bound): F1={full[0]:.3f}  EM={full[1]:.3f}  (n={full[2]})" if full else "full missing")
print("\nReal LongBench (narrativeqa) F1 at matched budget — higher=better:")
print(f"{'budget':8s} {'H2O':10s} {'XQP(2view)':12s} {'native':10s}")
for b in ["0.10", "0.20", "0.30"]:
    cells = []
    for pol in ["h2o", "xqp", "native"]:
        r = f1(f"{pol}_b{b}")
        cells.append(f"{r[0]:.3f}" if r else "--")
    print(f"{b:8s} {cells[0]:10s} {cells[1]:12s} {cells[2]:10s}")
print("\nΔ vs H2O (positive = learned beats H2O on real task quality):")
for b in ["0.10", "0.20", "0.30"]:
    h = f1(f"h2o_b{b}")
    for pol in ["xqp", "native"]:
        r = f1(f"{pol}_b{b}")
        if h and r:
            print(f"  b{b} {pol:8s}: {r[0]-h[0]:+.3f}  {'WIN' if r[0] > h[0] else 'lose/tie'}")
