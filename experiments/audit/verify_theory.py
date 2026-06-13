"""Numerically verify the decomposition A(method)-A_unif = S + D on real sampling cells,
and confirm D <= 0 (the concavity/budget-variance penalty)."""
import glob, numpy as np
import samp_audit
from protocol import step_outcome, permuted_random

print(f"{'cell':>22} {'B':>5} {'A_meth':>7} {'A_rand':>7} {'A_unif':>7} {'S(signal)':>9} {'D(budget)':>9} {'D<=0?':>6}")
for f in sorted(glob.glob("results/samp_*_gsm8k.jsonl") + glob.glob("results/samp_*_math.jsonl")):
    tag = f.split("/")[-1].replace("samp_", "").replace(".jsonl", "")
    try:
        outcome, budgets, answers = samp_audit.load(f)
    except Exception:
        continue
    alloc = samp_audit.asc_alloc(answers, 0.8)
    B = float(alloc.mean())
    A_meth = float(step_outcome(outcome, budgets, alloc).mean())
    A_rand, _ = permuted_random(outcome, budgets, alloc)
    # uniform at the grid level nearest B
    gi = int(np.clip(np.searchsorted(budgets, B, side="right") - 1, 0, len(budgets) - 1))
    A_unif = float(outcome[:, gi].mean())
    S = A_meth - A_rand; D = A_rand - A_unif
    print(f"{tag:>22} {B:>5.1f} {A_meth:>7.3f} {A_rand:>7.3f} {A_unif:>7.3f} {S:>+9.3f} {D:>+9.3f} {str(D<=1e-6):>6}")
print("\nProp.1: A_meth - A_unif = S + D (exact). Prop.2: D = A_rand - A_unif <= 0 under concavity.")
print("Signal-attributable gain is S = A_meth - A_rand (what the audit reports), NOT A_meth - A_unif.")
