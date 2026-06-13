"""De-confound: is captured% driven by base-accuracy REGIME or by SIGNAL-TYPE?

Design: hold SIGNAL-TYPE CONSTANT (all cells use the same self-consistency agreement signal,
method ASC.8) and vary base-accuracy across many cells = {model x task} plus MATH500 split
BY LEVEL (same task family, same signal, only difficulty changes). If captured% tracks
base-accuracy across these cells, the 'recoverable-regime' effect is real and NOT a
signal-type artifact (signal-type is fixed). Report captured% vs base-acc + correlation +
a quadratic fit (inverted-U expected: low at hopeless and at easy, high at mid).
"""
import json, glob, numpy as np
from protocol import audit_point
import samp_audit

CELLS = []  # (label, base_acc, captured, m_rand, p, n)


def cell_from(outcome, budgets, answers, label):
    base = float(outcome[:, 0].mean())
    r = audit_point(outcome, budgets, samp_audit.asc_alloc(answers, 0.8))
    CELLS.append((label, base, r["frac_captured"], r["method_minus_random"], r["p"], len(answers)))


def math_levels(path):
    """join MATH500 level per instance by line order (sample_gen reads in order)."""
    rows = [json.loads(l) for l in open("data/math500.jsonl")]
    return [str(rows[i].get("level", "?")) if i < len(rows) else "?" for i in range(10**6)]


for f in sorted(glob.glob("results/samp_*_*.jsonl")):
    tag = f.split("/")[-1].replace("samp_", "").replace(".jsonl", "")
    try:
        outcome, budgets, answers = samp_audit.load(f)
    except Exception:
        continue
    cell_from(outcome, budgets, answers, tag)
    if "math" in tag:                                   # split by difficulty level (same task+signal)
        lv = math_levels(f)
        R = [json.loads(l) for l in open(f)]
        groups = {"L1-2": [], "L3": [], "L4-5": []}
        for i, r in enumerate(R):
            L = lv[i]
            g = "L1-2" if L in ("Level 1", "Level 2", "1", "2") else ("L3" if L in ("Level 3", "3") else "L4-5")
            groups[g].append(i)
        for gname, idx in groups.items():
            if len(idx) < 30:
                continue
            sub_out = outcome[idx]; sub_ans = [answers[i] for i in idx]
            cell_from(sub_out, budgets, sub_ans, f"{tag}|{gname}")

# ---- analysis ----
CELLS.sort(key=lambda c: c[1])
print(f"=== De-confound: captured% vs base-accuracy, SIGNAL-TYPE FIXED (SC/ASC.8), {len(CELLS)} cells ===")
print(f"{'cell':>26} {'base_acc':>8} {'captured':>8} {'m-rand':>8} {'p':>7} {'n':>5}")
for label, base, capt, mr, p, n in CELLS:
    print(f"{label:>26} {base:>8.2f} {100*capt:>7.0f}% {mr:>+8.3f} {p:>7.3f} {n:>5}")

ba = np.array([c[1] for c in CELLS]); ca = np.array([c[2] for c in CELLS])
finite = np.isfinite(ca)
ba, ca = ba[finite], ca[finite]
r_lin = float(np.corrcoef(ba, ca)[0, 1])
# quadratic fit captured ~ a*base + b*base^2 + c  (inverted-U test)
X = np.column_stack([np.ones_like(ba), ba, ba ** 2])
coef, *_ = np.linalg.lstsq(X, ca, rcond=None)
pred = X @ coef
ss_res = ((ca - pred) ** 2).sum(); ss_tot = ((ca - ca.mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
peak = -coef[1] / (2 * coef[2]) if coef[2] != 0 else float("nan")
print(f"\nlinear  r(base_acc, captured) = {r_lin:+.2f}")
print(f"quadratic fit R^2 = {r2:.2f} ; vertex (peak base-acc) = {peak:.2f} ; concavity = {'inverted-U' if coef[2]<0 else 'U/linear'}")
print("INTERPRETATION: signal-type is held constant (all SC); if captured% rises then falls with")
print("base-accuracy (inverted-U, peak in mid-range), the 'recoverable-regime' effect is real and")
print("NOT a signal-type confound. If flat, captured% is signal-driven not regime-driven.")
