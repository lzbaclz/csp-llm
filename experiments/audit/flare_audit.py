"""Audit FLARE's gate: it retrieves when min-token-prob is LOW. Does ranking by FLARE's
confidence select better which queries to retrieve than matched-budget random? (binary budget)"""
import json, sys, numpy as np
from protocol import audit_point, binary_outcome_table


def audit(path, tag):
    R = [json.loads(l) for l in open(path)]
    mp = np.array([r["flare_minprob"] for r in R]); c = np.array([r["closed_correct"] for r in R]); o = np.array([r["open_correct"] for r in R])
    n = len(R); outcome, budgets = binary_outcome_table(c, o, 1.0)
    print(f"\n### FLARE gate face-slap [{tag}]  N={n}  closed={c.mean():.3f} open={o.mean():.3f} oracle={np.mean(np.maximum(o,c)):.3f}")
    print(f"  FLARE-conf corr with retrieve-helps(o>c): {np.corrcoef(mp,(o>c).astype(float))[0,1]:+.3f}  (gate retrieves low-conf)")
    print(f"  {'retr_rate':>9} {'FLARE':>7} {'random':>7} {'rand_ci':>16} {'oracle':>7} {'capt%':>6} {'F-rand':>8} {'ci':>16} {'p':>6}")
    order = np.argsort(mp)                                  # LOW min-prob (low confidence) -> retrieve first (FLARE policy)
    for p in [0.2, 0.3, 0.5, 0.7]:
        nr = int(p * n); alloc = np.zeros(n); alloc[order[:nr]] = 1.0
        r = audit_point(outcome, budgets, alloc)
        print(f"  {p:>8.2f} {r['acc_method']:>7.3f} {r['acc_random']:>7.3f} "
              f"[{r['rand_ci'][0]:.3f},{r['rand_ci'][1]:.3f}] {r['acc_oracle']:>7.3f} {100*r['frac_captured']:>5.0f}% "
              f"{r['method_minus_random']:>+8.3f} [{r['mr_ci'][0]:+.3f},{r['mr_ci'][1]:+.3f}] {r['p']:.3f}")


def main():
    for path, tag in [("results/flare_llama.jsonl", "Llama-3.1-8B"), ("results/flare_qwen.jsonl", "Qwen2.5-7B")]:
        try:
            audit(path, tag)
        except FileNotFoundError:
            print(f"(missing {path})")
    print("\n(face-slap = FLARE's F-rand CI includes 0 / capt% small => its confidence gate does")
    print(" not beat matched-budget random at choosing which queries to retrieve)")


if __name__ == "__main__":
    main()
