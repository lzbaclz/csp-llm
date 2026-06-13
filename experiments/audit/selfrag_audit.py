"""Audit Self-RAG's learned gate: does P([Retrieval]) select better which queries to
retrieve than matched-budget random? (binary budget: skip=closed-book, retrieve=open-book)."""
import json, sys, numpy as np
from protocol import audit_point, binary_outcome_table


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/selfrag_gate.jsonl"
    R = [json.loads(l) for l in open(path)]
    g = np.array([r["gate"] for r in R]); c = np.array([r["closed_correct"] for r in R]); o = np.array([r["open_correct"] for r in R])
    n = len(R)
    outcome, budgets = binary_outcome_table(c, o, 1.0)
    print(f"=== Self-RAG learned gate face-slap  N={n} ===")
    print(f"closed(no-retrieve) acc={c.mean():.3f}  open(retrieve) acc={o.mean():.3f}  oracle={np.mean(np.maximum(o,c)):.3f}")
    print(f"gate corr with retrieve-helps(o>c): {np.corrcoef(g, (o>c).astype(float))[0,1]:+.3f}")
    print(f"{'retr_rate':>9} {'selfrag':>8} {'random':>7} {'rand_ci':>16} {'oracle':>7} {'capt%':>6} {'gate-rand':>9} {'ci':>16} {'p':>6}")
    order = np.argsort(-g)                                   # HIGH P([Retrieval]) -> retrieve first (Self-RAG policy)
    for p in [0.2, 0.3, 0.5, 0.7]:
        nr = int(p * n); alloc = np.zeros(n); alloc[order[:nr]] = 1.0
        r = audit_point(outcome, budgets, alloc)
        print(f"  {p:>8.2f} {r['acc_method']:>8.3f} {r['acc_random']:>7.3f} "
              f"[{r['rand_ci'][0]:.3f},{r['rand_ci'][1]:.3f}] {r['acc_oracle']:>7.3f} {100*r['frac_captured']:>5.0f}% "
              f"{r['method_minus_random']:>+9.3f} [{r['mr_ci'][0]:+.3f},{r['mr_ci'][1]:+.3f}] {r['p']:.3f}")
    print("\n(face-slap = Self-RAG gate-rand CI includes 0 / capt% small => its learned retrieve-gate")
    print(" does not beat matched-budget random at choosing which queries to retrieve)")


if __name__ == "__main__":
    main()
