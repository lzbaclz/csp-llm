"""P5/RAG row of the unified audit: cast the RCRG retrieve-gating data into the shared
matched-budget protocol. Budget = retrieval (skip=0 uses closed-book, retrieve=1 uses
open-book). Gate retrieves the LOW-confidence fraction. Compare vs permuted-random + oracle.
"""
import json, sys, numpy as np
from protocol import audit_point, binary_outcome_table


def load(path):
    R = [json.loads(l) for l in open(path)]
    g = np.array([r["gate_agree"] for r in R]); o = np.array([r["open_correct"] for r in R]); c = np.array([r["closed_correct"] for r in R])
    return g, o, c


def audit(path, tag):
    g, o, c = load(path); n = len(g)
    outcome, budgets = binary_outcome_table(c, o, cost1=1.0)   # action0=skip(closed), action1=retrieve(open)
    print(f"\n### {tag}  N={n}  never(closed)={c.mean():.3f}  always(open)={o.mean():.3f}  oracle={np.mean(np.maximum(o,c)):.3f}")
    print(f"  {'retr_rate':>9} {'gate_acc':>8} {'random':>7} {'rand_ci':>16} {'oracle':>7} {'capt%':>6} {'g-rand':>8} {'ci':>16}")
    order = np.argsort(g)                                       # lowest confidence first -> retrieve
    for p in [0.2, 0.3, 0.5, 0.7]:
        nr = int(p * n); alloc = np.zeros(n); alloc[order[:nr]] = 1.0
        r = audit_point(outcome, budgets, alloc)
        print(f"  {p:>9.2f} {r['acc_method']:>8.3f} {r['acc_random']:>7.3f} "
              f"[{r['rand_ci'][0]:.3f},{r['rand_ci'][1]:.3f}] {r['acc_oracle']:>7.3f} "
              f"{100*r['frac_captured']:>5.0f}% {r['method_minus_random']:>+8.3f} [{r['mr_ci'][0]:+.3f},{r['mr_ci'][1]:+.3f}]")


def main():
    for path, tag in [("../rcrg/results/llama_bm25.jsonl", "RAG Llama-3.1-8B (confidence gate)"),
                      ("../rcrg/results/qwen_bm25.jsonl", "RAG Qwen2.5-7B (confidence gate)")]:
        try:
            audit(path, tag)
        except FileNotFoundError:
            print(f"(missing {path})")
    print("\n(capt% = share of oracle-over-random gap captured; g-rand>0 CI-excl-0 = gate beats random)")


if __name__ == "__main__":
    main()
