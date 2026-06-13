"""P3 audit: does small-model-confidence routing beat random routing at matched big-call
fraction? Binary budget {small=0, big=1}. Route the low-confidence p-fraction to big.
"""
import json, sys, numpy as np
from protocol import audit_point, binary_outcome_table


def load_small(path):
    """small model: per qi correct + confidence (from sampling agreement or conf_k)."""
    R = [json.loads(l) for l in open(path)]
    if "samples" in R[0]:                                  # from sample_gen: derive maj-correct + agreement
        import collections
        corr, conf = [], []
        for r in R:
            aa = [s["a"] for s in r["samples"]]
            cnt = collections.Counter(aa); maj, top = cnt.most_common(1)[0]
            corr.append(int(any(s["c"] for s in r["samples"] if s["a"] == maj)))
            conf.append(top / len(aa))
        return np.array(corr), np.array(conf)
    return np.array([r["correct"] for r in R]), np.array([r.get("conf") or 0.5 for r in R])


def load_big(path):
    R = [json.loads(l) for l in open(path)]
    return np.array([r["correct"] for r in R])


def audit(small_path, big_path, tag):
    sc, conf = load_small(small_path); bc = load_big(big_path)
    n = min(len(sc), len(bc)); sc, conf, bc = sc[:n], conf[:n], bc[:n]
    outcome, budgets = binary_outcome_table(sc, bc, cost1=1.0)
    print(f"\n### {tag}  N={n}  small_acc={sc.mean():.3f}  big_acc={bc.mean():.3f}  oracle(route-all-gainful)={np.mean(np.maximum(sc,bc)):.3f}")
    print(f"  {'big_frac':>8} {'router_acc':>10} {'random':>7} {'rand_ci':>16} {'oracle':>7} {'capt%':>6} {'r-rand':>8} {'ci':>16}")
    order = np.argsort(conf)                               # lowest confidence first -> route to big
    for p in [0.1, 0.2, 0.3, 0.5]:
        nb = int(p * n); alloc = np.zeros(n); alloc[order[:nb]] = 1.0
        r = audit_point(outcome, budgets, alloc)
        print(f"  {p:>8.2f} {r['acc_method']:>10.3f} {r['acc_random']:>7.3f} "
              f"[{r['rand_ci'][0]:.3f},{r['rand_ci'][1]:.3f}] {r['acc_oracle']:>7.3f} "
              f"{100*r['frac_captured']:>5.0f}% {r['method_minus_random']:>+8.3f} [{r['mr_ci'][0]:+.3f},{r['mr_ci'][1]:+.3f}]")


def main():
    # pairs: (small_file, big_file, tag)
    pairs = [
        ("results/samp_qwen_gsm8k.jsonl", "results/route_big_gsm8k.jsonl", "GSM8K (small=Qwen7B SC-conf, big=Qwen32B)"),
        ("results/route_small_mmlu.jsonl", "results/route_big_mmlu.jsonl", "MMLU (small=Qwen7B, big=Qwen32B)"),
    ]
    if len(sys.argv) > 1:
        pairs = [(sys.argv[1], sys.argv[2], "custom")]
    for s, b, t in pairs:
        try:
            audit(s, b, t)
        except FileNotFoundError as e:
            print(f"\n(missing for {t}: {e})")
    print("\n(capt% = share of oracle-over-random gap captured; r-rand>0 with CI excluding 0 = router beats random)")


if __name__ == "__main__":
    main()
