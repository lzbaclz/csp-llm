"""P4 audit: does difficulty-based reasoning-token allocation beat permuted-random at
matched avg tokens? Compare PROMPT-ONLY difficulty (cheap) vs MID-GEN early-consistency
(charged its probe cost) vs random vs oracle.
"""
import json, sys, numpy as np
from protocol import audit_point, step_outcome

CAPS = [64, 128, 256, 512, 1024]


def load(path):
    R = [json.loads(l) for l in open(path)]
    N = len(R); G = len(CAPS)
    outcome = np.zeros((N, G))
    diff = np.zeros(N); midgen_hard = np.zeros(N, bool)
    for i, r in enumerate(R):
        for g, b in enumerate(CAPS):
            outcome[i, g] = r["outcome"][str(b)]
        diff[i] = r.get("diff_prompt", 3)
        e = r.get("early", ["", ""]); midgen_hard[i] = not (e[0] == e[1] and e[0] != "")  # disagree => hard
    return outcome, np.array(CAPS, float), diff, midgen_hard


def audit(path, tag):
    outcome, budgets, diff, midhard = load(path)
    N = len(diff)
    print(f"\n### {tag}  N={N}  acc@{CAPS[0]}={outcome[:,0].mean():.3f} acc@{CAPS[-1]}={outcome[:,-1].mean():.3f}")
    print(f"  {'method':>24} {'avg_tok':>7} {'acc':>6} {'random':>7} {'oracle':>7} {'capt%':>6} {'m-rand':>8} {'mr_ci':>16} {'p':>6}")

    def report(name, alloc, est=None):
        r = audit_point(outcome, budgets, alloc, est_cost_N=est)
        print(f"  {name:>24} {r['B']:>7.0f} {r['acc_method']:>6.3f} {r['acc_random']:>7.3f} "
              f"{r['acc_oracle']:>7.3f} {100*r['frac_captured']:>5.0f}% {r['method_minus_random']:>+8.3f} "
              f"[{r['mr_ci'][0]:+.3f},{r['mr_ci'][1]:+.3f}] p={r['p']:.3f}")

    # PROMPT-ONLY difficulty -> cap map (monotone). The +8-token self-rating cost is added to
    # every instance (charged), but is negligible vs the reasoning budget.
    mapA = {1: 64, 2: 128, 3: 256, 4: 512, 5: 1024}
    mapB = {1: 128, 2: 256, 3: 512, 4: 1024, 5: 1024}
    report("prompt-diff mapA", np.array([mapA[int(d)] for d in diff], float) + 8.0)
    report("prompt-diff mapB", np.array([mapB[int(d)] for d in diff], float) + 8.0)

    # MID-GEN early-consistency: probe to 128 then decide -- agree(easy) STOP at 128, disagree
    # CONTINUE to 1024. The 128-token probe is INHERENT in the budget (no fake 'free' variant).
    alloc_mid = np.where(midhard, 1024.0, 128.0)
    report("mid-gen (probe in budget)", alloc_mid)
    frac_hard = float(midhard.mean())
    print(f"     (mid-gen fires hard on {100*frac_hard:.0f}% of instances)")


def main():
    for path, tag in [("results/ttc_gsm8k.jsonl", "TTC GSM8K (Qwen3-8B)"),
                      ("results/ttc_math.jsonl", "TTC MATH500 (Qwen3-8B)")]:
        try:
            audit(path, tag)
        except FileNotFoundError:
            print(f"(missing {path})")
    print("\n(prompt-only ~ cheap pre-gen signal; mid-gen observes early answers but costs probe tokens)")


if __name__ == "__main__":
    main()
