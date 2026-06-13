"""P2 audit: replay ESC / ASC adaptive-sampling stopping, then matched-budget protocol.

outcome_i(k) = is the majority vote of the first k samples correct?
Methods allocate #samples per query from answer agreement (stop when confident).
Audit: does adaptive sample-allocation beat the SAME #samples permuted across queries?
"""
import json, sys, collections, numpy as np
from protocol import audit_point

K_MAX = 40


def load(path):
    rows = [json.loads(l) for l in open(path)]
    N = len(rows); K = min(K_MAX, len(rows[0]["samples"]))
    outcome = np.zeros((N, K))                      # outcome_i(k) majority-of-first-(k+1) correct
    answers = []                                    # per-query list of (norm_answer, correct)
    for i, r in enumerate(rows):
        s = r["samples"][:K]; answers.append(s)
        cnt = collections.Counter()
        corr_of = {}
        for k in range(K):
            a = s[k]["a"]; cnt[a] += 1; corr_of[a] = s[k]["c"]
            maj = cnt.most_common(1)[0][0]
            outcome[i, k] = corr_of.get(maj, 0)
    budgets = np.arange(1, K + 1).astype(float)
    return outcome, budgets, answers


def esc_alloc(answers, w):
    """Early-Stopping SC: stop at end of first window of size w that is unanimous."""
    alloc = []
    for s in answers:
        K = len(s); stop = K
        for j in range(0, K, w):
            win = [x["a"] for x in s[j:j + w]]
            if len(win) == w and len(set(win)) == 1:
                stop = j + w; break
        alloc.append(stop)
    return np.array(alloc, float)


def asc_alloc(answers, t, nmin=3):
    """Adaptive-Consistency: stop when top-answer fraction >= t (after >= nmin samples)."""
    alloc = []
    for s in answers:
        K = len(s); cnt = collections.Counter(); stop = K
        for k in range(K):
            cnt[s[k]["a"]] += 1
            n = k + 1
            if n >= nmin and cnt.most_common(1)[0][1] / n >= t:
                stop = n; break
        alloc.append(stop)
    return np.array(alloc, float)


def audit_file(path, tag):
    outcome, budgets, answers = load(path)
    accK = outcome[:, -1].mean()
    print(f"\n### {tag}  N={len(answers)}  acc@1={outcome[:,0].mean():.3f}  acc@{len(budgets)}(full SC)={accK:.3f}")
    print(f"  {'method':>10} {'avg_smp':>7} {'acc':>6} {'random':>6} {'rand_ci':>16} {'oracle':>6} {'capt%':>6} {'m-rand':>8} {'mr_ci':>16}")
    pts = [("ESC w=3", esc_alloc(answers, 3)), ("ESC w=5", esc_alloc(answers, 5)),
           ("ASC t=.7", asc_alloc(answers, 0.7)), ("ASC t=.8", asc_alloc(answers, 0.8)), ("ASC t=.9", asc_alloc(answers, 0.9))]
    for name, alloc in pts:
        r = audit_point(outcome, budgets, alloc)
        print(f"  {name:>10} {r['B']:>7.1f} {r['acc_method']:>6.3f} {r['acc_random']:>6.3f} "
              f"[{r['rand_ci'][0]:.3f},{r['rand_ci'][1]:.3f}] {r['acc_oracle']:>6.3f} "
              f"{100*r['frac_captured']:>5.0f}% {r['method_minus_random']:>+8.3f} [{r['mr_ci'][0]:+.3f},{r['mr_ci'][1]:+.3f}]")


def main():
    files = sys.argv[1:] or [
        "results/samp_qwen_gsm8k.jsonl", "results/samp_qwen_math.jsonl",
        "results/samp_llama_gsm8k.jsonl", "results/samp_llama_math.jsonl"]
    for f in files:
        try:
            audit_file(f, f.split("/")[-1].replace(".jsonl", ""))
        except FileNotFoundError:
            print(f"\n(missing {f})")
    print("\n(capt% = fraction of oracle-over-random gap captured; m-rand>0 with CI excluding 0 = signal beats random)")


if __name__ == "__main__":
    main()
