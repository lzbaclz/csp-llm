"""RCRG ablations to preempt the adversarial review:
(1) Is the GATE SIGNAL real? AUROC of g for "safe-to-skip" (c>=o); and CRC Pareto with
    the real gate vs a SHUFFLED gate at matched risk -- a real signal gates MORE at equal
    coverage. (2) Does RCRG beat the cheap fix TARG+safety-margin (fixed eps-slack)?
"""
import json, sys, numpy as np
from rcrg import crc_threshold, targ_threshold, evaluate


def load(p):
    R = [json.loads(l) for l in open(p)]
    return (np.array([r["gate_agree"] for r in R]), np.array([r["open_correct"] for r in R]),
            np.array([r["closed_correct"] for r in R]))


def auroc(score, label):
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(len(score))
    pos = label == 1; npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0: return float("nan")
    return (ranks[pos].sum() - npos*(npos-1)/2) / (npos*nneg)


def main():
    L = load(sys.argv[1] if len(sys.argv) > 1 else "results/llama_bm25.jsonl")
    Q = load(sys.argv[2] if len(sys.argv) > 2 else "results/qwen_bm25.jsonl")
    EPS = 0.05
    for name, (g, o, c) in [("Llama", L), ("Qwen", Q)]:
        print(f"\n##### {name} #####")
        safe = (c >= o).astype(int)          # skipping costs nothing
        print(f"(1a) gate AUROC: safe-to-skip(c>=o)={auroc(g, safe):.3f}  closed-correct(c=1)={auroc(g, c):.3f}  (0.5=useless)")

        # (1b) real vs shuffled gate: CRC @ eps, 50/50 splits, retr% at matched (100%) coverage
        rng = np.random.RandomState(0); n = len(g)
        out = {"real": {"rf": [], "cov": []}, "shuffled": {"rf": [], "cov": []}}
        for _ in range(300):
            idx = rng.permutation(n); cal, te = idx[:n//2], idx[n//2:]
            gs = rng.permutation(g)                     # destroy g<->(o,c) link
            for tag, gg in [("real", g), ("shuffled", gs)]:
                t = crc_threshold(gg[cal], o[cal], c[cal], EPS)
                ev = evaluate(t, gg[te], o[te], c[te])
                out[tag]["rf"].append(ev["retr_freq"]); out[tag]["cov"].append(ev["loss_vs_always"] <= EPS)
        for tag in ["real", "shuffled"]:
            print(f"(1b) CRC gate={tag:>8}: retr%={100*np.mean(out[tag]['rf']):>4.0f}  cover={100*np.mean(out[tag]['cov']):>4.0f}%  "
                  f"(real should gate MORE = lower retr% at equal cover)")

        # (2) TARG+safety-margin vs CRC under DATASET shift (where naive fails). Does a FIXED
        #     margin generalize, or must it be tuned per-distribution (which CRC automates)?
        print("(2) under dataset-shift: TARG+fixed-margin (calib on easy->deploy hard, n=60)")
        # reuse C-style shift via dataset labels
        import json as _j
        R = [_j.loads(l) for l in open(sys.argv[1] if name == "Llama" else (sys.argv[2] if len(sys.argv) > 2 else "results/qwen_bm25.jsonl"))]
        ds = np.array([r["ds"] for r in R])
        helps = sorted(set(ds), key=lambda d: o[ds == d].mean() - c[ds == d].mean())
        easy, hard = helps[:3], helps[3:]
        cm = np.isin(ds, easy); tm = np.isin(ds, hard)
        for margin in [0.0, 0.02, 0.04]:
            covs, rfs = [], []
            for _ in range(300):
                ci = rng.choice(np.where(cm)[0], size=min(60, cm.sum()), replace=False)
                t = targ_threshold(g[ci], o[ci], c[ci], max(0.0, EPS - margin))
                ev = evaluate(t, g[tm], o[tm], c[tm])
                covs.append(ev["loss_vs_always"] <= EPS); rfs.append(ev["retr_freq"])
            print(f"    TARG margin={margin:>4}: cover={100*np.mean(covs):>4.0f}%  retr%={100*np.mean(rfs):>4.0f}")
        print("    (cf. wCRC ~68-91% cover. A fixed margin either still under-covers or over-retrieves;")
        print("     the correct margin is distribution-dependent -- which CRC/weighting computes automatically.)")


if __name__ == "__main__":
    main()
