#!/usr/bin/env python3
"""Which signal predicts per-block KV quantization sensitivity, and does adaptive
bit-allocation beat uniform at equal average bits?

Inputs: quant_sensitivity.py JSONL (per (prompt,block): sens + attn/vnorm/knorm/posn).
Outputs:
  (1) Relevance: Pearson/Spearman + AUC(predict top-20% most-sensitive block) per signal.
  (2) Redundancy: logistic-AUC for each signal alone vs all-signals vs leave-one-out
      (does attention add anything over value-norm/position?).
  (3) Head-to-head: keep fraction f of blocks high-precision, DEMOTE the rest to low-bit.
      total degradation = sum of sens over demoted blocks. Demote-lowest-by-signal vs
      uniform(random demote, = equal-avg-bits baseline) vs oracle(demote lowest-sens).
"""
import json, sys, numpy as np
from itertools import combinations

def auc(score, label):
    # rank-based AUC
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(len(score))
    pos = label == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0: return float('nan')
    return (ranks[pos].sum() - n_pos*(n_pos-1)/2) / (n_pos*n_neg)

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return np.corrcoef(rx, ry)[0,1]

def logistic_auc(X, y, iters=300, lr=0.5):
    # standardize + simple GD logistic, return in-sample AUC (proxy for predictive value)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Xs = np.hstack([Xs, np.ones((len(Xs),1))])
    w = np.zeros(Xs.shape[1])
    for _ in range(iters):
        p = 1/(1+np.exp(-Xs@w)); w -= lr*(Xs.T@(p-y))/len(y)
    return auc(Xs@w, y)

def main():
    path = sys.argv[1] if len(sys.argv)>1 else "results/quant/llama_b2.jsonl"
    rows = [json.loads(l) for l in open(path)]
    feats = ['attn','vnorm','knorm','posn']
    sens = np.array([r['sens'] for r in rows])
    F = {f: np.array([r[f] for r in rows]) for f in feats}
    # normalize sens within prompt (different prompts have different scales)
    pid = np.array([r['p'] for r in rows])
    sn = sens.copy()
    for p in np.unique(pid):
        m = pid==p; s=sens[m]
        sn[m] = (s - s.mean())/(s.std()+1e-8)
    hi = (sn > np.quantile(sn, 0.80)).astype(int)   # top-20% most sensitive

    print(f"=== {path}  (n={len(rows)} blocks, {len(np.unique(pid))} prompts) ===")
    print(f"sens: mean={sens.mean():.3f} median={np.median(sens):.3f} max={sens.max():.3f}\n")
    print("(1) RELEVANCE — which signal predicts quant-sensitivity?")
    print(f"{'signal':>8} {'Pearson':>9} {'Spearman':>9} {'AUC(top20%)':>12}")
    for f in feats:
        p = np.corrcoef(F[f], sn)[0,1]; sp = spearman(F[f], sn); a = auc(F[f], hi)
        print(f"{f:>8} {p:>+9.3f} {sp:>+9.3f} {a:>12.3f}")
    print("\n(2) REDUNDANCY — logistic AUC predicting top-20% sensitive")
    allX = np.column_stack([F[f] for f in feats])
    print(f"  all 4 signals:      AUC={logistic_auc(allX, hi):.3f}")
    for f in feats:
        print(f"  {f:>8} alone:      AUC={logistic_auc(F[f][:,None], hi):.3f}")
    for f in feats:
        others=[g for g in feats if g!=f]
        Xo=np.column_stack([F[g] for g in others])
        print(f"  drop {f:>8}:       AUC={logistic_auc(Xo, hi):.3f}  (unique value of {f})")

    print("\n(3) HEAD-TO-HEAD — keep fraction f high-precision, demote rest to low-bit.")
    print("    total degradation = sum(sens of demoted); lower=better. Per-prompt then mean.")
    print(f"{'keep f':>7} {'uniform':>9} {'by attn':>9} {'by vnorm':>9} {'by knorm':>9} {'by posn':>9} {'ORACLE':>9}")
    for f in [0.1,0.2,0.3,0.5]:
        deg={k:[] for k in ['uniform','attn','vnorm','knorm','posn','oracle']}
        for p in np.unique(pid):
            m=pid==p; s=sens[m]; n=len(s); ndem=int(round((1-f)*n))
            if ndem<=0: continue
            deg['uniform'].append((1-f)*s.mean()*n)   # E[random demote] * n  (≈ mean*ndem)
            deg['uniform'][-1]=s.mean()*ndem
            for sig in ['attn','vnorm','knorm','posn']:
                v=F[sig][m]; dem=np.argsort(v)[:ndem]   # demote LOWEST signal
                deg[sig].append(s[dem].sum())
            dem=np.argsort(s)[:ndem]; deg['oracle'].append(s[dem].sum())  # demote lowest sens
        row=[np.mean(deg[k]) for k in ['uniform','attn','vnorm','knorm','posn','oracle']]
        print(f"{f:>7.2f} "+" ".join(f"{v:>9.2f}" for v in row))
    print("\n  (win = a signal column << uniform and close to ORACLE; if all ~uniform,")
    print("   adaptive bit-allocation gives no benefit on this model.)")

if __name__=="__main__":
    main()
