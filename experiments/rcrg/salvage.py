"""RCRG salvage: does adding RETRIEVAL-QUALITY features to the gate (so it predicts o, not
just c) recover real, signal-isolating value at matched budget?

Honest protocol: out-of-fold (5-fold CV) predictions -> no train/test leakage. Compare, at
matched skip budget, the accuracy of: 2-signal gate vs g-alone vs random vs oracle, with
bootstrap CIs on (2-signal - g-alone) and (2-signal - random). Win = 2-signal beats BOTH
g-alone and random significantly on BOTH models.
"""
import json, sys, numpy as np

FEATS = ["bm25_top1", "bm25_top3_mean", "bm25_gap", "q_len", "overlap", "ret_len"]


def roc_auc_score(y, s):
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(len(s))
    pos = np.asarray(y) == 1; npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0: return float("nan")
    return float((ranks[pos].sum() - npos*(npos-1)/2) / (npos*nneg))


def _fit_logistic(X, y, iters=500, lr=0.3, l2=1e-3):
    Xb = np.hstack([X, np.ones((len(X), 1))]); w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        grad = Xb.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0]
        w -= lr * grad
    return w


def _predict(X, w):
    return 1 / (1 + np.exp(-(np.hstack([X, np.ones((len(X), 1))]) @ w)))


def load(model_path, feat_path):
    M = [json.loads(l) for l in open(model_path)]
    F = [json.loads(l) for l in open(feat_path)]
    assert len(M) == len(F)
    g = np.array([m["gate_agree"] for m in M]); o = np.array([m["open_correct"] for m in M]); c = np.array([m["closed_correct"] for m in M])
    X = np.column_stack([[f[k] for f in F] for k in FEATS] + [g])   # retrieval-quality + g
    return g, o, c, X


def oof_scores(X, y, seed=0, kf=5):
    """5-fold out-of-fold P(safe), pure numpy logistic."""
    rng = np.random.RandomState(seed); n = len(y); idx = rng.permutation(n)
    folds = np.array_split(idx, kf)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    pred = np.zeros(n)
    for i in range(kf):
        te = folds[i]; tr = np.concatenate([folds[j] for j in range(kf) if j != i])
        w = _fit_logistic(Xs[tr], y[tr].astype(float))
        pred[te] = _predict(Xs[te], w)
    return pred


def matched_budget(score, g, o, c, skipfrac, n_boot=1000, seed=0):
    """skip the top skipfrac by `score`; return acc + bootstrap of (this - random)."""
    rng = np.random.RandomState(seed); n = len(o); ns = int(skipfrac * n)
    accs, dr = [], []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        ss, oo, cc = score[idx], o[idx], c[idx]
        skip = np.argsort(-ss)[:ns]; mask = np.zeros(n, bool); mask[skip] = True
        acc = np.where(mask, cc, oo).mean(); accs.append(acc)
        rperm = rng.permutation(n)[:ns]; rmask = np.zeros(n, bool); rmask[rperm] = True
        dr.append(acc - np.where(rmask, cc, oo).mean())
    return np.mean(accs), np.percentile(dr, [2.5, 97.5]), np.mean(dr)


def main():
    feat = "results/features.jsonl"
    SKIP = 0.30
    for name, mp in [("Llama", "results/llama_bm25.jsonl"), ("Qwen", "results/qwen_bm25.jsonl")]:
        g, o, c, X = load(mp, feat)
        y = (c >= o).astype(int)                      # safe-to-skip
        # AUC: g-alone vs full feature set (out-of-fold)
        auc_g = roc_auc_score(y, g)
        p_full = oof_scores(X, y)
        auc_full = roc_auc_score(y, p_full)
        # only-retrieval-quality (no g)
        p_rq = oof_scores(X[:, :len(FEATS)], y); auc_rq = roc_auc_score(y, p_rq)
        print(f"\n##### {name} #####  safe-to-skip AUC: g-alone={auc_g:.3f}  retrieval-quality={auc_rq:.3f}  2-signal={auc_full:.3f}")
        oracle = np.mean(np.maximum(o, c)); always = o.mean()
        print(f"  always-retrieve={always:.4f}  oracle(skip {int(SKIP*100)}%)={oracle:.4f}")
        # matched-budget @ SKIP
        a_g, ci_g, _ = matched_budget(g, g, o, c, SKIP)
        a_f, ci_f, _ = matched_budget(p_full, g, o, c, SKIP)
        # 2-signal vs g-alone (paired bootstrap)
        rng = np.random.RandomState(1); n = len(o); ns = int(SKIP * n); d2 = []
        for _ in range(2000):
            idx = rng.choice(n, n, replace=True); oo, cc = o[idx], c[idx]
            sf = p_full[idx]; sg = g[idx]
            mf = np.zeros(n, bool); mf[np.argsort(-sf)[:ns]] = True
            mg = np.zeros(n, bool); mg[np.argsort(-sg)[:ns]] = True
            d2.append(np.where(mf, cc, oo).mean() - np.where(mg, cc, oo).mean())
        ci2 = np.percentile(d2, [2.5, 97.5])
        print(f"  matched skip {int(SKIP*100)}%: g-alone acc={a_g:.4f} (vs random {ci_g})  2-signal acc={a_f:.4f}")
        print(f"     2-signal - random = {a_f-always+ (always-a_f):.4f}  CI(2sig-random)={ci_f}")
        print(f"     2-signal - g-alone = {np.mean(d2):+.4f}  CI={ci2}  <- the salvage test (BOTH models must be >0)")


if __name__ == "__main__":
    main()
