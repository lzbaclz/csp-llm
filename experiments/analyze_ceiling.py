"""B — the serving-oracle predictability CEILING. Train every model class on every
feature subset against the SERVING oracle label, to answer definitively: can ANY
model on the deployment features beat H2O (accumulated attention) at predicting what
the model actually attends to in the loop? If the ceiling AUC ~= H2O's own AUC, no
learned selector can win; if it is much higher AND yields lower miss@budget, a
deployment-native scorer beats H2O.

    python experiments/analyze_ceiling.py --dir experiments/results/ceiling
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier
    HAVE_LGBM = False

COLS = ["prompt_id", "layer_pos", "attn_now", "within_accum", "native_cross", "recency", "label"]
# feature columns (exclude prompt_id, label); recency -> bounded exp form
FEAT = {"attn_now": 2, "within_accum": 3, "native_cross": 4, "recency": 5}


def load(path):
    d = json.load(open(path))
    r = np.array(d["rows"], dtype=np.float64)
    # X cols: 0=attn_now 1=within_accum(H2O) 2=native_cross 3=recency(bounded)
    X = np.stack([r[:, 2], r[:, 3], r[:, 4], np.exp(-r[:, 5] / 64.0)], axis=1)
    pid = r[:, 0].astype(int); y = r[:, 6]
    ok = np.isfinite(X).all(1) & np.isfinite(y)   # drop fp16-NaN rows (Qwen last layer)
    return pid[ok], X[ok], y[ok]


SUBSETS = {  # feature ablations to isolate what beats H2O
    "within_accum(H2O)": [1],
    "attn_now": [0],
    "accum+now": [0, 1],
    "accum+cross+recency(noNow)": [1, 2, 3],
    "ALL": [0, 1, 2, 3],
}


def split(pid):
    u = np.unique(pid); h = len(u) // 2
    return np.isin(pid, u[:h]), np.isin(pid, u[h:])


def recall_at(score, y, b):
    n = len(score); k = max(1, int(round(b * n)))
    keep = np.zeros(n, bool); keep[np.argpartition(-score, min(k - 1, n - 1))[:k]] = True
    return float((keep & (y > 0.5)).sum() / max(1, (y > 0.5).sum()))


def gbdt():
    return LGBMClassifier(n_estimators=200, num_leaves=31, verbose=-1) if HAVE_LGBM \
        else GradientBoostingClassifier(n_estimators=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/results/ceiling")
    ap.add_argument("--out", default="experiments/results/ceiling/SUMMARY.json")
    a = ap.parse_args()
    files = {os.path.basename(f)[5:-5]: f for f in sorted(glob.glob(os.path.join(a.dir, "feat_*.json")))}
    print(f"models: {list(files)} (GBDT={'lightgbm' if HAVE_LGBM else 'sklearn'})", flush=True)
    out = {"per_model": {}, "transfer": {}}
    data = {}
    for m, f in files.items():
        pid, X, y = load(f); data[m] = (pid, X, y)
        tr, te = split(pid)
        print(f"\n=== {m}: rows={len(y):,} pos_rate={y.mean():.3f} (train {tr.sum():,}/test {te.sum():,}) ===", flush=True)
        res = {"pos_rate": round(float(y.mean()), 3)}
        h2o_recall20 = recall_at(X[te, 1], y[te], 0.20)   # H2O = within_accum
        for sname, cols in SUBSETS.items():
            gb = gbdt().fit(X[tr][:, cols], y[tr])
            pg = gb.predict_proba(X[te][:, cols])[:, 1]
            auc = roc_auc_score(y[te], pg)
            r20 = recall_at(pg, y[te], 0.20)
            res[sname] = {"auc": round(float(auc), 4), "recall20": round(r20, 4),
                          "recall20_gain_vs_H2O": round(r20 - h2o_recall20, 4)}
            print(f"  {sname:28s} AUC={auc:.3f}  recall@20={r20:.3f} "
                  f"(vs H2O {h2o_recall20:.3f}, gain {r20-h2o_recall20:+.3f})", flush=True)
        # the clean (no-attn_now) result is the honest headline
        clean = res["accum+cross+recency(noNow)"]
        res["H2O_auc"] = res["within_accum(H2O)"]["auc"]
        res["clean_gain_recall20"] = clean["recall20_gain_vs_H2O"]
        out["per_model"][m] = res

    # transfer: train GBDT on llama, eval AUC on the others
    if "llama" in data:
        pid, X, y = data["llama"]; tr, _ = split(pid)
        gb = gbdt().fit(X[tr], y[tr])
        for m, (p2, X2, y2) in data.items():
            _, te2 = split(p2)
            out["transfer"][f"llama->{m}"] = round(roc_auc_score(y2[te2], gb.predict_proba(X2[te2])[:, 1]), 4)
        print(f"\ntransfer (GBDT trained on llama serving oracle): {out['transfer']}", flush=True)

    json.dump(out, open(a.out, "w"), indent=2)
    print("\nWROTE", a.out)


if __name__ == "__main__":
    main()
