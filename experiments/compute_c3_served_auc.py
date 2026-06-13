#!/usr/bin/env python
"""C3: served-oracle AUC of the 2-view scorer, with cluster bootstrap CI.

Reproduces transfer_gap.tex:20-22 point (~0.643) and adds a 95% CI plus a
per-layer (per-group) decomposition. Cluster unit = prompt_id (24 prompts),
the independent sampling unit; blocks within a prompt are highly correlated.
CPU-only, fixed seed.
"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score

SEED = 1234
N_BOOT = 1000
P = "/home/lzq/codes/csp-llm/experiments/results/serving_calib/calib_llama.json"
OUT = "/home/lzq/codes/csp-llm/experiments/results/served_oracle_ci/c3_served_auc.json"


def auc_safe(y, s):
    y = np.asarray(y)
    if y.min() == y.max():
        return np.nan  # one class -> AUC undefined
    return roc_auc_score(y, s)


def cluster_bootstrap_auc(y, s, clusters, n_boot, rng):
    """Resample whole clusters with replacement; pool their rows; recompute AUC."""
    uniq = np.unique(clusters)
    # pre-index rows per cluster
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in uniq}
    aucs = []
    k = len(uniq)
    for _ in range(n_boot):
        picks = rng.choice(uniq, size=k, replace=True)
        rows = np.concatenate([idx_by_cluster[c] for c in picks])
        a = auc_safe(y[rows], s[rows])
        if not np.isnan(a):
            aucs.append(a)
    aucs = np.array(aucs)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi), int(len(aucs)), float(aucs.mean()), float(aucs.std())


def main():
    d = json.load(open(P))
    cols = d["cols"]
    arr = np.array(d["rows"], dtype=float)
    ci = {c: i for i, c in enumerate(cols)}
    pid = arr[:, ci["prompt_id"]].astype(int)
    layer = arr[:, ci["layer_pos"]]
    score = arr[:, ci["score"]]
    label = arr[:, ci["label"]].astype(int)

    n = len(label)
    n_prompts = len(np.unique(pid))
    n_layers = len(np.unique(layer))

    # ---- Overall point ----
    overall_auc = float(auc_safe(label, score))

    # ---- Overall CI: cluster by prompt_id ----
    rng = np.random.default_rng(SEED)
    lo_p, hi_p, k_p, mean_p, std_p = cluster_bootstrap_auc(
        label, score, pid, N_BOOT, rng)

    # ---- Sensitivity: naive row-level (i.i.d.) bootstrap for contrast ----
    rng2 = np.random.default_rng(SEED + 1)
    row_aucs = []
    for _ in range(N_BOOT):
        idx = rng2.integers(0, n, size=n)
        a = auc_safe(label[idx], score[idx])
        if not np.isnan(a):
            row_aucs.append(a)
    row_aucs = np.array(row_aucs)
    lo_r, hi_r = np.percentile(row_aucs, [2.5, 97.5])

    # ---- Per-layer decomposition (the only group field available) ----
    per_layer = []
    rng3 = np.random.default_rng(SEED + 2)
    for L in np.unique(layer):
        m = layer == L
        yl, sl, pl = label[m], score[m], pid[m]
        pt = float(auc_safe(yl, sl))
        lo, hi, k, mn, sd = cluster_bootstrap_auc(yl, sl, pl, N_BOOT, rng3)
        per_layer.append({
            "layer_pos": round(float(L), 4),
            "n_rows": int(m.sum()),
            "n_prompts": int(len(np.unique(pl))),
            "pos_rate": float(yl.mean()),
            "auc_point": pt,
            "auc_ci95_lo": lo,
            "auc_ci95_hi": hi,
        })

    # summary across layers
    layer_pts = np.array([d["auc_point"] for d in per_layer])

    result = {
        "task": "C3_served_oracle_auc_with_ci",
        "data_file": P,
        "n_rows": int(n),
        "n_prompts": int(n_prompts),
        "n_layers": int(n_layers),
        "scorer": "2-view scorer probability (col 'score')",
        "target": "served-oracle binary label (col 'label')",
        "paper_claim_point": 0.643,
        "paper_ref": "transfer_gap.tex:20-22",
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "cluster_unit": "prompt_id (24 prompts)",
        "overall": {
            "auc_point": overall_auc,
            "cluster_bootstrap_prompt": {
                "ci95_lo": lo_p, "ci95_hi": hi_p,
                "n_valid_resamples": k_p,
                "boot_mean": mean_p, "boot_std": std_p,
            },
            "naive_row_bootstrap_FOR_CONTRAST_ONLY": {
                "ci95_lo": float(lo_r), "ci95_hi": float(hi_r),
                "note": "i.i.d. row resampling IGNORES within-prompt correlation; understates uncertainty; reported only to show the cluster CI is the honest one",
            },
        },
        "per_layer": per_layer,
        "per_layer_summary": {
            "min_auc": float(layer_pts.min()),
            "max_auc": float(layer_pts.max()),
            "mean_auc": float(layer_pts.mean()),
            "std_auc": float(layer_pts.std()),
        },
        "notes": [
            "No explicit workload/dataset field exists; columns are [prompt_id, layer_pos, score, label]. The only available group dimension besides prompt is layer_pos (normalized layer index, 32 layers), decomposed above.",
            "Both score and label vary across layers within a prompt, so each (prompt,layer) is a distinct served-oracle prediction problem.",
            "Cluster unit = prompt_id (24 independent prompts). Within a prompt, blocks are highly correlated, so the i.i.d. row bootstrap understates uncertainty; the prompt-cluster CI is the honest interval and is what we report.",
            "Only 24 clusters -> the cluster CI is itself estimated from few independent units; treat the interval as approximate.",
        ],
    }

    json.dump(result, open(OUT, "w"), indent=2)
    print("overall AUC point:", round(overall_auc, 4))
    print("prompt-cluster 95pct CI: [{:.4f}, {:.4f}] (valid resamples={})".format(lo_p, hi_p, k_p))
    print("naive row 95pct CI (contrast): [{:.4f}, {:.4f}]".format(lo_r, hi_r))
    print("per-layer AUC range: [{:.4f}, {:.4f}] mean={:.4f}".format(
        layer_pts.min(), layer_pts.max(), layer_pts.mean()))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
