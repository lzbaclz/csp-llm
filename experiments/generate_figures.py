"""Generate paper_icdm figures from icdm_full.json (+ wcet_gpu.json).

    python experiments/generate_figures.py --results experiments/results \
        --out paper_icdm/figures
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
FN = {"s_within": "within", "s_cross": "cross", "s_query": "query", "s_pos": "recency"}


def fig_redundancy(full, out):
    red = full["pooled"]["redundancy"]
    pairs = red["pairs"]
    labels = [p["pair"].replace("s_", "").replace("_within", "within").replace("_cross", "cross")
              .replace("_query", "query").replace("_pos", "recency").replace("~", "–") for p in pairs]
    ii = [p["interaction"] for p in pairs]
    colors = ["#d62728" if p["verdict"] == "synergistic" else
              "#1f77b4" if p["verdict"] == "redundant" else "#7f7f7f" for p in pairs]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    ax.barh(range(len(ii)), ii, color=colors)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(range(len(ii))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("interaction information  II = I(Xi;Xj|Y) − I(Xi;Xj)")
    ax.set_title("redundant (<0) · independent (≈0) · synergistic (>0)", fontsize=7.5)
    fig.savefig(os.path.join(out, "fig_redundancy.pdf")); plt.close(fig)


def fig_reliability(full, out):
    cal = full["pooled"]["calibration"]
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, label="perfect")
    for name, style in [("XQP-closed", "o-"), ("GBDT", "s-"), ("Quest(raw)", "^-")]:
        if name not in cal:
            continue
        rc = cal[name]["reliability"]
        conf = np.array([c if c is not None else np.nan for c in rc["confidence"]], float)
        acc = np.array([c if c is not None else np.nan for c in rc["accuracy"]], float)
        ece = cal[name]["ece"]
        ax.plot(conf, acc, style, ms=3, lw=1, label=f"{name} (ECE={ece:.3f})")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("empirical frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=6.5, loc="upper left")
    fig.savefig(os.path.join(out, "fig_reliability.pdf")); plt.close(fig)


def fig_pareto(full, out):
    par = full["pooled"]["pareto"]
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3))
    nv = par["by_n_views"]
    axes[0].plot([d["n_views"] for d in nv], [d["auc"] for d in nv], "o-", color="#1f77b4")
    axes[0].set_xlabel("# views computed"); axes[0].set_ylabel("AUC"); axes[0].set_xticks([1, 2, 3, 4])
    axes[0].set_title("accuracy vs. views", fontsize=8)
    bm = par["by_model"]
    axes[1].plot([d["params"] for d in bm], [d["auc"] for d in bm], "s-", color="#ff7f0e")
    for d in bm:
        axes[1].annotate(d["model"], (d["params"], d["auc"]), fontsize=6.5,
                         xytext=(2, 3), textcoords="offset points")
    axes[1].set_xscale("log"); axes[1].set_xlabel("# parameters"); axes[1].set_ylabel("AUC")
    axes[1].set_title("accuracy vs. model size", fontsize=8)
    fig.savefig(os.path.join(out, "fig_pareto.pdf")); plt.close(fig)


def fig_auc_horizon(full, out):
    avh = full["pooled"]["auc_vs_horizon"]
    hs = ["h1", "h4", "h16", "h64"]; xs = [1, 4, 16, 64]
    closed = [avh[h]["closed_auc"] for h in hs]
    best = [avh[h]["best_single_auc"] for h in hs]
    fig, ax = plt.subplots(figsize=(3.0, 2.3))
    ax.plot(xs, closed, "o-", label="XQP-closed (fused)", color="#1f77b4")
    ax.plot(xs, best, "s--", label="best single view", color="#7f7f7f")
    ax.set_xscale("log", base=2); ax.set_xticks(xs); ax.set_xticklabels(xs)
    ax.set_xlabel("prediction horizon (steps)"); ax.set_ylabel("AUC")
    ax.legend(fontsize=7)
    fig.savefig(os.path.join(out, "fig_auc_horizon.pdf")); plt.close(fig)


def fig_drift(full, out):
    dr = full["pooled"]["drift"]
    fig, ax = plt.subplots(figsize=(2.7, 2.3))
    names = ["static", "online", "refit\noracle"]
    vals = [dr.get("auc_static"), dr.get("auc_online"), dr.get("auc_refit_oracle")]
    ax.bar(names, vals, color=["#7f7f7f", "#1f77b4", "#2ca02c"])
    ax.set_ylabel("late-stream AUC (h4)")
    lo = min(v for v in vals if v) - 0.01
    ax.set_ylim(lo, max(vals) + 0.005)
    ax.set_title("train early steps → test late steps", fontsize=7.5)
    fig.savefig(os.path.join(out, "fig_drift.pdf")); plt.close(fig)


def fig_conformal(extra, out):
    c = extra["conformal"]
    ad = c["adaptive_g10"]["trajectory"]
    fc = c.get("fixed_split_conformal", c.get("fixed"))["trajectory"]
    fn = c.get("fixed_tau05", c.get("fixed"))["trajectory"]
    fck = "fixed_split_conformal" if "fixed_split_conformal" in c else "fixed"
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.2))
    axes[0].plot([t["step"] for t in fn], [t["miss"] for t in fn], "-", color="#d62728", lw=1,
                 label=f"fixed τ=.5 ({c['fixed_tau05']['mean_miss_2nd_half']:.2f})")
    axes[0].plot([t["step"] for t in fc], [t["miss"] for t in fc], "-", color="#2ca02c", lw=1,
                 label=f"fixed calib. ({c[fck]['mean_miss_2nd_half']:.2f})")
    axes[0].plot([t["step"] for t in ad], [t["miss"] for t in ad], "-", color="#1f77b4", lw=1,
                 label=f"adaptive ({c['adaptive_g10']['mean_miss_2nd_half']:.2f})")
    axes[0].axhline(c["alpha"], color="k", ls="--", lw=0.8, label=f"target α={c['alpha']}")
    axes[0].set_xlabel("decode step"); axes[0].set_ylabel("realized miss rate")
    axes[0].legend(fontsize=6.0, loc="upper right"); axes[0].set_title("coverage over the stream", fontsize=8)
    axes[1].plot([t["step"] for t in ad], [t["set_size"] for t in ad], "-", color="#1f77b4", lw=1, label="adaptive")
    axes[1].plot([t["step"] for t in fc], [t["set_size"] for t in fc], "-", color="#2ca02c", lw=1, label="fixed calib.")
    axes[1].plot([t["step"] for t in fn], [t["set_size"] for t in fn], "-", color="#d62728", lw=1, label="fixed τ=.5")
    axes[1].set_xlabel("decode step"); axes[1].set_ylabel("kept fraction |S|")
    axes[1].legend(fontsize=6.5); axes[1].set_title("set size (efficiency)", fontsize=8)
    fig.savefig(os.path.join(out, "fig_conformal.pdf")); plt.close(fig)


def fig_transfer(full, out):
    tm = full["transfer"]["standardized"]["matrix"]
    names = list(tm.keys())
    short = [n.replace("-Instruct", "").replace("-3.1", "3.1").replace("2.5", "2.5") for n in names]
    M = np.array([[tm[a][b] for b in names] for a in names])
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    im = ax.imshow(M, cmap="viridis", vmin=M.min(), vmax=M.max())
    ax.set_xticks(range(len(names))); ax.set_xticklabels(short, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("test model"); ax.set_ylabel("train model")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                    color="white" if M[i, j] < M.mean() else "black", fontsize=6.5)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title("cross-architecture transfer AUC", fontsize=8)
    fig.savefig(os.path.join(out, "fig_transfer.pdf")); plt.close(fig)


def fig_heavyhitter(hh, out):
    models = list(hh.keys())
    short = [m.replace("-Instruct", "").replace("-8B", "").replace("-7B", "") for m in models]
    hs = ["h1", "h4", "h16", "h64"]; xs = [1, 4, 16, 64]
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.3))
    for m, sh in zip(models, short):
        jv = hh[m]["jaccard_vs_h1"]
        axes[0].plot(xs, [jv[h] for h in hs], "o-", ms=3, lw=1, label=sh)
    axes[0].set_xscale("log", base=2); axes[0].set_xticks(xs); axes[0].set_xticklabels(xs)
    axes[0].set_xlabel("horizon $h$ (steps)"); axes[0].set_ylabel("salient-set Jaccard(t,\\,t$+h$)")
    axes[0].set_ylim(0.5, 1.0); axes[0].legend(fontsize=6.5)
    axes[0].set_title("temporally stable, slowly drifting", fontsize=8)
    g = [hh[m]["gini_mean"] for m in models]
    axes[1].bar(short, g, color="#ff7f0e")
    axes[1].set_ylabel("Gini of block attention"); axes[1].set_ylim(0.8, 1.0)
    axes[1].set_title("heavy-tailed concentration", fontsize=8)
    for t in axes[1].get_xticklabels():
        t.set_rotation(20); t.set_ha("right"); t.set_fontsize(7)
    fig.savefig(os.path.join(out, "fig_heavyhitter.pdf")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/results")
    ap.add_argument("--out", default="paper_icdm/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    full = json.load(open(os.path.join(a.results, "icdm_full.json")))
    for fn in (fig_redundancy, fig_reliability, fig_pareto, fig_auc_horizon, fig_drift, fig_transfer):
        try:
            fn(full, a.out); print("wrote", fn.__name__)
        except Exception as e:
            print("FAILED", fn.__name__, type(e).__name__, e)
    extra_path = os.path.join(a.results, "icdm_extra.json")
    if os.path.exists(extra_path):
        try:
            fig_conformal(json.load(open(extra_path)), a.out); print("wrote fig_conformal")
        except Exception as e:
            print("FAILED fig_conformal", type(e).__name__, e)
    hh_path = os.path.join(a.results, "heavy_hitter.json")
    if os.path.exists(hh_path):
        try:
            fig_heavyhitter(json.load(open(hh_path)), a.out); print("wrote fig_heavyhitter")
        except Exception as e:
            print("FAILED fig_heavyhitter", type(e).__name__, e)
    print("figures ->", a.out)


if __name__ == "__main__":
    main()
