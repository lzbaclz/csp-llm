"""Combined single-column figure (saves page budget): (a) per-view relevance
I(X;Z) -- the two attention-magnitude views (within/cross) dominate while
query/recency are near-zero; (b) calibration reliability -- the linear fit sits on
the diagonal out of the box (low ECE) while a class-balanced GBDT sags.
Data: experiments/results/icdm_full.json (pooled.redundancy / pooled.calibration)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})
OUT = "paper_icdm/figures"
RED, BLUE, GREY = "#d62728", "#2c7fb8", "#9e9e9e"
d = json.load(open("experiments/results/icdm_full.json"))["pooled"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(3.45, 1.68))

# (a) per-view relevance: ours = the two magnitude views (red), others grey
mi = d["redundancy"]["per_feature_mi"]
order = ["s_within", "s_cross", "s_query", "s_pos"]
labels = ["within", "cross", "query", "recency"]
vals = [mi[k] for k in order]
axA.bar(range(4), vals, color=[RED, RED, GREY, GREY], width=0.72)
axA.set_xticks(range(4)); axA.set_xticklabels(labels, rotation=28, ha="right")
axA.set_ylabel("relevance $I(X_i;Z)$")
axA.set_title("(a) two views carry the signal", fontsize=7.5)
axA.set_ylim(0, max(vals) * 1.22)
for i, v in enumerate(vals):
    axA.text(i, v + max(vals) * 0.02, f"{v:.2f}", ha="center", fontsize=6.3)

# (b) calibration reliability
cal = d["calibration"]
axB.plot([0, 1], [0, 1], "--", color=GREY, lw=0.8)
for name, style, col, short in [("XQP-closed", "o-", RED, "linear fit"),
                                ("GBDT", "s-", BLUE, "bal. GBDT")]:
    if name not in cal:
        continue
    rc = cal[name]["reliability"]
    conf = np.array([c if c is not None else np.nan for c in rc["confidence"]], float)
    acc = np.array([c if c is not None else np.nan for c in rc["accuracy"]], float)
    axB.plot(conf, acc, style, ms=2.4, lw=1.0, color=col,
             label=f"{short} (ECE {cal[name]['ece']:.3f})")
axB.set_xlabel("predicted prob."); axB.set_ylabel("empirical freq.")
axB.set_xlim(0, 1); axB.set_ylim(0, 1)
axB.set_title("(b) calibrated out of the box", fontsize=7.5)
axB.legend(fontsize=4.5, loc="upper left", frameon=True, framealpha=0.9,
           borderpad=0.2, handlelength=1.0, handletextpad=0.3, labelspacing=0.15,
           markerscale=0.85)
fig.tight_layout(pad=0.3)
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_redund_calib.pdf"))
print(f"WROTE {OUT}/fig_redund_calib.pdf | relevance={[round(v,3) for v in vals]} "
      f"| ECE linear={cal['XQP-closed']['ece']:.3f} GBDT={cal['GBDT']['ece']:.3f}")
