"""Corrected capstone figure: the served-oracle proxy is misleading. (a) on the
eps proxy H2O clearly 'beats' the learned scorer; (b) on real end-to-end task
quality (LongBench F1) they are tied. Averaged over all available datasets."""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
R = "experiments/results/e2e_confirm"
OUT = "paper_icdm/figures"

datasets = sorted({os.path.basename(f).rsplit("_", 1)[0]
                   for f in glob.glob(f"{R}/*_h2o.json")})


def agg(pol):
    eps, f1 = [], []
    for ds in datasets:
        f = f"{R}/{ds}_{pol}.json"
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        e = []
        for r in d["results"]:
            e += r.get("per_step_eps_measured", [])
        eps.append(np.mean(e)); f1.append(d["f1_mean"])
    return np.mean(eps), np.mean(f1)


h2o_e, h2o_f = agg("h2o")
xqp_e, xqp_f = agg("xqp")
print(f"datasets={datasets}\nH2O eps={h2o_e:.3f} F1={h2o_f:.3f} | XQP eps={xqp_e:.3f} F1={xqp_f:.3f}")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(3.5, 2.05))
# Style spec: baseline (H2O) = blue, ours (learned 2-view) = red.
c = ["#2c7fb8", "#d62728"]
axA.bar([0, 1], [h2o_e, xqp_e], color=c, width=0.6)
axA.set_ylim(0, max(h2o_e, xqp_e) * 1.22)
axA.set_xticks([0, 1]); axA.set_xticklabels(["H2O", "learned\n2-view"])
axA.set_ylabel("served-oracle miss $\\epsilon$ $\\downarrow$")
axA.set_title("(a) proxy: H2O looks better", fontsize=9)
for x, v in zip([0, 1], [h2o_e, xqp_e]):
    axA.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

axB.bar([0, 1], [h2o_f, xqp_f], color=c, width=0.6)
axB.set_ylim(0, max(h2o_f, xqp_f) * 1.34)
axB.set_xticks([0, 1]); axB.set_xticklabels(["H2O", "learned\n2-view"])
axB.set_ylabel("real task F1 $\\uparrow$")
axB.set_title("(b) reality: tied", fontsize=9)
for x, v in zip([0, 1], [h2o_f, xqp_f]):
    axB.text(x, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)
# In-figure honesty annotation: the powered equivalence test lives in the table.
axB.text(0.5, 0.99, "equivalent\nTOST $\\pm0.02$", transform=axB.transAxes,
         ha="center", va="top", fontsize=7, color="0.30")

fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_proxy_vs_quality.pdf"))
print("WROTE", os.path.join(OUT, "fig_proxy_vs_quality.pdf"))
