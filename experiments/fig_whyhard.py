"""Fig 1 (why-hard), phenomenon version: the served-oracle prediction problem is
genuinely harder than the offline-label one, and it varies with depth. Per-layer
served-oracle AUC (0.54-0.76, mean 0.64) sits far below the offline future-attention
AUC (0.95) at every layer. Data: experiments/results/served_oracle_ci/c3_served_auc.json
(2-view scorer vs served-oracle label, Llama-3.1, 32 layers, prompt-clustered 95% CI).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})
OUT = "paper_icdm/figures"
RED, GREY = "#d62728", "#9e9e9e"

d = json.load(open("experiments/results/served_oracle_ci/c3_served_auc.json"))
pl = sorted(d["per_layer"], key=lambda r: r["layer_pos"])
x = np.array([r["layer_pos"] for r in pl])
y = np.array([r["auc_point"] for r in pl])
lo = np.array([r["auc_ci95_lo"] for r in pl])
hi = np.array([r["auc_ci95_hi"] for r in pl])
mean_auc = d["overall"]["auc_point"]

fig, ax = plt.subplots(figsize=(2.95, 1.62))
# offline-label reference (easy problem)
ax.axhline(0.95, color=GREY, ls="--", lw=1.3, label="offline label (AUC 0.95)")
# served-oracle per-layer (hard problem), ours = red
ax.fill_between(x, lo, hi, color=RED, alpha=0.18, lw=0)
ax.plot(x, y, color=RED, lw=1.7, label="served-oracle (per layer)")
ax.axhline(mean_auc, color=RED, ls=":", lw=1.0, label=f"served-oracle mean ({mean_auc:.2f})")
ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.85), fontsize=5.3,
          framealpha=0.9, frameon=True, handlelength=1.35, handletextpad=0.4,
          borderpad=0.3, labelspacing=0.22)
ax.set_xlabel("layer depth (normalized)")
ax.set_ylabel("AUC")
ax.set_ylim(0.5, 1.0)
ax.set_xlim(0, 1)
fig.tight_layout(pad=0.3)
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_whyhard.pdf"))
print(f"WROTE {OUT}/fig_whyhard.pdf  (per-layer AUC {y.min():.2f}-{y.max():.2f}, mean {mean_auc:.2f})")
