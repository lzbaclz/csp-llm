"""Capstone figure: offline KV-saliency accuracy does NOT predict deployment
selection. (a) the scorer's AUC collapses from 0.95 (offline, future-attention
label) to 0.64 (deployment, served-oracle label); (b) consequently a trivial
accumulator (H2O) beats the learned selector at every budget in the serving loop."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
R = "experiments/results"
OUT = "paper_icdm/figures"

# --- deployment AUC + score separation from the serving-calibration log ---
d = json.load(open(f"{R}/serving_calib/calib_llama.json"))
rows = np.array(d["rows"]); score, label = rows[:, 2], rows[:, 3]
auc_deploy = roc_auc_score(label, score)
auc_offline = 0.948   # 2-view, h4, request-level (RESULTS headline / horizon_headtohead)

# --- deployment eps vs budget (H2O vs learned) ---
dd = json.load(open(f"{R}/design_deploy/SUMMARY.json"))["eps_oracle_FN"]
budgets = [0.10, 0.20, 0.30]
h2o = [dd["h2o"][f"b{b:.2f}"] for b in budgets]
learned = [dd["xqp"][f"b{b:.2f}"] for b in budgets]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(5.6, 2.3))

# Panel A: AUC collapse
bars = axA.bar([0, 1], [auc_offline, auc_deploy],
               color=["#2c7fb8", "#d95f0e"], width=0.6)
axA.axhline(0.5, ls=":", c="gray", lw=0.8)
axA.set_xticks([0, 1]); axA.set_xticklabels(["offline\n(future attn.)", "deployment\n(served oracle)"])
axA.set_ylabel("scorer AUC"); axA.set_ylim(0.5, 1.0)
for x, v in zip([0, 1], [auc_offline, auc_deploy]):
    axA.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
axA.annotate("", xy=(1, auc_deploy + 0.03), xytext=(0, auc_offline - 0.02),
             arrowprops=dict(arrowstyle="->", color="black", lw=1.1))
axA.set_title("(a) offline skill does not transfer", fontsize=9)

# Panel B: deployment eps vs budget
axB.plot(budgets, h2o, "o-", color="#2c7fb8", label="H2O (accumulation)")
axB.plot(budgets, learned, "s--", color="#d95f0e", label="learned 2-view")
axB.set_xlabel("HBM budget (kept fraction)")
axB.set_ylabel("served-oracle miss (lower=better)")
axB.set_xticks(budgets)
axB.legend(frameon=False, fontsize=8, loc="upper right")
axB.set_title("(b) H2O wins in the serving loop", fontsize=9)

fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_transfer_gap.pdf"))
plt.close(fig)
print(f"deploy AUC={auc_deploy:.3f}  offline AUC={auc_offline}")
print(f"H2O eps {h2o}  learned eps {learned}")
print("WROTE", os.path.join(OUT, "fig_transfer_gap.pdf"))
