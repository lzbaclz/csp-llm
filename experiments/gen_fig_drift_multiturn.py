"""Figure for E5: boundary-aligned missed-saliency around a topic switch.

Reads experiments/results/drift_multiturn.json and draws, per policy, the mean
per-step miss as a function of offset from a turn boundary (the switch at 0). The
story: fixed spikes and stays up; adaptive spikes then recovers below alpha.

    python experiments/gen_fig_drift_multiturn.py \
        --res experiments/results/drift_multiturn.json \
        --out paper_icdm/figures/fig_drift_multiturn.pdf
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "fixed_global": ("fixed (offline split-conformal)", "#c0392b", "o"),
    "coverage_per_layer": ("static per-layer (GuardKV E1)", "#e67e22", "s"),
    "adaptive_conformal": ("adaptive-conformal (ACI)", "#2471a3", "^"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="experiments/results/drift_multiturn.json")
    ap.add_argument("--out", default="paper_icdm/figures/fig_drift_multiturn.pdf")
    a = ap.parse_args()
    res = json.load(open(a.res))
    ba = res["boundary_aligned"]
    alpha = res["meta"]["alpha"]

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for key, (label, color, marker) in LABELS.items():
        d = ba[key]
        offs = sorted(int(o) for o in d)
        xs = [o for o in offs if d[str(o)] is not None]
        ys = [d[str(o)] for o in xs]
        ax.plot(xs, ys, marker=marker, ms=3, lw=1.6, color=color, label=label)
    ax.axvline(0, color="0.4", ls="--", lw=1.0)
    ax.axhline(alpha, color="0.5", ls=":", lw=1.0)
    ax.text(0.2, alpha + 0.004, fr"target $\alpha$={alpha}", fontsize=7, color="0.4")
    ax.annotate("topic switch", xy=(0, ax.get_ylim()[1]), xytext=(1.5, ax.get_ylim()[1]),
                fontsize=7, color="0.4", va="top")
    ax.set_xlabel("decode steps relative to turn boundary")
    ax.set_ylabel("missed-saliency rate")
    ax.set_title("Coverage across a topic switch (multi-turn)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
