"""Framework / overview figure (Fig 2). Single-column compact pipeline:
decode trace -> 4 cheap views (within/cross kept, query/pos shown dropped) ->
3-param 2-view calibrated scorer (ours, red) -> per-layer conformal budget ->
demote / prefetch. The figure itself encodes the paper's thesis: only the two
attention-magnitude views survive; query-key and recency are redundant."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "paper_icdm/figures"
BLUE, RED, GREY, DK = "#2c7fb8", "#d62728", "#9e9e9e", "#222222"

fig, ax = plt.subplots(figsize=(3.45, 1.72))
ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")


def box(x, y, w, h, text, ec=DK, fc="white", fs=7.0, lw=1.1, tc=DK, style="round,pad=0.02"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                ec=ec, fc=fc, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=5)


def arrow(x1, y1, x2, y2, ec=DK, lw=1.1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=8, ec=ec, fc=ec, lw=lw,
                                 shrinkA=0, shrinkB=0))


# Stage 1: decode trace
box(1, 24, 17, 12, "decode\ntrace", fc="#f2f2f2")
arrow(18, 30, 24, 30)

# Stage 2: 4 cheap views -- within/cross kept (blue), query/pos dropped (grey, struck)
ax.text(35, 50, "4 cheap views", ha="center", fontsize=6.6, color=DK)
box(24, 41, 22, 7, "within", ec=BLUE, tc=BLUE, fs=6.6, lw=1.3)
box(24, 32.5, 22, 7, "cross", ec=BLUE, tc=BLUE, fs=6.6, lw=1.3)
box(24, 24, 22, 7, "query", ec=GREY, tc=GREY, fs=6.6, lw=1.0)
box(24, 15.5, 22, 7, "pos", ec=GREY, tc=GREY, fs=6.6, lw=1.0)
# strike through the two dropped views
for yy in (27.5, 19.0):
    ax.plot([25.5, 44.5], [yy, yy], color=GREY, lw=1.3, zorder=6)
ax.text(35, 11.5, "query, pos: redundant", ha="center", fontsize=5.8,
        color=GREY, style="italic")

# arrows only from the two surviving (magnitude) views into the scorer
arrow(46, 44.5, 55, 37, ec=BLUE)
arrow(46, 36.0, 55, 34, ec=BLUE)

# Stage 3: 2-view calibrated scorer (ours)
box(55, 27, 21, 14, "2-view\ncalibrated\nscorer\n(3 params)",
    ec=RED, tc=RED, fs=6.8, lw=1.6, fc="#fdecec")
arrow(76, 34, 82, 34)

# Stage 4: per-layer conformal budget
box(82, 27, 17, 14, "per-layer\nconformal\nbudget", ec=DK, fs=6.6)

# Stage 5: action (down from budget)
arrow(90.5, 27, 87, 19)
box(72, 5, 27, 13, "demote /\nprefetch", ec=DK, fs=6.8, fc="#f2f2f2")

fig.tight_layout(pad=0.2)
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "fig_framework.pdf"), bbox_inches="tight")
print("WROTE", os.path.join(OUT, "fig_framework.pdf"))
