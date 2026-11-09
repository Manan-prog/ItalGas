"""Report figure: why a pipe makes an arch in the radar picture, and how the same idea works across the array.
Usage: python -m italgas_gpr.figures.report.arch_geometry out.png"""
import sys
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from italgas_gpr.io.paths import FIGURES

SURF, INK, MUTED, BLUE, ORANGE, AQUA = "#fcfcfb", "#1f1f1f", "#6b6b6b", "#2a78d6", "#eb6834", "#1baf7a"
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURF, gridspec_kw=dict(wspace=0.25))
for a in ax:
    a.set_facecolor(SURF); a.axhline(0, color=INK, lw=1); a.fill_between([-3, 3], 0, -2.2, color="#efe9dc", zorder=0)
    for sp in a.spines.values():
        sp.set_visible(False)
    a.set_yticks([]); a.set_ylim(-2.2, 1.3)
# A: along the street, a crossing pipe
a = ax[0]; z0 = 0.7; xs = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
a.scatter([0], [-z0], s=160, color=BLUE, zorder=5); a.text(0.15, -z0 + 0.12, "pipe (end on)", color=BLUE, fontsize=9, va="center")
for x in xs:
    a.plot([x, x], [0.05, 0.45], color=INK, lw=4, solid_capstyle="round"); a.plot([x, 0], [0, -z0], color=MUTED, lw=0.8, ls="--")
a.text(0, 0.75, "the antenna passes over the pipe;\nfarther away, the echo comes back later", ha="center", fontsize=9, color=INK)
a.set_xlim(-3, 3); a.set_xlabel("along the street (m)", fontsize=9)
ins = a.inset_axes([0.08, 0.04, 0.84, 0.30]); ins.set_facecolor("#f4f4f1")
xx = np.linspace(-2.2, 2.2, 200); ins.plot(xx, -np.sqrt(z0 ** 2 + xx ** 2) * 2 / 0.1, color=BLUE, lw=2)
ins.set_xlim(-2.2, 2.2); ins.set_yticks([]); ins.set_xticks([]); ins.text(0, -5, "what the radargram shows: an arch", ha="center", va="top", fontsize=8.5, color=BLUE); ins.set_ylim(-60, 0)
a.set_title("A  Pipe crossing the street", loc="left", fontsize=10.5, color=INK)
# B: across the street, a pipe running along the street under the array
b = ax[1]; z0 = 0.7; us = np.array([-0.751, -0.5, -0.25, 0.0, 0.25, 0.5, 0.751])
b.scatter([0.3], [-z0], s=160, color=AQUA, zorder=5); b.text(0.45, -z0 + 0.12, "pipe running along the street (end on)", color=AQUA, fontsize=9, va="center")
for u in us:
    b.plot([u, u], [0.05, 0.45], color=INK, lw=4, solid_capstyle="round"); b.plot([u, 0.3], [0, -z0], color=MUTED, lw=0.8, ls="--")
b.text(0, 0.75, "the 19 channels sit side by side;\nthe one above the pipe hears it first", ha="center", fontsize=9, color=INK)
b.set_xlim(-3, 3); b.set_xlabel("across the street (m)", fontsize=9)
ins = b.inset_axes([0.08, 0.04, 0.84, 0.30]); ins.set_facecolor("#f4f4f1")
uu = np.linspace(-0.751, 0.751, 100); ins.plot(uu, -np.sqrt(z0 ** 2 + (uu - 0.3) ** 2) * 2 / 0.1, color=AQUA, lw=2); ins.set_xlim(-2.2, 2.2); ins.set_ylim(-60, 0)
ins.set_yticks([]); ins.set_xticks([]); ins.text(0, -5, "across the 1.5 m array: a short, flat arch", ha="center", va="top", fontsize=8.5, color=AQUA)
b.set_title("B  Pipe along the street, seen across the array", loc="left", fontsize=10.5, color=INK)
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else str(FIGURES / "report" / "arch_geometry.png"), dpi=170, facecolor=SURF, bbox_inches="tight"); print("ok")
