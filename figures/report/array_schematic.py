"""Report figure: the radar array under the van, from the survey file (channel footprints measured, CLAUDE.md section 3).
Usage: python array_schematic.py out.png"""
import sys
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

SURF, INK, MUTED, BLUE, ORANGE, AQUA = "#fcfcfb", "#1f1f1f", "#6b6b6b", "#2a78d6", "#eb6834", "#1baf7a"
boxes = [-0.671, -0.335, 0.0, 0.335, 0.671]                       # antenna box centres (m), both rows
u200 = [-0.751, -0.671, -0.591, -0.503, -0.415, -0.335, -0.255, -0.168, -0.08, 0, 0.08, 0.168, 0.255, 0.335, 0.415, 0.503, 0.591, 0.671, 0.751]
u600 = [-0.751, -0.591, -0.415, -0.255, -0.08, 0.08, 0.255, 0.415, 0.591, 0.751]
fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURF); ax.set_facecolor(SURF)
# plan view: van outline, two rows of boxes, footprints
ax.add_patch(Rectangle((-1.0, -1.25), 2.0, 3.45, fc="#e9ebee", ec="#c9ccd1", lw=1, zorder=0)); ax.text(0, 2.0, "van (plan view)", ha="center", color=MUTED, fontsize=9)
for y, name, col, pitch in ((0.0, "200 MHz row (rear): 5 boxes, 19 channels", BLUE, u200), (0.9, "600 MHz row (front): 5 boxes, 10 channels", ORANGE, u600)):
    for b in boxes:
        ax.add_patch(Rectangle((b - 0.155, y - 0.14), 0.31, 0.28, fc="white", ec=col, lw=1.4, zorder=2))
    ax.scatter(pitch, [y] * len(pitch), s=18, color=col, zorder=3)
    ax.text(-1.05, y + 0.26, name, color=col, fontsize=9.5, ha="left", va="bottom")
ax.text(0, -0.42, "dots = where each channel looks down:\nevery 8 cm at 200 MHz, every 16 to 18 cm at 600 MHz", ha="center", va="top", color=INK, fontsize=9)
ax.annotate("", xy=(0.751, -0.85), xytext=(-0.751, -0.85), arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2)); ax.text(0, -0.99, "1.50 m strip seen in one pass", ha="center", color=INK, fontsize=9.5)
ax.add_patch(FancyArrow(1.25, 0.0, 0.0, 1.0, width=0.05, head_width=0.18, head_length=0.18, color=AQUA)); ax.text(1.42, 0.5, "travel", color=AQUA, fontsize=9.5, va="center")
ax.text(1.12, 1.55, "one scan every\n7.6 cm (200 MHz)\n4.5 cm (600 MHz)", color=MUTED, fontsize=8.5, va="center")
ax.set_xlim(-1.15, 1.95); ax.set_ylim(-1.3, 2.3); ax.set_aspect("equal"); ax.axis("off")
ax.set_title("Two rows of antennas look down at a 1.5 m strip under the van", loc="left", fontsize=11, color=INK)
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "array_schematic.png", dpi=170, facecolor=SURF, bbox_inches="tight"); print("ok")
