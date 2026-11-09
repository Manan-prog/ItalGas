"""Figure: where do buried utilities cross Corso A. Ricci, and how sure are we?
v2 detector, 14 Oct 11:31 swath. Usage (from figures/):
  python plot_utility_map_v2.py final_v2_<...>.json out.png
Panels: A overview; B and C street-aligned detail (zoom 21) with the probability that a
crossing utility passes each point (each detection spread by its measured position
uncertainty); D measured position uncertainty along the street."""
import sys
from pathlib import Path

import contextily as cx
import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import json  # noqa: E402

from basemap_tools import GSAT, street_aligned_tiles  # noqa: E402
from build_v2 import load_cfg  # noqa: E402
from eval_crossings_v2 import drain_targets  # noqa: E402
from geometry import merc_to_lonlat  # noqa: E402
from run_detection import su_to_map  # noqa: E402

SURF = "#fcfcfb"; BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"; INK = "#1f1f1f"; MUTED = "#6b6b6b"
ROOT = Path(__file__).resolve().parents[1]
cfg = load_cfg(str(ROOT / "pipeline/configs/dev_swath_v2_2.yaml"))
CACHE = Path(cfg["base"]["cache_dir"])
fin = json.load(open(CACHE / sys.argv[1])); out_png = sys.argv[2]
cube = np.load(CACHE / cfg["cube_file"]); s_nodes, xn, yn, k = cube["s_nodes"], cube["xn"], cube["yn"], float(cube["k"])
X = fin["crossings"]; T = fin["tracks"]; AD = fin["along_disagreement"]
recs = drain_targets(cube)
U_LO, U_HI = -0.58, 0.93
DETAIL = [(995.0, 1055.0, "B"), (5.0, 65.0, "C")]
cmap = LinearSegmentedColormap.from_list("p", [(0, (1, 1, 1, 0)), (0.35, to_rgba("#f7b267", 0.45)), (1.0, to_rgba("#b4321c", 0.9))])


def prob_map(s_grid, u_grid):
    """P(a crossing utility passes this point) = 1 - prod_i (1 - P_i g_i), g_i = Gaussian in the
    along-track offset from detection i's line with sigma = its along-track uncertainty."""
    S, U = np.meshgrid(s_grid, u_grid, indexing="ij")
    q = np.ones_like(S)
    for o in X:
        ds_ = S - (o["s0_m"] + U * np.tan(np.radians(o["phi_deg"])))
        q *= 1 - o["P"] * np.exp(-0.5 * (ds_ / o["sigma_along_m"]) ** 2)
    return 1 - q


fig = plt.figure(figsize=(21, 13), facecolor=SURF)
fig.suptitle("Where do buried utilities cross Corso A. Ricci, and how sure are we?  (radar only, provisional: no utility-company map yet)",
             x=0.01, y=0.99, ha="left", fontsize=14, color=INK)
fig.text(0.01, 0.955, "A: whole drive, every detected crossing.  B, C: 60 m close-ups; shading = probability that a buried utility "
         "crosses at that point, including how far our positions can be off.  D: how far off the positions can be along the street.",
         fontsize=9.5, color=MUTED)
gs = fig.add_gridspec(3, 2, width_ratios=[0.2, 0.8], height_ratios=[1, 1, 0.75], hspace=0.42, wspace=0.06,
                      left=0.01, right=0.965, top=0.9, bottom=0.06)
axA = fig.add_subplot(gs[:, 0]); axB = fig.add_subplot(gs[0, 1]); axC = fig.add_subplot(gs[1, 1]); axD = fig.add_subplot(gs[2, 1])
for ax in (axA, axB, axC, axD):
    ax.set_facecolor(SURF)
    for sp in ax.spines.values():
        sp.set_color("#cfcfcf")
    ax.tick_params(colors=MUTED, labelsize=8)

# ---- A overview
pad = 60 / k
axA.plot(xn, yn, color="white", lw=2.5, alpha=0.9, zorder=2)
for cls, col, sz in (("B", BLUE, 22), ("A", ORANGE, 46)):
    oo = [o for o in X if o["cls"] == cls]
    mx, my = su_to_map(np.array([o["s0_m"] for o in oo]), np.zeros(len(oo)), s_nodes, xn, yn, k)
    axA.scatter(mx, my, s=sz, color=col, edgecolors="white", linewidths=0.6, zorder=4)
mx, my = su_to_map(np.array([r["s0"] for r in recs]), np.zeros(len(recs)), s_nodes, xn, yn, k)
axA.scatter(mx, my, marker="x", s=28, color="white", linewidths=1.4, zorder=5)
for s0, s1, lab in DETAIL:
    xa, ya = su_to_map(np.array([s0, s1]), np.array([0, 0]), s_nodes, xn, yn, k)
    axA.plot(xa, ya, color=AQUA, lw=6, alpha=0.9, zorder=3, solid_capstyle="butt")
    axA.annotate(f"panel {lab}", (xa.mean(), ya.mean()), color="white", fontsize=10, fontweight="bold", xytext=(-70, 0), textcoords="offset points")
axA.set_xlim(xn.min() - pad, xn.max() + pad); axA.set_ylim(yn.min() - pad, yn.max() + pad); axA.set_aspect("equal")
cx.add_basemap(axA, source=GSAT, crs="EPSG:3857", zoom=18, attribution=False, alpha=0.75)
sb = 100 / k; x0b = xn.min() - pad + 15 / k; y0b = yn.min() - pad + 15 / k
axA.plot([x0b, x0b + sb], [y0b, y0b], color="white", lw=3, zorder=6); axA.text(x0b + sb / 2, y0b + 8 / k, "100 m", color="white", ha="center", fontsize=9)
lon, lat = merc_to_lonlat(np.array([xn.min() - pad, xn.max() + pad]), np.array([yn.min() - pad, yn.max() + pad]))
axA.text(0.005, 0.005, f"SW {lat[0]:.5f}N {lon[0]:.5f}E", transform=axA.transAxes, color="white", fontsize=8, va="bottom")
axA.text(0.995, 0.995, f"NE {lat[1]:.5f}N {lon[1]:.5f}E", transform=axA.transAxes, color="white", fontsize=8, va="top", ha="right")
axA.set_xticks([]); axA.set_yticks([])
nA = sum(o["cls"] == "A" for o in X); nB = len(X) - nA
axA.set_title(f"A  The 1.2 km drive\n{nA} crossings seen by both radar rows,\n{nB} seen by one", loc="left", fontsize=10.5, color=INK)
axA.legend(handles=[Line2D([], [], marker="o", ls="", color=ORANGE, mec="white", ms=8, label="crossing seen by BOTH radar rows (600 + 200 MHz)"),
                    Line2D([], [], marker="o", ls="", color=BLUE, mec="white", ms=6, label="crossing seen by ONE radar row"),
                    Line2D([], [], marker="x", ls="", color="k", ms=7, label="storm-drain pipe in municipal records"),
                    Line2D([], [], color=AQUA, lw=5, label="close-up panels B, C")],
           loc="upper center", fontsize=8, framealpha=0.92, bbox_to_anchor=(0.5, -0.005))

# ---- B, C details
HALF_W = 4.0
for ax, (s0, s1, lab) in zip((axB, axC), DETAIL):
    img, ext, hdg_deg, (xc, yc) = street_aligned_tiles(s0, s1, HALF_W, s_nodes, xn, yn, k)
    ax.imshow(img, extent=ext, origin="upper", zorder=1, interpolation="bilinear")
    sg = np.arange(s0, s1, 0.05); ug = np.arange(U_LO, U_HI, 0.04)
    P = prob_map(sg, ug)
    im = ax.imshow(P.T, extent=[s0, s1, U_LO, U_HI], origin="lower", cmap=cmap, vmin=0, vmax=1, aspect="auto", zorder=3)
    for uu in (U_LO, U_HI):
        ax.axhline(uu, color="white", lw=0.8, ls="-", alpha=0.8, zorder=4)
    ax.text(s0 + 0.5, U_HI + 0.12, "radar footprint (1.5 m wide)", color="white", fontsize=8, zorder=6)
    for o in X:
        if not (s0 - 2 < o["s0_m"] < s1 + 2):
            continue
        uu = np.array([U_LO, U_HI]); ss = o["s0_m"] + uu * np.tan(np.radians(o["phi_deg"]))
        col = ORANGE if o["cls"] == "A" else BLUE
        ax.plot(ss, uu, color=col, lw=2.2 if o["cls"] == "A" else 1.4, zorder=5, solid_capstyle="round")
        if o["cls"] == "A":
            ax.text(ss[1], U_HI + 0.55, f"{o['depth_m']:.1f} m deep", color="white", fontsize=7.5, ha="center", zorder=6,
                    bbox=dict(fc=col, ec="none", alpha=0.9, pad=1.2))
    for r in recs:
        if s0 < r["s0"] < s1:
            uu = np.array([-HALF_W, HALF_W]); ss = r["s0"] + uu * np.tan(np.radians(r["phi_deg"]))
            ax.plot(ss, uu, color="white", lw=1.3, ls=(0, (4, 3)), zorder=4)
    for t in T:
        if t["s_end"] < s0 or t["s_start"] > s1:
            continue
        ps, pu = np.array(t["path_s"]), np.array(t["path_u"])
        ax.plot(ps, pu, color=AQUA, lw=3, alpha=0.75, zorder=4, solid_capstyle="butt")
    ax.plot([s0 + 1, s0 + 11], [-HALF_W + 0.6] * 2, color="white", lw=3, zorder=7)
    ax.text(s0 + 6, -HALF_W + 0.9, "10 m", color="white", ha="center", fontsize=8, zorder=7)
    lonc, latc = merc_to_lonlat(np.array([xc]), np.array([yc]))
    ax.text(s1 - 0.5, -HALF_W + 0.4, f"centre {latc[0]:.5f}N {lonc[0]:.5f}E, travel heading {(90 - hdg_deg) % 360:.0f} deg",
            color="white", fontsize=7.5, ha="right", zorder=7)
    ax.set_xlim(s0, s1); ax.set_ylim(-HALF_W, HALF_W); ax.set_aspect("equal")
    ax.set_ylabel("across street (m)\n+ = left of travel", color=MUTED, fontsize=8.5)
    sig = np.interp(0.5 * (s0 + s1), AD["s"], np.abs(AD["along"]))
    ax.set_title(f"{lab}  Close-up, {s0:.0f} to {s1:.0f} m along the drive (position uncertainty here about {max(0.3, sig):.1f} m along the street)",
                 loc="left", fontsize=10.5, color=INK)
cb = fig.colorbar(im, ax=[axB, axC], fraction=0.012, pad=0.01)
cb.set_label("probability a buried utility crosses here", color=MUTED, fontsize=9); cb.ax.tick_params(colors=MUTED, labelsize=8)
axB.legend(handles=[Line2D([], [], color=ORANGE, lw=2.2, label="crossing, both radar rows (label = depth)"),
                    Line2D([], [], color=BLUE, lw=1.4, label="crossing, one radar row"),
                    Line2D([], [], color="white", lw=1.3, ls=(0, (4, 3)), label="storm-drain record (municipal)"),
                    Line2D([], [], color=AQUA, lw=3, label="wide reflector along street at 0.85 m (probable sewer collector)")],
           loc="lower left", fontsize=8.5, ncol=4, bbox_to_anchor=(0.0, 1.1), facecolor="#4a4a4a", edgecolor="none", labelcolor="white", framealpha=1)
axC.set_xlabel("distance along the drive (m)", color=MUTED, fontsize=9)

# ---- D position uncertainty
sD = np.array(AD["s"]); aD = np.abs(np.array(AD["along"])); cD = np.abs(np.array(AD["cross"]))
axD.plot(sD, aD, color=ORANGE, lw=1.6, label="along the street")
axD.plot(sD, cD, color=BLUE, lw=1.6, label="across the street")
axD.axhline(0.3, color=MUTED, lw=0.8, ls=":"); axD.text(5, 0.36, "0.3 m floor (radar picking)", color=MUTED, fontsize=8)
for s0, s1, lab in DETAIL:
    axD.axvspan(s0, s1, color=AQUA, alpha=0.15, lw=0); axD.text((s0 + s1) / 2, axD.get_ylim()[1] * 0.9 if False else 3.2, lab, color=AQUA, ha="center", fontsize=10, fontweight="bold")
axD.set_xlim(0, sD.max()); axD.set_ylim(0, 3.6)
axD.set_ylabel("disagreement (m)", color=MUTED, fontsize=9); axD.set_xlabel("distance along the drive (m)", color=MUTED, fontsize=9)
axD.legend(loc="upper center", fontsize=8.5, frameon=False, ncol=2)
axD.grid(axis="y", color="#e6e6e6", lw=0.6)
axD.set_title("D  How far can positions be off? Gap between the two GPS track solutions (real-time RTK vs post-processed); detections inherit it",
              loc="left", fontsize=10.5, color=INK)
fig.savefig(out_png, dpi=150, facecolor=SURF)
print("saved", out_png)
