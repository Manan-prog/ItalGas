"""Figure (INTERNAL, provisional): radar results vs the Comune's 2011 town-plan sketch of the gas and
water networks (PUC DF S.2.1, schematic, lines placed to several metres; user approval 3 Nov 2026,
not for redistribution). A Corso A. Ricci: the long 0.85 m reflector vs the drawn gas line.
B Via Nizza: merged drives, along-road candidates vs drawn gas and water lines.
Usage (from figures/): python plot_puc_overlay.py out.png"""
import json
import sys
from pathlib import Path

import contextily as cx
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from basemap_tools import GSAT  # noqa: E402
from map_layers import CACHE, PCMAP, decorate, draw_strips, strip_multipass, strip_single  # noqa: E402
from run_detection import su_to_map  # noqa: E402

SURF = "#fcfcfb"; INK = "#1f1f1f"; MUTED = "#6b6b6b"; GAS = "#f2c12e"; WATER = "#3ec7f0"; AQUA = "#1baf7a"
P = np.load(CACHE / "puc_su.npz")
fig = plt.figure(figsize=(18, 10.5), facecolor=SURF)
fig.suptitle("Radar finds vs the town-plan sketch of the gas network (INTERNAL, provisional: the sketch is schematic, lines placed to several metres)",
             x=0.02, y=0.975, ha="left", fontsize=13.5, color=INK, fontweight="bold")
fig.text(0.02, 0.928, "Yellow dots = gas main and cyan dots = water main as drawn on the Comune di Savona 2011 plan sheet DF S.2.1 (1:10,000). "
         "Strips = ground under the radar van, coloured by probability of a pipe.\nA: on Corso A. Ricci the long reflector 0.85 m deep (green) "
         "lies where the sketch puts the gas main under the van: 64 percent of its length vs 16 percent by chance (p < 0.001).", fontsize=10, color=MUTED)
gs = fig.add_gridspec(1, 2, wspace=0.06, left=0.02, right=0.98, top=0.86, bottom=0.12)

# ---- A Corso Ricci, s 1080-1220
cube = np.load(CACHE / "cube_Swath001_2025.10.14_11.31.18_2d9b11acb1.npz"); frd = {k: cube[k] for k in ("s_nodes", "xn", "yn", "k")}; k = float(frd["k"])
fin = json.load(open(sorted(CACHE.glob("final_v2_*.json"))[-1]))
S0, S1 = 1080.0, 1215.0
ax = fig.add_subplot(gs[0])
xs, ys = draw_strips(ax, [strip_single(frd, S0, S1, [o for o in fin["crossings"] if S0 - 3 < o["s0_m"] < S1 + 3], -0.58, 0.93)])
T = json.load(open(CACHE / "long_df5d12447d.json"))["arrays"]["2"]["tracks"]
for t in [t for t in T if t["q"] <= 0.05]:
    ps, pu = np.array(t["path_s"]), np.array(t["path_u"]); w = (ps >= S0) & (ps <= S1)
    if w.sum() > 2:
        mx, my = su_to_map(ps[w], pu[w], frd["s_nodes"], frd["xn"], frd["yn"], k); ax.plot(mx, my, color=AQUA, lw=3, zorder=5)
for cls, col in (("gas", GAS), ("water", WATER)):
    s_, u_ = P[f"ricci_{cls}_s"], P[f"ricci_{cls}_u"]; w = (s_ >= S0 - 20) & (s_ <= S1 + 20)
    mx, my = su_to_map(s_[w], u_[w], frd["s_nodes"], frd["xn"], frd["yn"], k); ax.scatter(mx, my, s=3, color=col, zorder=6, linewidths=0)
cxm, cym = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)); half = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys)) * 1.1
ax.set_xlim(cxm - half, cxm + half); ax.set_ylim(cym - half, cym + half); ax.set_aspect("equal")
cx.add_basemap(ax, source=GSAT, crs="EPSG:3857", zoom=20, attribution=False, alpha=0.9); decorate(ax, cxm, cym, half, k, 20)
ax.set_title("A.  Corso A. Ricci, 1080-1215 m: long reflector at 0.85 m along the drawn gas main", loc="left", fontsize=11.5, color=INK)

# ---- B Via Nizza, s 2840-2980 (merged drives)
REG = json.load(open(CACHE / "mp_registration.json"))["registration"]; fr = dict(np.load(CACHE / "mp_frame.npz")); kn = float(fr["k"])
ST = json.load(open(CACHE / "mp_shape_test.json")); summ = ST["summary"]; p_al = 1 - summ["frac_controls"] / summ["frac_features"]
cands = [x for x in ST["features"] if x["verdict"] == "pipe-like"]
S0, S1 = 2840.0, 2980.0
ax = fig.add_subplot(gs[1])
xs, ys = draw_strips(ax, [r for r in (strip_multipass(n, S0, S1, REG, fr, cands, p_al) for n in REG) if r is not None])
for cls, col in (("gas", GAS), ("water", WATER)):
    s_, u_ = P[f"nizza_{cls}_s"], P[f"nizza_{cls}_u"]; w = (s_ >= S0 - 20) & (s_ <= S1 + 20)
    mx, my = su_to_map(s_[w], u_[w], fr["s_nodes"], fr["xn"], fr["yn"], kn); ax.scatter(mx, my, s=3, color=col, zorder=6, linewidths=0)
cxm, cym = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)); half = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys)) * 1.1
ax.set_xlim(cxm - half, cxm + half); ax.set_ylim(cym - half, cym + half); ax.set_aspect("equal")
cx.add_basemap(ax, source=GSAT, crs="EPSG:3857", zoom=20, attribution=False, alpha=0.9); decorate(ax, cxm, cym, half, kn, 20)
ax.set_title("B.  Via Nizza, 2840-2980 m: along-road candidate 1 lies 0.6 m from both drawn lines", loc="left", fontsize=11.5, color=INK)

sm = plt.cm.ScalarMappable(cmap=PCMAP, norm=plt.Normalize(0, 1))
cax = fig.add_axes([0.2, 0.065, 0.26, 0.022]); cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_label("probability that a buried pipe passes under this spot", fontsize=10, color=INK); cb.ax.tick_params(labelsize=9)
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=GAS, ms=6, label="gas main, as drawn on the 2011 plan sketch"),
                    Line2D([], [], marker="o", ls="", color=WATER, ms=6, label="water main, as drawn on the 2011 plan sketch"),
                    Line2D([], [], color=AQUA, lw=3, label="long radar reflector 0.85 m deep (Corso Ricci)")],
           loc="lower right", bbox_to_anchor=(0.99, 0.02), fontsize=9.5, frameon=False)
fig.savefig(sys.argv[1], dpi=160, facecolor=SURF); print("saved", sys.argv[1])
