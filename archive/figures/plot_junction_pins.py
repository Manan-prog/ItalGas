"""Figure: along-street pipe locations pinned at junctions (user's idea, 3 Nov 2026). Where a drive crosses
another street, that street's main appears as a crossing pipe (both radar rows); such crossings that run
along the other street are 'pins'. Consecutive pins on the same street that agree (|du| <= 1 m, |dz| <= 0.25 m)
are joined as candidate straight pipelines. Usage (from figures/): python plot_junction_pins.py out.png"""
import json
import sys
from pathlib import Path

import contextily as cx
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from basemap_tools import GSAT  # noqa: E402
from geometry import merc_scale  # noqa: E402
from map_layers import decorate  # noqa: E402

SURF = "#fcfcfb"; INK = "#1f1f1f"; MUTED = "#6b6b6b"; PIN = "#eb6834"; SEG = "#2a78d6"
ROOT = Path(__file__).resolve().parents[1]; C = Path("/Users/manansarupria/ItalGas/cache/city")
cc = yaml.safe_load(open(ROOT / "pipeline/configs/citywide.yaml")); z = np.load(cc["swath_tracks"], allow_pickle=True)
T = [z[f"s{i}"] for i in range(len(z["days"]))]; k = float(merc_scale(T[0][:, 1]).mean())
cons = json.load(open(C / "anchors_consensus.json")); segs = json.load(open(C / "pinned_segments.json"))
import glob  # noqa: E402
from build_cube import swath_geometry  # noqa: E402
from build_v2 import load_cfg  # noqa: E402
from ids_reader import Swath  # noqa: E402
from run_detection import su_to_map  # noqa: E402
DET = load_cfg(str(ROOT / "pipeline" / cc["detector_config"])); files = sorted(glob.glob(str(Path(cc["raw_root"]) / "*" / "Survey_*" / "Swath*.path")))
for sg in segs:                                   # draw along the drive's own path at the pinned offset (pipes follow the street)
    f = files[sg["drive"]]; g = swath_geometry(Swath.load(str(Path(f).parent), Path(f).stem), DET["base"])
    ss = np.linspace(sg["s0"] + 1.19, sg["s1"] + 1.19, 200); uu = np.linspace(sg["u0"], sg["u1"], 200)
    mx, my = su_to_map(ss, uu, g["s_nodes"], g["xn"], g["yn"], float(g["k"])); sg["px"], sg["py"] = mx, my
    sg["straightness"] = float(np.hypot(mx[-1] - mx[0], my[-1] - my[0]) * k / (sg["s1"] - sg["s0"]))
    print(f"drive {sg['drive']}: {sg['s1'] - sg['s0']:.0f} m along the drive, straight-line distance ratio {sg['straightness']:.2f}")
fig = plt.figure(figsize=(18, 10.5), facecolor=SURF)
fig.suptitle("Along-street pipes pinned where another drive crosses the street (radar only, provisional)", x=0.02, y=0.975, ha="left",
             fontsize=14.5, color=INK, fontweight="bold")
fig.text(0.02, 0.93, f"Grey = every drive of the survey.  Orange dots = {len(cons)} pins: a pipe running along a street, seen as a crossing pipe by a drive that crosses that street "
         f"(large: confirmed by 2 or more crossing drives).\nBlue = {len(segs)} candidate pipelines where consecutive pins on the same street agree within 1 m sideways and 0.25 m in depth (5 of 22 pairs vs 1.9 by chance, p = 0.03), drawn along the street; dashed = road curves a lot, pairing doubtful.",
         fontsize=10, color=MUTED)
gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.05, left=0.02, right=0.98, top=0.86, bottom=0.08)
for ax_i in (0, 1):
    ax = fig.add_subplot(gs[ax_i])
    for tr in T:
        if len(tr) > 2:
            ax.plot(tr[:, 0], tr[:, 1], color="#cfd6de", lw=1.2 if ax_i == 0 else 5, alpha=0.9, zorder=3, solid_capstyle="butt")
    for c in cons:
        ax.scatter(c["x"], c["y"], s=(60 if c["support"] >= 2 else 18) * (1 if ax_i == 0 else 3), color=PIN, edgecolors="white", linewidths=0.6, zorder=6)
    for sg in segs:
        ok = sg["straightness"] >= 0.8
        ax.plot(sg["px"], sg["py"], color="white", lw=5 if ax_i == 0 else 7, zorder=4)
        ax.plot(sg["px"], sg["py"], color=SEG, lw=3 if ax_i == 0 else 4.5, zorder=5, ls="-" if ok else (0, (2, 2)))
    if ax_i == 0:
        allx = np.concatenate([t[:, 0] for t in T if len(t) > 2]); ally = np.concatenate([t[:, 1] for t in T if len(t) > 2])
        cxm, cym = np.median(allx), np.median(ally); half = 0.5 * max(np.quantile(allx, 0.98) - np.quantile(allx, 0.02), np.quantile(ally, 0.98) - np.quantile(ally, 0.02)) * 1.05
        zoom, bar, ttl = 15, 500, "A.  Whole survey"
    else:
        s0 = max([q for q in segs if q["straightness"] >= 0.8], key=lambda q: q["s1"] - q["s0"])
        cxm, cym = 0.5 * (s0["px"].min() + s0["px"].max()), 0.5 * (s0["py"].min() + s0["py"].max())
        half = 0.5 * max(np.ptp(s0["px"]), np.ptp(s0["py"])) * 1.2; zoom, bar = 19, 50
        ttl = f"B.  Longest pinned segment: {s0['s1'] - s0['s0']:.0f} m, {s0['u0']:+.1f} to {s0['u1']:+.1f} m from the drive, {s0['z0']:.2f}-{s0['z1']:.2f} m deep"
        for c in cons:
            if abs(c["x"] - cxm) < half and abs(c["y"] - cym) < half:
                ax.annotate(f"{c['depth_m']:.2f} m" + (f", {c['support']} drives" if c["support"] >= 2 else ""), (c["x"], c["y"]), xytext=(6, 6),
                            textcoords="offset points", fontsize=8.5, color="white", zorder=7, bbox=dict(fc=PIN, ec="none", alpha=0.85, pad=1))
    ax.set_xlim(cxm - half, cxm + half); ax.set_ylim(cym - half, cym + half); ax.set_aspect("equal")
    cx.add_basemap(ax, source=GSAT, crs="EPSG:3857", zoom=zoom, attribution=False, alpha=0.85); decorate(ax, cxm, cym, half, k, bar)
    ax.set_title(ttl, loc="left", fontsize=11.5, color=INK)
fig.legend(handles=[Line2D([], [], color="#cfd6de", lw=4, label="drives of the survey"),
                    Line2D([], [], marker="o", ls="", color=PIN, mec="white", ms=5, label="pin, 1 crossing drive"),
                    Line2D([], [], marker="o", ls="", color=PIN, mec="white", ms=9, label="pin, confirmed by 2 or more crossing drives"),
                    Line2D([], [], color=SEG, lw=4, label="candidate pipeline between agreeing pins, along the street"),
                    Line2D([], [], color=SEG, lw=4, ls=(0, (2, 2)), label="same, but the road curves a lot (pairing doubtful)")],
           loc="lower center", ncol=5, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.savefig(sys.argv[1], dpi=150, facecolor=SURF); print("saved", sys.argv[1])
