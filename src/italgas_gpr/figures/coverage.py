"""Investor figure 1: city-wide survey coverage of Savona.
One city-scale map of every drive of the Savona survey. Drives that were processed to an evidence raster (cache/city/
rframe_<i>.npz) are coloured by real-time positioning quality along the track (RTK fixed vs not fixed, every 0.5 m);
the other swaths of the track file are drawn in grey. Totals in the inset are computed from the files read here.
Usage: python -m italgas_gpr.figures.coverage [out.png]"""
from __future__ import annotations

import sys
from pathlib import Path

import contextily as cx
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from italgas_gpr.figures.basemap import GSAT  # noqa: E402
from italgas_gpr.geometry.frames import merc_scale, merc_to_lonlat  # noqa: E402
from italgas_gpr.io.paths import CITY, FIGURES, config_path, load_yaml  # noqa: E402

SURF, INK, MUTED, BLUE, ORANGE, AQUA = "#fcfcfb", "#1f1f1f", "#6b6b6b", "#2a78d6", "#eb6834", "#1baf7a"
RTK_FIX = 4                                   # GGA fix quality code for RTK fixed (measured, .nmea)

# ---- every swath track (EPSG:3857) and its day
T = np.load(load_yaml(config_path("survey.yaml"))["swath_tracks"], allow_pickle=True)
n_swaths = len(T["days"]); days = sorted(set(str(d) for d in T["days"]))
tracks = {i: T[f"s{i}"] for i in range(n_swaths) if len(T[f"s{i}"]) > 2}
k_all = float(np.mean([merc_scale(tr[:, 1]).mean() for tr in tracks.values()]))          # Web Mercator -> metres at this latitude
km_total = sum(np.sum(np.hypot(*np.diff(tr, axis=0).T)) * float(merc_scale(tr[:, 1]).mean()) for tr in tracks.values()) / 1e3

# ---- processed drives: frame + positioning quality every 0.5 m
frames = {int(f.stem.split("_")[1]): np.load(f) for f in CITY.glob("rframe_*.npz")}
segs_fix, segs_nofix, n_fix, n_q = [], [], 0, 0
km_processed = 0.0
for i, fr in frames.items():
    s, rq = fr["s"], fr["rt_q"]; sn, xn, yn = fr["s_nodes"], fr["xn"], fr["yn"]
    x = np.interp(s, sn, xn); y = np.interp(s, sn, yn); pts = np.c_[x, y]
    seg = np.stack([pts[:-1], pts[1:]], 1); fixed = rq[:-1] == RTK_FIX; known = np.isfinite(rq[:-1])
    segs_fix.append(seg[fixed]); segs_nofix.append(seg[~fixed & known]); segs_nofix.append(seg[~known])
    n_fix += int(fixed.sum()); n_q += int(known.sum()); km_processed += float(fr["length_m"]) / 1e3
segs_fix = np.concatenate(segs_fix); segs_nofix = np.concatenate(segs_nofix)
share_fix = n_fix / max(n_q, 1)

# ---- figure
allxy = np.concatenate(list(tracks.values())); x0, x1 = allxy[:, 0].min(), allxy[:, 0].max(); y0, y1 = allxy[:, 1].min(), allxy[:, 1].max()
cxm, cym = 0.5 * (x0 + x1), 0.5 * (y0 + y1); half = 0.5 * max(x1 - x0, y1 - y0) * 1.06
fig, ax = plt.subplots(figsize=(13, 13.6), facecolor=SURF)
fig.subplots_adjust(left=0.01, right=0.99, top=0.905, bottom=0.01)
fig.suptitle(f"Array-radar survey coverage: {km_total:.0f} km of street.  Savona, 13 to 17 October 2025", x=0.01, ha="left", fontsize=18, color=INK, weight="bold", y=0.975)
fig.text(0.01, 0.945, "Every drive of the array-radar survey, coloured by the real-time positioning quality of the vehicle. Basemap: Esri World Imagery, 75 % opacity.",
         fontsize=12.5, color=MUTED)
ax.set_facecolor(SURF); ax.set_xlim(cxm - half, cxm + half); ax.set_ylim(cym - half, cym + half); ax.set_aspect("equal")
# Esri World Imagery instead of Google satellite (6 Nov 2026): over this extent Google is a mosaic of two imagery dates with a
# visible seam at every zoom; Esri is one acquisition (a few clouds over the uninhabited hills in the north-west).
cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, crs="EPSG:3857", zoom=15, attribution=False, alpha=0.75)
for i, tr in tracks.items():
    if i not in frames:
        ax.plot(tr[:, 0], tr[:, 1], color="#c9c9c9", lw=2.0, alpha=0.95, zorder=3, solid_capstyle="round")
ax.add_collection(LineCollection(segs_nofix, colors=ORANGE, linewidths=2.6, zorder=4, capstyle="round"))
ax.add_collection(LineCollection(segs_fix, colors=AQUA, linewidths=2.6, zorder=5, capstyle="round"))
# scale bar, north arrow, corner coordinates
bar_m = 1000.0; xb, yb = cxm - half * 0.92, cym - half * 0.92
ax.plot([xb, xb + bar_m / k_all], [yb, yb], color="white", lw=6, zorder=7, solid_capstyle="butt")
ax.text(xb + bar_m / k_all / 2, yb + half * 0.025, "1 km", color="white", ha="center", fontsize=14, fontweight="bold", zorder=7)
ax.annotate("N", xy=(cxm + half * 0.92, cym + half * 0.90), xytext=(cxm + half * 0.92, cym + half * 0.76), color="white", ha="center",
            fontsize=15, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="white", lw=2.0), zorder=7)
lo_, la_ = merc_to_lonlat(np.array([cxm - half, cxm + half]), np.array([cym - half, cym + half]))
ax.text(0.01, 0.01, f"{la_[0]:.5f} N, {lo_[0]:.5f} E", transform=ax.transAxes, fontsize=10.5, color=INK, bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
ax.text(0.99, 0.99, f"{la_[1]:.5f} N, {lo_[1]:.5f} E", transform=ax.transAxes, fontsize=10.5, color=INK, ha="right", va="top", bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
# inset: totals from the files, and the legend in the same box style
txt = (f"{n_swaths} drives  ·  {km_total:.0f} km  ·  {len(days)} survey days\n"
       f"{len(frames)} drives ({km_processed:.0f} km) processed\n"
       f"RTK fixed on {100 * share_fix:.0f} % of the processed track\n"
       f"29 radar channels across a 1.5 m strip")
ax.text(0.02, 0.42, txt, transform=ax.transAxes, fontsize=14.5, color=INK, va="top", ha="left", linespacing=1.5,
        bbox=dict(fc="white", ec="none", alpha=0.94, boxstyle="round,pad=0.7"))
ax.legend(handles=[Line2D([], [], color=AQUA, lw=5, label="RTK fixed"), Line2D([], [], color=ORANGE, lw=5, label="Not fixed"),
                   Line2D([], [], color="#c9c9c9", lw=5, label="Not processed")], loc="lower right", fontsize=14.5, frameon=True, facecolor="white",
          edgecolor="none", framealpha=0.94, borderpad=0.9, labelspacing=0.7, handlelength=2.2)
fig.text(0.01, -0.012, "Positions: real-time RTK GNSS on the vehicle (measured); the fix state is the one reported by the receiver. "
         "No pipe position is verified against utility records yet (provisional).", fontsize=10.5, color=MUTED)
out = sys.argv[1] if len(sys.argv) > 1 else str(FIGURES / "investor_coverage_savona.png")
fig.savefig(out, dpi=170, facecolor=SURF, bbox_inches="tight")
print(out, dict(n_swaths=n_swaths, km_total=round(km_total, 1), days=days, n_processed=len(frames), km_processed=round(km_processed, 1), share_fix=round(share_fix, 3)))
