"""Figure: Via Nizza subset, three map layers on satellite imagery (north up):
(1) grey = where each drive's radar strip ran (registered);
(2) raster = probability that a pipe passes under each spot, combined over drives (highest of the drives
    covering the spot; crossings spread by 0.3 m registration residual, along-road candidates P 0.68);
(3) vectors = final pipelines:
    crossing pipes = crossing detections from different drives that agree (|dS0| <= 0.5 m, |dphi| <= 15 deg)
    merged into one line drawn across all the lanes that saw it (thick = confident: both radar rows of a drive or >= 3 drives,
    E33; thin = probable, P >= 0.9);
    along-road pipes = places where two drives independently show the same pipe arch (apex within 0.2 m).
Usage: python -m italgas_gpr.figures.crossing_map out.png"""
import json
import sys
from pathlib import Path

import contextily as cx
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from italgas_gpr.figures.basemap import GSAT  # noqa: E402
from italgas_gpr.figures.map_layers import CACHE, GREY, PCMAP, decorate, is_confident, merge_drives, multipass_drives, prob_on_cells  # noqa: E402
from italgas_gpr.geometry.frames import su_to_map  # noqa: E402

SURF = "#fcfcfb"; INK = "#1f1f1f"; MUTED = "#6b6b6b"; PIPE = "#101010"; ALONG = "#2a78d6"
REG = json.load(open(CACHE / "mp_registration.json"))["registration"]; fr = dict(np.load(CACHE / "mp_frame.npz")); k = float(fr["k"])
names = list(REG)
AP = json.load(open(CACHE / "mp_apex.json"))["features"]
P_ALONG = 0.68

drives = multipass_drives(REG)                       # crossings in TRUE coordinates, footprint edges along s


def coverage_at(n, S):
    d = drives[n]; return np.interp(S, d["s"], d["lo"], left=np.nan, right=np.nan), np.interp(S, d["s"], d["hi"], left=np.nan, right=np.nan)


# ---- crossing pipelines: merge agreeing detections across drives (map_layers.merge_drives; rule E33)
pipes = []
for g in merge_drives(drives, 0.5):
    us = [float(v[0]) for n in g["members"] for v in coverage_at(n, np.array([g["S0"]])) if np.isfinite(v[0])]
    if us:
        pipes.append(dict(g, u0=min(us), u1=max(us), confident=is_confident(g)))
strong = [p for p in pipes if p["confident"]]; weak = [p for p in pipes if not p["confident"] and p["P"] >= 0.9]
along = [f for f in AP if f["agree"]]
print(f"crossing pipelines: {len(strong)} confident (both rows or >= 3 drives within 0.5 m), {len(weak)} probable (P >= 0.9); along-road pipelines: {len(along)}")
json.dump(dict(crossing_pipes=pipes, along_road=along), open(CACHE / "vianizza_pipelines.json", "w"))

# ---- combined probability raster on a common true (s, u) grid
def raster(S0, S1, ds=0.1, du=0.08):
    sg = np.arange(S0, S1, ds); ug = np.arange(-3.5, 6.5, du)
    Sc, Uc = np.meshgrid(sg + ds / 2, ug + du / 2, indexing="ij"); best = np.full(Sc.shape, np.nan); cov = np.zeros(Sc.shape, bool)
    for n in names:
        lo, hi = coverage_at(n, Sc[:, 0]); inside = (Uc >= lo[:, None]) & (Uc <= hi[:, None])
        if not inside.any():
            continue
        X = [dict(s0=o["S0"], phi=o["phi"], P=o["P"]) for o in drives[n]["X"]]
        al = [dict(s0=f["s0"], s1=f["s1"], uc=float(np.mean([q for q in f["uc"] if q is not None])), P=P_ALONG) for f in along]
        P = prob_on_cells(Sc, Uc, X, al)
        best = np.where(inside, np.fmax(best, P), best); cov |= inside
    Sm, Um = np.meshgrid(np.r_[sg, sg[-1] + ds], np.r_[ug, ug[-1] + du], indexing="ij")
    mx, my = su_to_map(Sm.ravel(), Um.ravel(), fr["s_nodes"], fr["xn"], fr["yn"], k)
    return mx.reshape(Sm.shape), my.reshape(Sm.shape), best, cov


def line_map(S, U):
    return su_to_map(np.asarray(S, float), np.asarray(U, float), fr["s_nodes"], fr["xn"], fr["yn"], k)


fig = plt.figure(figsize=(19, 11), facecolor=SURF)
fig.suptitle("Via Nizza: where the drives went, the probability of a pipe, and the resulting pipelines (radar only, provisional)",
             x=0.02, y=0.988, ha="left", fontsize=14.5, color=INK, fontweight="bold")
fig.text(0.02, 0.952, "Grey = ground covered by the 6 drives.  Colour = probability that a pipe passes under the spot (highest over the drives).\n"
         "Black lines = crossing pipes (thick: confident = both radar rows, or 3 or more drives within 0.5 m; thin: probable, P >= 0.9).\nBlue = along-road candidate (same arch seen by 2 drives; unconfirmed).\n"
         "Positions can be off by about 1 m; the gap between the two groups of lanes has no data (crossing lines are drawn across it where both groups saw them).",
         fontsize=10, color=MUTED, va="top", linespacing=1.4)
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.05, left=0.02, right=0.98, top=0.845, bottom=0.11)
for ax_i, (S0, S1, zoom, ttl, bar) in enumerate(((2700.0, 3640.0, 18, "A.  The whole 940 m stretch (confident crossings only)", 100), (2850.0, 2990.0, 20, "B.  Close-up 2850-2990 m", 20))):
    ax = fig.add_subplot(gs[ax_i])
    mx, my, P, cov = raster(S0, S1, ds=0.1 if ax_i else 0.4, du=0.08 if ax_i else 0.16)
    ax.pcolormesh(mx, my, np.where(cov, 0.0, np.nan), cmap=GREY, vmin=0, vmax=1, shading="flat", zorder=3, rasterized=True)
    ax.pcolormesh(mx, my, np.ma.masked_invalid(np.where(P >= 0.03, P, np.nan)), cmap=PCMAP, vmin=0, vmax=1, shading="flat", zorder=4, rasterized=True)
    for p in (strong + weak if ax_i else strong):
        if not (S0 - 3 < p["S0"] < S1 + 3):
            continue
        uu = np.array([p["u0"], p["u1"]]); ss = p["S0"] + uu * np.tan(np.radians(p["phi"])); x, y = line_map(ss, uu)
        if ax_i:
            ax.plot(x, y, color="white", lw=(3.6 if p["confident"] else 2.0) + 1.4, zorder=5, solid_capstyle="round")
            ax.plot(x, y, color=PIPE, lw=3.6 if p["confident"] else 1.6, zorder=6, solid_capstyle="round")
        else:
            ax.plot(x, y, color=PIPE, lw=0.9, zorder=6)
    for f in along:
        if f["s1"] < S0 or f["s0"] > S1:
            continue
        uc = float(np.mean([q for q in f["uc"] if q is not None])); x, y = line_map([max(f["s0"], S0), min(f["s1"], S1)], [uc, uc])
        ax.plot(x, y, color="white", lw=6, zorder=7, solid_capstyle="round"); ax.plot(x, y, color=ALONG, lw=4, zorder=8, solid_capstyle="round")
    xs, ys = mx[np.isfinite(mx)], my[np.isfinite(my)]
    cxm, cym = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max()); half = 0.5 * max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.05
    ax.set_xlim(cxm - half, cxm + half); ax.set_ylim(cym - half, cym + half); ax.set_aspect("equal")
    cx.add_basemap(ax, source=GSAT, crs="EPSG:3857", zoom=zoom, attribution=False, alpha=0.9); decorate(ax, cxm, cym, half, k, bar)
    ax.set_title(ttl, loc="left", fontsize=11.5, color=INK)
sm = plt.cm.ScalarMappable(cmap=PCMAP, norm=plt.Normalize(0, 1))
cax = fig.add_axes([0.08, 0.06, 0.25, 0.022]); cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_label("probability that a pipe passes under this spot", fontsize=10, color=INK); cb.ax.tick_params(labelsize=9)
fig.legend(handles=[Patch(fc="#dfe6ee", ec="none", label="where the radar drives went"),
                    Line2D([], [], color=PIPE, lw=3.6, label=f"crossing pipe, confident ({len(strong)})"),
                    Line2D([], [], color=PIPE, lw=1.6, label=f"crossing pipe, probable ({len(weak)})"),
                    Line2D([], [], color=ALONG, lw=4, label=f"along-road CANDIDATE, unconfirmed ({len(along)} stretches)")],
           loc="lower right", bbox_to_anchor=(0.98, 0.015), fontsize=10, frameon=False, ncol=2)
fig.savefig(sys.argv[1], dpi=160, facecolor=SURF); print("saved", sys.argv[1])
