"""Figure: pipe probability of the radar strips of several drives, drawn where the strips lie on the satellite map.
A: north-up overview, every available piece of every listed drive. B: street-aligned ortho detail (zoom 21) along one
reference drive. Cell value = P = 1 - local false discovery rate of the along-street evidence (along/evidence.py),
max over depth, and max over drives where strips overlap (strips of different drives are NOT registered to each other:
E16 cross-track corrections were up to 0.4 m). Colour: red (P 0) -> neutral -> green (P 1), as requested.
Run: python -m italgas_gpr.figures.lane_raster <out.png> <drive,drive,...> <ref drive> <s0> <s1> [ground_grid_m]"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import contextily as cx
import matplotlib
import numpy as np
from scipy.ndimage import maximum_filter
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from italgas_gpr.figures.basemap import GSAT, street_aligned_tiles  # noqa: E402
from italgas_gpr.geometry.swath import swath_geometry  # noqa: E402
from italgas_gpr.processing.preprocess import load_cfg  # noqa: E402
from italgas_gpr.geometry.cube import to_su  # noqa: E402
from italgas_gpr.geometry.frames import merc_to_lonlat  # noqa: E402
from italgas_gpr.io.ids_reader import Swath  # noqa: E402
from italgas_gpr.geometry.frames import su_to_map  # noqa: E402

SURF, INK, MUTED = "#fcfcfb", "#1f1f1f", "#6b6b6b"; B_ALONG = 1.19   # m, ASSUMED along-travel bias (E25)
CMAP = LinearSegmentedColormap.from_list("rg", ["#c0392b", "#efe9dc", "#1a9850"])
from italgas_gpr.io.paths import CACHE, RAW, config_path  # noqa: E402


def drive_cells(i: int, det: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """All P cells of drive i from its piece files: map x, y (EPSG:3857) and P (max over depth); plus the drive frame."""
    files = sorted(glob.glob(f"{RAW}/*/Survey_*/Swath*.path")); f = files[i]
    sw = Swath.load(str(Path(f).parent), Path(f).stem); g = swath_geometry(sw, det["base"])
    X, Y, P = [], [], []
    for pf in sorted(glob.glob(str(CACHE / f"prob_raster_{i}_*.npz"))):
        d = np.load(pf); t = d["t_ns"]; it = (t >= d["t_search"][0]) & (t <= d["t_search"][1])
        plan = d["P"].astype(float)[:, :, it].max(2)                              # (n_win, n_u)
        plan = maximum_filter(plan, size=(1, 3))                                    # display: widen each 4 cm band to the detector's 12 cm peak separation
        cs, ug = d["centres_s"], d["ug"]; keep = (cs >= d["c0"]) & (cs < d["c1"])
        for off in np.arange(-0.2, 0.21, 0.1):                                       # fill each 0.5 m window along s
            S, U = np.meshgrid(cs[keep] + off, ug, indexing="ij")
            x, y = su_to_map(S.ravel() + B_ALONG, U.ravel(), g["s_nodes"], g["xn"], g["yn"], float(g["k"]))
            X.append(x); Y.append(y); P.append(plan[keep].ravel())
    if not X:
        return np.zeros(0), np.zeros(0), np.zeros(0), g
    return np.concatenate(X), np.concatenate(Y), np.concatenate(P), g


def rasterize(x, y, p, x0, y0, cell, nx, ny):
    """Max of p per cell of a regular grid (origin x0, y0; cell size in 3857 units); nan where empty."""
    ix = ((x - x0) / cell).astype(int); iy = ((y - y0) / cell).astype(int); ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    R = np.full((ny, nx), np.nan); np.fmax.at(R, (iy[ok], ix[ok]), p[ok]); return R


def main(out: str, drives: list[int], ref: int, s0: float, s1: float, cell_m: float = 0.1) -> None:
    det = load_cfg(config_path("detector.yaml"))
    cells = {i: drive_cells(i, det) for i in drives}; k = float(cells[ref][3]["k"])
    allx = np.concatenate([c[0] for c in cells.values()]); ally = np.concatenate([c[1] for c in cells.values()]); allp = np.concatenate([c[2] for c in cells.values()])
    print({i: len(c[2]) for i, c in cells.items()}, "cells; P>=0.5:", round(float(np.mean(allp >= 0.5)), 4))
    fig = plt.figure(figsize=(14, 9.5), facecolor=SURF)
    fig.suptitle("Where do the radar strips show along-street pipe evidence beyond chance?  Via Nizza, drives " + ", ".join(map(str, drives)), x=0.01, ha="left", fontsize=13, color=INK, weight="bold")
    fig.text(0.01, 0.94, "A: north-up overview, all pieces of the listed drives (P = 1 - local false discovery rate of the along-street evidence, max over depth; where strips overlap, the max over drives).\n"
             "B: ortho detail along the reference drive, satellite zoom 21, same values. Green = evidence beyond chance, red = none. Strips are placed by each drive's own RTK track (not registered to each other).",
             fontsize=8.5, color=MUTED, va="top")
    # A: overview
    ax = fig.add_axes([0.04, 0.40, 0.92, 0.47]); cell = cell_m / k
    x0, x1, y0, y1 = allx.min() - 40 / k, allx.max() + 40 / k, ally.min() - 40 / k, ally.max() + 40 / k
    from matplotlib.collections import LineCollection
    for i, (cx_, cy_, cp_, gi) in cells.items():                                  # along-track max of P per 1 m, as a coloured line
        if not len(cp_):
            continue
        si, _ = to_su(cx_, cy_, gi["s_nodes"], gi["xn"], gi["yn"], float(gi["k"])); b = np.round(si).astype(int); b -= b.min()
        pm = np.full(b.max() + 1, np.nan); np.fmax.at(pm, b, cp_); sb = np.arange(len(pm)) + np.round(si).min()
        lx, ly = su_to_map(sb + B_ALONG, np.zeros_like(sb), gi["s_nodes"], gi["xn"], gi["yn"], float(gi["k"]))
        seg = np.stack([np.c_[lx[:-1], ly[:-1]], np.c_[lx[1:], ly[1:]]], 1); ok = np.isfinite(pm[:-1])
        lc = LineCollection(seg[ok], cmap=CMAP, norm=plt.Normalize(0, 1), linewidths=3.0, zorder=4, alpha=0.9); lc.set_array(pm[:-1][ok]); ax.add_collection(lc)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
    cx.add_basemap(ax, source=GSAT, crs="EPSG:3857", zoom=19, attribution=False, alpha=0.75)
    g = cells[ref][3]; bx, by = su_to_map(np.array([s0, s0, s1, s1, s0]), np.array([-8.0, 8.0, 8.0, -8.0, -8.0]), g["s_nodes"], g["xn"], g["yn"], k)
    ax.plot(bx, by, color="white", lw=1.5, zorder=6); ax.text(bx[1], by[1], "  B", color="white", fontsize=11, fontweight="bold", zorder=6)
    L = 100.0; xb, yb = x0 + (x1 - x0) * 0.03, y0 + (y1 - y0) * 0.06
    ax.plot([xb, xb + L / k], [yb, yb], color="white", lw=4, zorder=6, solid_capstyle="butt"); ax.text(xb + L / k / 2, yb + (y1 - y0) * 0.02, f"{L:.0f} m", color="white", ha="center", fontsize=10, fontweight="bold", zorder=6)
    lo_, la_ = merc_to_lonlat(np.array([x0, x1]), np.array([y0, y1]))
    ax.text(0.01, 0.01, f"{la_[0]:.5f} N, {lo_[0]:.5f} E", transform=ax.transAxes, fontsize=8, color=INK, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    ax.text(0.99, 0.99, f"{la_[1]:.5f} N, {lo_[1]:.5f} E", transform=ax.transAxes, fontsize=8, color=INK, ha="right", va="top", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title("A  Overview (satellite at 75 %): each drive drawn as a 1 m line carrying the max of P across its 1.4 m strip", loc="left", fontsize=11, color=INK)
    # B: street-aligned detail along the reference drive
    half_w = 3.0; img, ext, hdg, _ = street_aligned_tiles(s0, s1, half_w, g["s_nodes"], g["xn"], g["yn"], k, zoom=21)
    ax = fig.add_axes([0.04, 0.13, 0.92, 0.24]); ax.imshow(img, extent=ext, origin="upper", aspect="equal", zorder=1)
    su = {i: to_su(c[0], c[1], g["s_nodes"], g["xn"], g["yn"], k) for i, c in cells.items() if len(c[2])}
    ss = np.concatenate([su[i][0] for i in su]); uu = np.concatenate([su[i][1] for i in su]); pp = np.concatenate([cells[i][2] for i in su])
    m = (ss >= s0) & (ss <= s1) & (np.abs(uu) <= half_w)
    nxb, nyb = int((s1 - s0) / cell_m) + 1, int(2 * half_w / cell_m) + 1
    Rb = rasterize(ss[m], uu[m], pp[m], s0, -half_w, cell_m, nxb, nyb)
    ax.imshow(np.ma.masked_invalid(Rb), extent=[s0, s0 + nxb * cell_m, -half_w, -half_w + nyb * cell_m], origin="lower", cmap=CMAP, vmin=0, vmax=1, alpha=0.85, interpolation="nearest", zorder=3)
    for i in drives:                                                              # significant E39 tracks of each drive, for reference
        f = CACHE / "city" / f"along_c_{i}.json"
        if not f.exists():
            continue
        for t in json.load(open(f))["tracks"]:
            if t["q"] <= 0.05 and t["length_m"] >= 10:
                ts, tu = to_su(np.array(t["x"]), np.array(t["y"]), g["s_nodes"], g["xn"], g["yn"], k); w = (ts >= s0) & (ts <= s1)
                if w.sum() > 1:
                    ax.plot(ts[w], tu[w], color="white", lw=1.0, zorder=5)
    for i in drives:                                                              # centre line of each drive's strip, labelled
        gi = cells[i][3]; S_i = np.nanmean(gi["s2"], 1); U_i = np.nanmean(gi["u2"], 1); ok = np.isfinite(S_i) & np.isfinite(U_i)
        xi, yi = su_to_map(S_i[ok] + B_ALONG, U_i[ok], gi["s_nodes"], gi["xn"], gi["yn"], float(gi["k"])); si, ui = to_su(xi, yi, g["s_nodes"], g["xn"], g["yn"], k)
        w = (si >= s0) & (si <= s1) & (np.abs(ui) < half_w)
        if w.sum() > 10:
            o = np.argsort(si[w]); ax.plot(si[w][o], ui[w][o], color="white", lw=0.6, ls=(0, (2, 2)), zorder=6)
            ax.text(s0 + 0.5, float(np.median(ui[w])), f"drive {i}", color="white", fontsize=7, va="center", zorder=7, bbox=dict(fc="black", ec="none", alpha=0.35, pad=1))
    ax.set_xlim(s0, s1); ax.set_ylim(-half_w, half_w); ax.set_aspect("equal")
    ax.set_xlabel(f"metres along drive {ref}", fontsize=9); ax.set_ylabel("across (m, left +)", fontsize=9); ax.tick_params(labelsize=8)
    ax.plot([s0 + 3, s0 + 13], [-half_w + 0.8, -half_w + 0.8], color="white", lw=4, zorder=6, solid_capstyle="butt"); ax.text(s0 + 8, -half_w + 1.3, "10 m", color="white", ha="center", fontsize=9, fontweight="bold", zorder=6)
    ax.set_title(f"B  Ortho view along drive {ref}, s {s0:.0f}-{s1:.0f} m (white solid: significant single-drive tracks, E39; dotted: each drive's strip centre)", loc="left", fontsize=11, color=INK)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1)); cax = fig.add_axes([0.3, 0.075, 0.4, 0.016])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal"); cb.set_label("P = 1 - local false discovery rate of the along-street evidence (green: beyond chance; not 'certainly a pipe')", fontsize=9, color=INK); cb.ax.tick_params(labelsize=8)
    fig.text(0.01, 0.004, "Evidence: 200 MHz, option C cleaning, 2 m stack, across-strip focusing, spot-minus-line contrast; null = 8 antenna-order shuffles per 600 m piece; lfdr with pi0 = 1, monotone; for display each across-strip cell takes the max of its 12 cm neighbourhood.\n"
             "Along-travel bias 1.19 m ASSUMED. Red/green as requested: not colour-blind safe; a neutral midpoint separates the two.", fontsize=7.2, color=MUTED, va="bottom")
    fig.savefig(out, dpi=160, facecolor=SURF); print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1], [int(v) for v in sys.argv[2].split(",")], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]) if len(sys.argv) > 6 else 0.1)
