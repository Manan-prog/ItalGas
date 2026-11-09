"""Why green raster cells carry no along-street line: one Via Nizza stretch (the screenshot of 4 Nov 2026, lat 44.297636 N)
seen by six drives. A: the page's raster (max over drives and depths) with every saved track (shown and hidden) in the
drive-37 frame; B: the same cells drive by drive; C: depth of the strongest cell per 0.5 m window where P >= 0.5.
Run from figures/: python plot_green_vs_lines.py -> savona_green_vs_lines_nizza.png"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "pipeline")); sys.path.insert(0, str(HERE))
from basemap_tools import street_aligned_tiles  # noqa: E402
from build_cube import swath_geometry  # noqa: E402
from build_v2 import load_cfg  # noqa: E402
from ids_reader import Swath  # noqa: E402
from mosaic_test import pass_cells  # noqa: E402

from paths import CACHE, RAW  # noqa: E402
DRIVES = [37, 46, 40]; REF = 37; LAT = 44.297636; BIAS = 1.19   # drives on this carriageway (36, 39, 45 run the other way, 4 m across); BIAS m ASSUMED (E25)
DS, DU = 0.5, 0.04; HALF_W = 2.5; BEFORE, AFTER = 35.0, 25.0
sys.path.insert(0, str(HERE / "web")); from web_common import RG_STOPS  # noqa: E402  (the page's red -> neutral -> green ramp)
RG = LinearSegmentedColormap.from_list("rg", [(float(s), (r / 255, g_ / 255, b / 255)) for s, r, g_, b, _ in RG_STOPS])
NOTE = {40: "no raster for the last 44 m of drive 40 (piece shorter than 60 m is skipped)"}
COL = {36: "#2a78d6", 37: "#eb6834", 39: "#1baf7a", 40: "#7b4fbf", 45: "#b5651d", 46: "#2b2b2b"}


def project(x, y, s_nodes, xn, yn, k):
    """Map (EPSG:3857) -> (s, u) in the reference frame by nearest dense node; u (m) positive to the left of travel."""
    sd = np.arange(s_nodes[0], s_nodes[-1], 0.1); xd = np.interp(sd, s_nodes, xn); yd = np.interp(sd, s_nodes, yn)
    tx, ty = np.gradient(xd), np.gradient(yd); n = np.hypot(tx, ty); tx, ty = tx / n, ty / n
    _, j = cKDTree(np.c_[xd, yd]).query(np.c_[x, y]); dx, dy = x - xd[j], y - yd[j]
    return sd[j] + (tx[j] * dx + ty[j] * dy) * k, (tx[j] * dy - ty[j] * dx) * k


def main() -> None:
    det = load_cfg(str(ROOT / "pipeline" / "configs" / "dev_swath_v2_2.yaml"))
    files = sorted(Path(RAW).glob("*/Survey_*/Swath*.path")); f = files[REF]
    sw = Swath.load(str(f.parent), f.stem); g = swath_geometry(sw, det["base"]); frame = {k: g[k] for k in ("s_nodes", "xn", "yn", "k")}; k = float(g["k"])
    # screenshot point: the drive-37 real-time track point closest to LAT, projected into the frame
    t = np.load(CACHE / "swaths.npz")[f"s{REF}"]; y_lat = np.log(np.tan(np.pi / 4 + np.radians(LAT) / 2)) * 6378137; p = t[np.abs(t[:, 1] - y_lat).argmin()]
    s_ref, _ = project(np.array([p[0]]), np.array([p[1]]), g["s_nodes"], g["xn"], g["yn"], k); s_ref = float(s_ref[0])
    s0, s1 = s_ref - BEFORE, s_ref + AFTER; s_grid = np.arange(s0, s1, DS); u_grid = np.arange(-HALF_W, HALF_W + 1e-9, DU)
    cells, tracks = {}, {}
    for i in DRIVES:
        c = pass_cells(i, det, frame, dict(A=0.0, B=0.0), CACHE)
        if c is None:
            continue
        S, U, P, Z, _ = c; m = (S >= s0) & (S < s1) & (np.abs(U) <= HALF_W); cells[i] = (S[m], U[m], P[m], Z[m])
        J = json.load(open(CACHE / "city" / f"along_c_{i}.json")); tr = []
        for tt in J["tracks"]:
            ps, pu = project(np.array(tt["x"]), np.array(tt["y"]), g["s_nodes"], g["xn"], g["yn"], k)
            if ps.max() >= s0 and ps.min() <= s1:
                tr.append(dict(s=ps, u=pu, q=tt["q"], L=tt["length_m"], z=tt["depth_m"], shown=tt["q"] <= 0.05 and tt["length_m"] >= 10))
        tracks[i] = tr

    def grid(S, U, P):
        R = np.zeros((len(s_grid), len(u_grid)), np.float32); ii = np.round((S - s0) / DS).astype(int); jj = np.round((U + HALF_W) / DU).astype(int)
        ok = (ii >= 0) & (ii < len(s_grid)) & (jj >= 0) & (jj < len(u_grid)); np.fmax.at(R, (ii[ok], jj[ok]), P[ok]); return maximum_filter(R, size=(1, 3))

    union = np.max([grid(*cells[i][:3]) for i in cells], 0)
    img, ext, _, _ = street_aligned_tiles(s0, s1, HALF_W + 1.0, g["s_nodes"], g["xn"], g["yn"], k, zoom=21)
    fig = plt.figure(figsize=(14, 10.5), facecolor="#fcfcfb"); gs = fig.add_gridspec(2 + len(DRIVES), 1, height_ratios=[2.6, 0.25] + [1.0] * len(DRIVES), hspace=0.35)
    # ---- A
    ax = fig.add_subplot(gs[0]); ax.imshow(img, extent=ext, origin="upper", alpha=0.75, aspect="auto")
    mk = np.ma.masked_less(union, 0.02); ax.imshow(mk.T, extent=[s0, s1, -HALF_W, HALF_W], origin="lower", cmap=RG, vmin=0, vmax=1, aspect="auto", alpha=0.9, interpolation="nearest")
    for i in DRIVES:
        for tr in tracks.get(i, []):
            ax.plot(tr["s"], tr["u"], color="#1baf7a" if tr["shown"] else "white", lw=2.2 if tr["shown"] else 1.4, ls="-" if tr["shown"] else "--", alpha=0.95)
    ax.axvline(s_ref, color="white", lw=1, ls=":"); ax.text(s_ref, HALF_W - 0.15, "screenshot latitude", color="white", fontsize=8, ha="center", va="top")
    ax.set_xlim(s0, s1); ax.set_ylim(-HALF_W, HALF_W); ax.set_ylabel("across (m, + left)"); ax.set_xlabel(f"along drive {REF} (m)")
    ax.set_title("A  page raster: max over the drives of this carriageway and all depths; solid = track drawn on the page (q <= 0.05, >= 10 m), dashed = saved track hidden by those rules", fontsize=9.5, loc="left")
    cax = fig.add_subplot(gs[1]); cb = plt.colorbar(plt.cm.ScalarMappable(cmap=RG, norm=plt.Normalize(0, 1)), cax=cax, orientation="horizontal")
    cb.set_label("P: share of cells with this much evidence that pipe-free (channel-shuffled) data does not produce", fontsize=8.5); cax.tick_params(labelsize=8)
    # ---- B
    for r, i in enumerate(DRIVES):
        ax = fig.add_subplot(gs[2 + r]); R = grid(*cells[i][:3]) if i in cells else np.zeros((len(s_grid), len(u_grid)))
        ax.imshow(np.ma.masked_less(R, 0.02).T, extent=[s0, s1, -HALF_W, HALF_W], origin="lower", cmap=RG, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_facecolor("#e9e9e6")
        for tr in tracks.get(i, []):
            ax.plot(tr["s"], tr["u"], color="#1baf7a" if tr["shown"] else "#333", lw=2 if tr["shown"] else 1.2, ls="-" if tr["shown"] else "--")
            inside = (tr["s"] >= s0) & (tr["s"] <= s1); sx = float(np.clip(tr["s"][inside].mean(), s0 + 7, s1 - 7)) if inside.any() else s0 + 7
            ax.text(sx, float(tr["u"][inside].mean()) - 0.55 if inside.any() else 0.0, f"q={tr['q']:.2f}, {tr['L']:.0f} m, depth {tr['z']:.2f} m", fontsize=7.5, ha="center", color="#111")
        n50 = int((cells[i][2] >= 0.5).sum()) if i in cells else 0
        ax.set_ylim(-HALF_W, HALF_W); ax.set_xlim(s0, s1); ax.set_ylabel(f"drive {i}", fontsize=9, color=COL[i]); ax.tick_params(labelsize=7.5)
        ax.text(s0 + 0.5, HALF_W - 0.1, f"B{r + 1}  cells P >= 0.5 in this window: {n50}; saved tracks here: {len(tracks.get(i, []))}" + (f"; {NOTE[i]}" if i in NOTE else ""), fontsize=8, va="top")
        if r < len(DRIVES) - 1:
            ax.set_xticklabels([])
    fig.axes[-1].set_xlabel(f"along drive {REF} (m); B: one drive per row, same cells as in A before taking the max over drives")
    fig.suptitle(f"Via Nizza around {LAT:.6f} N: why the green cells have no line over them (13 Oct 2025)", fontsize=12, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.972, "A: the page's raster is the max of the drives in B (36, 39 and 45 drove the other carriageway, 4 m across, and do not touch this strip).\n"
                          "B: no drive shows a continuous green run past 2765 m; each has short patches at different offsets and depths, and the tracks found here have q 0.06-0.19.\n"
                          "Track rules: 0.5 m hops within 0.12 m sideways and 2 ns (0.10 m) in depth, >= 3 m, then drawn only if q <= 0.05 and >= 10 m (q: share of tracks this strong that shuffled data gives).",
             fontsize=8.5, va="top")
    # ---- C as an inset-like extra axes at the bottom right of A? keep as separate small panel under B
    out = HERE / "savona_green_vs_lines_nizza.png"; fig.savefig(out, dpi=150, bbox_inches="tight"); print("saved", out)
    # depth scatter (C) as its own figure appended below would crowd; print the numbers instead
    for i in cells:
        S, U, P, Z = cells[i]; m = P >= 0.5
        print(f"drive {i}: cells {len(P)}, P>=0.5 {m.sum()}, depth of those: median {np.median(Z[m]) if m.any() else np.nan:.2f} m, 10-90 % {np.quantile(Z[m], [0.1, 0.9]) if m.any() else '-'}, u 10-90 % {np.quantile(U[m], [0.1, 0.9]) if m.any() else '-'}")


if __name__ == "__main__":
    main()
