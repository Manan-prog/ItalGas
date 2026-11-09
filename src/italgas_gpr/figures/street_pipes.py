"""Team figure: what the radar van finds on one stretch of street, with the echoes behind three of the finds.
Top: 135 m of Via Nizza from above (satellite, street-aligned), the radar strip under the van shaded, pipes crossing the
road found by both radar rows as blue bars (depth written next to each), pipes running along the road as green lines.
Bottom: the 200 MHz centre-channel radargram (cleaned, signed) +-4 m around three of the crossings: a pipe makes an
arch because the van sees it before and after passing over it. Usage: python -m italgas_gpr.figures.street_pipes out.png"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

from italgas_gpr.figures.basemap import street_aligned_tiles  # noqa: E402
from italgas_gpr.geometry.swath import swath_geometry  # noqa: E402
from italgas_gpr.processing.preprocess import load_cfg, preprocess_array  # noqa: E402
from italgas_gpr.crossing.score import to_s_grid  # noqa: E402
from italgas_gpr.geometry.cube import to_su  # noqa: E402
from italgas_gpr.geometry.frames import merc_to_lonlat  # noqa: E402
from italgas_gpr.io.ids_reader import Swath  # noqa: E402
from italgas_gpr.geometry.frames import su_to_map  # noqa: E402
from italgas_gpr.io.paths import CACHE, CITY, FIGURES, config_path, load_yaml  # noqa: E402

SURF, INK, MUTED, BLUE, ORANGE, AQUA = "#fcfcfb", "#1f1f1f", "#6b6b6b", "#2a78d6", "#eb6834", "#1baf7a"
DET = load_cfg(config_path("detector.yaml")); V = DET["migration"]["velocity_m_ns"]; DS = DET["grid"]["ds_m"]
BIAS = load_yaml(config_path("crossing_scan.yaml"))["along_travel_bias_m"]     # m, ASSUMED (E25)
DRIVE = 45; S0, S1 = 2150.0, 2285.0; HW = 6.0; HALF = 4.0; ZMAX = 1.6; SHOW = (217, 222, 230)      # drive index, window (own frame), inset crossings

R = json.load(open(CITY / f"scan_{DRIVE}.json")); cr = R["crossings"]; L = json.load(open(CITY / f"rline_{DRIVE}.json")); fr = np.load(CITY / f"rframe_{DRIVE}.npz")
sn, xn, yn, k = fr["s_nodes"], fr["xn"], fr["yn"], float(fr["k"])
conf = [(j, c) for j, c in enumerate(cr) if c["cls"] == "A" and c["P"] >= 0.99 and S0 <= c["s0_m"] + BIAS < S1]
lines = [l for l in L["lines"] if l["s1"] + BIAS > S0 and l["s0"] + BIAS < S1]
# radargram of the centre channel of the 200 MHz row over the window (cleaned, signed), on a 5 cm grid
f = Path(R["file"]); sw = Swath.load(str(f.parent), f.stem); g = swath_geometry(sw, DET["base"]); u_ch = np.nanmedian(g["u2"], 0); c = int(np.argmin(np.abs(u_ch)))
sc = g["s2"][:, c]; idx = np.nonzero((sc > S0 - BIAS - 30) & (sc < S1 - BIAS + 30))[0]; sl = slice(int(idx[0]), int(idx[-1]) + 1)
pp = preprocess_array(sw, 2, DET, scans=sl); dt = pp["dt_ns"]; s_grid = np.arange(S0 - BIAS - 10, S1 - BIAS + 10, DS)
Zc = to_s_grid(pp["X"][:, c], sc[sl], s_grid); nz = int(round(2 * ZMAX / V / dt)); depth = np.arange(nz) * dt * V / 2

fig = plt.figure(figsize=(16, 6.7), facecolor=SURF)
fig.text(0.01, 0.99, f"What the radar van finds on one street: {len(conf)} pipes crossing the road and {len(lines)} running along it, in {S1 - S0:.0f} m of Via Nizza, Savona",
         fontsize=15, color=INK, weight="bold", va="top")
fig.text(0.01, 0.945, "Top: the street from above; shaded = the 1.5 m radar strip under the van; blue bars = pipes crossing the road, found by both radar rows, with their depth; green = pipes running along the road.\n"
         "Bottom: the radar echo behind three of the crossings. A buried pipe makes an arch because the van sees it before and after passing over it.", fontsize=9.6, color=MUTED, va="top", linespacing=1.4)
gs = fig.add_gridspec(2, 3, left=0.04, right=0.99, top=0.845, bottom=0.12, hspace=0.25, wspace=0.12, height_ratios=[1.35, 3.0])   # strip row = axes width x 12/135 m, no distortion with aspect auto
ax = fig.add_subplot(gs[0, :]); ax.set_facecolor(SURF)
tiles, ext, hdg, _ = street_aligned_tiles(S0, S1, HW, sn, xn, yn, k, zoom=21)
ax.imshow(tiles, extent=ext, origin="upper", aspect="auto", alpha=0.92)
ss = np.linspace(S0, S1, 80); lo = np.interp(ss, fr["s"] + BIAS, fr["lo"]); hi = np.interp(ss, fr["s"] + BIAS, fr["hi"])
ax.fill_between(ss, lo, hi, color="white", alpha=0.28, lw=0)
for l in lines:
    s_, u_ = to_su(np.array(l["x"]), np.array(l["y"]), sn, xn, yn, k); ax.plot(s_, u_, color=AQUA, lw=4, solid_capstyle="round")
    ax.text(s_.mean(), u_.mean() + 0.9, f"{l['depth_m']:.2f} m deep", color=AQUA, fontsize=8.5, ha="center", fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
for q, (j, cc) in enumerate(conf):
    sm = cc["s0_m"] + BIAS; uu = np.array([cc["u0"], cc["u1"]]); ax.plot(sm + uu * np.tan(np.radians(cc["phi_deg"])), uu, color=BLUE, lw=4, solid_capstyle="round")
    top = j in SHOW or (q % 2 == 0 and not any(abs(cr[k]["s0_m"] - cc["s0_m"]) < 6 for k in SHOW))
    near = [cr[k]["s0_m"] for kk, (k, _) in enumerate(conf) if kk < q and abs(cr[k]["s0_m"] - cc["s0_m"]) < 7 and ((k in SHOW) == top or (kk % 2 == 0) == top)]
    ax.text(sm + (4.5 if near else 0), HW - 0.6 if top else -HW + 0.6, f"{cc['depth_m']:.2f} m", color=BLUE, fontsize=8.5, ha="center", va="top" if top else "bottom", fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
ax.set_xlim(S0, S1); ax.set_ylim(-HW, HW); ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
ax.plot([S0 + 2, S0 + 12], [-HW + 1.0, -HW + 1.0], color="white", lw=4, solid_capstyle="butt"); ax.text(S0 + 7, -HW + 1.4, "10 m", color="white", ha="center", fontsize=9.5, fontweight="bold")
ax.annotate("", xy=(S0 + 52, HW - 1.1), xytext=(S0 + 44, HW - 1.1), arrowprops=dict(arrowstyle="->", color="white", lw=1.6)); ax.text(S0 + 53, HW - 1.1, "direction of travel", color="white", fontsize=8.5, va="center")
xy0 = su_to_map(np.array([S0]), np.array([-HW]), sn, xn, yn, k); xy1 = su_to_map(np.array([S1]), np.array([HW]), sn, xn, yn, k)
lo_, la_ = merc_to_lonlat(np.array([xy0[0][0], xy1[0][0]]), np.array([xy0[1][0], xy1[1][0]]))
ax.text(0.995, 0.5, f"{la_[0]:.5f} N, {lo_[0]:.5f} E", transform=ax.transAxes, fontsize=7.5, color=INK, ha="right", va="center", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
ax.text(0.005, 0.5, f"{la_[1]:.5f} N, {lo_[1]:.5f} E", transform=ax.transAxes, fontsize=7.5, color=INK, va="center", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
for p, j in enumerate(SHOW):
    cc = cr[j]; b = fig.add_subplot(gs[1, p]); b.set_facecolor(SURF); sm = cc["s0_m"]
    w = (s_grid >= sm - HALF) & (s_grid <= sm + HALF); img = Zc[w][:, :nz].T; lim = np.percentile(np.abs(img), 99)
    b.imshow(img, cmap="gray", vmin=-lim, vmax=lim, aspect="auto", extent=[-HALF, HALF, depth[-1], 0], interpolation="nearest")
    b.axhline(cc["depth_m"], color=BLUE, lw=1.1, ls="--"); b.plot(0, 0.02, marker="v", color=BLUE, ms=11, clip_on=False)
    b.set_title(f"{'ABC'[p]}  pipe crossing at {cc['depth_m']:.2f} m, evidence P = {cc['P']:.2f}", loc="left", fontsize=10, color=INK)
    b.set_xlabel("metres along the street from the pipe", fontsize=9); b.set_ylabel("depth (m)" if p == 0 else "", fontsize=9); b.tick_params(labelsize=8.5, colors=MUTED)
    for sp in ("top", "right"):
        b.spines[sp].set_visible(False)
    fig.add_artist(ConnectionPatch(xyA=(0, 0), coordsA=b.transData, xyB=(sm + BIAS, HW - 1.3), coordsB=ax.transData, color=BLUE, lw=0.9, ls=":", alpha=0.9))
fig.text(0.01, 0.004, "Drive 2025-10-13 10:01 Swath046, IDS Stream UP (200 and 600 MHz rows). Depths at 0.10 m/ns (derived from the data). Positions from the real-time GPS track, moved 1.19 m forward (assumed),\n"
         "uncertain by 0.3 to 1 m. No utility map has been compared yet: every pipe here is provisional.", fontsize=8.3, color=MUTED, linespacing=1.4)
out = sys.argv[1] if len(sys.argv) > 1 else str(FIGURES / "team_street_pipes.png")
fig.savefig(out, dpi=170, facecolor=SURF, bbox_inches="tight"); print(out, len(conf), "crossings,", len(lines), "lines")
