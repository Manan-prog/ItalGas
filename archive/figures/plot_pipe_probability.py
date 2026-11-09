"""Figure: where are buried utilities under the 14 Oct 11:31 swath, and how sure are we?
Panels: A overview map with per-100 m maximum probability; B detail map (zoom 21) with
the plan-view probability raster and extracted segments; C the swath-local probability
strip (s, u) for both frequencies; D calibration curves. Usage:
python plot_pipe_probability.py det_dev.npz cube.npz out.png s_detail_start s_detail_end"""
import json
import sys

import contextily as cx
import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, "../pipeline")
from geometry import merc_scale, merc_to_lonlat
from run_detection import su_to_map

SURF = "#fcfcfb"; BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"; INK = "#1f1f1f"; MUTED = "#6b6b6b"
GSAT = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
det_file, cube_file, out_png, s0, s1 = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
d = np.load(det_file); z = np.load(cube_file)
sg, ug = d["sg"], d["ug"]; DS, DU = 0.10, 0.08
s_nodes, xn, yn, k = z["s_nodes"], z["xn"], z["yn"], float(z["k"])
feats = json.load(open(det_file.replace(".npz", ".geojson")))["features"]

# sequential colormap: transparent -> orange (single hue, light->dark), used for probability
cmap = LinearSegmentedColormap.from_list("p", [(0, (1, 1, 1, 0)), (0.5, to_rgba("#f7b267", 0.55)), (1.0, to_rgba("#b4321c", 0.95))])

fig = plt.figure(figsize=(18, 19), facecolor=SURF)
gs = fig.add_gridspec(4, 2, height_ratios=[1.3, 0.75, 0.8, 0.6], width_ratios=[1, 1], hspace=0.32, wspace=0.12)
axA = fig.add_subplot(gs[0, 0]); axD = fig.add_subplot(gs[0, 1]); axB = fig.add_subplot(gs[1, :]); axC = fig.add_subplot(gs[2, :]); axE = fig.add_subplot(gs[3, :])
for ax in (axA, axB, axC, axD, axE):
    ax.set_facecolor(SURF)
    for sp in ax.spines.values(): sp.set_color("#cfcfcf")
    ax.tick_params(colors=MUTED, labelsize=8)

# ---- A: overview, track coloured by max probability per 10 m (both frequencies agree map)
pb = d["prob_both"]; pany = d["prob_any"]
seg_len = 10.0; nseg = int(np.ceil(sg[-1] / seg_len))
pmax = np.array([(pany[int(i * seg_len / DS): int((i + 1) * seg_len / DS)] > 0.9).mean() if int(i * seg_len / DS) < len(sg) else 0 for i in range(nseg)])
xs = np.interp(np.arange(nseg) * seg_len + seg_len / 2, s_nodes, xn); ys = np.interp(np.arange(nseg) * seg_len + seg_len / 2, s_nodes, yn)
axA.plot(xn, yn, color="#ffffff", lw=4, alpha=0.9, zorder=2)
sc = axA.scatter(xs, ys, c=pmax, cmap="YlOrRd", vmin=0, vmax=0.15, s=28, zorder=3, edgecolors="none")
pad = 120 / k
axA.set_xlim(xn.min() - pad, xn.max() + pad); axA.set_ylim(yn.min() - pad, yn.max() + pad)
cx.add_basemap(axA, source=GSAT, crs="EPSG:3857", zoom=17, attribution=False, alpha=0.75)
cb = fig.colorbar(sc, ax=axA, fraction=0.035, pad=0.02); cb.set_label("fraction of footprint with P > 0.9 per 10 m (either frequency)", color=MUTED, fontsize=8); cb.ax.tick_params(colors=MUTED, labelsize=8)
# detail box
xa, ya = su_to_map(np.array([s0, s1]), np.array([0, 0]), s_nodes, xn, yn, k)
axA.plot(xa, ya, color=BLUE, lw=3, zorder=4); axA.annotate("panel B", (xa[0], ya[0]), color=BLUE, fontsize=9, fontweight="bold", xytext=(8, 8), textcoords="offset points")
# scale bar 200 m
sb = 200 / k; x0b = xn.min() - pad + 30 / k; y0b = yn.min() - pad + 30 / k
axA.plot([x0b, x0b + sb], [y0b, y0b], color="white", lw=3); axA.text(x0b + sb / 2, y0b + 12 / k, "200 m", color="white", ha="center", fontsize=8)
lon, lat = merc_to_lonlat(np.array([xn.min() - pad, xn.max() + pad]), np.array([yn.min() - pad, yn.max() + pad]))
axA.text(0.01, 0.01, f"SW {lat[0]:.5f}N {lon[0]:.5f}E", transform=axA.transAxes, color="white", fontsize=7, va="bottom")
axA.text(0.99, 0.99, f"NE {lat[1]:.5f}N {lon[1]:.5f}E", transform=axA.transAxes, color="white", fontsize=7, va="top", ha="right")
axA.set_xticks([]); axA.set_yticks([])
axA.set_title("A  Swath 14 Oct 11:31, Corso A. Ricci (1.2 km): density of confident linear reflectors", fontsize=10, loc="left", color=INK)

# ---- B: street-aligned detail: basemap tiles rotated so the track runs left->right
from contextily import bounds2img
from scipy.ndimage import rotate as nd_rotate
i0, i1 = int(s0 / DS), int(s1 / DS)
sc_mid = 0.5 * (s0 + s1)
xc, yc = su_to_map(np.array([sc_mid]), np.array([0.0]), s_nodes, xn, yn, k); xc, yc = float(xc[0]), float(yc[0])
tx = np.interp(sc_mid, s_nodes, np.gradient(xn)); ty = np.interp(sc_mid, s_nodes, np.gradient(yn)); hdg = np.arctan2(ty, tx)
half_len = (s1 - s0) / 2 + 4; half_w = 7.0
R = (half_len + half_w) / k
img, ext = bounds2img(xc - R, yc - R, xc + R, yc + R, zoom=21, source=GSAT, ll=False)
# pixel size (3857 units) of the tile mosaic; rotate about the image centre, then crop
px = (ext[1] - ext[0]) / img.shape[1]
rot = nd_rotate(img, -np.degrees(hdg), reshape=False, order=1, mode="constant", cval=0)   # sign verified on a synthetic line (E08)
cx_px = (xc - ext[0]) / px; cy_px = (ext[3] - yc) / px          # image centre of our point (row from top)
# after rotation about the array centre, our point moved: compute its new pixel position
cxa, cya = (img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2
th = np.radians(np.degrees(hdg))
dxp, dyp = cx_px - cxa, cy_px - cya
# scipy rotate (reshape=False) rotates the image content counter-clockwise by angle in array coords
rx = cxa + dxp * np.cos(th) - dyp * np.sin(th); ry = cya + dxp * np.sin(th) + dyp * np.cos(th)
# ground metres per pixel = px * k
mpp = px * k
a0 = int(rx - half_len / mpp); a1 = int(rx + half_len / mpp); b0 = int(ry - half_w / mpp); b1 = int(ry + half_w / mpp)
crop = rot[b0:b1, a0:a1]
# NOTE: in the rotated image, +x pixel = along track (increasing s), image rows increase downward = to the RIGHT of travel (u negative)
axB.imshow(crop, extent=[sc_mid - half_len, sc_mid + half_len, -half_w, half_w], origin="upper", zorder=1, interpolation="bilinear")
axB.imshow(pany[i0:i1].T, extent=[s0, s1, ug[0] - DU / 2, ug[-1] + DU / 2], origin="lower", cmap=cmap, vmin=0.5, vmax=1, zorder=3, aspect="auto", interpolation="nearest")
axB.axhline(ug[0] - DU / 2, color="white", lw=0.6, alpha=0.6, zorder=4); axB.axhline(ug[-1] + DU / 2, color="white", lw=0.6, alpha=0.6, zorder=4)
for f in feats:
    pr = f["properties"]; sa, sb_ = pr["s_m"]; ua, ub = pr["u_m"]
    if sb_ < s0 or sa > s1: continue
    col = {"600": BLUE, "200": AQUA, "both": ORANGE}[pr["source"]]
    axB.plot([sa, sb_], [ua, ub], color=col, lw=2.4 if pr["source"] == "both" else 1.6, zorder=5, solid_capstyle="round")
# public records: sewer collector (Regione Liguria 2023) if available
import os
if os.path.exists("/Users/manansarupria/ItalGas/cache/collector_dev.npz"):
    cc = np.load("/Users/manansarupria/ItalGas/cache/collector_dev.npz")
    axB.plot(cc["s"], cc["u"], color="white", lw=1.2, ls=(0, (4, 3)), zorder=6, label="public sewer collector record")
# repeat passes (13 Oct) segments, mapped into this swath's (s,u)
sys.path.insert(0, "../pipeline")
from cube import to_su
for tag, mk in (("rep029", "D"), ("rep031", "D")):
    gj = f"/Users/manansarupria/ItalGas/cache/det_{tag}.geojson"
    if not os.path.exists(gj): continue
    for f in json.load(open(gj))["features"]:
        if f["properties"]["source"] == "both": continue
        (lo0, la0), (lo1, la1) = f["geometry"]["coordinates"]
        x_ = np.radians(np.array([lo0, lo1])) * 6378137.0; y_ = 6378137.0 * np.log(np.tan(np.pi / 4 + np.radians(np.array([la0, la1])) / 2))
        s_, u_ = to_su(x_, y_, s_nodes, xn, yn, k)
        if s_.max() < s0 or s_.min() > s1 or np.abs(u_).max() > half_w: continue
        axB.plot(s_, u_, color="#ffd166", lw=1.4, zorder=5, alpha=0.95)
axB.set_xlim(sc_mid - half_len, sc_mid + half_len); axB.set_ylim(-half_w, half_w); axB.set_aspect("equal")
axB.set_xlabel("along-track s (m)", color=MUTED, fontsize=9); axB.set_ylabel("u (m, left +)", color=MUTED, fontsize=9)
lon, lat = merc_to_lonlat(np.array([xc]), np.array([yc]))
axB.text(0.01, 0.02, f"centre {lat[0]:.5f}N {lon[0]:.5f}E, heading {np.degrees(np.pi/2-hdg)%360:.0f} deg", transform=axB.transAxes, color="white", fontsize=7, va="bottom")
axB.plot([s0, s0 + 10], [-half_w + 0.8] * 2, color="white", lw=3, zorder=7); axB.text(s0 + 5, -half_w + 1.3, "10 m", color="white", ha="center", fontsize=8, zorder=7)
axB.set_title(f"B  Street-aligned detail s = {s0:.0f} to {s1:.0f} m: P(utility) raster, segments, repeat pass, public record", fontsize=10, loc="left", color=INK)
from matplotlib.lines import Line2D
axB.legend(handles=[Line2D([], [], color=BLUE, lw=1.6, label="600 MHz segment"), Line2D([], [], color=AQUA, lw=1.6, label="200 MHz segment"), Line2D([], [], color=ORANGE, lw=2.4, label="both frequencies"),
                    Line2D([], [], color="#ffd166", lw=1.4, label="13 Oct repeat pass"), Line2D([], [], color="white", lw=1.2, ls=(0, (4, 3)), label="public sewer collector (Regione Liguria 2023)")],
           loc="upper right", fontsize=7, framealpha=0.8)

# ---- C: swath-local strips, both frequencies, with segments
for row, (arr, name) in enumerate(((1, "600 MHz"), (2, "200 MHz"))):
    p = d[f"prob{arr}"][i0:i1]
    off = row * 1.7
    axC.imshow(p.T, extent=[s0, s1, ug[0] - off, ug[-1] - off], origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    axC.text(s0 + 0.3, ug[-1] - off - 0.05, name, color="white", fontsize=9, va="top", fontweight="bold")
    for f in feats:
        pr = f["properties"]
        if pr["source"] != ("600" if arr == 1 else "200"): continue
        sa, sb_ = pr["s_m"]; ua, ub = pr["u_m"]
        if sb_ < s0 or sa > s1: continue
        axC.plot([sa, sb_], [ua - off, ub - off], color=BLUE if arr == 1 else AQUA, lw=1.6)
axC.set_xlim(s0, s1); axC.set_ylim(ug[0] - 1.7, ug[-1]); axC.set_xlabel("along-track distance s (m)", color=MUTED, fontsize=9); axC.set_ylabel("across-track u (m, left +)", color=MUTED, fontsize=9)
axC.set_yticks([ug[0], 0, ug[-1], ug[0] - 1.7, -1.7, ug[-1] - 1.7]); axC.set_yticklabels([f"{ug[0]:.1f}", "0", f"{ug[-1]:.1f}", f"{ug[0]:.1f}", "0", f"{ug[-1]:.1f}"])
axC.set_title("C  Plan-view P(utility) in swath coordinates (max over depth 0.2 to 2.2 m) with Hough segments", fontsize=10, loc="left", color=INK)

# ---- D: calibration curves
for arr, col, name in ((1, BLUE, "600 MHz"), (2, AQUA, "200 MHz")):
    c = d[f"calib_centres{arr}"]; h = d[f"calib_hist{arr}"]; f0 = d[f"calib_f0{arr}"]
    axD.plot(c, h, color=col, lw=1.6, label=f"{name} all pixels"); axD.plot(c, f0, color=col, lw=1, ls="--", label=f"{name} null (0.95 log-normal)")
axD.set_yscale("log"); axD.set_ylim(1e-5, 3); axD.set_xlim(-3, 6); axD.set_xlabel("log line evidence (max over depth)", color=MUTED, fontsize=9); axD.set_ylabel("density", color=MUTED, fontsize=9)
axD.legend(fontsize=7, loc="upper right"); axD.grid(color="#e6e6e6", lw=0.5)
axD.set_title("D  Calibration: empirical-Bayes two-groups fit", fontsize=10, loc="left", color=INK)

# ---- E: depth-vs-s of segments and probability-length summary
for f in feats:
    pr = f["properties"]
    col = {"600": BLUE, "200": AQUA, "both": ORANGE}[pr["source"]]
    axE.plot(pr["s_m"], [pr["depth_m_at_v0.10"]] * 2, color=col, lw=1.2, alpha=0.8)
axE.invert_yaxis(); axE.set_xlim(sg[0], sg[-1]); axE.set_xlabel("s (m)", color=MUTED, fontsize=9); axE.set_ylabel("depth (m, v = 0.10 m/ns)", color=MUTED, fontsize=9); axE.grid(color="#e6e6e6", lw=0.5)
axE.set_title("E  Segment depth along the swath (depth is velocity-dependent, plan position is not)", fontsize=10, loc="left", color=INK)

fig.suptitle("Where are buried utilities under this street, and how sure are we? (provisional: no ground truth yet)", fontsize=13, color=INK, x=0.01, ha="left", y=0.985)
fig.text(0.01, 0.962, "A overview, density of confident pixels per 10 m; D calibration of evidence to probability; B street-aligned zoom with probability raster, segments, repeat pass and public record; C probability strips per frequency; E segment depths.", fontsize=9, color=MUTED)
plt.savefig(out_png, dpi=160, bbox_inches="tight", facecolor=SURF)
print("saved", out_png)
