"""Figure: probability that a buried pipe passes under each spot, drawn on the satellite map where
each drive's radar strip really lies (north up). Via Nizza, six drives of 13 Oct 2025.
Usage (from figures/): python plot_map_probability.py out.png
Probability per spot = 1 - prod(1 - P_i g_i) over detections i of that drive:
  crossing pipes: P_i from the column-shift null (both radar rows merged), spread along the street
  with a Gaussian of 0.3 m (registration residual 0.23 m, E16);
  along-road candidates (pipe-shaped repeated features, E17): P = 1 - 0.06/0.19 = 0.68 (share of
  pipe-shaped features not explained by the random-place rate), spread 0.15 m across."""
import json
import sys
from pathlib import Path

import contextily as cx
import matplotlib
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap, to_rgba

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from basemap_tools import GSAT  # noqa: E402
from build_v2 import load_cfg  # noqa: E402
from geometry import merc_to_lonlat  # noqa: E402
from run_detection import su_to_map  # noqa: E402

SURF = "#fcfcfb"; INK = "#1f1f1f"; MUTED = "#6b6b6b"
ROOT = Path(__file__).resolve().parents[1]
mc = yaml.safe_load(open(ROOT / "pipeline/configs/multipass_aurelia.yaml"))
det = load_cfg(str(ROOT / "pipeline" / mc["detector_config"])); CACHE = Path(det["base"]["cache_dir"])
out_png = sys.argv[1]
REG = json.load(open(CACHE / "mp_registration.json"))["registration"]
fr = np.load(CACHE / "mp_frame.npz"); s_nodes, xn, yn, k = fr["s_nodes"], fr["xn"], fr["yn"], float(fr["k"])
ST = json.load(open(CACHE / "mp_shape_test.json"))
summ = ST["summary"]; P_ALONG = 1 - summ["frac_controls"] / summ["frac_features"]
cands = [x for x in ST["features"] if x["verdict"] == "pipe-like"]
SIG_S = 0.3


def merged_crossings(R: dict) -> list[dict]:
    a = [c for c in (R["arrays"].get("1") or {}).get("crossings", []) if c["q"] <= 0.05]
    b = [c for c in (R["arrays"].get("2") or {}).get("crossings", []) if c["q"] <= 0.05]
    used = set(); out = []
    for c in a:
        j = next((i for i, d in enumerate(b) if i not in used and abs(c["s0_m"] - d["s0_m"]) <= 0.3 and abs(c["phi_deg"] - d["phi_deg"]) <= 15
                  and abs(c["depth_m"] - d["depth_m"]) <= 0.2), None)
        if j is None:
            out.append(dict(s0=c["s0_m"], phi=c["phi_deg"], P=c["p_real"]))
        else:
            used.add(j); d = b[j]
            out.append(dict(s0=0.5 * (c["s0_m"] + d["s0_m"]), phi=c["phi_deg"], P=1 - (1 - c["p_real"]) * (1 - d["p_real"])))
    out += [dict(s0=d["s0_m"], phi=d["phi_deg"], P=d["p_real"]) for i, d in enumerate(b) if i not in used]
    return out


def strip(name: str, s0: float, s1: float, ds: float = 0.1, n_u: int = 20):
    """Map-coordinate mesh (corners) and probability of one drive's strip between s0 and s1 (true s)."""
    g = REG[name]; d = np.load(CACHE / f"mp_{name}_2.npz"); sg = d["s_grid"]; Ug = d["Ug"]
    R = json.load(open(CACHE / f"mp_{name}.json")); X = merged_crossings(R)
    s_raw = np.arange(s0 + g["A"], s1 + g["A"] + 1e-9, ds)                         # raw s of the wanted true range
    lo = np.array([np.interp(s_raw, sg, np.nanmin(np.where(np.isfinite(Ug), Ug, np.inf), 0), left=np.nan, right=np.nan)])[0] - 0.04
    hi = np.array([np.interp(s_raw, sg, np.nanmax(np.where(np.isfinite(Ug), Ug, -np.inf), 0), left=np.nan, right=np.nan)])[0] + 0.04
    ok = np.isfinite(lo) & np.isfinite(hi) & (hi - lo < 2.0) & (hi > lo)
    if ok.sum() < 5:
        return None
    s_raw = s_raw[ok]; lo = lo[ok]; hi = hi[ok]
    f = np.linspace(0, 1, n_u + 1)
    S = np.repeat(s_raw[:, None], n_u + 1, 1); U = lo[:, None] + f[None, :] * (hi - lo)[:, None]    # corners, raw frame
    Sc = 0.5 * (S[1:, 1:] + S[:-1, :-1]); Uc = 0.5 * (U[1:, 1:] + U[:-1, :-1])                     # cell centres
    q = np.ones_like(Sc)
    for o in X:
        dd = Sc - (o["s0"] + Uc * np.tan(np.radians(o["phi"])))
        q *= 1 - o["P"] * np.exp(-0.5 * (dd / SIG_S) ** 2)
    for c in cands:                                                                  # candidates are in true coordinates
        inside = (Sc - g["A"] >= c["s0"]) & (Sc - g["A"] <= c["s1"])
        q *= 1 - P_ALONG * inside * np.exp(-0.5 * ((Uc - g["B"] - c["uc"]) / 0.15) ** 2)
    P = 1 - q
    mx, my = su_to_map(S.ravel() - g["A"], U.ravel() - g["B"], s_nodes, xn, yn, k)
    return mx.reshape(S.shape), my.reshape(S.shape), P


cmap = LinearSegmentedColormap.from_list("p", [(0.0, to_rgba("#f6e3b8", 0.8)), (0.15, to_rgba("#f2d9a6", 0.85)),
                                               (0.5, to_rgba("#eb8a3c", 0.95)), (1.0, to_rgba("#9e1f12", 1.0))])
GREY = LinearSegmentedColormap.from_list("g", [to_rgba("#dfe6ee", 0.6), to_rgba("#dfe6ee", 0.6)])
fig = plt.figure(figsize=(18, 10.5), facecolor=SURF)
fig.suptitle("Where are buried pipes under Via Nizza? Probability from six drives of the radar van, on the map (provisional)",
             x=0.02, y=0.975, ha="left", fontsize=14.5, color=INK, fontweight="bold")
fig.text(0.02, 0.93, "Each coloured strip is the 1.5 m wide band of ground under the van's 10 antennas during one drive. Colour = probability that a buried pipe passes "
         "under that spot.\nRed bars across a strip = a pipe crossing the street. A red band along a strip = a pipe running along the street "
         "(only one strong candidate here, panel B). Positions can be off by about 1 m.", fontsize=10, color=MUTED)
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.06, left=0.02, right=0.98, top=0.86, bottom=0.12)
VIEWS = [(2820.0, 2960.0, 20, "A.  A 140 m stretch of Via Nizza, six drives"), (2870.0, 2915.0, 21, "B.  Close-up: the along-road pipe candidate (about 0.7 m deep)")]
for ax_i, (s0, s1, zoom, ttl) in enumerate(VIEWS):
    ax = fig.add_subplot(gs[ax_i])
    xs, ys = [], []
    strips = [r for r in (strip(n, s0, s1) for n in REG) if r is not None]
    for mx, my, P in strips:                     # 1) every drive's strip in grey (bottom layer)
        ax.pcolormesh(mx, my, np.zeros_like(P), cmap=GREY, vmin=0, vmax=1, shading="flat", zorder=3, edgecolors="none", rasterized=True)
        xs += [mx.min(), mx.max()]; ys += [my.min(), my.max()]
    for mx, my, P in strips:                     # 2) probabilities of all drives on top, so no strip hides a detection
        ax.pcolormesh(mx, my, np.ma.masked_less(P, 0.03), cmap=cmap, vmin=0, vmax=1, shading="flat", zorder=4, edgecolors="none", rasterized=True)
    # along-road path that passed the chance test in the path-linking step (E21)
    NET = json.load(open(CACHE / "mp_network.json"))
    for pth in [q for q in NET["dp"]["paths"] if q["significant"]]:
        ps, pu = np.array(pth["path_s"]), np.array(pth["path_u"]); w_ = (ps >= s0) & (ps <= s1)
        if w_.sum() > 2:
            px, py = su_to_map(ps[w_], pu[w_], s_nodes, xn, yn, k)
            ax.plot(px, py, color="white", lw=2.6, zorder=5); ax.plot(px, py, color="#2a78d6", lw=1.4, ls=(0, (4, 2)), zorder=6)
    # GPS track of the reference drive (dashed), for orientation
    sgr = np.linspace(s0, s1, 200); tx, ty = su_to_map(sgr, np.zeros_like(sgr), s_nodes, xn, yn, k)
    ax.plot(tx, ty, ls=(0, (3, 3)), color="black", lw=0.8, zorder=4)
    cxm, cym = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)); half = 0.5 * max(max(xs) - min(xs), max(ys) - min(ys)) * 1.08
    ax.set_xlim(cxm - half, cxm + half); ax.set_ylim(cym - half, cym + half); ax.set_aspect("equal")
    cx.add_basemap(ax, source=GSAT, crs="EPSG:3857", zoom=zoom, attribution=False, alpha=0.9)
    L = 20.0 if ax_i == 0 else 5.0
    x0b, y0b = cxm - half * 0.92, cym - half * 0.9
    ax.plot([x0b, x0b + L / k], [y0b, y0b], color="white", lw=4, zorder=6, solid_capstyle="butt")
    ax.text(x0b + L / k / 2, y0b + half * 0.03, f"{L:.0f} m", color="white", ha="center", fontsize=10, zorder=6, fontweight="bold")
    ax.annotate("N", xy=(cxm + half * 0.9, cym + half * 0.88), xytext=(cxm + half * 0.9, cym + half * 0.72), color="white", ha="center",
                fontsize=11, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5), zorder=6)
    lo_, la_ = merc_to_lonlat(np.array([cxm - half, cxm + half]), np.array([cym - half, cym + half]))
    ax.text(0.01, 0.01, f"{la_[0]:.5f} N, {lo_[0]:.5f} E", transform=ax.transAxes, fontsize=8, color=INK, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    ax.text(0.99, 0.99, f"{la_[1]:.5f} N, {lo_[1]:.5f} E", transform=ax.transAxes, fontsize=8, color=INK, ha="right", va="top", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(ttl, loc="left", fontsize=11.5, color=INK)
    if ax_i == 0:
        vx, vy = su_to_map(np.array([2870, 2870, 2915, 2915, 2870]), np.array([3.0, 5.6, 5.6, 3.0, 3.0]), s_nodes, xn, yn, k)
        ax.plot(vx, vy, color="white", lw=1.5, zorder=5); ax.text(vx[2], vy[2], "  B", color="white", fontsize=11, fontweight="bold", zorder=6)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
cax = fig.add_axes([0.25, 0.065, 0.3, 0.022]); cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
cb.set_label("probability that a buried pipe passes under this spot", fontsize=10, color=INK); cb.ax.tick_params(labelsize=9)
fig.legend(handles=[Line2D([], [], ls=(0, (4, 2)), color="#2a78d6", lw=1.6, label="linked along-road path (passes the chance test, wanders: provisional)"),
                    Line2D([], [], ls=(0, (3, 3)), color="black", lw=1, label="GPS track of one drive"),
                    Patch(fc=to_rgba("#dfe6ee", 0.8), ec="none", label="radar strip, nothing found")],
           loc="lower right", bbox_to_anchor=(0.99, 0.02), fontsize=9.5, frameon=False, ncol=1)
fig.savefig(out_png, dpi=160, facecolor=SURF); print("saved", out_png)
