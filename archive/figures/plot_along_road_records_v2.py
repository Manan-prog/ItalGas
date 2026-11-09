"""Figure: along-road detection on held-out swaths vs municipal records. Top: per-swath
targets, hits and chance. Rows below: for recorded pipes running inside the footprint, the
200 MHz radar across the van (2 m stack at the record middle) and along the street under
the nearest channel, with the record's position and recorded depth marked.
Usage (from figures/): python plot_along_road_records_v2.py along_summary_<hash>.json out.png"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from along_road import match  # noqa: E402
from build_v2 import load_cfg  # noqa: E402

SURF = "#fcfcfb"; BLUE = "#2a78d6"; AQUA = "#1baf7a"; ORANGE = "#eb6834"; INK = "#1f1f1f"; MUTED = "#6b6b6b"
ROOT = Path(__file__).resolve().parents[1]
cfg = load_cfg(str(ROOT / "pipeline/configs/dev_swath_v2_2.yaml")); CACHE = Path(cfg["base"]["cache_dir"]); V = cfg["migration"]["velocity_m_ns"]
S = json.load(open(CACHE / sys.argv[1])); out = sys.argv[2]
rows = S["swaths"]
items = []
for r in rows:
    res = json.load(open(CACHE / f"along_{r['tag']}.json")); npz = np.load(CACHE / f"along_{r['tag']}.npz")
    trk = res["arrays"]["1"]["tracks"] + res["arrays"]["2"]["tracks"]
    h = match(trk, res["targets"], 0.05, 0.5, None, 0.3)
    for i, (g, hh) in enumerate(zip(res["targets"], h)):
        items.append(dict(tag=r["tag"], i=i, g=g, hit=bool(hh), npz=npz))
# show all hits, then misses with the largest recorded diameter (strongest expected reflectors)
def diam(g):
    try:
        return float(str(g["diam"]).split("*")[0])
    except ValueError:
        return 0.0
hits = [x for x in items if x["hit"]]; miss = sorted([x for x in items if not x["hit"]], key=lambda x: -diam(x["g"]))
show = hits[:1] + miss[:2]
import copy
from build_v2 import geometry_for, grid_Z, preprocess_array  # noqa: E402
from ids_reader import Swath  # noqa: E402
TRACK_SWATHS = [("savona 13.10.2025 matt_2025.10.13_09.51.03", "Survey_2025-10-13_10-01-51", "Swath026", "20251013_095103_10-01-51_Swath026", [0, 1]),
                ("savona 1410 matt_2025.10.15_09.07.28", "Survey_2025-10-15_09-11-09", "Swath010", "20251015_090728_09-11-09_Swath010", [0])]
tracks_show = []
for sess, surv, swn, tag, idx in TRACK_SWATHS:
    c_ = copy.deepcopy(cfg); c_["base"]["survey_dir"] = f"/Users/manansarupria/ItalGas/Savona/{sess}/{surv}"; c_["base"]["swath"] = swn; c_["cube_file"] = None
    sw = Swath.load(c_["base"]["survey_dir"], swn); geo = geometry_for(c_, sw)
    sg = np.arange(0, float(min(np.nanmax(geo["s1"]), np.nanmax(geo["s2"]))), 0.05)
    pp = preprocess_array(sw, 2, c_); Zt = grid_Z(pp["X"], geo["s2"], sg); ut = np.nanmedian(geo["u2"], 0)
    sig = [t for t in json.load(open(CACHE / f"along_{tag}.json"))["arrays"]["2"]["tracks"] if t["q"] <= 0.05]
    for i_ in idx:
        tracks_show.append(dict(tag=tag, t=sig[i_], Z=Zt, u=ut, s=sg, dt=pp["dt_ns"]))

NR = len(show) + len(tracks_show)
fig = plt.figure(figsize=(18, 4.2 + 2.6 * NR), facecolor=SURF)
P = S["pooled"]
fig.subplots_adjust()
fig.suptitle(f"Held-out test: along-road detections do NOT match recorded pipes: {sum(r['hits'] for r in rows)} of {P['n_targets']} hit, "
             f"{P['chance_frac'] * P['n_targets']:.0f} expected by chance", x=0.01, y=0.997, ha="left", fontsize=14, color=INK)
fig.text(0.01, 0.955, "B to D: what the radar shows at recorded pipes (orange + = recorded position and depth).  E to G: the strongest along-road tracks the detector confirmed (orange = track), which look like layers that end or lie partly under the van, not compact pipes.", fontsize=9.5, color=INK)
fig.text(0.01, 0.967, "12 swaths on 4 days not used to build the method. Records: Comune di Savona storm-drain and sewer pipes inside the 1.5 m radar footprint "
         "for at least 10 m, recorded 0.2 to 1.5 m deep.\nHit = a confirmed along-road track within 0.5 m across and overlapping 30 percent of the record.",
         fontsize=9.5, color=MUTED)
gs = fig.add_gridspec(1 + NR, 2, height_ratios=[1.3] + [1] * NR, width_ratios=[0.33, 0.67], hspace=0.75, wspace=0.12,
                      left=0.05, right=0.985, top=0.93, bottom=0.04)
ax = fig.add_subplot(gs[0, :])
x = np.arange(len(rows))
ax.bar(x - 0.27, [r["n_targets"] for r in rows], 0.26, color="#cfcfcf", label="recorded pipes along the footprint")
ax.bar(x, [r["hits"] for r in rows], 0.26, color=AQUA, label="hit by a confirmed track")
ax.bar(x + 0.27, [r["chance_mean"] for r in rows], 0.26, color=ORANGE, alpha=0.8, label="hits expected by chance")
ax.set_xticks(x); ax.set_xticklabels([r["tag"].replace("_", "\n", 2) for r in rows], fontsize=7)
ax.set_ylabel("count", color=MUTED); ax.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper left")
ax.set_title("A  Per swath", loc="left", fontsize=10.5, color=INK)
for k, it in enumerate(show):
    g = it["g"]; z = it["npz"]; i = it["i"]
    u = z["u_ch2"]; dt = float(z["dt2"]); X = z[f"x{i}_2"].astype(np.float32); B = z[f"b{i}_2"].astype(np.float32); bs = z[f"bs{i}_2"]
    o = np.argsort(u); uu = u[o]; e = np.r_[uu[0] - 0.04, (uu[1:] + uu[:-1]) / 2, uu[-1] + 0.04]
    a1 = fig.add_subplot(gs[1 + k, 0])
    a1.pcolormesh(e, np.r_[np.arange(X.shape[1]), X.shape[1]] * dt * V / 2, X[o].T, cmap="gray", vmin=-3, vmax=3)
    a1.plot(g["u0"], g["z"], "+", color=ORANGE, ms=14, mew=2); a1.axvspan(g["u0"] - 0.5, g["u0"] + 0.5, color=ORANGE, alpha=0.08)
    a1.set_ylim(1.8, 0); a1.invert_xaxis(); a1.set_ylabel("depth (m)", color=MUTED, fontsize=8.5)
    a1.set_xlabel("across the van u (m)", color=MUTED, fontsize=8)
    verdict = "HIT" if it["hit"] else "MISS"
    a1.set_title(f"{'BCDEFGH'[k]}  RECORD {verdict}: {g['kind']} pipe {g['diam']} mm {g['material']}, recorded {g['z']:.2f} m",
                 loc="left", fontsize=9.5, color=AQUA if it["hit"] else INK)
    a2 = fig.add_subplot(gs[1 + k, 1])
    a2.imshow(B.T, cmap="gray", vmin=-4, vmax=4, aspect="auto", extent=[bs[0], bs[1], B.shape[1] * dt * V / 2, 0])
    a2.plot([max(g["s0"], bs[0]), min(g["s1"], bs[1])], [g["z"]] * 2, color=ORANGE, lw=1.2, ls="--")
    a2.set_ylim(1.8, 0); a2.set_xlabel("along the street (m)", color=MUTED, fontsize=8)
    a2.set_title(f"{g['street']}: along the street under the channel at u = {bs[2]:+.2f} m (record u = {g['u0']:+.2f} m, {it['tag']})", loc="left", fontsize=9, color=MUTED)
LAB = ["E  layer starting and ending abruptly at 0.33 m, left side only (slab or patch edge?)",
       "F  reflector at 1.08 m for 25 m, tilted across the van (layer or pipe: undecidable)",
       "G  reflector at 0.95 m for 100 m, Via Ciantagaletto (pipe or layer: undecidable)"]
for k, it in enumerate(tracks_show):
    t = it["t"]; Zt = it["Z"]; u = it["u"]; sg = it["s"]; dt = it["dt"]
    sm = 0.5 * (t["s_start"] + t["s_end"]); w = (sg > sm - 1) & (sg < sm + 1); X = Zt[:, w].mean(1)
    o = np.argsort(u); uu = u[o]; e = np.r_[uu[0] - 0.04, (uu[1:] + uu[:-1]) / 2, uu[-1] + 0.04]
    a1 = fig.add_subplot(gs[1 + len(show) + k, 0])
    a1.pcolormesh(e, np.r_[np.arange(X.shape[1]), X.shape[1]] * dt * V / 2, X[o].T, cmap="gray", vmin=-3, vmax=3)
    a1.plot(t["u_mean"], t["depth_m"], "+", color=ORANGE, ms=14, mew=2); a1.set_ylim(1.8, 0); a1.invert_xaxis()
    a1.set_ylabel("depth (m)", color=MUTED, fontsize=8.5); a1.set_xlabel("across the van u (m)", color=MUTED, fontsize=8)
    a1.set_title(LAB[k], loc="left", fontsize=9.5, color=BLUE)
    a2 = fig.add_subplot(gs[1 + len(show) + k, 1])
    c = int(np.argmin(np.abs(u - t["u_mean"]))); w2 = (sg > t["s_start"] - 10) & (sg < t["s_end"] + 10)
    a2.imshow(Zt[c, w2].T, cmap="gray", vmin=-4, vmax=4, aspect="auto", extent=[sg[w2][0], sg[w2][-1], Zt.shape[2] * dt * V / 2, 0])
    a2.plot([t["s_start"], t["s_end"]], [t["depth_m"] + 0.12] * 2, color=ORANGE, lw=1.5)
    a2.set_ylim(1.8, 0); a2.set_xlabel("along the street (m)", color=MUTED, fontsize=8)
    a2.set_title(f"along the street under the channel at u = {u[c]:+.2f} m ({it['tag']}); orange bar just below the track", loc="left", fontsize=9, color=MUTED)
for a_ in fig.axes:
    a_.set_facecolor(SURF); a_.tick_params(colors=MUTED, labelsize=8)
    for sp in a_.spines.values():
        sp.set_color("#cfcfcf")
fig.savefig(out, dpi=130, facecolor=SURF); print("saved", out)
