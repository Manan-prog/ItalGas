"""Figure: can the method find a pipe running ALONG the road? Injection-recovery test on the
development swath (Corso A. Ricci): synthetic along-road pipes added to the real radar data.
Usage (from figures/): python plot_injection_v2.py injection_<hash>.json out.png"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from along_road import inject_longitudinal  # noqa: E402
from build_v2 import cfg_hash, load_cfg  # noqa: E402

SURF = "#fcfcfb"; BLUE = "#2a78d6"; AQUA = "#1baf7a"; ORANGE = "#eb6834"; INK = "#1f1f1f"; MUTED = "#6b6b6b"
ROOT = Path(__file__).resolve().parents[1]
cfg = load_cfg(str(ROOT / "pipeline/configs/dev_swath_v2_2.yaml")); CACHE = Path(cfg["base"]["cache_dir"]); V = cfg["migration"]["velocity_m_ns"]
R = json.load(open(CACHE / sys.argv[1])); out = sys.argv[2]
amps = [r["amp"] for r in R["runs"]]
rec = {a: [np.mean(r["arrays"][a]["hit"]) for r in R["runs"]] for a in ("1", "2")}
anyq = {a: [np.mean(r["arrays"][a]["hit_any_q"]) for r in R["runs"]] for a in ("1", "2")}
ra = R["real_crossing_amp"]
lo = min(ra["1"]["q25"], ra["2"]["q25"]); hi = max(ra["1"]["q75"], ra["2"]["q75"])

fig = plt.figure(figsize=(18, 9.5), facecolor=SURF)
fig.suptitle("Can the method find a pipe running ALONG the road?  Test: hide fake pipes in the real radar data and count how many come back",
             x=0.01, y=0.985, ha="left", fontsize=14, color=INK)
fig.text(0.01, 0.935, "60 fake pipes per frequency (12 at each strength), 20 m long, 0.4 to 1.3 m deep, at random places and positions across the van, "
         "added to the real Corso A. Ricci data. Strength = peak amplitude in units of the typical radar level at that depth.\n"
         "A: share recovered (measured at strengths 1, 2, 3, 5, 8).  B to E: what one fake pipe of typical real strength (5) at 0.7 m looks like to the radar.",
         fontsize=9.5, color=MUTED)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.25, 1, 1], hspace=0.45, wspace=0.22, left=0.05, right=0.985, top=0.85, bottom=0.17)
axA = fig.add_subplot(gs[:, 0])
axA.axvspan(lo, hi, color="#e9e4d8", zorder=0)
axA.text((lo + hi) / 2, 1.06, "how strong real pipes\non this street are\n(crossings, middle half)", ha="center", va="bottom", fontsize=8.5, color=MUTED)
for a, col, lab in (("2", AQUA, "200 MHz"), ("1", BLUE, "600 MHz")):
    axA.plot(amps, rec[a], "-o", color=col, lw=2, ms=7, label=f"{lab}: recovered and passing the chance test", zorder=3)
    axA.plot(amps, anyq[a], "--o", color=col, lw=1.4, ms=4, mfc="white", alpha=0.9, label=f"{lab}: found, but not confirmed", zorder=2)
axA.set_xlabel("strength of the fake pipe (x typical level)", color=MUTED); axA.set_ylabel("share of fake pipes", color=MUTED)
axA.set_ylim(-0.03, 1.2); axA.set_xlim(0.5, 8.5); axA.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); axA.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
axA.grid(color="#e6e6e6", lw=0.6); axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8.5, frameon=False)
axA.set_title("A  200 MHz finds every along-road pipe of real strength; 600 MHz cannot", loc="left", fontsize=10.5, color=INK)
fig.text(0.05, 0.035, "Why 600 MHz fails: its channels are 0.16-0.18 m apart, more than a quarter wavelength (0.04 m), so the curve across the van is under-sampled.  "
         "Upper bound: the fake pipes are ideal (straight, constant depth and offset, same velocity as the detector); real pipes wander under the van.",
         fontsize=8.5, color=MUTED, ha="left", va="bottom")

# example: amp 5 pipe at 0.7 m, 200 MHz
d = np.load(CACHE / f"v2grid_2_{cfg_hash(cfg)}.npz"); s = d["s_grid"]; u = d["u_ch"]; dt = float(d["dt_full_ns"])
w = (s > 590) & (s < 650)
Z0 = d["Z"][:, w].astype(np.float32); sw_ = s[w]
seg = dict(s0=605.0, s1=625.0, u0=0.30, z=0.7, amp=5.0)
Zi = inject_longitudinal(Z0, u, sw_, dt, V, [seg], R["f_mhz"]["2"])
o = np.argsort(u); uu = u[o]; e = np.r_[uu[0] - 0.04, (uu[1:] + uu[:-1]) / 2, uu[-1] + 0.04]
mid = (sw_ > 614) & (sw_ < 616)
for k, (ZZ, ttl) in enumerate(((Z0, "B  Across the van at s = 615 m, real data only"), (Zi, "C  Same place with the fake pipe added"))):
    ax = fig.add_subplot(gs[0, 1 + k])
    ax.pcolormesh(e, np.r_[np.arange(ZZ.shape[2]), ZZ.shape[2]] * dt * V / 2, ZZ[o][:, mid].mean(1).T, cmap="gray", vmin=-3, vmax=3)
    ax.set_ylim(1.6, 0); ax.invert_xaxis()
    if k == 1:
        ax.plot(seg["u0"], seg["z"], "+", color=ORANGE, ms=14, mew=2)
    ax.set_title(ttl, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("across the van u (m, + = left)", color=MUTED, fontsize=8.5); ax.set_ylabel("depth (m)", color=MUTED, fontsize=8.5)
c = int(np.argmin(np.abs(u - seg["u0"])))
for k, (ZZ, ttl) in enumerate(((Z0, f"D  Along the street, channel at u = {u[c]:+.2f} m, real data"), (Zi, "E  Same with the fake pipe (605 to 625 m)"))):
    ax = fig.add_subplot(gs[1, 1 + k])
    ax.imshow(ZZ[c].T, cmap="gray", vmin=-4, vmax=4, aspect="auto", extent=[sw_[0], sw_[-1], ZZ.shape[2] * dt * V / 2, 0])
    ax.set_ylim(1.6, 0)
    if k == 1:
        ax.annotate("", xy=(625, 0.62), xytext=(605, 0.62), arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.2))
    ax.set_title(ttl, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("along the street (m)", color=MUTED, fontsize=8.5); ax.set_ylabel("depth (m)", color=MUTED, fontsize=8.5)
for a_ in fig.axes:
    a_.set_facecolor(SURF); a_.tick_params(colors=MUTED, labelsize=8)
    for sp in a_.spines.values():
        sp.set_color("#cfcfcf")
fig.savefig(out, dpi=150, facecolor=SURF); print("saved", out)
