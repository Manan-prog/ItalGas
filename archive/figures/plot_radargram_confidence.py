"""Radargrams at a high, a medium and a low P(linear reflector) location, 600 MHz, dev swath.

Question answered: what does the raw-ish radar data look like where the detector is
confident vs where it is not?
Rows: (A) along-track B-scan of the channel nearest the location (processed, NOT migrated);
(B) cross-section across the 10 x 600 MHz channels at that position (mean of +-0.25 m);
(C) the detector's plan-view probability around the location.
Usage: python plot_radargram_confidence.py  (paths from pipeline/configs/dev_swath.yaml)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from ids_reader import Swath  # noqa: E402
from processing import (align_traces, background_removal, bandpass, dewow,  # noqa: E402
                        first_break_sample, sec_gain, shift_time_zero)

cfg = yaml.safe_load(open(ROOT / "pipeline/configs/dev_swath.yaml"))
CACHE = Path(cfg["cache_dir"])
CUBE = CACHE / "cube_Swath001_2025.10.14_11.31.18_2d9b11acb1.npz"
DET = CACHE / "det_dev.npz"
V = cfg["processing"]["velocity_m_ns"]          # DERIVED 0.10 +- 0.02 m/ns (E03)
# locations chosen from det_dev.npz (600 MHz, u = +0.46 m column of the plan grid)
CASES = [("High confidence", 130.0, 0.46), ("Medium confidence", 255.0, 0.46), ("Low confidence", 618.0, 0.46)]
HALF_S = 12.0      # m shown either side along track
ARR = 1            # 600 MHz

pc = cfg["processing"]
sw = Swath.load(cfg["survey_dir"], cfg["swath"])
chans = sw.config.channels_of(ARR)
dt = chans[0].dt_ns
cube = np.load(CUBE)
S = cube["s1"]; U = cube["u1"]                 # (n_scans, n_channels) m
det = np.load(DET)
prob, sg, ug = det["prob1"], det["sg"], det["ug"]

proc = []
for ch in chans:
    x = np.asarray(sw.data[ARR][:, ch.slot, :]).astype(np.float32)
    x = dewow(x, dt, pc["dewow_win_ns"][ARR])
    n0 = first_break_sample(x.mean(0), pc["t0_frac"])
    x = align_traces(x, x.mean(0), n0)
    x = shift_time_zero(x, n0)
    x = background_removal(x, None)
    x = bandpass(x, dt, *pc["bandpass_mhz"][ARR])
    x = sec_gain(x, dt, 0.0, pc["gain_power"])
    proc.append(x)
proc = np.stack(proc, 1)                       # (n_scans, n_channels, n_samples)
t_ns = np.arange(proc.shape[2]) * dt
depth = V * t_ns / 2
u_ch = np.nanmedian(U, 0)                      # m, lateral position of each channel footprint
order = np.argsort(u_ch)
clip = np.percentile(np.abs(proc[:, :, 10:300]), 99)

fig = plt.figure(figsize=(16, 13), facecolor="#fcfcfb")
fig.suptitle("Does a high detector P mean a visible reflector? No: at P = 0.96 the radargram shows only noise (600 MHz, Corso A. Ricci, 14 Oct 11:31)",
             x=0.01, y=0.985, ha="left", fontsize=14)
fig.text(0.01, 0.94, "A: radargram along the street for the antenna nearest the point (vertical line). "
         "B: slice ACROSS the street through all 10 antennas at that point (a pipe running along the street is a "
         "compact bright spot here). C: detector probability seen from above; dashed line = A, cross = point.",
         fontsize=9.5, color="#555", wrap=True)
gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1.0, 0.55], hspace=0.45, wspace=0.22,
                      left=0.06, right=0.98, top=0.87, bottom=0.05)
for j, (name, s0, u0) in enumerate(CASES):
    i_g = np.argmin(np.abs(sg - s0)); j_g = np.argmin(np.abs(ug - u0))
    p0 = prob[i_g, j_g]; t0 = det["t_ns1"][i_g, j_g]
    c = int(np.argmin(np.abs(u_ch - u0)))
    s_c = S[:, c]
    m = (s_c > s0 - HALF_S) & (s_c < s0 + HALF_S)
    ax = fig.add_subplot(gs[0, j])
    ax.imshow(proc[m, c, :].T, cmap="gray", vmin=-clip, vmax=clip, aspect="auto",
              extent=[s_c[m][0], s_c[m][-1], depth[-1], 0])
    ax.axvline(s0, color="#eb6834", lw=1, ls=":")
    ax.axhline(V * t0 / 2, color="#2a78d6", lw=0.8, ls="--")
    ax.set_ylim(3.0, 0); ax.set_xlim(s0 - HALF_S, s0 + HALF_S)
    ax.set_title(f"{'ABC'[0]}{j+1}  {name}: P = {p0:.2f}\nchannel at u = {u_ch[c]:+.2f} m, s = {s0:.0f} m",
                 loc="left", fontsize=10.5)
    ax.set_xlabel("distance along street s (m)")
    if j == 0:
        ax.set_ylabel("depth (m, at v = 0.10 m/ns)")
    ax.text(0.99, 0.02, "blue dashed = depth the detector picked", transform=ax.transAxes, ha="right",
            fontsize=8, color="#2a78d6", bbox=dict(fc="white", ec="none", alpha=0.7))

    ax = fig.add_subplot(gs[1, j])
    mm = np.abs(S[:, c] - s0) < 0.25
    xs = np.abs(hilbert(proc[mm].mean(0), axis=-1))[order]   # (n_channels, n_samples) envelope
    uu = u_ch[order]
    edges = np.r_[uu[0] - 0.04, (uu[1:] + uu[:-1]) / 2, uu[-1] + 0.04]
    ax.pcolormesh(edges, np.r_[depth, depth[-1] + V * dt / 2], xs.T, cmap="gray_r", vmin=0, vmax=np.percentile(xs[:, 10:300], 99), shading="flat")
    ax.plot(u0, V * t0 / 2, "+", color="#eb6834", ms=14, mew=2)
    ax.set_ylim(3.0, 0); ax.invert_xaxis()
    ax.set_title(f"B{j+1}  slice across the street at s = {s0:.0f} m",
                 loc="left", fontsize=10.5)
    ax.set_xlabel("across-street position u (m, + = left of travel)")
    if j == 0:
        ax.set_ylabel("depth (m)")

    ax = fig.add_subplot(gs[2, j])
    w = np.abs(sg - s0) < HALF_S
    ax.imshow(prob[w].T, origin="lower", aspect="auto", cmap="Oranges", vmin=0, vmax=1,
              extent=[sg[w][0], sg[w][-1], ug[0] - 0.04, ug[-1] + 0.04])
    ax.axhline(u_ch[c], color="#2a78d6", lw=0.8, ls="--")
    ax.plot(s0, u0, "x", color="k", ms=8, mew=1.5)
    ax.set_title(f"C{j+1}  detector P seen from above (white 0, dark orange 1)", loc="left", fontsize=10.5)
    ax.set_xlabel("distance along street s (m)")
    if j == 0:
        ax.set_ylabel("u (m)")
for a in fig.axes:
    for sp in a.spines.values():
        sp.set_color("#999")
    a.tick_params(colors="#555", labelsize=8.5)
out = ROOT / "figures/savona_radargram_confidence_dev.png"
fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
print("saved", out)
