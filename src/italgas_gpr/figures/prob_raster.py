"""Figure: where on the 200 MHz B-scans is the along-street evidence beyond chance?
Input: cache/prob_raster_<drive>_<s0>.npz (along/evidence.py) and the E39 tracks of that drive.
Run: python -m italgas_gpr.figures.prob_raster <drive> <s0> <stretch start m> <stretch length m> -> figures/savona_prob_raster_<drive>_<s0>.png"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from italgas_gpr.io.paths import CACHE, FIGURES  # noqa: E402
BLUE, ORANGE, AQUA, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#1f1f1f", "#8a8a8a", "#fcfcfb"
V = 0.10   # m/ns, ASSUMED velocity for the depth axis labels (same as the detector)


def main(drive: int, s0: int, a: float, L: float) -> None:
    d = np.load(CACHE / f"prob_raster_{drive}_{s0}.npz"); s = d["s"]; t = d["t_ns"]; P = d["P"].astype(float); cs = d["centres_s"]; ug = d["ug"]
    B = d["B"].astype(float); u_ch = d["u_ch"]; cols = d["cols"]; t_lo, t_hi = d["t_search"]
    tracks = [x for x in json.load(open(CACHE / "city" / f"along_c_{drive}.json"))["tracks"] if x["q"] <= 0.05 and x["length_m"] >= 10]
    it = (t >= t_lo) & (t <= t_hi); plan = P[:, :, it].max(2)                                   # (n_win, n_u)
    cmap = LinearSegmentedColormap.from_list("p", [(0, (0, 0, 0, 0)), (0.5, (*matplotlib.colors.to_rgb(ORANGE), 0.0)), (0.6, (*matplotlib.colors.to_rgb(ORANGE), 0.55)), (1, (*matplotlib.colors.to_rgb(ORANGE), 1.0))])
    fig = plt.figure(figsize=(14, 10.5), facecolor=SURF)
    fig.suptitle(f"Where on the B-scans is the along-street evidence beyond chance?  Drive {drive}, piece {int(d['c0'])}-{int(d['c1'])} m", x=0.01, ha="left", fontsize=13, color=INK, weight="bold")
    fig.text(0.01, 0.962, "A: plan view of P = 1 - local false discovery rate, max over depth (grey 0 to black 1), with the significant tracks (aqua) and the five channels shown below (dotted).\n"
             "B-F: the five single-column B-scans of the stretch marked in A (grey = cleaned traces, option C), orange = P >= 0.5. "
             "P means 'chance alone rarely gives this much evidence here', not 'this is a pipe'.", fontsize=8.5, color=MUTED, va="top")
    gs = fig.add_gridspec(6, 1, left=0.06, right=0.95, top=0.905, bottom=0.075, hspace=0.5, height_ratios=[1.6, 1, 1, 1, 1, 1])
    ax = fig.add_subplot(gs[0]); ax.set_facecolor(SURF)
    ax.imshow(plan.T, extent=[cs[0], cs[-1], ug[0], ug[-1]], aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=1, alpha=0.9, interpolation="nearest")
    for x in tracks:
        if x["s_end"] > cs[0] and x["s_start"] < cs[-1]:
            ax.plot(x["path_s"], x["path_u"], color=AQUA, lw=1.6)
    ax.axvspan(a, a + L, color=BLUE, alpha=0.12, lw=0); ax.text(a + L / 2, ug[-1] + 0.03, "stretch shown below", ha="center", fontsize=7.5, color=BLUE)
    for y in u_ch[cols]: ax.axhline(y, color=MUTED, lw=0.4, ls=":")
    ax.set_xlim(cs[0], cs[-1]); ax.set_ylabel("across the strip, u (m, left +)", fontsize=8.5); ax.set_xlabel("along the drive, s (m)", fontsize=8.5)
    ax.set_title(f"A  Plan view; cells with P >= 0.5: {100*float(d['frac_P50']):.2f}% of the searched volume, P >= 0.9: {100*float(d['frac_P90']):.3f}%", fontsize=10, loc="left", color=INK)
    ax.tick_params(labelsize=8)
    ms = (s >= a) & (s <= a + L); im = (t <= min(t_hi + 6, t[-1]))
    for k, (c, lab) in enumerate(zip(cols, ("B", "C", "D", "E", "F"))):
        ax = fig.add_subplot(gs[k + 1]); ax.set_facecolor(SURF)
        b = B[k][ms][:, im]; vmax = float(np.percentile(np.abs(b), 99))
        ax.imshow(b.T, extent=[s[ms][0], s[ms][-1], t[im][-1], t[im][0]], aspect="auto", cmap="Greys", vmin=-vmax, vmax=vmax, interpolation="nearest")
        iu = int(np.argmin(np.abs(ug - u_ch[c]))); js = np.clip(np.searchsorted(cs, s[ms]), 0, len(cs) - 1)
        Pc = P[js, iu][:, im]; ax.imshow(Pc.T, extent=[s[ms][0], s[ms][-1], t[im][-1], t[im][0]], aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.axhline(t_lo, color=MUTED, lw=0.6, ls=":"); ax.axhline(t_hi, color=MUTED, lw=0.6, ls=":")
        for x in tracks:
            ps, pu, pt = np.array(x["path_s"]), np.array(x["path_u"]), np.array(x["path_t"]); m = (ps >= a) & (ps <= a + L) & (np.abs(pu - u_ch[c]) <= 0.1)
            if m.sum() >= 2: ax.plot(ps[m], pt[m], color=AQUA, lw=1.2)
        ax.set_xlim(s[ms][0], s[ms][-1]); ax.set_ylim(t[im][-1], t[im][0]); ax.set_ylabel("time (ns)", fontsize=8); ax.tick_params(labelsize=7.5)
        ax2 = ax.twinx(); ax2.set_ylim(ax.get_ylim()[0] * V / 2, ax.get_ylim()[1] * V / 2); ax2.set_ylabel("depth (m, v 0.10 ASSUMED)", fontsize=7, color=MUTED); ax2.tick_params(labelsize=7, colors=MUTED)
        ax.set_title(f"{lab}  channel {int(c)} at u = {u_ch[c]:+.2f} m", fontsize=9, loc="left", color=INK)
        if k == 4: ax.set_xlabel("along the drive, s (m)", fontsize=8.5)
    fig.text(0.01, 0.03, "Dotted lines in B-F: search window (0.2 m to where the 200 MHz signal falls to 1.3 x its floor). Evidence score: 2 m coherent stack, across-strip Kirchhoff focusing,\n"
             "spot-minus-line contrast (v2.2). Null: 8 channel-order shuffles of the same piece. lfdr with pi0 = 1 (conservative), monotone in the score. Depth axis: v = 0.10 m/ns ASSUMED.", fontsize=7.2, color=MUTED, va="top")
    out = FIGURES / f"savona_prob_raster_{drive}_{s0}.png"; fig.savefig(out, dpi=160, facecolor=SURF); print("saved", out)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
