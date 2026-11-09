"""E45 figure: after registration, do drives in different lanes show along-street evidence at the same ground position?
Inputs: cache/mosaic_test.npz + .json (pipeline/mosaic_test.py), cache/mp_frame.npz (reference frame of Swath041).
Run (from figures/): python plot_mosaic_test.py -> figures/savona_mosaic_test_nizza.png"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import maximum_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from basemap_tools import street_aligned_tiles  # noqa: E402
from paths import CACHE  # noqa: E402

SURF, INK, MUTED, BLUE, ORANGE, AQUA = "#fcfcfb", "#1f1f1f", "#6b6b6b", "#2a78d6", "#eb6834", "#1baf7a"
CMAP = LinearSegmentedColormap.from_list("rg", ["#c0392b", "#efe9dc", "#1a9850"])


def main() -> None:
    d = np.load(CACHE / "mosaic_test.npz"); J = json.load(open(CACHE / "mosaic_test.json")); s, u = d["s_grid"], d["u_grid"]
    passes = list(J["passes"]); drive = J["passes"]; T5 = J["thresholds"]["0.5"]; T9 = J["thresholds"]["0.9"]
    Rmax = np.full((len(s), len(u)), np.nan)
    for p in passes:
        R = np.where(d[f"cov_{p}"], d[f"R_{p}"], np.nan); Rmax = np.fmax(Rmax, R)
    Rshow = maximum_filter(np.nan_to_num(Rmax, nan=-1), size=(1, 3)); Rshow = np.where(np.isfinite(Rmax), Rshow, np.nan)   # display: 12 cm widening
    agree = np.zeros_like(Rmax, bool)
    for p in passes:
        agree |= d[f"agree_{p}"]
    fig = plt.figure(figsize=(14, 11), facecolor=SURF)
    fig.suptitle("After registration, do drives in different lanes see along-street evidence at the same ground position?  Via Nizza, 6 drives", x=0.01, ha="left", fontsize=12.5, color=INK, weight="bold")
    fig.text(0.01, 0.957, "A: the six registered strips in the frame of drive 40 (across-street axis stretched 8x). Colour = P, the share of cells with this much evidence that pipe-free (antenna-shuffled) data does not produce.\n"
             "Blue = evidence repeated by a drive in another lane (strip centres >= 0.5 m apart) within 10 cm. B: satellite ortho detail of the longest repeated stretch. C: repeat fraction against two pipe-free references.",
             fontsize=8.3, color=MUTED, va="top")
    # A: registered plan mosaic
    ax = fig.add_axes([0.05, 0.56, 0.93, 0.33]); ax.set_facecolor("#e9e9e9")
    ax.imshow(Rshow.T, extent=[s[0], s[-1], u[0], u[-1]], origin="lower", aspect="auto", cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    ys, xs = np.nonzero(agree.T); ax.scatter(s[xs], u[ys], s=1.5, color=BLUE, lw=0, zorder=4)
    for p in passes:
        c = d[f"cen_{p}"]; ax.plot(s, c, color="white", lw=0.6, ls=(0, (2, 2)), zorder=3); ax.text(s[0] + 3, np.nanmedian(c), f"drive {drive[p]}", color="white", fontsize=7.5, va="center", zorder=5, bbox=dict(fc="black", ec="none", alpha=0.35, pad=1))
    best = T5["runs"][0] if T5["runs"] else None
    if best:
        ax.add_patch(plt.Rectangle((best["s0"] - 20, best["u_median"] - 1.2), best["length_m"] + 40, 2.4, fill=False, ec="white", lw=1.2, zorder=6)); ax.text(best["s0"] - 20, best["u_median"] + 1.3, "B", color="white", fontsize=10, fontweight="bold", zorder=6)
    ax.set_xlim(s[0], s[-1]); ax.set_ylim(u[0], u[-1]); ax.set_xlabel("metres along the reference drive (40)", fontsize=9); ax.set_ylabel("across the street (m, left of drive 40 +)", fontsize=9); ax.tick_params(labelsize=8)
    ax.set_title(f"A  Registered mosaic: {T5['agree_rows_m']:.0f} m of the {s[-1]-s[0]:.0f} m window carry evidence repeated from another lane (P >= 0.5); {len(T5['runs'])} stretches >= 10 m", loc="left", fontsize=10.5, color=INK)
    # B: ortho detail around the longest run
    fr = dict(np.load(CACHE / "mp_frame.npz")); k = float(fr["k"])
    if best:
        b0, b1 = best["s0"] - 20, best["s1"] + 20; half_w = 4.0
        img, ext, _, _ = street_aligned_tiles(b0, b1, half_w, fr["s_nodes"], fr["xn"], fr["yn"], k, zoom=21)
        ax = fig.add_axes([0.05, 0.27, 0.93, 0.24]); ax.imshow(img, extent=ext, origin="upper", aspect="equal", zorder=1)
        m = (s >= b0) & (s <= b1); mu = (u >= -half_w) & (u <= half_w)
        ax.imshow(np.ma.masked_invalid(Rshow[m][:, mu]).T, extent=[s[m][0] - 0.25, s[m][-1] + 0.25, u[mu][0] - 0.02, u[mu][-1] + 0.02], origin="lower", aspect="equal", cmap=CMAP, vmin=0, vmax=1, alpha=0.8, interpolation="nearest", zorder=3)
        A = agree[m][:, mu]; ys, xs = np.nonzero(A.T); ax.scatter(s[m][xs], u[mu][ys], s=6, color=BLUE, lw=0, zorder=5)
        for p in passes:
            c = d[f"cen_{p}"][m]
            if np.nanmedian(np.abs(c)) < half_w:
                ax.plot(s[m], c, color="white", lw=0.6, ls=(0, (2, 2)), zorder=4); ax.text(b0 + 0.5, np.nanmedian(c), f"drive {drive[p]}", color="white", fontsize=7, va="center", zorder=6, bbox=dict(fc="black", ec="none", alpha=0.35, pad=1))
        ax.plot([b0 + 2, b0 + 12], [-half_w + 0.6, -half_w + 0.6], color="white", lw=4, zorder=6, solid_capstyle="butt"); ax.text(b0 + 7, -half_w + 1.0, "10 m", color="white", ha="center", fontsize=9, fontweight="bold", zorder=6)
        ax.set_xlim(b0, b1); ax.set_ylim(-half_w, half_w); ax.set_xlabel("metres along drive 40", fontsize=9); ax.set_ylabel("across (m)", fontsize=9); ax.tick_params(labelsize=8)
        ax.set_title(f"B  Longest repeated stretch: s {best['s0']:.0f}-{best['s1']:.0f} m ({best['length_m']:.0f} m), u {best['u_median']:+.2f} m, depth {best['depth_median']:.2f} m (v 0.10 ASSUMED), drives {', '.join(str(drive[p]) for p in best['passes'])}", loc="left", fontsize=10.5, color=INK)
    # C: statistic vs nulls
    ax = fig.add_axes([0.05, 0.06, 0.42, 0.15]); ax.set_facecolor(SURF)
    fn = d["f_shift"]; ax.hist(fn[np.isfinite(fn)], bins=30, color="#cfcfcf", label="shift reference (200 draws): other drives moved 20-200 m along the street")
    ax.axvline(T5["f_mirror"], color=ORANGE, lw=2, label=f"mirror reference (strips reflected about their centre): {T5['f_mirror']:.2f}")
    ax.axvline(T5["f_real"], color=AQUA, lw=2.5, label=f"real data: {T5['f_real']:.2f} of {T5['n_tested']} tested cells repeated (p = {T5['p_shift']:.3f} against shift)")
    ax.set_xlabel("fraction of evidence cells (P >= 0.5) repeated by a drive in another lane", fontsize=8.5); ax.set_yticks([]); ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.3, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    ax.set_title(f"C  Repeat fraction against pipe-free references (at P >= 0.9: real {T9['f_real']:.2f} of {T9['n_tested']}, mirror {T9['f_mirror']:.2f}, p = {T9['p_shift']:.2f})", loc="left", fontsize=10.5, color=INK)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, 1)); cax = fig.add_axes([0.62, 0.075, 0.3, 0.015]); cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("P: share of cells with this much evidence that pipe-free data does not produce", fontsize=8); cb.ax.tick_params(labelsize=7.5)
    fig.text(0.01, 0.004, "Registration: E16 (cross-track residual 0.04 m, along-track 0.23 m RMS). Evidence: 200 MHz, option C cleaning, v2.2 along-street score, 8 antenna-order shuffles per 600 m piece, local FDR with pi0 = 1.\n"
             "Drives 36 and 39 have no RTK fix; their placement relies on the E16 overlap registration. Red/green as requested (not colour-blind safe); a neutral midpoint separates the two.", fontsize=7, color=MUTED, va="bottom")
    out = ROOT / "figures" / "savona_mosaic_test_nizza.png"; fig.savefig(out, dpi=160, facecolor=SURF); print("saved", out)


if __name__ == "__main__":
    main()
