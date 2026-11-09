"""E42 figure: do the along-street tracks move with the ground (slope b = -1) or with the van (b = 0)?
Inputs: cache/city/dither_test.json (real tracks), cache/dither_injection_<win>.json (controls G / V).
Run: python plot_dither_test.py -> figures/savona_dither_test.png"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; CACHE = Path.home() / "ItalGas" / "cache"
sys.path.insert(0, str(ROOT / "pipeline"))
from dither import pooled_b, slope, y_at  # noqa: E402

BLUE, ORANGE, AQUA, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#1f1f1f", "#8a8a8a", "#fcfcfb"


def detrended(ps, pu, yv):
    """Residuals of pu and yv after removing offset + trend in s (the quantities whose ratio is b)."""
    A = np.c_[np.ones(len(ps)), ps - ps.mean()]; P = A @ np.linalg.pinv(A)
    return yv - P @ yv, pu - P @ pu


def main() -> None:
    T = json.load(open(CACHE / "city" / "dither_test.json")); S = T["summary"]
    ctrl = {"G": [], "V": []}
    for f in sorted(glob.glob(str(CACHE / "dither_injection_*.json"))):
        d = json.load(open(f))
        for mode in ("G", "V"):
            for r in d["runs"].get(mode, {}).get("results", []):
                if r["found"] and r["test"].get("n_fixed", 0) >= 5 and "b" in r["test"]:
                    ctrl[mode].append(dict(r["test"], win=d["window"]))
    fig = plt.figure(figsize=(13, 8.6), facecolor=SURF)
    fig.suptitle("Do the along-street tracks move with the ground (slope b = -1) or with the van (b = 0)?", x=0.01, ha="left", fontsize=13, color=INK, weight="bold")
    fig.text(0.01, 0.95, "A-C: detrended instrument position of the track against the van's sideways position, with the fitted slope (each track drawn as its +-1 sd end points).\n"
             "D: slope of every track against how much the van moved sideways under it.  E: pooled slope of the real tracks against chance (van motion read 50-500 m away).",
             fontsize=8.5, color=MUTED, va="top")
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.985, top=0.88, bottom=0.12, hspace=0.45, wspace=0.28)
    # A-C: pooled detrended scatter
    panels = [("A  Fake pipes fixed in the GROUND", ctrl["G"], BLUE), ("B  Fake pipes fixed to the VAN", ctrl["V"], ORANGE), ("C  Real tracks (58, 12 drives)", None, AQUA)]
    for k, (title, rows, col) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k]); ax.set_facecolor(SURF)
        if rows is None:                                      # real tracks: recompute detrended points from the saved paths? not saved -> use per-track b and sd_d
            xs, ys = [], []
            for r in T["tracks"]:
                f = r["sg2_60"]
                if f.get("n_fixed", 0) >= 5 and "b" in f and np.isfinite(f["b"]):
                    # represent each track by its two sufficient statistics: spread of y (sd_d) and implied spread of u (b sd_d)
                    xs += [-f["sd_d"], f["sd_d"]]; ys += [-f["b"] * f["sd_d"], f["b"] * f["sd_d"]]
            xs, ys = np.array(xs), np.array(ys); bp = S["pooled"]["all"]["b_pool"]
            ax.scatter(xs, ys, s=9, color=col, alpha=0.55, lw=0, label="tracks (+-1 sd end points)")
            lab = f"pooled slope b = {bp:+.2f} (95% {S['pooled']['all']['ci95'][0]:+.2f} to {S['pooled']['all']['ci95'][1]:+.2f})"
        else:
            xs = np.array([x for r in rows for x in (-r["sd_d"], r["sd_d"])]); ys = np.array([y for r in rows for y in (-r["b"] * r["sd_d"], r["b"] * r["sd_d"])])
            bp = pooled_b(rows); ax.scatter(xs, ys, s=16, color=col, alpha=0.7, lw=0)
            lab = f"pooled slope b = {bp:+.2f} ({len(rows)} recovered fakes)"
        lim = max(0.05, float(np.max(np.abs(xs))) * 1.15) if len(xs) else 0.3
        xx = np.array([-lim, lim]); ax.plot(xx, -xx, color=MUTED, lw=1, ls="--", label="ground-fixed (b = -1)"); ax.plot(xx, 0 * xx, color=MUTED, lw=1, ls=":", label="van-fixed (b = 0)")
        if np.isfinite(bp):
            ax.plot(xx, bp * xx, color=col, lw=2, label=lab)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_xlabel("van sideways position, detrended (m)", fontsize=8.5); ax.set_ylabel("track position in the instrument, detrended (m)", fontsize=8.5)
        ax.set_title(title, fontsize=10, loc="left", color=INK); ax.tick_params(labelsize=8); ax.legend(fontsize=7.2, loc="upper right", frameon=False)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    # D: b vs sd_d
    ax = fig.add_subplot(gs[1, :2]); ax.set_facecolor(SURF)
    for mode, col, lab in (("G", BLUE, "fake, ground-fixed"), ("V", ORANGE, "fake, van-fixed")):
        ax.scatter([r["sd_d"] for r in ctrl[mode]], [r["b"] for r in ctrl[mode]], s=34, color=col, alpha=0.85, lw=0, label=lab, zorder=3)
    rr = [r["sg2_60"] for r in T["tracks"] if r["sg2_60"].get("n_fixed", 0) >= 5 and "b" in r["sg2_60"]]
    cl = [r["cluster"] for r in T["tracks"] if r["sg2_60"].get("n_fixed", 0) >= 5 and "b" in r["sg2_60"]]
    ax.scatter([r["sd_d"] for r, c in zip(rr, cl) if not c], [r["b"] for r, c in zip(rr, cl) if not c], s=22, color=AQUA, alpha=0.8, lw=0, label="real track", zorder=2)
    ax.scatter([r["sd_d"] for r, c in zip(rr, cl) if c], [r["b"] for r, c in zip(rr, cl) if c], s=30, facecolor="none", edgecolor=AQUA, lw=1.2, label="real track, leftmost-antenna cluster", zorder=2)
    ax.axhline(-1, color=MUTED, lw=1, ls="--"); ax.axhline(0, color=MUTED, lw=1, ls=":"); ax.axvline(0.05, color=MUTED, lw=0.8, ls="-")
    ax.text(0.051, 2.6, "pre-registered\ntestable limit", fontsize=7.5, color=MUTED, va="top")
    ax.set_xscale("log"); ax.set_xlim(0.002, 1.0); ax.set_ylim(-3, 3)
    ax.set_xlabel("sideways motion of the van under the track, sd after detrending (m, log)", fontsize=9); ax.set_ylabel("slope b", fontsize=9)
    ax.set_title("D  Slope of every track; real tracks sit where the van barely moved, so single slopes are noisy", fontsize=10, loc="left", color=INK)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False, ncol=2); ax.tick_params(labelsize=8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    # E: pooled null
    ax = fig.add_subplot(gs[1, 2]); ax.set_facecolor(SURF)
    nulls = []
    bn = np.array(T.get("b_null", []), float)
    pa = S["pooled"]["all"]
    # reconstruct pooled null draws are not stored; show the per-track shift null spread instead via summary numbers
    ax.axvspan(pa["ci95"][0], pa["ci95"][1], ymax=0.55, color=AQUA, alpha=0.15, lw=0, label="95% bootstrap interval, real")
    ax.axvline(pa["b_pool"], ymax=0.55, color=AQUA, lw=2.2, label=f"real tracks, pooled b = {pa['b_pool']:+.2f}\n(chance mean {pa['null_mean']:+.2f}, 5th pct {pa['null_q05']:+.2f}, p = {pa['p']:.3f})")
    if ctrl["G"]: ax.axvline(pooled_b(ctrl["G"]), ymax=0.55, color=BLUE, lw=2.2, label=f"fakes in the ground, b = {pooled_b(ctrl['G']):+.2f}")
    if ctrl["V"]: ax.axvline(pooled_b(ctrl["V"]), ymax=0.55, color=ORANGE, lw=2.2, label=f"fakes on the van, b = {pooled_b(ctrl['V']):+.2f}")
    ax.axvline(-1, ymax=0.55, color=MUTED, lw=1, ls="--"); ax.axvline(0, ymax=0.55, color=MUTED, lw=1, ls=":")
    ax.set_xlim(-1.6, 0.8); ax.set_ylim(0, 1); ax.set_yticks([]); ax.set_xlabel("pooled slope b", fontsize=9)
    ax.set_title("E  Pooled slope against chance", fontsize=10, loc="left", color=INK); ax.legend(fontsize=7.2, loc="upper left", frameon=False)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    fig.text(0.01, 0.008, "Van sideways position: 200 MHz array centre from the real-time RTK track against a 60 m local-quadratic reference line; RTK-fixed points only. "
             "Slope b from path_u = a0 + a1 s + b y_van per track; pooled b = sum(w b) / sum(w), w = n sd_d^2. Along-travel bias 1.19 m ASSUMED. "
             f"Positioning: RT minus post-processed sideways noise {S['positioning_noise']['sd_rt_minus_pp_30m_median_m']*100:.1f} cm (30 m windows), attenuation of b {S['positioning_noise']['attenuation']:.2f}.",
             fontsize=7.2, color=MUTED, wrap=True)
    out = ROOT / "figures" / "savona_dither_test.png"; fig.savefig(out, dpi=170, facecolor=SURF); print("saved", out)


if __name__ == "__main__":
    main()
