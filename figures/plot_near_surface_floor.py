"""Where does high evidence (P >= 0.9) recur across all drives? Justification of the near-surface floor
(configs/raster_lines.yaml t_min_ns, DECISIONS 2026-11-04). Reads every prob_raster_<drive>_<piece>.npz, pools the
cells on the global lateral lattice (U0, 0.04 m) and the common time axis, and plots the rate of cells with P >= 0.9.
Run from figures/: python plot_near_surface_floor.py -> near_surface_floor.png"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "pipeline"))
from mosaic_fast import DU, U0  # noqa: E402

from paths import CACHE, load_yaml  # noqa: E402

CFG = load_yaml(ROOT / "pipeline" / "configs" / "raster_lines.yaml")
P_MIN, T_FLOOR = CFG["p_min"], CFG["t_min_ns"]
BLUE, ORANGE, AQUA, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#fcfcfb"


def rate_map() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """hit[j, k], cells[j, k] on the lattice (j: lateral, k: time sample) over the pieces' own search windows (no floor);
    drives[j, k] = number of drives with at least one hit there. Also dt (ns) and the number of drives."""
    NJ, NK = int(round(10 / DU)) + 1, 512; hit = np.zeros((NJ, NK)); cells = np.zeros((NJ, NK)); dr = {}; dt = None; ids = set()
    for f in sorted(glob.glob(str(CACHE / "prob_raster_*_*.npz"))):
        i = int(re.match(r"prob_raster_(\d+)_", Path(f).name)[1]); ids.add(i); d = np.load(f); t = d["t_ns"]; dt = float(t[1] - t[0])
        it = np.nonzero((t >= d["t_search"][0]) & (t <= d["t_search"][1]))[0]; cs = d["centres_s"]; keep = (cs >= d["c0"]) & (cs < d["c1"])
        h = d["P"][keep][:, :, it].astype(np.float32) >= P_MIN; j0 = int(round((float(d["ug"][0]) - U0) / DU)); k0 = int(it[0])
        hit[j0:j0 + h.shape[1], k0:k0 + len(it)] += h.sum(0); cells[j0:j0 + h.shape[1], k0:k0 + len(it)] += h.shape[0]
        for j, k in zip(*np.nonzero(h.sum(0) > 0)):
            dr.setdefault((j0 + j, k0 + k), set()).add(i)
    drives = np.zeros((NJ, NK), int)
    for (j, k), s in dr.items():
        drives[j, k] = len(s)
    return hit, cells, drives, dt, len(ids)


def main(out: Path) -> None:
    hit, cells, drives, dt, n_drives = rate_map(); js = np.nonzero(cells.sum(1) > 0)[0]; ks = np.nonzero(cells.sum(0) > 0)[0]
    u = U0 + DU * js; t = ks * dt
    centre = js[(u > 0.08) & (u < 0.18)]; edge = js[u > 0.78]; other = np.array([j for j in js if j not in set(centre) | set(edge)])

    def rate(jj, kk):
        return 1e4 * hit[np.ix_(jj, kk)].sum(0) / np.maximum(1, cells[np.ix_(jj, kk)].sum(0))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURF)
    fig.suptitle(f"Where does high evidence (P ≥ {P_MIN}) recur across all {n_drives} drives?  Rate of such cells, pooled over the survey", x=0.01, ha="left", fontsize=12, weight="bold")
    fig.text(0.01, 0.905, f"A: rate by two-way time for three groups of lateral columns (depth ≈ 0.05 m/ns × t, velocity ASSUMED).   "
             f"B: rate by lateral position above and below the floor of {T_FLOOR} ns.", fontsize=9, color="#444")
    a = ax[0]; kk = ks[t <= 30]
    for jj, col, lab in ((centre, ORANGE, "columns at u 0.10–0.16 m (centre antenna)"), (edge, BLUE, "columns at u 0.80–0.86 m (outer antenna)"), (other, AQUA, "all other columns")):
        a.plot(kk * dt, np.maximum(rate(jj, kk), 0.05), color=col, lw=2, label=lab)
    a.axvline(T_FLOOR, color="#777", lw=1, ls="--"); a.text(T_FLOOR + 0.3, 150, f"floor {T_FLOOR} ns\n(not evaluated above it)", fontsize=8, color="#555", va="top")
    a.set_yscale("log"); a.set_xlabel("two-way time (ns)"); a.set_ylabel(f"cells with P ≥ {P_MIN} per 10 000 cells"); a.set_title("A", loc="left", weight="bold")
    a.legend(frameon=False, fontsize=8, loc="upper right")
    b = ax[1]; above = ks[(t >= 3.9) & (t < T_FLOOR)]; below = ks[(t >= T_FLOOR) & (t <= 25)]
    for kk, col, lab in ((above, ORANGE, f"3.9–{T_FLOOR} ns (above the floor)"), (below, BLUE, f"{T_FLOOR}–25 ns (evaluated band)")):
        r = np.array([1e4 * hit[j, kk].sum() / max(1, cells[j, kk].sum()) for j in js]); b.plot(u, np.maximum(r, 0.05), color=col, lw=2, label=lab)
    for y in (0.671, 0.335, 0.0, -0.335, -0.671):
        b.axvline(y + 0.175, color="#bbb", lw=0.8, ls=":")
    b.text(0.175 + 0.02, 0.07, "dotted: antenna columns (u = y + 0.175 m)", fontsize=7.5, color="#777")
    b.set_yscale("log"); b.set_xlabel("lateral position u in the raster (m)"); b.set_title("B", loc="left", weight="bold"); b.legend(frameon=False, fontsize=8, loc="upper center")
    for x in ax:
        x.set_facecolor(SURF); x.grid(True, color="#e6e6e6", lw=0.6); [x.spines[s].set_visible(False) for s in ("top", "right")]; x.spines["left"].set_color("#999"); x.spines["bottom"].set_color("#999")
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(out, dpi=150, facecolor=SURF); print(f"saved {out}")
    print("max number of drives sharing one (lateral, time) cell above the floor: %d; in the evaluated band: %d" % (drives[:, above].max(), drives[:, below].max()))
    print("rate above floor / evaluated band, all columns: %.1f / %.2f per 1e4" % (1e4 * hit[:, above].sum() / cells[:, above].sum(), 1e4 * hit[:, below].sum() / cells[:, below].sum()))


if __name__ == "__main__":
    main(Path(__file__).with_name("near_surface_floor.png"))
