"""Report figure: one crossing pipe through the processing chain. Raw traces -> cleaned -> migrated (focused) -> detection.
Drive 45 (2025-10-13 10:01 Swath046), crossing 217 (0.40 m). Usage: python processing_steps.py out.png"""
import json
import sys
from pathlib import Path
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipeline"))
from build_cube import swath_geometry  # noqa: E402
from build_v2 import load_cfg, preprocess_array  # noqa: E402
from ids_reader import Swath  # noqa: E402
from paths import CACHE, CITY  # noqa: E402
from processing import envelope, kirchhoff_migrate  # noqa: E402

SURF, INK, MUTED, BLUE, ORANGE, AQUA = "#fcfcfb", "#1f1f1f", "#6b6b6b", "#2a78d6", "#eb6834", "#1baf7a"
DET = load_cfg(str(ROOT / "pipeline/configs/dev_swath_v2_2.yaml")); V = DET["migration"]["velocity_m_ns"]; AP = DET["migration"]["aperture_m"]
DRIVE, J, HALF, ZMAX = 45, 217, 4.0, 1.6
R = json.load(open(CITY / f"scan_{DRIVE}.json")); c = R["crossings"][J]; s0 = c["s0_m"]
f = Path(R["file"]); sw = Swath.load(str(f.parent), f.stem); g = swath_geometry(sw, DET["base"]); u_ch = np.nanmedian(g["u2"], 0); ch = int(np.argmin(np.abs(u_ch)))
sc = g["s2"][:, ch]; idx = np.nonzero((sc > s0 - 30) & (sc < s0 + 30))[0]; sl = slice(int(idx[0]), int(idx[-1]) + 1)
pp = preprocess_array(sw, 2, DET, scans=sl); X = pp["X"][:, ch]; dt = pp["dt_ns"]; dx = pp["dx_m"]
mig = envelope(kirchhoff_migrate(X, dx, dt, V, AP)); s_sl = sc[sl]; w = (s_sl > s0 - HALF) & (s_sl < s0 + HALF)
raw = np.asarray(sw.data[2][sl, ch, :]).astype(np.float32); raw -= raw.mean(1, keepdims=True); dt_raw = sw.config.channels_of(2)[ch].dt_ns
nz = int(round(2 * ZMAX / V / dt)); depth = np.arange(nz) * dt * V / 2; ext = [-HALF, HALF, depth[-1], 0]
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURF, gridspec_kw=dict(wspace=0.22))
t_raw = np.arange(raw.shape[1]) * dt_raw; ct = t_raw <= 40
rw = raw[w][:, ct]; lim = np.percentile(np.abs(rw), 99)
ax[0].imshow(rw.T, cmap="gray", vmin=-lim, vmax=lim, aspect="auto", extent=[-HALF, HALF, t_raw[ct][-1], 0], interpolation="nearest")
ax[0].set_ylabel("recorded time (ns)", fontsize=9); ax[0].set_title("1  As recorded", loc="left", fontsize=10, color=INK)
cl = X[w][:, :nz]; lim = np.percentile(np.abs(cl), 99)
ax[1].imshow(cl.T, cmap="gray", vmin=-lim, vmax=lim, aspect="auto", extent=ext, interpolation="nearest")
ax[1].set_ylabel("depth (m)", fontsize=9); ax[1].set_title("2  Cleaned: the arch appears", loc="left", fontsize=10, color=INK)
mg = mig[w][:, :nz]; ax[2].imshow(mg.T, cmap="gray_r", vmin=0, vmax=np.percentile(mg, 99.5), aspect="auto", extent=ext, interpolation="nearest")
ax[2].scatter([0], [c["depth_m"]], s=260, facecolors="none", edgecolors="#de1e1e", linewidths=2.2); ax[2].set_title(f"3  Focused: a spot at {c['depth_m']:.2f} m", loc="left", fontsize=10, color=INK)
for a in ax:
    a.set_facecolor(SURF); a.set_xlabel("metres along the street from the pipe", fontsize=9); a.tick_params(labelsize=8.5, colors=MUTED)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
fig.text(0.01, -0.03, f"Drive {R['swath']} of 2025-10-13 10:01, 200 MHz centre channel. Cleaning = remove the average trace, filter, equalise the noise with depth. Focusing = Kirchhoff migration at 0.10 m/ns (derived from the data).", fontsize=8.3, color=MUTED)
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "processing_steps.png", dpi=170, facecolor=SURF, bbox_inches="tight"); print("ok")
