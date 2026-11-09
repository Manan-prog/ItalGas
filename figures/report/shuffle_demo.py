"""Report figure: the along-street method on one real line. A: the 19 channels stacked over 2 m, side by side (signed).
B: focused across the street with the channels in their true positions: a spot. C: the same with the channel order shuffled:
the spot is gone. Drive 45 (2025-10-13 10:01 Swath046), its first line. Usage: python shuffle_demo.py out.png"""
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
from longitudinal import lateral_migrate  # noqa: E402

SURF, INK, MUTED, BLUE, ORANGE, AQUA, RED = "#fcfcfb", "#1f1f1f", "#6b6b6b", "#2a78d6", "#eb6834", "#1baf7a", "#de1e1e"
DET = load_cfg(str(ROOT / "pipeline/configs/dev_swath_v2_2.yaml")); V = DET["migration"]["velocity_m_ns"]
DRIVE, N, ZMAX = 45, 0, 1.6; U = np.arange(-0.9, 0.9 + 1e-9, 0.04)
m = json.load(open(CITY / f"rline_{DRIVE}.json")); ln = m["lines"][N]; s_mid = 0.5 * (ln["s0"] + ln["s1"])
f = Path(m["file"]); sw = Swath.load(str(f.parent), f.stem); g = swath_geometry(sw, DET["base"]); u_ch = np.nanmedian(g["u2"], 0)
sc = np.nanmedian(g["s2"], 1); idx = np.nonzero((sc > s_mid - 30) & (sc < s_mid + 30))[0]; sl = slice(int(idx[0]), int(idx[-1]) + 1)
pp = preprocess_array(sw, 2, DET, scans=sl); X = pp["X"]; S = g["s2"][sl]; dt = pp["dt_ns"]; nz = int(round(2 * ZMAX / V / dt)); depth = np.arange(nz) * dt * V / 2
st = np.stack([np.nanmean(X[np.abs(S[:, k] - s_mid) <= 1.0, k, :], 0) for k in range(X.shape[1])])[None]        # (1, 19, n_t)
rng = np.random.default_rng(0); perm = rng.permutation(len(u_ch))
true = lateral_migrate(np.nan_to_num(st), u_ch, U, dt, V)[0][:, :nz]; shuf = lateral_migrate(np.nan_to_num(st), u_ch[perm], U, dt, V)[0][:, :nz]
vmax = np.percentile(true, 99.5)
fig, ax = plt.subplots(1, 3, figsize=(13, 4.3), facecolor=SURF, gridspec_kw=dict(wspace=0.25))
o = np.argsort(u_ch); sig = st[0][o][:, :nz]; lim = np.percentile(np.abs(sig), 99)
ax[0].imshow(sig.T, cmap="gray", vmin=-lim, vmax=lim, aspect="auto", extent=[u_ch.min(), u_ch.max(), depth[-1], 0], interpolation="nearest")
ax[0].set_title("1  The 19 channels, averaged over 2 m", loc="left", fontsize=10, color=INK)
for a, img, t in ((ax[1], true, "2  Focused, true positions: a spot"), (ax[2], shuf, "3  Focused, shuffled order: no spot")):
    a.imshow(img.T, cmap="gray_r", vmin=0, vmax=vmax, aspect="auto", extent=[U[0], U[-1], depth[-1], 0], interpolation="nearest"); a.set_title(t, loc="left", fontsize=10, color=INK)
ax[1].scatter([ln["u_mean"]], [ln["depth_m"]], s=260, facecolors="none", edgecolors=RED, linewidths=2.2)
for a in ax:
    a.set_facecolor(SURF); a.invert_xaxis(); a.set_xlabel("across the street (m, + left of travel)", fontsize=9); a.tick_params(labelsize=8.5, colors=MUTED)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
ax[0].set_ylabel("depth (m)", fontsize=9)
fig.text(0.01, -0.03, f"Drive {m['swath']} of 2025-10-13 10:01 at {s_mid:.0f} m along the drive; line {N} of the page (depth {ln['depth_m']:.2f} m, evidence P {ln['P_mean']:.2f}). Focusing across the channels at 0.10 m/ns.", fontsize=8.3, color=MUTED)
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "shuffle_demo.png", dpi=170, facecolor=SURF, bbox_inches="tight"); print("ok")
