"""Figure: what a confident crossing looks like in the raw radar vs a rejected one.
Each row is one detection; the five left panels are the 200 MHz radargram along the
street under each of the five antenna columns (left of travel to right), the last panel
is the 600 MHz radargram under the centre column. A buried pipe crossing the street
shows the SAME hyperbola in every column, at positions that line up along the pipe
(orange ticks = where the detector expects the top of the hyperbola).
Usage (from figures/): python plot_crossing_evidence_v2.py out.png"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from build_v2 import cfg_hash, load_cfg  # noqa: E402

SURF = "#fcfcfb"; ORANGE = "#eb6834"; BLUE = "#2a78d6"; INK = "#1f1f1f"; MUTED = "#6b6b6b"
ROOT = Path(__file__).resolve().parents[1]
cfg = load_cfg(str(ROOT / "pipeline/configs/dev_swath_v2_2.yaml")); CACHE = Path(cfg["base"]["cache_dir"])
h = cfg_hash(cfg)
D = {a: np.load(CACHE / f"v2grid_{a}_{h}.npz") for a in (1, 2)}
s = D[1]["s_grid"]; V = cfg["migration"]["velocity_m_ns"]
COLS = [(0.85, "far left"), (0.51, "left"), (0.17, "centre"), (-0.16, "right"), (-0.50, "far right")]
ROWS = [  # (s0 m, phi deg, depth m, verdict line) from final_v2 / xing json (q, P printed there)
    (843.27, 5.0, 0.48, "CONFIDENT (P = 1.00, both radar rows): same hyperbola in all five columns, crossing at right angles"),
    (60.38, -40.0, 0.62, "CONFIDENT (P = 1.00, both radar rows): oblique pipe, the hyperbola moves along the street from column to column"),
    (437.05, 15.0, 0.66, "PROBABLE TRENCH, NOT A CLEAN PIPE (P = 0.91, 600 MHz only): no hyperbola, but the layers break along one line across the street"),
    (172.70, 30.0, 0.47, "REJECTED (q = 0.51): bright in some columns but not lined up across the array, as often happens by chance"),
]
HALF = 3.0
fig, ax = plt.subplots(len(ROWS), 6, figsize=(20, 3.3 * len(ROWS) + 1.2), facecolor=SURF,
                       gridspec_kw=dict(hspace=0.62, wspace=0.12, left=0.04, right=0.995, top=0.83, bottom=0.05))
fig.suptitle("Why some detections are trusted and others are not: the same buried pipe must appear under every antenna column",
             x=0.01, ha="left", fontsize=14, color=INK)
fig.text(0.01, 0.935, "Each row = one detection. Panels 1 to 5: 200 MHz radargram along the street under each antenna column "
         "(left to right across the van). Panel 6: 600 MHz under the centre column. Orange tick and dotted line = where the detector "
         "expects the top of the pipe's hyperbola; grey dashed = detected depth. Noise-equalised amplitude, clipped at 4 times the "
         "typical level at each depth. Grey vertical band in D = scans blanked because of radio interference.", fontsize=9.5, color=MUTED, wrap=True)
for i, (s0, phi, dep, verdict) in enumerate(ROWS):
    for j in range(6):
        a = 2 if j < 5 else 1
        u = D[a]["u_ch"]; dt = float(D[a]["dt_full_ns"]); Z = D[a]["Z"]
        uc, name = COLS[j] if j < 5 else (0.17, "centre")
        c = int(np.argmin(np.abs(u - uc)))
        sc_ = s0 + u[c] * np.tan(np.radians(phi))
        w = (s > sc_ - HALF) & (s < sc_ + HALF)
        img = Z[c, w].astype(np.float32).T
        A = ax[i, j]
        A.imshow(img, cmap="gray", vmin=-4, vmax=4, aspect="auto", extent=[sc_ - HALF, sc_ + HALF, img.shape[0] * dt * V / 2, 0])
        A.axvline(sc_, color=ORANGE, lw=0.8, ls=":")
        A.plot([sc_], [0.04], marker="v", color=ORANGE, ms=8, clip_on=False)
        A.axhline(dep, color="#bbbbbb", lw=0.7, ls="--")
        A.set_ylim(1.6 if a == 2 else 1.25, 0)
        A.set_title(f"{'200' if a == 2 else '600'} MHz, {name} column (u = {u[c]:+.2f} m)", fontsize=8.5, loc="left", color=INK)
        A.tick_params(colors=MUTED, labelsize=7.5)
        for sp in A.spines.values():
            sp.set_color("#cfcfcf")
        if j == 0:
            A.set_ylabel("depth (m)", color=MUTED, fontsize=8.5)
        if i == len(ROWS) - 1:
            A.set_xlabel("along the street (m)", color=MUTED, fontsize=8.5)
    col = ORANGE if verdict.startswith("CONF") else (BLUE if verdict.startswith("PROB") else MUTED)
    ax[i, 0].text(0.0, 1.32, f"{'ABCD'[i]}  s = {s0:.0f} m, {dep:.2f} m deep:  {verdict}", transform=ax[i, 0].transAxes,
                  fontsize=10.5, color=col, fontweight="bold")
fig.savefig(sys.argv[1], dpi=150, facecolor=SURF)
print("saved", sys.argv[1])
