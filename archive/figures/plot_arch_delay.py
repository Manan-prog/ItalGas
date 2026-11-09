"""Figure (E28): can the across-array arch of a pipe be measured by TIMING each channel's reflection
(cross-correlation), instead of imaging it? Panel A: the test of the method itself on two confirmed
crossing pipes, timed ALONG the street with the same trace pitch as the array: measured reflector
depth per trace (dots) against the arch the pipe must make (dashed). Panels B, C: the same timing
across the array at the two Via Nizza candidates, one line per drive, with the hyperbola and line fits.
Usage (from figures/): python plot_arch_delay.py out.png"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from build_v2 import load_cfg  # noqa: E402

SURF = "#fcfcfb"; ORANGE = "#eb6834"; BLUE = "#2a78d6"; AQUA = "#1baf7a"; INK = "#1f1f1f"; MUTED = "#6b6b6b"
ROOT = Path(__file__).resolve().parents[1]
cfg = load_cfg(str(ROOT / "pipeline/configs/dev_swath_v2_2.yaml")); CACHE = Path(cfg["base"]["cache_dir"])
R = json.load(open(CACHE / "arch_delay_vianizza.json")); C = R["config"]
COLS = [BLUE, ORANGE, AQUA]

fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), facecolor=SURF, gridspec_kw=dict(wspace=0.25))
fig.suptitle("Can a pipe's arch be measured by timing each channel's reflection? No: even over confirmed crossing pipes the timing follows a flat reflector",
             x=0.01, ha="left", fontsize=14, color=INK, weight="bold", y=1.02)
fig.text(0.01, 0.955, "A: confirmed crossing pipes (Swath046, both radar rows, q = 0) timed ALONG the street at the array's trace pitch: dots = timed depth, dashed = arch a pipe at that depth must make.  "
         "B, C: the same timing ACROSS the array at the two Via Nizza candidates, one drive per colour; solid = best hyperbola, dotted = best line.", fontsize=9.5, color=MUTED)
# A: positive control
a = ax[0]; a.set_facecolor(SURF)
for k, p in enumerate(R["positive"]):
    u = np.array(p["u"]); z = np.array(p["z"]); r = np.array(p["corr"]); ok = (z > 0) & (r >= C["min_corr"])
    zh = p["z_hint"]; uu = np.linspace(-0.95, 0.95, 200)
    a.plot(uu, np.sqrt(zh ** 2 + uu ** 2), ls="--", color=COLS[k], lw=1.2)
    a.plot(u[ok], z[ok], "o", color=COLS[k], ms=5, label=f"crossing at s {p['s0_drive']:.0f} m, {zh:.2f} m deep: timed arch {100 * p.get('measured_over_expected_at_edge', float('nan')):.0f}% of expected at the edges")
a.set_ylim(1.3, 0.3); a.set_xlabel("distance along the street from the pipe (m)", fontsize=9); a.set_ylabel("depth of the timed reflection (m)", fontsize=9)
a.set_title("A  Method test on confirmed crossing pipes", loc="left", fontsize=10, color=INK)
a.legend(fontsize=8, frameon=False, loc="lower left")
# B, C: targets
for j, name in enumerate(("place1", "place2")):
    b = ax[j + 1]; b.set_facecolor(SURF); rows = [t for t in R["targets"] if t["name"] == name]
    for k, t in enumerate(rows):
        u = np.array(t["u"]); z = np.array(t["z"]); r = np.array(t["corr"]); ok = (z > 0) & (r >= C["min_corr"])
        err = t.get("pick_sd_m", 0.0)
        b.errorbar(u[ok], z[ok], yerr=err, fmt="o", color=COLS[k], ms=4.5, elinewidth=0.8, capsize=0,
                   label=f"{t['drive']}: hyperbola misfit {t['rms_hyperbola_m']:.3f} m, line misfit {t['rms_line_m']:.3f} m")
        uu = np.linspace(u[ok].min(), u[ok].max(), 200)
        b.plot(uu, np.sqrt(t["z0"] ** 2 + (uu - t["uc"]) ** 2), color=COLS[k], lw=1.0)
        A = np.c_[np.ones(ok.sum()), u[ok]]; coef = np.linalg.lstsq(A, z[ok], rcond=None)[0]; b.plot(uu, coef[0] + coef[1] * uu, color=COLS[k], lw=1.0, ls=":")
    tt = next(x for x in C["targets"] if x["name"] == name)
    b.set_title(f"{'BC'[j]}  {name}: s {tt['s0']:.0f}-{tt['s1']:.0f} m, expected {tt['expect']}", loc="left", fontsize=10, color=INK)
    b.set_ylim(tt["z_hint"] + 0.45, tt["z_hint"] - 0.35); b.set_xlabel("across the street u (m, + left of the reference drive)", fontsize=9); b.set_ylabel("depth of the timed reflection (m)", fontsize=9)
    b.invert_xaxis(); b.legend(fontsize=8, frameon=False, loc="lower left")
for a in ax:
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.tick_params(labelsize=8.5, colors=MUTED)
n_ctrl = sum(v["n"] for v in R["summary"].values()); n_pl = sum(v["pipe_like"] for v in R["summary"].values())
fig.text(0.01, -0.04, f"Timing: median trace across positions removed, then chained adjacent-trace cross-correlation (gate +-{C['chain_gate_half_ns']} ns, step limit {C['chain_max_lag_ns']} ns, x{C['upsample']} resampling); "
         f"per-channel statics from {n_ctrl} random control windows ({n_pl} of them pipe-shaped). Depths at v = 0.10 m/ns (DERIVED). Error bars = pick scatter between 4 m sub-windows.", fontsize=8.5, color=MUTED)
out = sys.argv[1] if len(sys.argv) > 1 else "savona_arch_delay_vianizza.png"
fig.savefig(out, dpi=150, facecolor=SURF, bbox_inches="tight"); print(out)
