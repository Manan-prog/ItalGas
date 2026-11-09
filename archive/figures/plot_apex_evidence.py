"""Figure: do the two Via Nizza along-road candidates (E24, two-drive apex agreement) look like
pipes in the raw radar of EACH drive separately?
Each row is one drive at one place. Left: across-street section of that drive alone (200 MHz,
coherent stack over +-1 m along the street at the feature's midpoint, envelope), with the pipe arch
z(u) = sqrt(z0^2 + (u - uc)^2) fitted to that drive alone (orange) and its apex (cross). Right: the
signed radargram along the street of the single channel of that drive that passed closest to the
fitted apex; dashed line = fitted depth, bar = extent of the repeated feature.
Usage (from figures/): python plot_apex_evidence.py out.png"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from build_v2 import load_cfg  # noqa: E402
from register_multipass import cross_section  # noqa: E402

SURF = "#fcfcfb"; ORANGE = "#eb6834"; BLUE = "#2a78d6"; AQUA = "#1baf7a"; INK = "#1f1f1f"; MUTED = "#6b6b6b"
ROOT = Path(__file__).resolve().parents[1]
cfg = load_cfg(str(ROOT / "pipeline/configs/dev_swath_v2_2.yaml")); CACHE = Path(cfg["base"]["cache_dir"])
V = cfg["migration"]["velocity_m_ns"]                      # m/ns, DERIVED (E03)
REG = json.load(open(CACHE / "mp_registration.json"))["registration"]
APEX = json.load(open(CACHE / "mp_apex.json"))["features"]
agree = [f for f in APEX if f.get("agree")]

# The two places, built from the agreeing features (drive -> (uc, z0) fitted in that drive alone).
PLACES = []
for key, title in (("046", "PLACE 1: s 2876-2914 m, u about +4.4 m, 0.68 m deep, seen by 3 drives (2 drive pairs)"),
                   ("041", "PLACE 2: s 2915-2950 m, u about +0.1 m, 0.63 m deep, seen by 1 drive pair")):
    fs = [f for f in agree if f["a"].endswith(key)]
    f0 = max(fs, key=lambda f: f["s1"] - f["s0"])
    drives = {}
    for f in fs:
        for k, n in enumerate((f["a"], f["b"])):
            drives.setdefault(n, dict(uc=f["uc"][k], z0=f["z0"][k], s0=f["s0"], s1=f["s1"], partner=f["b"] if k == 0 else f["a"]))
    PLACES.append(dict(title=title, s0=f0["s0"], s1=f0["s1"], u=f0["u"], z=f0["z0"][0], drives=drives))

rows = [(p, n, d) for p in PLACES for n, d in p["drives"].items()]
D = {n: np.load(CACHE / f"mp_{n}_2.npz") for n in {n for _, n, _ in rows}}
u_grid = np.arange(-3.5, 6.5 + 1e-9, 0.04)
HALF_W = 0.8; PAD = 10.0

fig, ax = plt.subplots(len(rows), 2, figsize=(19, 3.0 * len(rows) + 1.4), facecolor=SURF,
                       gridspec_kw=dict(width_ratios=[1, 2.6], hspace=0.95, wspace=0.10))
fig.suptitle("Via Nizza: do the two along-road candidates look like a pipe in each drive on its own?",
             x=0.01, ha="left", fontsize=15, color=INK, weight="bold", y=0.995)
fig.text(0.01, 0.972, "Left: across-street section of ONE drive (200 MHz, +-1 m stack, envelope) with the arch fitted to that drive (orange) and its apex (cross). "
         "Right: signed radargram along the street of that drive's channel closest to the apex; dashed = fitted depth; bar = repeated feature.",
         fontsize=10.5, color=MUTED)
letters = "ABCDEFGH"
for r, (p, n, d) in enumerate(rows):
    g = REG[n]; z = D[n]; dt = float(z["dt"]); s_grid = z["s_grid"]; s_mid = 0.5 * (d["s0"] + d["s1"])
    # left: across-street section of this drive alone at s_mid (true frame), as used by the apex test
    X, depth = cross_section({n: z}, REG, s_grid, u_grid, s_mid, V)
    cz = depth <= 1.8
    a = ax[r, 0]; a.set_facecolor(SURF)
    sec = X[cz]; vmax = np.nanpercentile(sec, 99.5)
    a.imshow(sec, aspect="auto", cmap="gray_r", vmin=0, vmax=vmax, extent=[u_grid[0], u_grid[-1], depth[cz][-1], 0], interpolation="nearest")
    uu = np.linspace(d["uc"] - HALF_W, d["uc"] + HALF_W, 200)
    a.plot(uu, np.sqrt(d["z0"] ** 2 + (uu - d["uc"]) ** 2), color=ORANGE, lw=1.4)
    a.plot(d["uc"], d["z0"], "+", color=ORANGE, ms=14, mew=2)
    a.set_xlim(d["uc"] + 1.2, d["uc"] - 1.2)                  # u positive to the left of travel, shown on the left
    a.set_ylim(1.8, 0); a.set_xlabel("across the street u (m, + left of the reference drive)", fontsize=9); a.set_ylabel("depth (m)", fontsize=9)
    a.set_title(f"{letters[r]}  {n}: arch at u = {d['uc']:+.2f} m, {d['z0']:.2f} m\n(partner {d['partner']} agrees within {abs(d['uc'] - p['drives'][d['partner']]['uc']):.2f} m)",
                fontsize=10, loc="left", color=INK)
    # right: along-street signed radargram of the channel of this drive nearest the fitted apex
    s_true = s_grid - g["A"]; ws = (s_true > d["s0"] - PAD) & (s_true < d["s1"] + PAD)
    U_true = z["Ug"][:, ws] - g["B"]; uch = np.nanmedian(U_true, 1); c = int(np.nanargmin(np.abs(uch - d["uc"])))
    B = z["Z"][c][ws].astype(np.float32); tz = np.arange(B.shape[1]) * dt * V / 2; ct = tz <= 1.8
    b = ax[r, 1]; b.set_facecolor(SURF)
    lim = np.nanpercentile(np.abs(B[:, ct]), 99)
    b.imshow(B[:, ct].T, aspect="auto", cmap="gray", vmin=-lim, vmax=lim, extent=[s_true[ws][0], s_true[ws][-1], tz[ct][-1], 0], interpolation="nearest")
    b.axhline(d["z0"], color=ORANGE, lw=1.0, ls="--")
    b.plot([d["s0"], d["s1"]], [1.72, 1.72], color=ORANGE, lw=3, solid_capstyle="butt")
    b.set_ylim(1.8, 0); b.set_xlabel("along the street s (m, reference drive frame)", fontsize=9)
    b.set_title(f"{n}: channel at u = {uch[c]:+.2f} m; this drive sits {g['A']:+.2f} m along and {g['B']:+.2f} m across from the reference drive",
                fontsize=10, loc="left", color=MUTED)
    b.tick_params(labelleft=False)
    if n == next(iter(p["drives"])):
        fig.text(0.01, ax[r, 0].get_position().y1 + 0.028, p["title"], fontsize=12, color=BLUE, weight="bold")
for a in ax.ravel():
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.tick_params(labelsize=8.5, colors=MUTED)
fig.text(0.01, 0.004, "Data: 13 Oct 2025 session 10-01-51, drives registered to Swath041 (E16); depths at v = 0.10 m/ns (DERIVED, E03); "
         "positions in the reference drive's frame, real-time RTK track, about 1 m absolute uncertainty. No utility-company ground truth.", fontsize=8.5, color=MUTED)
out = sys.argv[1] if len(sys.argv) > 1 else "savona_apex_evidence_vianizza.png"
fig.savefig(out, dpi=150, facecolor=SURF, bbox_inches="tight")
print(out, [(n, round(d["uc"], 2), round(d["z0"], 2)) for _, n, d in rows])
