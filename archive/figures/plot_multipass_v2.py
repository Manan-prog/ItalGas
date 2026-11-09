"""Figure: six passes over one street merged into one radar map after registration with
crossing pipes and overlap correlation; do pipes running along the road appear?
Usage (from figures/): python plot_multipass_v2.py out.png s0 s1"""
import json
import sys
from pathlib import Path

import contextily as cx
import matplotlib
import numpy as np
import yaml
from scipy.ndimage import label
from scipy.signal import hilbert

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from basemap_tools import GSAT  # noqa: E402
from build_v2 import load_cfg  # noqa: E402
from geometry import merc_to_lonlat  # noqa: E402
from register_multipass import line_evidence, place_interp, slice_map  # noqa: E402
from run_detection import su_to_map  # noqa: E402

SURF = "#fcfcfb"; BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"; INK = "#1f1f1f"; MUTED = "#6b6b6b"
ROOT = Path(__file__).resolve().parents[1]
mc = yaml.safe_load(open(ROOT / "pipeline/configs/multipass_aurelia.yaml"))
det = load_cfg(str(ROOT / "pipeline" / mc["detector_config"])); CACHE = Path(det["base"]["cache_dir"]); V = det["migration"]["velocity_m_ns"]
out_png, z0s, z1s = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
REG = json.load(open(CACHE / "mp_registration.json")); reg = REG["registration"]; sol = REG["solution"]
fr = np.load(CACHE / "mp_frame.npz"); s_nodes, xn, yn, k = fr["s_nodes"], fr["xn"], fr["yn"], float(fr["k"])
names = list(reg)
D = {n: np.load(CACHE / f"mp_{n}_2.npz") for n in names}
s = D[names[0]]["s_grid"]; ds = float(s[1] - s[0])
u_grid = np.arange(mc["mosaic"]["u_range_m"][0], mc["mosaic"]["u_range_m"][1] + 1e-9, mc["mosaic"]["du_m"]); du = float(u_grid[1] - u_grid[0])
SL = mc["mosaic"]["depth_slices_m"]; ZI = 2                                   # 0.7-0.9 m slice for the map panels

def mosaic(zi, use_reg):
    maps = {}
    for n in names:
        g = reg[n]; A, B = (g["A"], g["B"]) if use_reg else (0.0, 0.0)
        maps[n] = place_interp(slice_map(D[n]["G"][0], float(D[n]["dtg"]), V, *SL[zi]), D[n]["Ug"], s, A, B, u_grid)
    st = np.stack(list(maps.values()))
    with np.errstate(all="ignore"):
        return np.nanmean(st, 0), maps

M_raw, _ = mosaic(ZI, False); M_reg, per_reg = mosaic(ZI, True)
# repeated along-road features: both passes of an overlapping pair show narrow-line evidence > 1 over >= 10 m
feats = []
for zi in range(len(SL)):
    _, per = mosaic(zi, True)
    L = {n: line_evidence(np.nan_to_num(per[n]), np.isfinite(per[n]).astype(np.float32), ds, du) for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = np.isfinite(L[a]) & np.isfinite(L[b])
            if both.sum() < 2000:
                continue
            lab, nl = label((L[a] > 1.0) & (L[b] > 1.0) & both)
            for j in range(1, nl + 1):
                ii, jj = np.nonzero(lab == j)
                Ls = (ii.max() - ii.min()) * ds; W = (jj.max() - jj.min() + 1) * du
                if Ls >= 10 and W <= 0.6:
                    feats.append(dict(a=a, b=b, zi=zi, s0=float(s[ii.min()]), s1=float(s[ii.max()]), u=float(u_grid[jj].mean()), len_m=float(Ls), width_m=float(W)))
feats.sort(key=lambda f: -f["len_m"])
print(len(feats), "repeated along-road features; top:", [(round(f["s0"]), round(f["s1"]), round(f["u"], 2), SL[f["zi"]], f["a"][-3:], f["b"][-3:]) for f in feats[:6]])

fig = plt.figure(figsize=(19, 17), facecolor=SURF)
fig.suptitle("Six passes over Via Nizza merged into one radar map: one strong along-road pipe candidate (three passes, 0.7 m deep) and weaker ones (provisional)",
             x=0.01, y=0.993, ha="left", fontsize=13.5, color=INK)
fig.text(0.01, 0.972, f"13 Oct, street near 44.285 N 8.447 E (window {mc['window_s_m'][0]:.0f}-{mc['window_s_m'][1]:.0f} m along the reference pass). Passes are "
         "registered with crossing pipes (along the street) and with overlapping radar data (across). 200 MHz, migrated amplitude.",
         fontsize=9.5, color=MUTED)
gs = fig.add_gridspec(4, 3, height_ratios=[1.15, 0.8, 0.8, 1.0], width_ratios=[1, 1, 1], hspace=0.62, wspace=0.18, left=0.05, right=0.985, top=0.94, bottom=0.04)
axA = fig.add_subplot(gs[0, 0]); axR = fig.add_subplot(gs[0, 1:])
for n in names:
    g = reg[n]; d = D[n]
    uc = np.nanmedian(d["Ug"], 0) - g["B"]; ok = np.isfinite(uc)
    mx, my = su_to_map(s[ok] - g["A"], uc[ok], s_nodes, xn, yn, k)
    axA.plot(mx, my, color=BLUE if g["direction"] > 0 else ORANGE, lw=1.6, alpha=0.9)
xs_, ys_ = su_to_map(np.array([s[0], s[-1]]), np.array([0, 0]), s_nodes, xn, yn, k)
pad = 40 / k; X0, X1 = min(xs_) - pad, max(xs_) + pad; Y0, Y1 = min(ys_) - pad, max(ys_) + pad
axA.set_xlim(X0, X1); axA.set_ylim(Y0, Y1); axA.set_aspect("equal")
cx.add_basemap(axA, source=GSAT, crs="EPSG:3857", zoom=18, attribution=False, alpha=0.75)
lo_, la_ = merc_to_lonlat(np.array([X0, X1]), np.array([Y0, Y1]))
axA.text(0.01, 0.01, f"SW {la_[0]:.5f}N {lo_[0]:.5f}E", transform=axA.transAxes, color="white", fontsize=7.5)
axA.text(0.99, 0.99, f"NE {la_[1]:.5f}N {lo_[1]:.5f}E", transform=axA.transAxes, color="white", fontsize=7.5, ha="right", va="top")
sb = 100 / k; axA.plot([X0 + 15 / k, X0 + 15 / k + sb], [Y0 + 25 / k] * 2, color="white", lw=3); axA.text(X0 + 15 / k + sb / 2, Y0 + 32 / k, "100 m", color="white", ha="center", fontsize=8)
axA.set_xticks([]); axA.set_yticks([])
axA.legend(handles=[Line2D([], [], color=BLUE, lw=2, label="3 passes, one direction"), Line2D([], [], color=ORANGE, lw=2, label="3 passes, other direction")],
           loc="lower right", fontsize=8, framealpha=0.9)
axA.set_title("A  The six passes (registered)", loc="left", fontsize=10.5, color=INK)
# registration summary
rows = [(n, reg[n]) for n in names]
axR.axis("off")
txt = ["B  Registration result (relative to the reference pass Swath041)", "",
       f"{'pass':10s} {'direction':>10s} {'along-street shift':>20s} {'across-street shift':>21s}"]
for n, g in rows:
    txt.append(f"{n:10s} {('same' if g['direction'] > 0 else 'opposite'):>10s} {g['A']:+19.2f} m {g['B']:+20.2f} m")
pr = REG["pairs"]
ov = [p for p in pr if p["overlap"]]
txt += ["", "Overlapping pairs: agreement of the two radar maps (correlation) before -> after shifting:"]
txt += ["   " + ", ".join(f"{p['a'][-3:]}-{p['b'][-3:]}: {p['overlap']['r_zero']:.2f} -> {p['overlap']['r_best']:.2f}" for p in ov)]
txt += ["", f"Loop closure across the street: {sol['resid_B_rms']:.2f} m RMS.   Along the street: {sol['resid_A_rms']:.2f} m RMS.",
        "Opposite-direction passes sit 2.3-2.65 m from same-direction passes along the street (3 of 3 passes):",
        f"the signature of a position error along the direction of travel of about {abs(sol['eps_along_travel_m']):.1f} m (timing or lever arm), applied here.",
        "Independent check: moving the 14 Oct Corso Ricci detections 1.15 m forward raises storm-drain hits from 7 to 9 of 21 (chance 4.3)."]
axR.text(0.0, 1.0, "\n".join(txt), va="top", ha="left", family="monospace", fontsize=8.8, color=INK, transform=axR.transAxes)

S0, S1 = 2850.0, 3000.0
w = (s >= S0) & (s <= S1)
for row, (M, ttl) in enumerate(((M_raw, "C  Depth 0.7-0.9 m, passes as the GPS placed them"), (M_reg, "D  Same after registration (shifts from crossing ties and overlap correlation; the 1.5-2.5 m band between the two direction groups has no data)"))):
    ax = fig.add_subplot(gs[1 + row, :])
    ax.imshow(M[w].T, origin="lower", aspect="auto", extent=[S0, S1, u_grid[0], u_grid[-1]], cmap="magma", vmin=0, vmax=6, interpolation="nearest")
    if row == 1:
        for f in feats:
            if f["zi"] == ZI and f["s1"] > S0 and f["s0"] < S1:
                ax.plot([max(f["s0"], S0), min(f["s1"], S1)], [f["u"]] * 2, color=AQUA, lw=2.2)
    if row == 1:
        sys.path.insert(0, str(ROOT / "pipeline"))
        from along_road import longitudinal_records  # noqa: E402
        recs = longitudinal_records([ROOT / "data_public/gfi_all_condotte_acque_bianche.json", ROOT / "data_public/gfi_all_condotte_acque_nere.json"],
                                    s_nodes, xn, yn, k, (u_grid[0], u_grid[-1]), min_len=5, depth_m=(0.0, 5.0))
        for r_ in recs:
            if r_["s1"] > S0 and r_["s0"] < S1:
                ax.plot([max(r_["s0"], S0), min(r_["s1"], S1)], [r_["u0"]] * 2, color="white", lw=1.2, ls=(0, (4, 3)))
                ax.text(max(r_["s0"], S0) + 1, r_["u0"] + 0.15, f"{r_['kind']} {r_['diam']} mm, {r_['z']:.1f} m deep", color="white", fontsize=7)
    ax.set_ylabel("across street (m)\n+ = left of reference", color=MUTED, fontsize=8.5)
    ax.set_title(ttl + ("   aqua = along-road feature repeated in two passes; white dashed = municipal record" if row == 1 else ""), loc="left", fontsize=10.5, color=INK)
    if row == 1:
        ax.set_xlabel("distance along the reference pass (m)", color=MUTED, fontsize=9)

# cross-sections: two pipe-shaped repeated features and one layer-shaped one, with the shape test
from register_multipass import cross_section  # noqa: E402
ST = json.load(open(CACHE / "mp_shape_test.json"))
pipes = sorted([x for x in ST["features"] if x["verdict"] == "pipe-like"], key=lambda x: -x["len_m"])
layers = sorted([x for x in ST["features"] if x["verdict"] == "layer-like"], key=lambda x: -x["len_m"])
show = [pipes[0], next(x for x in pipes if abs(x["u"] - pipes[0]["u"]) > 1.5), layers[0]]
summ = ST["summary"]
for c_i, f in enumerate(show):
    ax = fig.add_subplot(gs[3, c_i])
    X, zax = cross_section(D, reg, s, u_grid, f["s_mid"], V)
    ax.imshow(X, origin="upper", aspect="auto", extent=[u_grid[0], u_grid[-1], zax[-1], 0], cmap="gray_r", vmin=0, vmax=3, interpolation="nearest")
    ax.set_ylim(1.6, 0); ax.set_xlim(f["u"] + 2.2, f["u"] - 2.2)
    ax.plot(f["edge_u"], f["edge_z"], ".", color=AQUA, ms=4, label="top edge of the reflector")
    uu = np.linspace(f["u"] - 0.8, f["u"] + 0.8, 60)
    ax.plot(uu, np.sqrt(f["z0"] ** 2 + (uu - f["uc"]) ** 2), color=ORANGE, lw=1.6, label=f"pipe curve fit, misfit {f['rms_hyperbola_m'] * 100:.1f} cm")
    ax.plot([], [], " ", label=f"flat/tilted layer fit, misfit {f['rms_plane_m'] * 100:.1f} cm")
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.85)
    verdict = "PIPE-SHAPED" if f["verdict"] == "pipe-like" else "LAYER-SHAPED"
    ax.set_title(f"{'EFG'[c_i]}  {verdict}: s {f['s0']:.0f}-{f['s1']:.0f} m, u {f['uc']:+.2f} m, {f['z0']:.2f} m deep (passes {f['a'][-3:]}+{f['b'][-3:]})",
                 loc="left", fontsize=9.2, color=ORANGE if c_i < 2 else MUTED)
    ax.set_xlabel("across street (m, + = left of reference)", color=MUTED, fontsize=8.5); ax.set_ylabel("depth (m)", color=MUTED, fontsize=8.5)
fig.text(0.05, 0.248, f"E-G: radar cross-sections across the merged passes (2 m along-street stack). Shape test: {summ['pipe_like_features']} of {summ['n_features_tested']} "
         f"repeated along-road features are pipe-shaped vs {summ['frac_controls'] * 100:.0f} percent of {summ['n_controls']} random places (p = {summ['p_binomial']:.3f}).\n"
         "E is repeated by 3 independent passes over 40 m. The nearest record is the 600 mm sewer 1-2 m away but recorded 2.6 m deep: E (0.7 m) is either an unrecorded utility "
         "(gas, water and telecom maps are not public) or the record is wrong.", fontsize=9, color=INK)
for a_ in fig.axes:
    a_.set_facecolor(SURF); a_.tick_params(colors=MUTED, labelsize=8)
    for sp in a_.spines.values():
        sp.set_color("#cfcfcf")
fig.savefig(out_png, dpi=140, facecolor=SURF); print("saved", out_png)
json.dump(feats, open(CACHE / "mp_repeated_features.json", "w"))
