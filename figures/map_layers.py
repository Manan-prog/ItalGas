"""Shared pieces for north-up probability maps (single implementation for plot_map_probability.py
and plot_puc_overlay.py): colour maps, radar-strip probability meshes, map decorations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from geometry import merc_to_lonlat  # noqa: E402
from run_detection import su_to_map  # noqa: E402

from paths import CACHE  # noqa: E402
PCMAP = LinearSegmentedColormap.from_list("p", [(0.0, to_rgba("#f6e3b8", 0.8)), (0.15, to_rgba("#f2d9a6", 0.85)),
                                                (0.5, to_rgba("#eb8a3c", 0.95)), (1.0, to_rgba("#9e1f12", 1.0))])
GREY = LinearSegmentedColormap.from_list("g", [to_rgba("#dfe6ee", 0.6), to_rgba("#dfe6ee", 0.6)])
SIG_S = 0.3
SIG_U = 0.15                    # m, smallest across-street spread of an along-street track in the glow (ASSUMED, value used since E17)


def merged_crossings(R: dict) -> list[dict]:
    """Crossings of one drive (both radar rows merged; q <= 0.05): s0 (m, at u = 0), phi (deg), P."""
    a = [c for c in (R["arrays"].get("1") or {}).get("crossings", []) if c["q"] <= 0.05]
    b = [c for c in (R["arrays"].get("2") or {}).get("crossings", []) if c["q"] <= 0.05]
    used = set(); out = []
    for c in a:
        j = next((i for i, d in enumerate(b) if i not in used and abs(c["s0_m"] - d["s0_m"]) <= 0.3 and abs(c["phi_deg"] - d["phi_deg"]) <= 15
                  and abs(c["depth_m"] - d["depth_m"]) <= 0.2), None)
        if j is None:
            out.append(dict(s0=c["s0_m"], phi=c["phi_deg"], P=c["p_real"], depth=c["depth_m"], rows=1))
        else:
            used.add(j); d = b[j]
            out.append(dict(s0=0.5 * (c["s0_m"] + d["s0_m"]), phi=c["phi_deg"], P=1 - (1 - c["p_real"]) * (1 - d["p_real"]),
                            depth=0.5 * (c["depth_m"] + d["depth_m"]), rows=2))
    out += [dict(s0=d["s0_m"], phi=d["phi_deg"], P=d["p_real"], depth=d["depth_m"], rows=1) for i, d in enumerate(b) if i not in used]
    return out


def multipass_drives(REG: dict, tag: str = "") -> dict:
    """Per registered drive of a multi-pass cluster: crossings in TRUE coordinates (S0 = s0 - A - B tan(phi)) and the
    200 MHz strip edges lo/hi (m, true u) along true s. Raw-frame positions become true by s - A, u - B."""
    out = {}
    for n, g in REG.items():
        d = np.load(CACHE / f"mp_{tag}{n}_2.npz"); Ug = d["Ug"]; s_all = d["s_grid"]
        X = merged_crossings(json.load(open(CACHE / f"mp_{tag}{n}.json")))
        for o in X:
            o["S0"] = o["s0"] - g["A"] - g["B"] * np.tan(np.radians(o["phi"]))
        lo = np.nanmin(np.where(np.isfinite(Ug), Ug, np.inf), 0) - 0.04 - g["B"]; hi = np.nanmax(np.where(np.isfinite(Ug), Ug, -np.inf), 0) + 0.04 - g["B"]
        ok = np.isfinite(lo) & np.isfinite(hi) & (hi - lo < 2.0)
        out[n] = dict(X=X, s=(s_all - g["A"])[ok], lo=lo[ok], hi=hi[ok])
    return out


def merge_drives(DRV: dict, tol_s: float = 0.5, tol_phi: float = 15.0, window=None) -> list[dict]:
    """Group crossings of registered drives that agree (|dS0| <= tol_s m, |dphi| <= tol_phi deg, at most one per drive),
    strongest first, nearest member per drive. Returns [dict(S0, phi, P (max), depth (median), n_drives, rows (max), members)]."""
    allx = [(n, o) for n, d in DRV.items() for o in d["X"] if window is None or window[0] <= o["S0"] <= window[1]]
    if not allx:
        return []
    S = np.array([o["S0"] for _, o in allx]); PH = np.array([o["phi"] for _, o in allx]); NM = np.array([n for n, _ in allx])
    used = np.zeros(len(allx), bool); out = []
    for i in np.argsort([-o["P"] for _, o in allx]):
        if used[i]:
            continue
        cand = np.nonzero(~used & (NM != NM[i]) & (np.abs(S - S[i]) <= tol_s) & (np.abs(PH - PH[i]) <= tol_phi))[0]
        grp = [i]; seen = {NM[i]}
        for j in cand[np.argsort(np.abs(S[cand] - S[i]))]:
            if NM[j] not in seen:
                grp.append(j); seen.add(NM[j])
        used[grp] = True; mem = [allx[q][1] for q in grp]
        out.append(dict(S0=float(S[grp].mean()), phi=float(PH[grp].mean()), P=float(max(m["P"] for m in mem)),
                        depth=float(np.median([m["depth"] for m in mem])), n_drives=len(grp), rows=int(max(m["rows"] for m in mem)),
                        members=[str(NM[q]) for q in grp]))
    return out


def is_confident(g: dict) -> bool:
    """Crossing-line confidence rule (DECISIONS 2026-11-03): both radar rows of one drive, or >= 3 registered drives within 0.5 m."""
    return g["rows"] == 2 or g["n_drives"] >= 3


def prob_on_cells(Sc, Uc, crossings, along=(), sig_s=SIG_S, mass=False):
    """P(pipe passes this cell) = 1 - prod(1 - P_i g_i). crossings: dict(s0, phi, P[, sig]); along: dict(s0, s1, uc, P)
    or dict(s0, s1, ps, pu, P[, sig]) for a track that wanders (u interpolated along its path ps, pu, m).
    g_i = exp(-d^2 / 2 sig^2); with mass=True its peak is scaled by min(1, SIG_S / sig) for crossings and by
    min(1, SIG_U / sig) for along-street tracks, so a poorly placed detection is spread wider AND dimmer
    (its integral across the pipe does not grow with the uncertainty)."""
    q = np.ones_like(Sc)
    for o in crossings:
        dd = Sc - (o["s0"] + Uc * np.tan(np.radians(o["phi"]))); sg = o.get("sig", sig_s)
        q *= 1 - o["P"] * (min(1.0, SIG_S / sg) if mass else 1.0) * np.exp(-0.5 * (dd / sg) ** 2)
    for c in along:
        inside = (Sc >= c["s0"]) & (Sc <= c["s1"]); sg = c.get("sig", SIG_U)
        uc = np.interp(Sc, c["ps"], c["pu"]) if "ps" in c else c["uc"]
        q *= 1 - c["P"] * (min(1.0, SIG_U / sg) if mass else 1.0) * inside * np.exp(-0.5 * ((Uc - uc) / sg) ** 2)
    return 1 - q


def strip_multipass(name, s0, s1, REG, frame, cands, p_along, tag="", ds=0.1, n_u=20):
    """Mesh corners in map coordinates and P for one registered drive of a multi-pass cluster (true s0..s1)."""
    g = REG[name]; d = np.load(CACHE / f"mp_{tag}{name}_2.npz"); sg = d["s_grid"]; Ug = d["Ug"]
    X = merged_crossings(json.load(open(CACHE / f"mp_{tag}{name}.json")))
    s_raw = np.arange(s0 + g["A"], s1 + g["A"] + 1e-9, ds)
    lo = np.interp(s_raw, sg, np.nanmin(np.where(np.isfinite(Ug), Ug, np.inf), 0), left=np.nan, right=np.nan) - 0.04
    hi = np.interp(s_raw, sg, np.nanmax(np.where(np.isfinite(Ug), Ug, -np.inf), 0), left=np.nan, right=np.nan) + 0.04
    ok = np.isfinite(lo) & np.isfinite(hi) & (hi - lo < 2.0) & (hi > lo)
    if ok.sum() < 5:
        return None
    s_raw, lo, hi = s_raw[ok], lo[ok], hi[ok]
    f = np.linspace(0, 1, n_u + 1); S = np.repeat(s_raw[:, None], n_u + 1, 1); U = lo[:, None] + f[None, :] * (hi - lo)[:, None]
    Sc = 0.5 * (S[1:, 1:] + S[:-1, :-1]); Uc = 0.5 * (U[1:, 1:] + U[:-1, :-1])
    along = [dict(s0=c["s0"] + g["A"], s1=c["s1"] + g["A"], uc=c["uc"] + g["B"], P=p_along) for c in cands]
    P = prob_on_cells(Sc, Uc, X, along)
    mx, my = su_to_map(S.ravel() - g["A"], U.ravel() - g["B"], frame["s_nodes"], frame["xn"], frame["yn"], float(frame["k"]))
    return mx.reshape(S.shape), my.reshape(S.shape), P


def strip_single(frame, s0, s1, crossings, u_lo, u_hi, ds=0.1, n_u=20):
    """Mesh and P for a single drive in its own frame (crossings from finalize_v2: s0_m, phi_deg, P, sigma_along_m)."""
    S, U = np.meshgrid(np.arange(s0, s1 + 1e-9, ds), np.linspace(u_lo, u_hi, n_u + 1), indexing="ij")
    Sc = 0.5 * (S[1:, 1:] + S[:-1, :-1]); Uc = 0.5 * (U[1:, 1:] + U[:-1, :-1])
    X = [dict(s0=o["s0_m"], phi=o["phi_deg"], P=o["P"], sig=o["sigma_along_m"]) for o in crossings]
    P = prob_on_cells(Sc, Uc, X)
    mx, my = su_to_map(S.ravel(), U.ravel(), frame["s_nodes"], frame["xn"], frame["yn"], float(frame["k"]))
    return mx.reshape(S.shape), my.reshape(S.shape), P


def draw_strips(ax, strips):
    """Grey footprint of every strip first, then probabilities of all strips on top."""
    xs, ys = [], []
    for mx, my, P in strips:
        ax.pcolormesh(mx, my, np.zeros_like(P), cmap=GREY, vmin=0, vmax=1, shading="flat", zorder=3, edgecolors="none", rasterized=True)
        xs += [mx.min(), mx.max()]; ys += [my.min(), my.max()]
    for mx, my, P in strips:
        ax.pcolormesh(mx, my, np.ma.masked_less(P, 0.03), cmap=PCMAP, vmin=0, vmax=1, shading="flat", zorder=4, edgecolors="none", rasterized=True)
    return xs, ys


def decorate(ax, cxm, cym, half, k, bar_m, ink="#1f1f1f"):
    """Scale bar, north arrow and corner coordinates on a north-up EPSG:3857 axis."""
    x0b, y0b = cxm - half * 0.92, cym - half * 0.9
    ax.plot([x0b, x0b + bar_m / k], [y0b, y0b], color="white", lw=4, zorder=7, solid_capstyle="butt")
    ax.text(x0b + bar_m / k / 2, y0b + half * 0.03, f"{bar_m:.0f} m", color="white", ha="center", fontsize=10, zorder=7, fontweight="bold")
    ax.annotate("N", xy=(cxm + half * 0.9, cym + half * 0.88), xytext=(cxm + half * 0.9, cym + half * 0.72), color="white", ha="center",
                fontsize=11, fontweight="bold", arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5), zorder=7)
    lo_, la_ = merc_to_lonlat(np.array([cxm - half, cxm + half]), np.array([cym - half, cym + half]))
    ax.text(0.01, 0.01, f"{la_[0]:.5f} N, {lo_[0]:.5f} E", transform=ax.transAxes, fontsize=8, color=ink, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    ax.text(0.99, 0.99, f"{la_[1]:.5f} N, {lo_[1]:.5f} E", transform=ax.transAxes, fontsize=8, color=ink, ha="right", va="top", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    ax.set_xticks([]); ax.set_yticks([])
