"""E45 registered multi-drive mosaic test, Via Nizza (rules: DECISIONS 2026-11-04 E45).
Per-drive P rasters (along/evidence.py, own frame) -> reference frame of Swath041 through the channel footprints
(swath_geometry on the saved reference frame) -> E16 registration (s - A, u - B) -> common grid (0.5 m x 0.04 m).
Statistic f: evidence cells (P >= thr) of a pass that are repeated (within +-0.10 m across, same s) by a pass whose
registered strip centre is >= 0.5 m away ('independent lane'), among evidence cells that such a pass covers.
Nulls: mirror (each pass reflected about its own strip centre) and shift (non-reference passes shifted 20-200 m along s,
200 draws). Run: python -m italgas_gpr.lanes.agreement configs/lane_registration.yaml -> cache/mosaic_test.npz + .json"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from italgas_gpr.io.paths import RAW, config_path, load_yaml, swath_files
from scipy.ndimage import binary_closing, maximum_filter1d

from italgas_gpr.geometry.swath import swath_geometry
from italgas_gpr.processing.preprocess import load_cfg
from italgas_gpr.io.ids_reader import Swath
from italgas_gpr.along.evidence import search_columns, search_samples

_RL = load_yaml(config_path("along_lines.yaml"))                            # evaluated band, one source
T_MIN_NS, U_MIN_M, U_MAX_M = float(_RL["t_min_ns"]), float(_RL["u_min_m"]), float(_RL["u_max_m"])

PASSES = {"Swath041": 40, "Swath038": 37, "Swath047": 46, "Swath046": 45, "Swath040": 39, "Swath037": 36}   # E16 pass -> drive index
DS, DU, U0, U1 = 0.5, 0.04, -3.5, 6.5; TOL_CELLS = 2; MIN_SEP = 0.5; N_SHIFT = 200


def frame_map(i: int, det: dict, frame: dict, centre: bool = False):
    """Map from drive i's own (s, u) to the reference frame (s_ref, u_ref) through the 200 MHz channel footprints:
    per scan a linear map u_ref = a + b u_own, interpolated along own s. Returns f(s, u) -> (s_ref, u_ref); with
    centre=True also returns the strip-centre u_ref as a third output."""
    files = swath_files(det["base"].get("raw_root", RAW)); f = files[i]
    sw = Swath.load(str(Path(f).parent), Path(f).stem); go = swath_geometry(sw, det["base"]); gr = swath_geometry(sw, det["base"], frame)
    So, Uo, Sr, Ur = go["s2"], go["u2"], gr["s2"], gr["u2"]
    sc_o = np.nanmean(So, 1); sc_r = np.nanmean(Sr, 1); uc_o = np.nanmean(Uo, 1); uc_r = np.nanmean(Ur, 1)
    b = (Ur[:, -1] - Ur[:, 0]) / (Uo[:, -1] - Uo[:, 0]); a = uc_r - b * uc_o                       # per-scan linear map u_ref = a + b u_own
    ok = np.isfinite(sc_o) & np.isfinite(sc_r) & np.isfinite(a) & np.isfinite(b); o = np.argsort(sc_o[ok])
    sc_o, sc_r, a, b, uc_r = sc_o[ok][o], sc_r[ok][o], a[ok][o], b[ok][o], uc_r[ok][o]

    def fm(s, u):
        sr = np.interp(s, sc_o, sc_r); ur = np.interp(s, sc_o, a) + np.interp(s, sc_o, b) * u
        return (sr, ur, np.interp(s, sc_o, uc_r)) if centre else (sr, ur)
    fm.s_range = (float(sc_o[0]), float(sc_o[-1]))
    return fm


def pass_cells(i: int, det: dict, frame: dict, reg: dict, cache: Path):
    """All P cells of drive i in the registered reference frame: (s_true, u_true, P, depth_m, strip-centre u_true per cell)."""
    fm = frame_map(i, det, frame, centre=True); sc_o = fm.s_range
    S, U, P, Z, C = [], [], [], [], []
    for pf in sorted(glob.glob(str(cache / f"prob_raster_{i}_*.npz"))):
        d = np.load(pf); t = d["t_ns"]; it = search_samples(d, T_MIN_NS); iu = search_columns(d, U_MIN_M, U_MAX_M)       # evaluated band: raster_lines.yaml
        if not len(it) or not len(iu):
            continue
        P3 = d["P"].astype(np.float32)[:, iu][:, :, it]; plan = P3.max(2); zi = P3.argmax(2); depth = 0.05 * t[it][zi]   # v 0.10 m/ns ASSUMED
        cs, ug = d["centres_s"], d["ug"][iu]; keep = (cs >= d["c0"]) & (cs < d["c1"]); cs = cs[keep]; plan = plan[keep]; depth = depth[keep]
        inside = (cs >= sc_o[0]) & (cs <= sc_o[1]); cs, plan, depth = cs[inside], plan[inside], depth[inside]
        SS0, UG = np.meshgrid(cs, ug, indexing="ij"); SS, UU, CR = fm(SS0, UG); cr = CR[:, 0]
        S.append(SS.ravel() - reg["A"]); U.append(UU.ravel() - reg["B"]); P.append(plan.ravel()); Z.append(depth.ravel()); C.append(np.repeat(cr - reg["B"], len(ug)))
    return (np.concatenate(S), np.concatenate(U), np.concatenate(P), np.concatenate(Z), np.concatenate(C)) if S else None


def to_grid(s, u, p, z, c, s_grid, u_grid, mirror=False):
    """Max of p per grid cell; coverage; strip centre per s row; depth at the max. mirror: reflect u about the centre."""
    if mirror:
        u = 2 * c - u
    R = np.full((len(s_grid), len(u_grid)), np.nan, np.float32); Zg = np.full_like(R, np.nan); cen = np.full(len(s_grid), np.nan)
    i = np.round((s - s_grid[0]) / DS).astype(int); j = np.round((u - U0) / DU).astype(int); ok = (i >= 0) & (i < len(s_grid)) & (j >= 0) & (j < len(u_grid))
    np.fmax.at(R, (i[ok], j[ok]), p[ok])
    order = np.lexsort((-p[ok], j[ok], i[ok])); ii, jj, zz = i[ok][order], j[ok][order], z[ok][order]         # first of each cell = max p
    first = np.r_[True, (ii[1:] != ii[:-1]) | (jj[1:] != jj[:-1])]; Zg[ii[first], jj[first]] = zz[first]
    cs = np.full(len(s_grid), np.nan); np.fmax.at(cs, i[ok], c[ok]); cen = cs                                  # centre is constant per s row
    cov = binary_closing(np.isfinite(R), structure=np.ones((1, 3)))
    return np.nan_to_num(R, nan=0.0), cov, cen, Zg


def agreement(R, cov, cen, thr, shifts=None):
    """f and the agreeing-cell mask. R, cov: dict pass -> (n_s, n_u); cen: pass -> (n_s,). shifts: pass -> rows to roll."""
    names = list(R); n_s = len(next(iter(cen.values())))
    Rr = {p: (np.roll(R[p], shifts.get(p, 0), 0) if shifts else R[p]) for p in names}
    Cv = {p: (np.roll(cov[p], shifts.get(p, 0), 0) if shifts else cov[p]) for p in names}
    Cn = {p: (np.roll(cen[p], shifts.get(p, 0)) if shifts else cen[p]) for p in names}
    num = den = 0; agree = {p: np.zeros_like(R[p], bool) for p in names}; den_mask = {p: np.zeros_like(R[p], bool) for p in names}
    for p in names:
        ev = Rr[p] >= thr; covered = np.zeros_like(ev); hit = np.zeros_like(ev)
        for q in names:
            if q == p:
                continue
            indep = np.abs(Cn[p] - Cn[q]) >= MIN_SEP; indep &= np.isfinite(Cn[p]) & np.isfinite(Cn[q])
            covered |= Cv[q] & indep[:, None]
            hit |= (maximum_filter1d(Rr[q], 2 * TOL_CELLS + 1, axis=1) >= thr) & Cv[q] & indep[:, None]
        den_mask[p] = ev & covered; agree[p] = ev & covered & hit
        den += int(den_mask[p].sum()); num += int(agree[p].sum())
    return (num / den if den else np.nan), num, den, agree, den_mask


def main(cfg_path: str) -> None:
    mc = load_yaml(config_path(cfg_path)); det = load_cfg(config_path(mc["detector_config"])); cache = Path(det["base"]["cache_dir"])
    frame = dict(np.load(cache / "mp_frame.npz")); frame["k"] = float(frame["k"]); REG = json.load(open(cache / "mp_registration.json"))["registration"]
    w0, w1 = mc["window_s_m"]; s_grid = np.arange(w0, w1, DS); u_grid = np.arange(U0, U1 + 1e-9, DU)
    R, cov, cen, Zg, Rm, covm, cenm = {}, {}, {}, {}, {}, {}, {}
    for name, i in PASSES.items():
        cells = pass_cells(i, det, frame, REG[name], cache)
        if cells is None:
            print(name, "no raster"); continue
        R[name], cov[name], cen[name], Zg[name] = to_grid(*cells, s_grid, u_grid)
        Rm[name], covm[name], cenm[name], _ = to_grid(*cells, s_grid, u_grid, mirror=True)
        print(f"{name} drive {i}: cells {len(cells[0])}, covered rows {int(cov[name].any(1).sum())}/{len(s_grid)}, centre u median {np.nanmedian(cen[name]):+.2f}, P>=0.5 {float(np.mean(R[name][cov[name]] >= 0.5)):.3f}", flush=True)
    out = dict(window=[w0, w1], passes={n: PASSES[n] for n in R}, thresholds={})
    rng = np.random.default_rng(det["seed"] + 45); lo, hi = int(20 / DS), int(200 / DS)
    for thr in (0.5, 0.9):
        f, num, den, agree, dm = agreement(R, cov, cen, thr)
        fm, numm, denm, _, _ = agreement(Rm, covm, cenm, thr)
        fn = []
        for _ in range(N_SHIFT):
            sh = {p: int(rng.integers(lo, hi) * rng.choice([-1, 1])) for p in R if p != mc["reference"]}
            fn.append(agreement(R, cov, cen, thr, shifts=sh)[0])
        fn = np.array(fn); okn = np.isfinite(fn)
        rows = np.zeros(len(s_grid), bool)
        for p in agree:
            rows |= agree[p].any(1)
        runs = []; k = 0
        while k < len(rows):                                       # continuous agreeing stretches >= 10 m
            if rows[k]:
                j = k
                while j < len(rows) and rows[j]:
                    j += 1
                if (j - k) * DS >= 10:
                    us, zs = [], []
                    for p in agree:
                        ii, jj = np.nonzero(agree[p][k:j]); us += list(u_grid[jj]); zs += list(Zg[p][k:j][ii, jj])
                    runs.append(dict(s0=float(s_grid[k]), s1=float(s_grid[j - 1]), length_m=float((j - k) * DS), u_median=float(np.median(us)), u_iqr=[float(np.percentile(us, 25)), float(np.percentile(us, 75))],
                                     depth_median=float(np.nanmedian(zs)), passes=[p for p in agree if agree[p][k:j].any()]))
                k = j
            else:
                k += 1
        res = dict(f_real=f, n_agree=num, n_tested=den, f_mirror=fm, n_mirror=[numm, denm], f_shift_mean=float(np.nanmean(fn)), f_shift_q95=float(np.nanquantile(fn, 0.95)),
                   p_shift=float((np.sum(fn[okn] >= f) + 1) / (okn.sum() + 1)), agree_rows_m=float(rows.sum() * DS), runs=sorted(runs, key=lambda r: -r["length_m"]))
        out["thresholds"][str(thr)] = res
        print(f"thr {thr}: f_real {f:.3f} ({num}/{den}), mirror {fm:.3f} ({numm}/{denm}), shift mean {res['f_shift_mean']:.3f} q95 {res['f_shift_q95']:.3f}, p {res['p_shift']:.3f}, agreeing length {res['agree_rows_m']:.0f} m, runs>=10m {len(runs)}", flush=True)
        if thr == 0.5:
            np.savez_compressed(cache / "mosaic_test.npz", s_grid=s_grid, u_grid=u_grid, **{f"R_{p}": R[p] for p in R}, **{f"cov_{p}": cov[p] for p in R}, **{f"cen_{p}": cen[p] for p in R},
                                **{f"agree_{p}": agree[p] for p in R}, **{f"Z_{p}": Zg[p] for p in R}, f_shift=fn)
    json.dump(out, open(cache / "mosaic_test.json", "w"), indent=1); print("saved")


if __name__ == "__main__":
    main(sys.argv[1])
