"""E46: registered multi-drive mosaic test for one region, with self-registration on the P rasters and
registration-matched nulls (rules: DECISIONS 2026-11-04 E46). Reuses mosaic_test (E45) for the grid and the statistic.
Run: python mosaic_region.py <region> <reference drive> <drive,drive,...> -> cache/mosaic_region_<region>.npz + .json"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing

from build_cube import swath_geometry
from build_v2 import load_cfg
from ids_reader import Swath
from mosaic_test import DS, DU, U0, U1, agreement, pass_cells, to_grid

ROOT = Path(__file__).resolve().parents[1]; RAW = "/Users/manansarupria/ItalGas/Savona"
DA_RANGE = np.arange(-3.0, 3.01, 0.5); DB_RANGE = np.round(np.arange(-1.0, 1.001, 0.04), 3); MIN_CELLS = 5000; MIN_GAIN = 0.05; N_SHIFT = 100


def corr_overlap(Rr, Cr, R, C) -> tuple[float, int]:
    m = Cr & C
    if m.sum() < MIN_CELLS:
        return np.nan, int(m.sum())
    a, b = Rr[m], R[m]
    if a.std() < 1e-6 or b.std() < 1e-6:
        return np.nan, int(m.sum())
    return float(np.corrcoef(a, b)[0, 1]), int(m.sum())


def register(Rr, Cr, cells, s_grid, u_grid, search_A=True, mirror=False) -> dict:
    """Best (dA, dB) of one drive's cells against the reference raster: cells moved by (-dA, -dB) then gridded."""
    s, u, p, z, c = cells
    best = dict(dA=0.0, dB=0.0, r=np.nan, n=0); r0 = None
    for dA in (DA_RANGE if search_A else [0.0]):
        for dB in DB_RANGE:
            R, C, _, _ = to_grid(s - dA, u - dB, p, z, c - dB, s_grid, u_grid, mirror=mirror)
            r, n = corr_overlap(Rr, Cr, R, C)
            if dA == 0.0 and dB == 0.0:
                r0 = r
            if np.isfinite(r) and (not np.isfinite(best["r"]) or r > best["r"]):
                best = dict(dA=float(dA), dB=float(dB), r=r, n=n)
    best["r0"] = r0; best["accepted"] = bool(np.isfinite(best["r"]) and r0 is not None and np.isfinite(r0) and best["r"] - r0 >= MIN_GAIN and best["n"] >= MIN_CELLS)
    if not best["accepted"]:
        best.update(dA=0.0, dB=0.0)
    return best


def main(region: str, ref: int, drives: list[int]) -> None:
    det = load_cfg(str(ROOT / "pipeline" / "configs" / "dev_swath_v2_2.yaml")); cache = Path(det["base"]["cache_dir"])
    files = sorted(glob.glob(f"{RAW}/*/Survey_*/Swath*.path")); f = files[ref]
    sw = Swath.load(str(Path(f).parent), Path(f).stem); g = swath_geometry(sw, det["base"]); frame = {k: g[k] for k in ("s_nodes", "xn", "yn", "k")}
    np.savez(cache / f"mosaic_frame_{region}.npz", **frame)
    w0, w1 = 0.0, float(g["s_nodes"][-1]); s_grid = np.arange(w0, w1, DS); u_grid = np.arange(U0, U1 + 1e-9, DU)
    zero = dict(A=0.0, B=0.0); C = {}
    for i in [ref] + [d for d in drives if d != ref]:
        c = pass_cells(i, det, frame, zero, cache)
        if c is not None and len(c[0]):
            C[i] = c
    names = {i: f"d{i}" for i in C}
    Rr, Cr, cen_r, Z_r = to_grid(*C[ref], s_grid, u_grid)
    reg = {ref: dict(dA=0.0, dB=0.0, r=1.0, n=int(Cr.sum()), accepted=True, r0=1.0)}
    R, cov, cen, Zg = {names[ref]: Rr}, {names[ref]: Cr}, {names[ref]: cen_r}, {names[ref]: Z_r}
    for i in C:
        if i == ref:
            continue
        reg[i] = register(Rr, Cr, C[i], s_grid, u_grid)
        s, u, p, z, c = C[i]; R[names[i]], cov[names[i]], cen[names[i]], Zg[names[i]] = to_grid(s - reg[i]["dA"], u - reg[i]["dB"], p, z, c - reg[i]["dB"], s_grid, u_grid)
        print(f"{region}: drive {i} vs {ref}: dA {reg[i]['dA']:+.1f} dB {reg[i]['dB']:+.2f} r {reg[i]['r']:.3f} (r0 {reg[i]['r0'] if reg[i]['r0'] is None else round(reg[i]['r0'], 3)}, n {reg[i]['n']}) accepted {reg[i]['accepted']}", flush=True)
    out = dict(region=region, reference=ref, drives=list(C), registration={str(i): reg[i] for i in reg}, window=[w0, w1], thresholds={})
    rng = np.random.default_rng(det["seed"] + 46); lo, hi = int(20 / DS), int(200 / DS)
    for thr in (0.5, 0.9):
        f, num, den, agree, _ = agreement(R, cov, cen, thr)
        # mirror null, re-registered: each non-reference drive mirrored about its own centre, then dB re-optimised (dA kept)
        Rm, covm, cenm = {names[ref]: Rr}, {names[ref]: Cr}, {names[ref]: cen_r}
        regm = {}
        for i in C:
            if i == ref:
                continue
            s, u, p, z, c = C[i]; cm = (s - reg[i]["dA"], u, p, z, c)
            rm = register(Rr, Cr, cm, s_grid, u_grid, search_A=False, mirror=True); regm[i] = rm
            Rm[names[i]], covm[names[i]], cenm[names[i]], _ = to_grid(s - reg[i]["dA"], u - rm["dB"], p, z, c - rm["dB"], s_grid, u_grid, mirror=True)
        _, _, _, _, _ = (None,) * 5
        Rm_ref_m, covm_ref_m, cenm_ref_m, _ = to_grid(*C[ref], s_grid, u_grid, mirror=True)       # the reference is mirrored too (as E45)
        Rm[names[ref]], covm[names[ref]], cenm[names[ref]] = Rm_ref_m, covm_ref_m, cenm_ref_m
        fm, numm, denm, _, _ = agreement(Rm, covm, cenm, thr)
        # shift null, re-registered per draw
        fn = []
        for _ in range(N_SHIFT):
            Rs, covs, cens = {names[ref]: Rr}, {names[ref]: Cr}, {names[ref]: cen_r}
            for i in C:
                if i == ref:
                    continue
                s, u, p, z, c = C[i]; sh = float(rng.integers(lo, hi) * rng.choice([-1, 1]) * DS)
                ss = w0 + (s - reg[i]["dA"] + sh - w0) % (w1 - w0)
                rs = register(Rr, Cr, (ss, u, p, z, c), s_grid, u_grid, search_A=False)
                Rs[names[i]], covs[names[i]], cens[names[i]], _ = to_grid(ss, u - rs["dB"], p, z, c - rs["dB"], s_grid, u_grid)
            fn.append(agreement(Rs, covs, cens, thr)[0])
        fn = np.array(fn); okn = np.isfinite(fn)
        rows = np.zeros(len(s_grid), bool)
        for p_ in agree:
            rows |= agree[p_].any(1)
        runs = []; k = 0
        while k < len(rows):
            if rows[k]:
                j = k
                while j < len(rows) and rows[j]:
                    j += 1
                if (j - k) * DS >= 10:
                    us, zs, ps = [], [], []
                    for p_ in agree:
                        ii, jj = np.nonzero(agree[p_][k:j]); us += list(u_grid[jj]); zs += list(Zg[p_][k:j][ii, jj])
                        if len(ii): ps.append(p_)
                    runs.append(dict(s0=float(s_grid[k]), s1=float(s_grid[j - 1]), length_m=float((j - k) * DS), u_median=float(np.median(us)), depth_median=float(np.nanmedian(zs)), passes=ps))
                k = j
            else:
                k += 1
        res = dict(f_real=f, n_agree=num, n_tested=den, f_mirror=fm, n_mirror=[numm, denm], mirror_registration={str(i): regm[i] for i in regm},
                   f_shift_mean=float(np.nanmean(fn)) if okn.any() else None, f_shift_q95=float(np.nanquantile(fn, 0.95)) if okn.any() else None,
                   p_shift=float((np.sum(fn[okn] >= f) + 1) / (okn.sum() + 1)) if okn.any() and np.isfinite(f) else None, agree_rows_m=float(rows.sum() * DS), runs=sorted(runs, key=lambda r: -r["length_m"]))
        out["thresholds"][str(thr)] = res
        print(f"{region} thr {thr}: f_real {f:.3f} ({num}/{den}), mirror(re-reg) {fm:.3f} ({numm}/{denm}), shift(re-reg) mean {res['f_shift_mean']} q95 {res['f_shift_q95']}, p {res['p_shift']}, agreeing {res['agree_rows_m']:.0f} m, runs {len(runs)}", flush=True)
        if thr == 0.5:
            np.savez_compressed(cache / f"mosaic_region_{region}.npz", s_grid=s_grid, u_grid=u_grid, **{f"R_{p_}": R[p_] for p_ in R}, **{f"cov_{p_}": cov[p_] for p_ in R}, **{f"cen_{p_}": cen[p_] for p_ in R},
                                **{f"agree_{p_}": agree[p_] for p_ in R}, **{f"Z_{p_}": Zg[p_] for p_ in R}, f_shift=fn)
    json.dump(out, open(cache / f"mosaic_region_{region}.json", "w"), indent=1); print("saved", region)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), [int(v) for v in sys.argv[3].split(",")])
