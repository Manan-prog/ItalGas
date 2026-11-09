"""E47: two-lane test for every lane component of the survey, on gridded P rasters with integer shifts (fast).
Same statistic, nulls and acceptance rules as E46 (DECISIONS 2026-11-04 E47). One process per component.
Run: python -m italgas_gpr.lanes.register [n_workers] [component reference ...] -> cache/city/lane_<ref>.json / .npz, mosaic_frame_c<ref>.npz,
     cache/city/lane_summary.json"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from italgas_gpr.io.paths import CACHE, RAW, config_path  # noqa: E402
DS, DU, U0, U1 = 0.5, 0.04, -3.5, 6.5; TOL_CELLS = 2; MIN_SEP = 0.5
DA_ROWS = np.arange(-6, 7); DB_COLS = np.arange(-25, 26); MIN_CELLS = 5000; MIN_GAIN = 0.05; N_SHIFT = 100


def shift2(R, cov, cen, dr: int, dc: int):
    """Move a raster by dr rows (s) and dc columns (u) without wrap-around (cells shifted in are uncovered)."""
    R2 = np.roll(R, (dr, dc), (0, 1)); C2 = np.roll(cov, (dr, dc), (0, 1)); c2 = np.roll(cen, dr) + dc * DU
    if dr > 0: C2[:dr] = False; c2[:dr] = np.nan
    elif dr < 0: C2[dr:] = False; c2[dr:] = np.nan
    if dc > 0: C2[:, :dc] = False
    elif dc < 0: C2[:, dc:] = False
    return R2, C2, c2


def corr(Rr, Cr, R, C) -> tuple[float, int]:
    m = Cr & C; n = int(m.sum())
    if n < MIN_CELLS:
        return np.nan, n
    a = Rr[m]; b = R[m]; sa, sb = a.std(), b.std()
    return (float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb)) if sa > 1e-6 and sb > 1e-6 else np.nan), n


def register(Rr, Cr, R, cov, cen, search_A=True) -> dict:
    best = dict(dr=0, dc=0, r=np.nan, n=0); r0 = None
    for dr in (DA_ROWS if search_A else [0]):
        for dc in DB_COLS:
            R2, C2, _ = shift2(R, cov, cen, int(dr), int(dc)); r, n = corr(Rr, Cr, R2, C2)
            if dr == 0 and dc == 0:
                r0 = r
            if np.isfinite(r) and (not np.isfinite(best["r"]) or r > best["r"]):
                best = dict(dr=int(dr), dc=int(dc), r=r, n=n)
    acc = bool(np.isfinite(best["r"]) and r0 is not None and np.isfinite(r0) and best["r"] - r0 >= MIN_GAIN and best["n"] >= MIN_CELLS)
    if not acc:
        best.update(dr=0, dc=0)
    return dict(best, r0=r0, accepted=acc, dA=best["dr"] * DS, dB=best["dc"] * DU)


def agreement(R, cov, cen, thr):
    """f and agreeing cells (as mosaic_test.agreement, without the shift option)."""
    from scipy.ndimage import maximum_filter1d
    names = list(R); num = den = 0; agree = {}
    for p in names:
        ev = R[p] >= thr; covered = np.zeros_like(ev); hit = np.zeros_like(ev)
        for q in names:
            if q == p:
                continue
            indep = (np.abs(cen[p] - cen[q]) >= MIN_SEP) & np.isfinite(cen[p]) & np.isfinite(cen[q])
            covered |= cov[q] & indep[:, None]; hit |= (maximum_filter1d(R[q], 2 * TOL_CELLS + 1, axis=1) >= thr) & cov[q] & indep[:, None]
        agree[p] = ev & covered & hit; den += int((ev & covered).sum()); num += int(agree[p].sum())
    return (num / den if den else np.nan), num, den, agree


def component(ref: int, drives: list[int]) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from italgas_gpr.geometry.swath import swath_geometry
    from italgas_gpr.processing.preprocess import load_cfg
    from italgas_gpr.io.ids_reader import Swath
    from italgas_gpr.lanes.agreement import pass_cells, to_grid
    t0 = time.time(); det = load_cfg(config_path("detector.yaml"))
    files = sorted(glob.glob(f"{RAW}/*/Survey_*/Swath*.path")); f = files[ref]
    sw = Swath.load(str(Path(f).parent), Path(f).stem); g = swath_geometry(sw, det["base"]); frame = {k: g[k] for k in ("s_nodes", "xn", "yn", "k")}
    np.savez(CACHE / f"mosaic_frame_c{ref}.npz", **frame)
    w0, w1 = 0.0, float(g["s_nodes"][-1]); s_grid = np.arange(w0, w1, DS); u_grid = np.arange(U0, U1 + 1e-9, DU)
    C = {}
    for i in drives:
        c = pass_cells(i, det, frame, dict(A=0.0, B=0.0), CACHE)
        if c is not None and len(c[0]):
            C[i] = c
    if ref not in C or len(C) < 2:
        return dict(reference=ref, drives=list(C), status="no rasters" if ref not in C else "no partner rasters")
    G = {i: to_grid(*C[i], s_grid, u_grid) for i in C}; Gm = {i: to_grid(*C[i], s_grid, u_grid, mirror=True) for i in C}
    Rr, Cr, cen_r, Z_r = G[ref]; reg = {ref: dict(dr=0, dc=0, dA=0.0, dB=0.0, r=1.0, r0=1.0, n=int(Cr.sum()), accepted=True)}
    R, cov, cen, Zg = {ref: Rr}, {ref: Cr}, {ref: cen_r}, {ref: Z_r}
    for i in C:
        if i == ref:
            continue
        reg[i] = register(Rr, Cr, G[i][0], G[i][1], G[i][2])
        R[i], cov[i], cen[i] = shift2(G[i][0], G[i][1], G[i][2], reg[i]["dr"], reg[i]["dc"]); Zg[i] = np.roll(G[i][3], (reg[i]["dr"], reg[i]["dc"]), (0, 1))
    out = dict(reference=ref, drives=list(C), status="ok", window=[w0, w1], registration={str(i): reg[i] for i in reg}, thresholds={})
    rng = np.random.default_rng(1000 + ref); lo, hi = int(20 / DS), int(200 / DS)
    for thr in (0.5, 0.9):
        f, num, den, agree = agreement(R, cov, cen, thr)
        Rm, covm, cenm = {ref: Gm[ref][0]}, {ref: Gm[ref][1]}, {ref: Gm[ref][2]}; regm = {}
        for i in C:
            if i == ref:
                continue
            Rb, Cb, cb = shift2(Gm[i][0], Gm[i][1], Gm[i][2], reg[i]["dr"], 0); rm = register(Rr, Cr, Rb, Cb, cb, search_A=False); regm[i] = rm
            Rm[i], covm[i], cenm[i] = shift2(Rb, Cb, cb, 0, rm["dc"])
        fm, numm, denm, _ = agreement(Rm, covm, cenm, thr)
        fn = []
        for _ in range(N_SHIFT):
            Rs, covs, cens = {ref: Rr}, {ref: Cr}, {ref: cen_r}
            for i in C:
                if i == ref:
                    continue
                sh = int(rng.integers(lo, hi) * rng.choice([-1, 1]))
                Rb = np.roll(G[i][0], sh + reg[i]["dr"], 0); Cb = np.roll(G[i][1], sh + reg[i]["dr"], 0); cb = np.roll(G[i][2], sh + reg[i]["dr"])   # wrapped, as E45
                rs = register(Rr, Cr, Rb, Cb, cb, search_A=False); Rs[i], covs[i], cens[i] = shift2(Rb, Cb, cb, 0, rs["dc"])
            fn.append(agreement(Rs, covs, cens, thr)[0])
        fn = np.array(fn); okn = np.isfinite(fn)
        rows = np.zeros(len(s_grid), bool)
        for p in agree:
            rows |= agree[p].any(1)
        runs = []; k = 0
        while k < len(rows):
            if rows[k]:
                j = k
                while j < len(rows) and rows[j]:
                    j += 1
                if (j - k) * DS >= 10:
                    us, zs, ps = [], [], []
                    for p in agree:
                        ii, jj = np.nonzero(agree[p][k:j]); us += list(u_grid[jj]); zs += list(Zg[p][k:j][ii, jj])
                        if len(ii): ps.append(int(p))
                    runs.append(dict(s0=float(s_grid[k]), s1=float(s_grid[j - 1]), length_m=float((j - k) * DS), u_median=float(np.median(us)), depth_median=float(np.nanmedian(zs)), passes=ps))
                k = j
            else:
                k += 1
        out["thresholds"][str(thr)] = dict(f_real=None if not np.isfinite(f) else f, n_agree=num, n_tested=den, f_mirror=None if not np.isfinite(fm) else fm, n_mirror=[numm, denm],
                                           mirror_registration={str(i): regm[i] for i in regm}, f_shift=[None if not np.isfinite(x) else float(x) for x in fn],
                                           f_shift_mean=float(np.nanmean(fn)) if okn.any() else None, f_shift_q95=float(np.nanquantile(fn, 0.95)) if okn.any() else None,
                                           p_shift=float((np.sum(fn[okn] >= f) + 1) / (okn.sum() + 1)) if okn.any() and np.isfinite(f) else None,
                                           agree_rows_m=float(rows.sum() * DS), runs=sorted(runs, key=lambda r: -r["length_m"]))
        if thr == 0.5:
            np.savez_compressed(CACHE / "city" / f"lane_{ref}.npz", s_grid=s_grid, u_grid=u_grid, **{f"R_{p}": R[p] for p in R}, **{f"cov_{p}": cov[p] for p in R},
                                **{f"cen_{p}": cen[p] for p in R}, **{f"agree_{p}": agree[p] for p in R}, **{f"Z_{p}": Zg[p] for p in R})
    out["seconds"] = round(time.time() - t0); json.dump(out, open(CACHE / "city" / f"lane_{ref}.json", "w"))
    return out


def main(n_workers: int = 7, only: list[int] | None = None) -> None:
    comps = json.load(open(CACHE / "city" / "lane_components.json"))["components"]
    if only:
        comps = [c for c in comps if c["reference"] in only]
    t0 = time.time(); results = []
    with ProcessPoolExecutor(n_workers) as ex:
        for res in ex.map(component, [c["reference"] for c in comps], [c["drives"] for c in comps]):
            results.append(res); T = res.get("thresholds", {}).get("0.5", {})
            print(f"component {res['reference']} {res['drives']}: {res['status']}" + (f", f {T.get('f_real')} ({T.get('n_agree')}/{T.get('n_tested')}), mirror {T.get('f_mirror')}, shift q95 {T.get('f_shift_q95')}, p {T.get('p_shift')}, runs {len(T.get('runs', []))}, {res.get('seconds')}s" if T else ""), flush=True)
    ok = [r for r in results if r["status"] == "ok" and r["thresholds"]["0.5"]["f_real"] is not None]
    pos = [r for r in ok if r["thresholds"]["0.5"]["f_mirror"] is not None and r["thresholds"]["0.5"]["f_real"] > r["thresholds"]["0.5"]["f_mirror"] and r["thresholds"]["0.5"]["p_shift"] <= 0.05]
    num = sum(r["thresholds"]["0.5"]["n_agree"] for r in ok); den = sum(r["thresholds"]["0.5"]["n_tested"] for r in ok)
    # pooled shift null: draw-wise pooled fraction over components (same draw index), weights = tested cells
    pooled_null = []
    for d in range(N_SHIFT):
        a = b = 0
        for r in ok:
            fs = r["thresholds"]["0.5"]["f_shift"][d]
            if fs is not None:
                a += fs * r["thresholds"]["0.5"]["n_tested"]; b += r["thresholds"]["0.5"]["n_tested"]
        pooled_null.append(a / b if b else np.nan)
    pooled_null = np.array(pooled_null); f_pool = num / den if den else None
    from scipy.stats import binomtest
    summ = dict(n_components=len(comps), n_tested_components=len(ok), n_positive=len(pos), expected_positive_on_pipe_free=0.05 * len(ok),
                p_binomial=float(binomtest(len(pos), len(ok), 0.05, alternative="greater").pvalue) if ok else None,
                f_pooled=f_pool, n_agree=num, n_tested=den, f_pooled_null_mean=float(np.nanmean(pooled_null)) if den else None,
                f_pooled_null_q95=float(np.nanquantile(pooled_null, 0.95)) if den else None,
                p_pooled=float((np.sum(pooled_null[np.isfinite(pooled_null)] >= f_pool) + 1) / (np.isfinite(pooled_null).sum() + 1)) if den else None,
                positive=[r["reference"] for r in pos], runs_total_m=sum(x["length_m"] for r in pos for x in r["thresholds"]["0.5"]["runs"]),
                seconds=round(time.time() - t0))
    json.dump(summ, open(CACHE / "city" / "lane_summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7, [int(v) for v in sys.argv[2:]] or None)
