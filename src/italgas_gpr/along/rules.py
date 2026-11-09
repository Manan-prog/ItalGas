"""E50 (DECISIONS 2026-11-05): choose the along-street line rule by what a second lane reproduces.
For each candidate rule (cell threshold, run-mean threshold, gap, lateral gate) and each E47 lane group, lines are derived
per drive from the evidence rasters, mapped into the group's registered frame (E47 dA, dB), and scored by the length of
drive A's line rows on which drive B (same rule) has a line within +-tol_cells, minus the same with B's lines shifted along
the street by >= SHIFT_MIN_M (N_SHIFT draws). Selection on the 13-14 Oct groups, report on the 15-17 Oct groups.
Run: python -m italgas_gpr.along.rules [n_workers] -> cache/city/rule_compare.json. Peaks per (drive, p_cell) are cached in
cache/city/rulecmp/."""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d

from italgas_gpr.along import lines as rl  # noqa: E402
from italgas_gpr.processing.preprocess import load_cfg  # noqa: E402
from italgas_gpr.lanes.register import DS, DU, U0  # noqa: E402
from italgas_gpr.lanes.agreement import frame_map  # noqa: E402
from italgas_gpr.io.paths import config_path

CACHE = rl.CACHE; CITY = CACHE / "city"; PK = CITY / "rulecmp"
GRID = dict(p_cell=[0.5, 0.7, 0.9], p_run=[0.0, 0.7, 0.85], max_gap=[1, 4, 8], gate_u_m=[0.12, 0.20])
MIN_LEN_M = 10.0; N_SHIFT = 20; SHIFT_MIN_M = 20.0; SEED = 5002; TOL = rl.CFG["tol_cells"]; MIN_SEP = rl.CFG["min_sep_m"]
LAT_MIN_CELLS, LAT_MAX_CELLS = 2 * TOL + 1, 15                                   # across-shift null: 0.20-0.60 m (amendment, DECISIONS 2026-11-05)
SELECT_REFS = [40, 56, 63, 82, 85, 88]                                           # lane groups surveyed 13-14 Oct (pre-registered)
RULES = [dict(p_cell=a, p_run=b, max_gap=c, gate_u_m=d) for a, b, c, d in itertools.product(*GRID.values())]
CURRENT = dict(p_cell=0.9, p_run=0.0, max_gap=1, gate_u_m=0.12)


def rule_key(r: dict) -> str:
    return f"c{r['p_cell']}_r{r['p_run']}_g{r['max_gap']}_u{r['gate_u_m']}"


def peaks(i: int, p_cell: float) -> dict | None:
    """Local maxima of P >= p_cell of drive i (raster_lines.drive_peaks), cached as npz."""
    f = PK / f"peaks_{i}_{p_cell}.npz"
    if f.exists():
        d = np.load(f); return dict(keys=d["keys"], vals=d["vals"], n_win=int(d["n_win"]), ug=d["ug"], dt=float(d["dt"]))
    pk = rl.drive_peaks(i, CACHE, p_cell)
    if pk is None:
        return None
    np.savez(f, keys=pk["keys"], vals=pk["vals"], n_win=pk["n_win"], ug=pk["ug"], dt=pk["dt"]); return pk


def make_peaks(args) -> str:
    i, p = args; os.environ.setdefault("OMP_NUM_THREADS", "1"); t = time.time(); pk = peaks(i, p)
    return f"peaks {i} p{p}: {0 if pk is None else len(pk['keys'])} in {time.time() - t:.0f} s"


def runs_for_rule(pk: dict, rule: dict) -> list:
    """Runs of the rule: linked with its gap and lateral gate from cells >= p_cell, kept when mean P >= p_run."""
    runs = rl.link_runs(pk, MIN_LEN_M, rule=dict(gate_u_m=rule["gate_u_m"], max_gap=rule["max_gap"]))
    return [r for r in runs if r[1].mean() >= rule["p_run"]]


def group_eval(ref: int) -> dict:
    """All rules on one lane group. Returns {rule_key: dict(line_m, covered_rows, rep_rows, null_rows)}."""
    os.environ.setdefault("OMP_NUM_THREADS", "1"); t0 = time.time()
    J = json.load(open(CITY / f"lane_{ref}.json")); Z = np.load(CITY / f"lane_{ref}.npz"); frame = dict(np.load(CACHE / f"mosaic_frame_c{ref}.npz"))
    det = load_cfg(config_path(rl.CFG["detector_config"])); s_grid, u_grid = Z["s_grid"], Z["u_grid"]; nrow, ncol = len(s_grid), len(u_grid)
    drives = [int(p) for p in J["drives"] if f"R_{p}" in Z.files]; cov = {i: Z[f"cov_{i}"] for i in drives}; cen = {i: Z[f"cen_{i}"] for i in drives}
    fms = {i: frame_map(i, det, frame) for i in drives}; reg = {i: J["registration"][str(i)] for i in drives}
    pks = {(i, p): peaks(i, p) for i in drives for p in GRID["p_cell"]}
    rng = np.random.default_rng(SEED + ref); shift_rows = int(round(SHIFT_MIN_M / DS))
    shifts = rng.integers(shift_rows, max(shift_rows + 1, nrow - shift_rows), size=N_SHIFT) if nrow > 2 * shift_rows + 1 else np.array([], int)
    lat = rng.integers(LAT_MIN_CELLS, LAT_MAX_CELLS + 1, size=N_SHIFT) * rng.choice([-1, 1], size=N_SHIFT)   # across-shift null (amendment)
    out = {}
    for rule in RULES:
        line_m = 0.0; rc = {}; Lg = {}; active = cells_n = 0
        for i in drives:
            pk = pks[(i, rule["p_cell"])]; runs = [] if pk is None else runs_for_rule(pk, rule); line_m += sum(r[2] for r in runs)
            g = np.zeros((nrow, ncol), bool); rows, cols = [], []
            for cells, _, _ in runs:
                s = cells[:, 0] * DS; u = pk["ug"][cells[:, 1]]; sr, ur = fms[i](s, u); sr = sr + reg[i]["dA"]; ur = ur + reg[i]["dB"]
                row = np.round((sr - s_grid[0]) / DS).astype(int); col = np.round((ur - U0) / DU).astype(int)
                ok = (row >= 0) & (row < nrow) & (col >= 0) & (col < ncol); rows.append(row[ok]); cols.append(col[ok])
            row = np.concatenate(rows) if rows else np.zeros(0, int); col = np.concatenate(cols) if cols else np.zeros(0, int)
            g[row, col] = True; rc[i] = (row, col); Lg[i] = maximum_filter1d(g.astype(np.uint8), 2 * TOL + 1, axis=1).astype(bool)
            active += int(g.any(1).sum()); cells_n += int(g.sum())
        covered = rep = 0; null = 0.0; null_lat = 0.0
        for i in drives:
            row, col = rc[i]
            if not len(row):
                continue
            for j in drives:
                if j == i:
                    continue
                indep = np.abs(cen[i][row] - cen[j][row]) >= MIN_SEP; cv = cov[j][row, col] & indep & np.isfinite(cen[i][row]) & np.isfinite(cen[j][row])
                if not cv.any():
                    continue
                covered += int(cv.sum()); rep += int((Lg[j][row, col] & cv).sum())
                if len(shifts):
                    null += np.mean([(Lg[j][(row + k) % nrow, col] & cv).sum() for k in shifts])
                null_lat += np.mean([(Lg[j][row, np.clip(col + k, 0, ncol - 1)] & cv).sum() for k in lat])
        out[rule_key(rule)] = dict(line_m=round(line_m, 1), covered_rows=covered, rep_rows=rep, null_rows=round(float(null), 2), null_lat_rows=round(float(null_lat), 2),
                                   active_rows=active, cells=cells_n)
    out["_meta"] = dict(ref=ref, drives=drives, n_rows=nrow, seconds=round(time.time() - t0)); return out


def main(n_workers: int = 6) -> None:
    PK.mkdir(exist_ok=True); refs = [int(f.stem.split("_")[1]) for f in CITY.glob("lane_*.json") if f.stem != "lane_summary" and json.load(open(f)).get("status") == "ok"]
    drives = sorted({int(p) for r in refs for p in json.load(open(CITY / f"lane_{r}.json"))["drives"]}); t0 = time.time()
    print(f"{len(refs)} lane groups, {len(drives)} drives, {len(RULES)} rules", flush=True)
    with ProcessPoolExecutor(n_workers) as ex:
        for m in ex.map(make_peaks, [(i, p) for i in drives for p in GRID["p_cell"]]):
            print(m, flush=True)
    print(f"peaks done {time.time() - t0:.0f} s", flush=True)
    with ProcessPoolExecutor(n_workers) as ex:
        groups = dict(zip(refs, ex.map(group_eval, refs)))
    for r, g in groups.items():
        print(f"group {r}: {g['_meta']['seconds']} s", flush=True)

    def agg(sel):
        res = {}
        for rule in RULES:
            k = rule_key(rule); L = sum(groups[r][k]["line_m"] for r in sel); c = sum(groups[r][k]["covered_rows"] for r in sel)
            rep = sum(groups[r][k]["rep_rows"] for r in sel); nul = sum(groups[r][k]["null_rows"] for r in sel); nl = sum(groups[r][k]["null_lat_rows"] for r in sel)
            act = sum(groups[r][k]["active_rows"] for r in sel); cel = sum(groups[r][k]["cells"] for r in sel)
            res[k] = dict(rule=rule, line_km=round(L / 1e3, 2), covered_m=round(c * DS), repeated_m=round(rep * DS), null_m=round(nul * DS, 1), null_lat_m=round(nl * DS, 1),
                          excess_m=round((rep - nul) * DS, 1), excess_lat_m=round((rep - nl) * DS, 1), excess_min_m=round((rep - max(nul, nl)) * DS, 1),
                          share=None if c == 0 else round(rep / c, 3), null_share=None if c == 0 else round(nul / c, 3), null_lat_share=None if c == 0 else round(nl / c, 3),
                          active_km=round(act * DS / 1e3, 2), cells_per_active_row=None if act == 0 else round(cel / act, 2))
        return res
    test_refs = [r for r in refs if r not in SELECT_REFS]; S = agg([r for r in refs if r in SELECT_REFS]); T = agg(test_refs)
    chosen = max(S, key=lambda k: S[k]["excess_min_m"]); cur = rule_key(CURRENT)
    out = dict(grid=GRID, min_len_m=MIN_LEN_M, n_shift=N_SHIFT, shift_min_m=SHIFT_MIN_M, seed=SEED, select_refs=SELECT_REFS, test_refs=test_refs,
               selection=S, test=T, chosen=chosen, current=cur, groups=groups, seconds=round(time.time() - t0))
    json.dump(out, open(CITY / "rule_compare.json", "w"), indent=1)
    print("\nselection set, top 10 by the smaller excess (along-shift and across-shift nulls):")
    for k in sorted(S, key=lambda k: -S[k]["excess_min_m"])[:10]:
        v = S[k]; print(f"  {k:24s} lines {v['line_km']:6.2f} km (active {v['active_km']:5.2f} km, {v['cells_per_active_row']} cells/row)  covered {v['covered_m']:5d} m  repeated {v['repeated_m']:5d} m  "
                        f"null along {v['null_m']:6.1f} across {v['null_lat_m']:6.1f}  excess_min {v['excess_min_m']:6.1f} m  share {v['share']}")
    print(f"  current {cur}: {S[cur]}")
    print("\ntest set (the result):")
    for k in (chosen, cur):
        print(f"  {k:28s} {T[k]}")
    print(f"done {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
