"""E52 (6 Nov 2026): bridging test for along-street lines. Two lines of the adopted rule (configs/raster_lines.yaml) that
end and start on the same lateral position and depth (within the linking gates) with a gap of up to joins.gap_max_m are
candidates for a JOIN; the join's support is the share of gap windows with a peak (P >= p_min) within +-tol_cells across
and +-gate_ns in time of the straight corridor between the two line ends. A join is accepted when support >= joins.support_min.
Joins are judged as lines were in E50: by the rows a drive in an independent lane reproduces (its own lines + joins of the
same variant, within +-tol_cells) against along- and across-shifted nulls, chosen on the 13-14 Oct lane groups and
reported on the 15-17 Oct groups. Run: python raster_joins.py [n_workers] -> cache/city/join_compare.json."""
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

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import raster_lines as rl  # noqa: E402
import rule_compare as rc  # noqa: E402
from build_v2 import load_cfg  # noqa: E402
from mosaic_fast import DS, DU, U0  # noqa: E402
from mosaic_test import frame_map  # noqa: E402

CFG = rl.CFG; CACHE = rl.CACHE; CITY = CACHE / "city"
GRID = dict(gap_max_m=[10.0, 20.0, 30.0], support_min=[0.0, 0.25, 0.5])
VARIANTS = [dict(gap_max_m=a, support_min=b) for a, b in itertools.product(*GRID.values())]
N_SHIFT = 20; SHIFT_MIN_M = 20.0; SEED = 5201; TOL = CFG["tol_cells"]; MIN_SEP = CFG["min_sep_m"]
LAT_MIN_CELLS, LAT_MAX_CELLS = 2 * TOL + 1, 15


def vkey(v: dict) -> str:
    return f"gap{v['gap_max_m']:.0f}_s{v['support_min']}"


def peak_table(pk: dict) -> dict[int, np.ndarray]:
    """window -> (n, 2) array of (iu, it) of the peaks with P >= p_min in that window."""
    keys = pk["keys"]; out = {}
    if not len(keys):
        return out
    cuts = np.flatnonzero(np.diff(keys[:, 0])) + 1
    for w, blk in zip(keys[np.r_[0, cuts], 0], np.split(keys[:, 1:], cuts)):
        out[int(w)] = blk
    return out


def corridor_support(tab: dict, cells: np.ndarray, tol_u: int, tol_t: int) -> float:
    """Share of corridor windows (w, iu, it) that hold a peak within +-tol_u across and +-tol_t in time."""
    hit = 0
    for w, iu, it in cells:
        b = tab.get(int(w))
        if b is not None and np.any((np.abs(b[:, 0] - iu) <= tol_u) & (np.abs(b[:, 1] - it) <= tol_t)):
            hit += 1
    return hit / len(cells)


def find_joins(lines: list, pk: dict, gap_max_m: float, support_min: float, rng: np.random.Generator | None = None,
               n_null: int = 20) -> list[dict]:
    """Candidate joins between merged lines [(cells (n, 3), P, length_m, members)] of one drive. A line end may join one
    line start and vice versa (best support, then shortest gap). Null support (optional): the same corridor evaluated on
    the drive's peaks shifted >= SHIFT_MIN_M along the street."""
    if len(lines) < 2:
        return []
    tab = peak_table(pk); n_win = pk["n_win"]
    gate_u = max(1, int(round(CFG["gate_u_m"] / DU))); gate_t = max(1, int(round(CFG["gate_ns"] / pk["dt"])))
    tol_t = gate_t; gap_max = int(round(gap_max_m / DS))
    ends = np.array([ln[0][-1] for ln in lines]); starts = np.array([ln[0][0] for ln in lines])
    cand = []
    for a, (wa, ua, ta) in enumerate(ends):
        for b, (wb, ub, tb) in enumerate(starts):
            gap = wb - wa - 1
            if gap < 1 or gap > gap_max or abs(ub - ua) > gate_u or abs(tb - ta) > gate_t:
                continue
            ws = np.arange(wa + 1, wb); f = (ws - wa) / (wb - wa)
            cells = np.c_[ws, np.round(ua + f * (ub - ua)).astype(int), np.round(ta + f * (tb - ta)).astype(int)]
            sup = corridor_support(tab, cells, TOL, tol_t)
            if sup < support_min:
                continue
            cand.append(dict(a=a, b=b, w0=int(wa), w1=int(wb), gap_m=float(gap * DS), cells=cells, support=float(sup)))
    cand.sort(key=lambda j: (-j["support"], j["gap_m"])); used_a, used_b, out = set(), set(), []
    for j in cand:
        if j["a"] in used_a or j["b"] in used_b:
            continue
        used_a.add(j["a"]); used_b.add(j["b"]); out.append(j)
    if rng is not None and out:
        shift_w = int(round(SHIFT_MIN_M / DS))
        if n_win > 2 * shift_w + 1:
            shifts = rng.integers(shift_w, n_win - shift_w, size=n_null)
            for j in out:
                c = j["cells"].copy(); vals = []
                for k in shifts:
                    c[:, 0] = (j["cells"][:, 0] + k) % n_win; vals.append(corridor_support(tab, c, TOL, tol_t))
                j["support_null_mean"] = float(np.mean(vals)); j["support_null_q95"] = float(np.quantile(vals, 0.95))
    return sorted(out, key=lambda j: j["w0"])


def drive_lines_joins(i: int) -> dict | None:
    """Merged lines of the adopted rule and joins for every variant, from the E50 peak cache."""
    pk = rc.peaks(i, CFG["p_min"])
    if pk is None:
        return None
    raw = rl.link_runs(pk, CFG["min_len_m"]); lines = rl.merge_runs(raw, CFG["merge_tol_cells"], CFG["merge_min_share"])
    joins = {vkey(v): find_joins(lines, pk, v["gap_max_m"], v["support_min"]) for v in VARIANTS}
    return dict(pk=pk, lines=lines, joins=joins)


def group_eval(ref: int) -> dict:
    """One lane group: for lines alone and for lines + joins of each variant, rows of drive A reproduced by drive B
    (independent lane) within +-TOL, against along- and across-shifted nulls. Join rows are also scored separately."""
    os.environ.setdefault("OMP_NUM_THREADS", "1"); t0 = time.time()
    J = json.load(open(CITY / f"lane_{ref}.json")); Z = np.load(CITY / f"lane_{ref}.npz"); frame = dict(np.load(CACHE / f"mosaic_frame_c{ref}.npz"))
    det = load_cfg(str(rl.ROOT / "pipeline" / CFG["detector_config"])); s_grid, u_grid = Z["s_grid"], Z["u_grid"]; nrow, ncol = len(s_grid), len(u_grid)
    drives = [int(p) for p in J["drives"] if f"R_{p}" in Z.files]; cov = {i: Z[f"cov_{i}"] for i in drives}; cen = {i: Z[f"cen_{i}"] for i in drives}
    fms = {i: frame_map(i, det, frame) for i in drives}; reg = {i: J["registration"][str(i)] for i in drives}
    D = {i: drive_lines_joins(i) for i in drives}
    rng = np.random.default_rng(SEED + ref); shift_rows = int(round(SHIFT_MIN_M / DS))
    shifts = rng.integers(shift_rows, max(shift_rows + 1, nrow - shift_rows), size=N_SHIFT) if nrow > 2 * shift_rows + 1 else np.array([], int)
    lat = rng.integers(LAT_MIN_CELLS, LAT_MAX_CELLS + 1, size=N_SHIFT) * rng.choice([-1, 1], size=N_SHIFT)

    def to_frame(i, cells_list):
        rows, cols = [], []
        for cells in cells_list:
            s = cells[:, 0] * DS; u = D[i]["pk"]["ug"][np.clip(cells[:, 1], 0, len(D[i]["pk"]["ug"]) - 1)]; sr, ur = fms[i](s, u); sr = sr + reg[i]["dA"]; ur = ur + reg[i]["dB"]
            row = np.round((sr - s_grid[0]) / DS).astype(int); col = np.round((ur - U0) / DU).astype(int)
            ok = (row >= 0) & (row < nrow) & (col >= 0) & (col < ncol); rows.append(row[ok]); cols.append(col[ok])
        return (np.concatenate(rows), np.concatenate(cols)) if rows else (np.zeros(0, int), np.zeros(0, int))

    def score(rc_query, Lg):
        covered = rep = 0; null = null_lat = 0.0
        for i in drives:
            row, col = rc_query[i]
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
        return dict(covered_rows=covered, rep_rows=rep, null_rows=round(float(null), 2), null_lat_rows=round(float(null_lat), 2))

    out = {}; line_rc = {i: to_frame(i, [ln[0] for ln in D[i]["lines"]]) if D[i] else (np.zeros(0, int), np.zeros(0, int)) for i in drives}
    line_m = sum(ln[2] for i in drives if D[i] for ln in D[i]["lines"])

    def dil(rcs):
        g = np.zeros((nrow, ncol), bool); g[rcs[0], rcs[1]] = True
        return maximum_filter1d(g.astype(np.uint8), 2 * TOL + 1, axis=1).astype(bool)
    Lg_lines = {i: dil(line_rc[i]) for i in drives}
    out["lines"] = dict(line_m=round(line_m, 1), n_lines=sum(len(D[i]["lines"]) for i in drives if D[i]), **score(line_rc, Lg_lines))
    for v in VARIANTS:
        k = vkey(v); join_rc = {i: to_frame(i, [j["cells"] for j in D[i]["joins"][k]]) if D[i] else (np.zeros(0, int), np.zeros(0, int)) for i in drives}
        both = {i: (np.r_[line_rc[i][0], join_rc[i][0]], np.r_[line_rc[i][1], join_rc[i][1]]) for i in drives}; Lg_both = {i: dil(both[i]) for i in drives}
        n_j = sum(len(D[i]["joins"][k]) for i in drives if D[i]); m_j = sum(j["gap_m"] for i in drives if D[i] for j in D[i]["joins"][k])
        out[k] = dict(variant=v, n_joins=n_j, join_m=round(m_j, 1), joins_vs_lines=score(join_rc, Lg_lines), joins_vs_both=score(join_rc, Lg_both), all_vs_both=score(both, Lg_both))
    out["_meta"] = dict(ref=ref, drives=drives, n_rows=nrow, seconds=round(time.time() - t0)); return out


def main(n_workers: int = 5) -> None:
    refs = [int(f.stem.split("_")[1]) for f in CITY.glob("lane_*.json") if f.stem != "lane_summary" and json.load(open(f)).get("status") == "ok"]; t0 = time.time()
    print(f"{len(refs)} lane groups, {len(VARIANTS)} join variants", flush=True)
    with ProcessPoolExecutor(n_workers) as ex:
        groups = dict(zip(refs, ex.map(group_eval, refs)))
    for r, g in groups.items():
        print(f"group {r}: {g['_meta']['seconds']} s, lines {g['lines']['n_lines']}, joins gap20_s0.25 {g['gap20_s0.25']['n_joins']}", flush=True)

    def agg(sel):
        res = {}
        def add(key, pick):
            c = sum(pick(groups[r])["covered_rows"] for r in sel); rep = sum(pick(groups[r])["rep_rows"] for r in sel)
            nul = sum(pick(groups[r])["null_rows"] for r in sel); nl = sum(pick(groups[r])["null_lat_rows"] for r in sel)
            return dict(covered_m=round(c * DS), repeated_m=round(rep * DS), null_m=round(nul * DS, 1), null_lat_m=round(nl * DS, 1), excess_min_m=round((rep - max(nul, nl)) * DS, 1),
                        share=None if c == 0 else round(rep / c, 3), null_share=None if c == 0 else round(nul / c, 3), null_lat_share=None if c == 0 else round(nl / c, 3))
        res["lines"] = dict(line_km=round(sum(groups[r]["lines"]["line_m"] for r in sel) / 1e3, 2), n_lines=sum(groups[r]["lines"]["n_lines"] for r in sel), **add("lines", lambda g: g["lines"]))
        for v in VARIANTS:
            k = vkey(v)
            res[k] = dict(variant=v, n_joins=sum(groups[r][k]["n_joins"] for r in sel), join_km=round(sum(groups[r][k]["join_m"] for r in sel) / 1e3, 2),
                          joins_vs_lines=add(k, lambda g, k=k: g[k]["joins_vs_lines"]), joins_vs_both=add(k, lambda g, k=k: g[k]["joins_vs_both"]), all_vs_both=add(k, lambda g, k=k: g[k]["all_vs_both"]))
        return res
    test_refs = [r for r in refs if r not in rc.SELECT_REFS]; S = agg([r for r in refs if r in rc.SELECT_REFS]); T = agg(test_refs)
    keys = [vkey(v) for v in VARIANTS]
    chosen = max(keys, key=lambda k: S[k]["joins_vs_both"]["excess_min_m"])
    out = dict(grid=GRID, line_rule={k: CFG[k] for k in ("p_min", "max_gap", "gate_u_m", "gate_ns", "min_len_m", "tol_cells", "min_sep_m")}, n_shift=N_SHIFT, shift_min_m=SHIFT_MIN_M, seed=SEED,
               select_refs=rc.SELECT_REFS, test_refs=test_refs, selection=S, test=T, chosen=chosen, groups=groups, seconds=round(time.time() - t0))
    json.dump(out, open(CITY / "join_compare.json", "w"), indent=1)
    for name, R in (("selection", S), ("test", T)):
        L = R["lines"]; print(f"\n{name}: lines {L['n_lines']} ({L['line_km']} km) covered {L['covered_m']} m repeated {L['repeated_m']} m share {L['share']} null along {L['null_share']} across {L['null_lat_share']}")
        for k in keys:
            v = R[k]; a = v["joins_vs_both"]; b = v["joins_vs_lines"]
            print(f"  {k:14s} joins {v['n_joins']:4d} ({v['join_km']:5.2f} km)  join rows covered {a['covered_m']:5d} m  repeated by partner lines+joins {a['repeated_m']:4d} m (share {a['share']}, null {a['null_share']}/{a['null_lat_share']}, excess {a['excess_min_m']})"
                  f"  by partner lines only {b['repeated_m']:4d} m (share {b['share']})")
    print(f"\nchosen on selection: {chosen}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
