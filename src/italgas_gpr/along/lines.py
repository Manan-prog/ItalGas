"""Along-street pipe lines derived from the per-cell evidence rasters, one rule set (configs/along_lines.yaml).
Per drive: cells with P >= p_min that are local maxima are linked window to window (0.5 m) within gate_u_m across and
gate_ns in time, bridging max_gap empty windows; runs of min_len_m or more become lines. Each line carries its mean and
lowest P, depth, length, the positioning quality at its middle and, where the drive belongs to a repeat-drive group
(E47), the agreement of the other drives at the same ground position after registration.
Also writes a light frame per drive (rframe_<i>.npz: nodes, strip extent, positioning quality) for the web page.
Run: python -m italgas_gpr.along.lines [n_workers] [cache dir] -> cache/city/rline_<i>.json, rframe_<i>.npz, raster_lines.json"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from italgas_gpr.geometry.swath import swath_geometry  # noqa: E402
from italgas_gpr.processing.preprocess import load_cfg  # noqa: E402
from italgas_gpr.crossing.scan import DS_Q, at, quality_profiles  # noqa: E402
from italgas_gpr.io.ids_reader import Swath  # noqa: E402
from italgas_gpr.along.focus import link_tracks, window_peaks  # noqa: E402
from italgas_gpr.lanes.register import DS, DU, U0  # noqa: E402
from italgas_gpr.lanes.agreement import frame_map  # noqa: E402
from italgas_gpr.along.evidence import search_columns, search_samples  # noqa: E402
from italgas_gpr.geometry.frames import su_to_map  # noqa: E402

from italgas_gpr.io.paths import CACHE, config_path, load_yaml  # noqa: E402

CFG = load_yaml(config_path("along_lines.yaml"))


class PeakScore:
    """score[(w, iu, it)] for the linker, holding P only at the peaks (the full P cube of a drive is too large)."""
    def __init__(self, keys: np.ndarray, vals: np.ndarray):
        self.d = {(int(a), int(b), int(c)): float(v) for (a, b, c), v in zip(keys, vals)}

    def __getitem__(self, key):
        return self.d[(int(key[0]), int(key[1]), int(key[2]))]


def drive_peaks(i: int, cache: Path, p_min: float) -> dict | None:
    """Local maxima of P >= p_min over all pieces of drive i: keys (n, 3) = (window, iu, it) on the drive's common grids, vals = P."""
    files = sorted(glob.glob(str(cache / f"prob_raster_{i}_*.npz")))
    if not files:
        return None
    keys, vals = [], []
    Ds = [np.load(pf) for pf in files]; ugs = [d["ug"] for d in Ds]                 # one open per piece (arrays load lazily)
    j_lo = min(int(round((float(u[0]) - U0) / DU)) for u in ugs); j_hi = max(int(round((float(u[-1]) - U0) / DU)) for u in ugs)
    ug = U0 + DU * np.arange(j_lo, j_hi + 1); dt = None; snap_max = 0.0            # common lattice of the drive anchored at U0 (as mosaic_fast does);
    for d, ugp in zip(Ds, ugs):                                                   # a piece's own origin (u.min + 0.06 m, along/evidence.py) is snapped to it, error <= DU/2 = 0.02 m
        t = d["t_ns"]; it = search_samples(d, CFG["t_min_ns"]); iu = search_columns(d, CFG["u_min_m"], CFG["u_max_m"])
        cs = d["centres_s"]; keep = np.nonzero((cs >= d["c0"]) & (cs < d["c1"]))[0]; dt = float(t[1] - t[0])
        off = int(round((float(ugp[0]) - U0) / DU)) - j_lo; snap = abs(float(ugp[0]) - float(ug[off])); snap_max = max(snap_max, snap)
        assert snap <= DU / 2 + 1e-6, f"piece origin {ugp[0]:.4f} m is more than half a cell off the lattice"
        if not len(iu) or not len(it):
            continue
        P = d["P"].astype(np.float32)[keep][:, iu][:, :, it]                      # evaluated band only (floor, lateral exclusion)
        pk = window_peaks(P, p_min, max(1, int(round(CFG["sep_u_m"] / DU))), max(1, int(round(CFG["sep_ns"] / dt))), (0, P.shape[2]))
        if len(pk):
            w = np.round(cs[keep][pk[:, 0]] / DS).astype(int)                    # global window index = s / 0.5 m
            keys.append(np.c_[w, pk[:, 1] + off + iu[0], pk[:, 2] + it[0]]); vals.append(P[pk[:, 0], pk[:, 1], pk[:, 2]])
    n_win = int(round(max(float(d["c1"]) for d in Ds) / DS))
    if not keys:
        return dict(ug=ug, dt=dt, keys=np.zeros((0, 3), int), vals=np.zeros(0), n_win=n_win, snap_m=snap_max)
    keys = np.concatenate(keys); vals = np.concatenate(vals); o = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    n_win = max(n_win, int(keys[:, 0].max()) + 1)                                  # a centre just below c1 can round up to n_win
    return dict(ug=ug, dt=dt, keys=keys[o], vals=vals[o], n_win=n_win, snap_m=snap_max)


def link_runs(pk: dict, min_len_m: float, perm: np.ndarray | None = None, rule: dict | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    """Link peaks window to window (gates from the config) into runs of min_len_m or more. perm: optional window
    permutation (new index of each window) applied before linking = chance reference that keeps every window's
    cells but destroys continuity between blocks. Returns [(cells (n, 3) in ORIGINAL window indices, P)]."""
    keys, vals = pk["keys"], pk["vals"]
    if not len(keys):
        return []
    sc = PeakScore(keys, vals)
    if perm is not None:
        kp = keys.copy(); kp[:, 0] = perm[keys[:, 0]]; o = np.lexsort((kp[:, 2], kp[:, 1], kp[:, 0])); kp, vp = kp[o], vals[o]
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm)); scp = PeakScore(kp, vp)
    else:
        kp, vp, inv, scp = keys, vals, None, sc
    R_ = CFG if rule is None else {**CFG, **rule}                                   # rule: optional override of gate_u_m, gate_ns, max_gap (E50)
    tr = link_tracks(kp, scp, max(1, int(round(R_["gate_u_m"] / DU))), max(1, int(round(R_["gate_ns"] / pk["dt"]))), R_["max_gap"])
    out = []
    for t in tr:
        length = float((t[-1, 0] - t[0, 0]) * DS)
        if length >= min_len_m:
            P = np.array([scp[p] for p in t]); tt = t.copy()
            if inv is not None:
                tt[:, 0] = inv[t[:, 0]]
            out.append((tt, P, length))
    return out


def chance_lines(pk: dict, min_len_m: float) -> dict:
    """Lines the rule produces on block-shuffled windows of the same drive (configs/along_lines.yaml: chance)."""
    c = CFG["chance"]; rng = np.random.default_rng(c["seed"]); n, m = [], []
    for _ in range(c["n_draws"]):
        perm = block_perm(pk["n_win"], c["block_windows"], rng); runs = []
        for cells, P, L in link_runs(pk, min_len_m, perm):                              # merge in the SHUFFLED window order (lengths are meaningful there)
            cp = cells.copy(); cp[:, 0] = perm[cells[:, 0]]; o = np.argsort(cp[:, 0]); runs.append((cp[o], P[o], L))
        r = merge_runs(runs, CFG["merge_tol_cells"], CFG["merge_min_share"]); n.append(len(r)); m.append(sum(x[2] for x in r))
    return dict(n_draws=c["n_draws"], block_m=c["block_windows"] * DS, lines_mean=float(np.mean(n)), lines_q95=float(np.quantile(n, 0.95)), length_mean_m=float(np.mean(m)))


def merge_runs(runs: list, tol_cells: int, min_share: float) -> list:
    """Plan-view merge (E50 caveat, DECISIONS 2026-11-05): runs that lie on top of each other in plan are ONE line.
    Two runs join when, on the windows they share, their lateral cells are within tol_cells on at least min_share of the
    shorter run's windows (transitive). Returns [(cells (n, 3) = window, mean lateral cell, SHALLOWEST time cell; P = max
    of the members per window; length_m; members = [(depth cells median, mean P, length) per run])]. Loop over runs."""
    n = len(runs)
    if n == 0:
        return []
    maps = [dict(zip(c[:, 0].tolist(), zip(c[:, 1].tolist(), c[:, 2].tolist(), P.tolist()))) for c, P, _ in runs]
    w0 = np.array([c[:, 0].min() for c, _, _ in runs]); w1 = np.array([c[:, 0].max() for c, _, _ in runs]); parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in np.argsort(w0):
        for j in np.nonzero((w0 > w0[i]) | ((w0 == w0[i]) & (np.arange(n) > i)))[0]:
            if w0[j] > w1[i]:
                continue
            close = sum(abs(maps[i][w][0] - maps[j][w][0]) <= tol_cells for w in maps[i].keys() & maps[j].keys())
            if close and close / min(len(maps[i]), len(maps[j])) >= min_share:
                parent[find(i)] = find(j)
    groups = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(k)
    out = []
    for mem in groups.values():
        ws = sorted(set().union(*[maps[k].keys() for k in mem])); cells = []; P = []
        for w in ws:
            here = [maps[k][w] for k in mem if w in maps[k]]
            cells.append((w, int(round(np.mean([h[0] for h in here]))), min(h[1] for h in here))); P.append(max(h[2] for h in here))
        cells = np.array(cells, int); members = [(float(np.median(runs[k][0][:, 2])), float(runs[k][1].mean()), float(runs[k][2])) for k in mem]
        out.append((cells, np.array(P), float((cells[-1, 0] - cells[0, 0]) * DS), sorted(members)))
    return sorted(out, key=lambda o: o[0][0, 0])


def block_perm(n_win: int, block: int, rng) -> np.ndarray:
    """Permutation of window indices that moves whole blocks of `block` consecutive windows (keeps the 2 m stack's
    within-block correlation, breaks continuity across blocks)."""
    nb = int(np.ceil(n_win / block)); order = rng.permutation(nb)
    src = np.concatenate([np.arange(b * block, (b + 1) * block) for b in order])     # new position -> old window
    src = src[src < n_win]; perm = np.empty(n_win, int); perm[src] = np.arange(len(src)); return perm


def drive_runs(i: int, cache: Path) -> dict | None:
    """Lines of drive i from its raster pieces (own-drive frame). Returns dict(ug, dt, runs=[(cells, P)])."""
    pk = drive_peaks(i, cache, CFG["p_min"])
    if pk is None:
        return None
    raw = link_runs(pk, CFG["min_len_m"]); merged = merge_runs(raw, CFG["merge_tol_cells"], CFG["merge_min_share"])
    return dict(ug=pk["ug"], dt=pk["dt"], runs=merged, n_raw_runs=len(raw), chance=chance_lines(pk, CFG["min_len_m"]), peaks_per_window=len(pk["keys"]) / max(pk["n_win"], 1))


def drive_lines(args) -> str:
    """One drive: frame (rframe_<i>.npz) and lines (rline_<i>.json) in map coordinates."""
    i, cache = args; os.environ.setdefault("OMP_NUM_THREADS", "1")
    det = load_cfg(config_path(CFG["detector_config"])); files = sorted(glob.glob(f"{CFG['raw_root']}/*/Survey_*/Swath*.path")); f = files[i]
    rr = drive_runs(i, cache)
    if rr is None:
        return f"{i}: no rasters"
    sw = Swath.load(str(Path(f).parent), Path(f).stem); g = swath_geometry(sw, det["base"]); sn, xn, yn, k = g["s_nodes"], g["xn"], g["yn"], float(g["k"])
    q = quality_profiles(sw, g, dict(pp_fail_m=CFG["pp_fail_m"]))
    lo = np.full(len(q["s"]), np.nan); hi = np.full(len(q["s"]), np.nan)
    for a in (1, 2):
        if f"s{a}" not in g:
            continue
        sc = np.nanmedian(g[f"s{a}"], 1); b = np.clip((sc / DS_Q).round().astype(int), 0, len(q["s"]) - 1)
        np.fmin.at(lo, b, np.nanmin(g[f"u{a}"], 1)); np.fmax.at(hi, b, np.nanmax(g[f"u{a}"], 1))
    np.savez(cache / "city" / f"rframe_{i}.npz", s_nodes=sn, xn=xn, yn=yn, k=k, **{kk: np.asarray(v, np.float32) for kk, v in q.items()},
             lo=lo.astype(np.float32), hi=hi.astype(np.float32), length_m=float(sn[-1]))
    B = CFG["along_travel_bias_m"]; v = CFG["velocity_m_ns"]; lines = []
    for t, P, _, mem in rr["runs"]:
        s = t[:, 0] * DS; u = rr["ug"][t[:, 1]]; z = v * (t[:, 2] * rr["dt"]) / 2
        x, y = su_to_map(s + B, u, sn, xn, yn, k)
        lines.append(dict(s0=float(s[0]), s1=float(s[-1]), length_m=float(s[-1] - s[0]), n_windows=int(len(s)), P_mean=round(float(P.mean()), 3), P_min=round(float(P.min()), 3),
                          depth_m=round(float(np.median(z)), 2), depth_range_m=[round(float(z.min()), 2), round(float(z.max()), 2)], u_mean=round(float(u.mean()), 2),
                          n_runs=len(mem), runs=[dict(depth_m=round(v * d * rr["dt"] / 2, 2), P_mean=round(p, 3), length_m=round(L, 1)) for d, p, L in mem],
                          s=np.round(s, 2).tolist(), u=np.round(u, 3).tolist(), x=np.round(x, 2).tolist(), y=np.round(y, 2).tolist(), **at(q, 0.5 * (s[0] + s[-1]))))
    out = dict(index=i, file=f, session=Path(f).parents[1].name, survey=Path(f).parent.name, swath=Path(f).stem, length_m=float(sn[-1]), n_pieces=len(glob.glob(str(cache / f"prob_raster_{i}_*.npz"))),
               peaks_per_window=round(rr["peaks_per_window"], 2), n_raw_runs=rr["n_raw_runs"], chance=rr["chance"], lines=lines)
    json.dump(out, open(cache / "city" / f"rline_{i}.json", "w"))
    return f"{i}: {len(lines)} lines, {sum(l['length_m'] for l in lines):.0f} m; by chance {rr['chance']['lines_mean']:.1f} lines, {rr['chance']['length_mean_m']:.0f} m"


def repeat_attributes(cache: Path) -> dict:
    """For every line of a drive in a repeat-drive group: rows covered by an independent-lane partner, rows where that
    partner shows P >= p_min within +-tol_cells, per partner; self-check = share of the line's own rows found in its own
    registered grid (should be near 1; verifies the registration sign)."""
    det = load_cfg(config_path(CFG["detector_config"])); out = {}
    for jf in sorted((cache / "city").glob("lane_*.json")):
        if jf.stem == "lane_summary":
            continue
        J = json.load(open(jf))
        if J.get("status") != "ok":
            continue
        ref = J["reference"]; T = J["thresholds"]["0.5"]; Z = np.load(cache / "city" / f"lane_{ref}.npz"); frame = dict(np.load(cache / f"mosaic_frame_c{ref}.npz"))
        s_grid, u_grid = Z["s_grid"], Z["u_grid"]; R = {int(p): Z[f"R_{p}"] for p in J["drives"] if f"R_{p}" in Z.files}
        cov = {p: Z[f"cov_{p}"] for p in R}; cen = {p: Z[f"cen_{p}"] for p in R}
        for i in R:
            lf = cache / "city" / f"rline_{i}.json"
            if not lf.exists():
                continue
            L = json.load(open(lf)); reg = J["registration"][str(i)]; fm = frame_map(i, det, frame)
            for n, ln in enumerate(L["lines"]):
                sr, ur = fm(np.array(ln["s"]), np.array(ln["u"])); sr = sr + reg["dA"]; ur = ur + reg["dB"]
                row = np.round((sr - s_grid[0]) / DS).astype(int); col = np.round((ur - U0) / DU).astype(int)
                ok = (row >= 0) & (row < len(s_grid)) & (col >= 0) & (col < len(u_grid)); row, col = row[ok], col[ok]
                if not len(row):
                    continue
                tol = CFG["tol_cells"]; c0, c1 = np.maximum(col - tol, 0), np.minimum(col + tol + 1, len(u_grid))
                def hits(Rj):
                    return np.array([Rj[r_, a:b].max() >= CFG["p_min"] for r_, a, b in zip(row, c0, c1)])
                selfhit = float(hits(R[i]).mean()); partners = {}
                for j in R:
                    if j == i:
                        continue
                    indep = np.abs(cen[i][row] - cen[j][row]) >= CFG["min_sep_m"]; covered = cov[j][row, col] & indep & np.isfinite(cen[i][row]) & np.isfinite(cen[j][row])
                    if covered.sum() == 0:
                        continue
                    h = hits(R[j]) & covered
                    partners[str(j)] = dict(covered_rows=int(covered.sum()), hit_rows=int(h.sum()), share=round(float(h.sum() / covered.sum()), 3),
                                            repeats=bool(covered.sum() >= CFG["min_covered_rows"] and h.sum() / covered.sum() >= CFG["repeat_min_share"]))
                out[f"{i}:{n}"] = dict(component=ref, f_real=T["f_real"], f_shift_q95=T["f_shift_q95"], p_shift=T["p_shift"], self_check=round(selfhit, 3), rows=int(len(row)),
                                       partners=partners, n_repeating=sum(p["repeats"] for p in partners.values()), covered_by_any=int(sum(p["covered_rows"] > 0 for p in partners.values()) > 0))
    return out


def main(n_workers: int = 7, cache: Path = CACHE) -> None:
    drives = sorted({int(Path(f).stem.split("_")[2]) for f in glob.glob(str(cache / "prob_raster_*_*.npz"))}); t0 = time.time()
    print(f"{len(drives)} drives with rasters, {n_workers} workers", flush=True)
    with ProcessPoolExecutor(n_workers) as ex:
        for msg in ex.map(drive_lines, [(i, cache) for i in drives]):
            print(msg, flush=True)
    rep = repeat_attributes(cache); feats = []; per_drive = {}
    summ = dict(n_drives=len(drives), n_lines=0, length_m=0.0, chance_lines=0.0, chance_length_m=0.0, n_repeated=0, length_repeated_m=0.0, n_covered=0, length_covered_m=0.0, self_check_median=None)
    for i in drives:
        lf = cache / "city" / f"rline_{i}.json"
        if not lf.exists():
            continue
        L = json.load(open(lf)); c = L["chance"]; per_drive[str(i)] = dict(n_lines=len(L["lines"]), length_m=round(sum(l["length_m"] for l in L["lines"]), 1), chance_lines=round(c["lines_mean"], 2),
                                                                           chance_length_m=round(c["length_mean_m"], 1), peaks_per_window=L["peaks_per_window"], drive_length_m=round(L["length_m"]))
        summ["chance_lines"] += c["lines_mean"]; summ["chance_length_m"] += c["length_mean_m"]
        for n, ln in enumerate(L["lines"]):
            r = rep.get(f"{i}:{n}"); props = {k_: v_ for k_, v_ in ln.items() if k_ not in ("s", "u", "x", "y")}
            props.update(drive=i, session=L["session"], survey=L["survey"], swath=L["swath"], repeat=r, chance_share=round(c["lines_mean"] / max(len(L["lines"]), 1), 3))
            feats.append(dict(type="Feature", geometry=dict(type="LineString", coordinates=np.c_[ln["x"], ln["y"]].tolist()), properties=props))
            summ["n_lines"] += 1; summ["length_m"] += ln["length_m"]
            if r and r["covered_by_any"]:
                summ["n_covered"] += 1; summ["length_covered_m"] += ln["length_m"]
            if r and r["n_repeating"] > 0:
                summ["n_repeated"] += 1; summ["length_repeated_m"] += ln["length_m"]
    sc = [r["self_check"] for r in rep.values()]; summ["self_check_median"] = float(np.median(sc)) if sc else None; summ["seconds"] = round(time.time() - t0)
    json.dump(dict(config=CFG, summary=summ, per_drive=per_drive, features=feats), open(cache / "city" / "raster_lines.json", "w"))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7, Path(sys.argv[2]) if len(sys.argv) > 2 else CACHE)
