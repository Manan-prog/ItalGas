"""Whole-survey scan (3 Nov 2026): on every drive, crossing pipes (600 and 200 MHz, column-shift null,
merged into both-row / one-row detections) and along-street tracks (200 MHz, channel-permutation null).
Drives are processed in pieces (config chunk_m) in memory; only detections and a small frame file are saved.
Every detection carries the positioning quality at its place: RT fix quality, post-processed standard
deviation, and |RT - PP| along and across travel. Map positions include the ASSUMED along-travel bias.
Branches (config 'branches'): 'crossings' -> cache/city/scan_<swath index>.json (+ scan_<i>_frame.npz);
'along' -> cache/city/along<along_tag>_<swath index>.json, with the along-street preprocessing overrides of the config
('along_preprocess', e.g. background = signature: the day's instrument signature, DECISIONS 2026-11-03);
'ends' -> cache/city/ends_<swath index>.json: end points of partial crossings per array (terminations.detect_ends,
config section 'ends'), each with the map position of the end point and the positioning quality there.
Run (workers share the drives, longest first):  python citywide_scan.py <config> <worker> <n_workers> [indices]
or, one command for the whole survey:            python citywide_scan.py <config> --workers <n>   (starts n shards as subprocesses)"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from build_cube import swath_geometry
from dither import ground_to_map, shift_at, sideways, y_at
from build_v2 import grid_Z, instrument_signature, load_cfg, preprocess_array
from cube import to_su
from finalize_v2 import merge_crossings
from geometry import lonlat_to_merc
from ids_reader import Swath
from paths import load_yaml
from run_crossings_v2 import detect_crossings, migrate_grid
from run_detection import su_to_map
from run_longitudinal_v2 import detect_longitudinal
from terminations import detect_ends
from run_multipass import longest_run
from trajectory_qc import along_cross_disagreement

ROOT = Path(__file__).resolve().parents[1]
DS_Q = 0.5                                     # m, sampling of the quality / coverage profiles along s


def quality_profiles(sw: Swath, g: dict, cc: dict) -> dict:
    """Positioning quality along the drive's own s (every DS_Q m): rt_q (GGA fix quality, 4 = RTK fixed),
    pp_sd_m (post-processed standard deviation), d_along_m / d_cross_m (|PP - RT|, median-filtered;
    nan where the post-processed solution is missing or off by more than pp_fail_m)."""
    sn, xn, yn, k = g["s_nodes"], g["xn"], g["yn"], float(g["k"]); s = np.arange(0.0, float(sn[-1]), DS_Q)
    out = dict(s=s)
    def onto(lon, lat, val, lat_max=5.0):
        x, y = lonlat_to_merc(lon, lat); ss, uu = to_su(x, y, sn, xn, yn, k); m = np.abs(uu) < lat_max
        if m.sum() < 2:
            return np.full(len(s), np.nan)
        o = np.argsort(ss[m]); i = np.clip(np.searchsorted(ss[m][o], s), 0, m.sum() - 1)
        return np.asarray(val[m][o][i], float)
    out["rt_q"] = onto(sw.gga_rt["lon"], sw.gga_rt["lat"], sw.gga_rt["q"]) if sw.gga_rt is not None and len(sw.gga_rt) else np.full(len(s), np.nan)
    tq = sw.trj_quality
    out["pp_sd_m"] = onto(tq["lon"], tq["lat"], tq["sd_m"]) if tq is not None and len(tq) else np.full(len(s), np.nan)
    out["d_along_m"] = np.full(len(s), np.nan); out["d_cross_m"] = np.full(len(s), np.nan)
    if sw.gga_pp is not None and len(sw.gga_pp) > 10:
        try:
            dq = along_cross_disagreement(sw.path, sw.gga_pp, k)
            for key in ("along", "cross"):
                d = np.abs(median_filter(dq[key], 51, mode="nearest")); d = np.where(d > cc["pp_fail_m"], np.nan, d)
                out[f"d_{key}_m"] = np.interp(s, dq["s_rt"], d, left=np.nan, right=np.nan)
        except Exception:                          # PP track not alignable in time: leave unknown
            pass
    return out


def at(q: dict, s0: float) -> dict:
    """Positioning quality at along-track position s0 (m); None where unknown."""
    i = int(np.clip(round(s0 / DS_Q), 0, len(q["s"]) - 1)); f = lambda v: None if not np.isfinite(v) else round(float(v), 3)
    return dict(rt_q=f(q["rt_q"][i]), pp_sd_m=f(q["pp_sd_m"][i]), d_along_m=f(q["d_along_m"][i]), d_cross_m=f(q["d_cross_m"][i]))


def day_files(f: str, raw_root: str) -> list[str]:
    """All drives recorded on the same date as drive f (session folders carry the date as YYYY.MM.DD)."""
    day = Path(f).parents[1].name.split("_")[-2]
    return sorted(glob.glob(str(Path(raw_root) / f"*{day}*" / "Survey_*" / "Swath*.path")))


def along_tracks(sw: Swath, g: dict, q: dict, sl: slice, s_grid: np.ndarray, c0: float, c1: float, det: dict, cc: dict,
                 rng: np.random.Generator, raw: np.ndarray | None = None, signature: np.ndarray | None = None, sv: dict | None = None) -> list[dict]:
    """Along-street tracks of one piece (200 MHz): preprocessing with det['preprocess'], detect_longitudinal, tracks
    starting inside [c0, c1) with q <= keep_q and length >= 5 m, path decimated x4, map coordinates and quality added."""
    sn, xn, yn, k = g["s_nodes"], g["xn"], g["yn"], float(g["k"]); B = cc["along_travel_bias_m"]
    pp = preprocess_array(sw, 2, det, scans=sl, raw=raw, signature=signature); S = g["s2"][sl]; u_ch = np.nanmedian(g["u2"][sl], 0)
    Z = grid_Z(pp["X"], S, s_grid)
    y_win = (lambda ss: shift_at(sv, ss)) if sv is not None else None                 # E43 ground frame: u_ground = u_frame + y_van - uc
    tracks, _ = detect_longitudinal(Z, u_ch, s_grid, pp["dt_ns"], pp["snr"], det, rng, y_win=y_win); out = []
    for t in tracks:
        if not (c0 <= t["s_start"] < c1) or t["q"] > cc["keep_q"] or t["length_m"] < 5.0:
            continue
        ps, pu = np.array(t["path_s"][::4]), np.array(t["path_u"][::4])
        if sv is not None:
            mx, my = ground_to_map(sv, ps + 0.0, pu); t["frame"] = "ground"; t["y_range_m"] = round(float(np.ptp(np.nan_to_num(y_at(sv, ps)))), 3)
        else:
            mx, my = su_to_map(ps + B, pu, sn, xn, yn, k); t["frame"] = "instrument"
        t.update(path_s=ps.tolist(), path_u=pu.tolist(), path_t=t["path_t"][::4], x=np.round(mx, 3).tolist(), y=np.round(my, 3).tolist(),
                 **at(q, 0.5 * (t["s_start"] + t["s_end"])))
        out.append(t)
    return out


def along_det(det: dict, cc: dict) -> dict:
    """Detector config with the along-street preprocessing and longitudinal overrides applied."""
    return dict(det, preprocess=dict(det["preprocess"], **cc.get("along_preprocess", {})),
                longitudinal=dict(det["longitudinal"], **cc.get("along_longitudinal", {})))


def scan_swath(f: str, det: dict, cc: dict, rng: np.random.Generator) -> tuple[dict, dict]:
    sw = Swath.load(str(Path(f).parent), Path(f).stem); br = cc.get("branches", ["crossings", "along"])
    detA = along_det(det, cc)
    sig = instrument_signature(day_files(f, cc["raw_root"]), 2, detA) if "along" in br and detA["preprocess"].get("background") == "signature" else None
    res = dict(file=f, session=Path(f).parents[1].name, survey=Path(f).parent.name, swath=Path(f).stem, crossings=[], per_array={"1": [], "2": []}, tracks=[],
               ends={"1": [], "2": []})
    if "ends" in br:
        det = dict(det, ends=cc["ends"])
    if 2 not in sw.index or len(sw.index[2]) < cc["min_scans"]:
        res["status"] = "too short"; return res, {}
    g = swath_geometry(sw, det["base"]); sn, xn, yn, k = g["s_nodes"], g["xn"], g["yn"], float(g["k"]); L = float(sn[-1])
    q = quality_profiles(sw, g, cc); B = cc["along_travel_bias_m"]
    sc = {a: np.nanmedian(g[f"s{a}"], 1) for a in (1, 2)}
    sv = sideways(g, B, cc["L_ref_m"]) if cc.get("frame") in ("ground", "ground_mirror") else None
    if cc.get("frame") == "ground_mirror":                 # E43 mirror control: wrong sign of the van's sideways position
        sv["y"] = -sv["y"]
    # coverage: lateral extent of the channel footprints of both arrays along s (m)
    lo = np.full(len(q["s"]), np.nan); hi = np.full(len(q["s"]), np.nan)
    for a in (1, 2):
        b = np.clip((sc[a] / DS_Q).round().astype(int), 0, len(q["s"]) - 1); umin = np.nanmin(g[f"u{a}"], 1); umax = np.nanmax(g[f"u{a}"], 1)
        np.fmin.at(lo, b, umin); np.fmax.at(hi, b, umax)
    frame = dict(s_nodes=sn, xn=xn, yn=yn, k=k, **{kk: np.asarray(v, np.float32) for kk, v in q.items()}, lo=lo.astype(np.float32), hi=hi.astype(np.float32))
    C, P, ds = cc["chunk_m"], cc["pad_m"], det["grid"]["ds_m"]; v = det["migration"]["velocity_m_ns"]
    for c0 in np.arange(0.0, L, C):
        c1 = min(L, c0 + C); s_grid = np.arange(max(0.0, c0 - P), min(L, c1 + P), ds)
        if len(s_grid) * ds < 60.0:
            continue
        X = {1: [], 2: []}
        for a in (1, 2):
            sl = longest_run((sc[a] > s_grid[0] - 15) & (sc[a] < s_grid[-1] + 15))
            if sl is None or sl.stop - sl.start < 200:
                continue
            if a == 2 and "along" in br:
                res["tracks"] += along_tracks(sw, g, q, sl, s_grid, c0, c1, detA, cc, rng, signature=sig, sv=sv)
            if "crossings" not in br and "ends" not in br:
                continue
            pp = preprocess_array(sw, a, det, scans=sl); S = g[f"s{a}"][sl]; u_ch = np.nanmedian(g[f"u{a}"][sl], 0)
            G = migrate_grid(pp["X"], S, s_grid, pp["dx_m"], pp["dt_ns"], det)
            if "ends" in br:
                ends, _ = detect_ends(G, u_ch, s_grid, pp["dt_ns"] * det["grid"]["t_down"], pp["snr"], pp["dt_ns"], det, rng)
                for e in ends:
                    if c0 <= e["s0_m"] < c1 and e["q"] <= cc["keep_q"]:
                        se = e["s0_m"] + e["u_end_m"] * np.tan(np.radians(e["phi_deg"])); mx, my = su_to_map(np.array([se + B]), np.array([e["u_end_m"]]), sn, xn, yn, k)
                        res["ends"][str(a)].append(dict(e, s_end_m=round(float(se), 2), x=round(float(mx[0]), 3), y=round(float(my[0]), 3), **at(q, se)))
            if "crossings" not in br:
                del G, pp; continue
            cands, _ = detect_crossings(G, u_ch, s_grid, pp["dt_ns"] * det["grid"]["t_down"], pp["snr"], pp["dt_ns"], det, rng); del G
            X[a] = [c for c in cands if c0 <= c["s0_m"] < c1]
            res["per_array"][str(a)] += [dict(s0_m=c["s0_m"], phi_deg=c["phi_deg"], depth_m=c["depth_m"], q=c["q"], p_real=c["p_real"], score=c["score"])
                                         for c in X[a] if c["q"] <= cc["keep_q"]]
            del pp
        for o in merge_crossings(X[1], X[2], det["crossing"]["q_max"]):
            j = int(np.clip(round(o["s0_m"] / DS_Q), 0, len(q["s"]) - 1)); u0, u1 = float(lo[j]), float(hi[j])
            if not np.isfinite(u0):
                u0, u1 = -0.75, 0.75
            uu = np.array([u0, u1]); ss = o["s0_m"] + B + uu * np.tan(np.radians(o["phi_deg"]))
            mx, my = su_to_map(ss, uu, sn, xn, yn, k)
            res["crossings"].append(dict(o, u0=round(u0, 3), u1=round(u1, 3), x=np.round(mx, 3).tolist(), y=np.round(my, 3).tolist(), **at(q, o["s0_m"])))
    res.update(status="ok", length_m=round(L, 1), branches=br, along_preprocess=detA["preprocess"] if "along" in br else None,
               along_longitudinal=detA["longitudinal"] if "along" in br else None)
    return res, frame


def main(cfg_path: str, worker: int, n_workers: int) -> None:
    cc = load_yaml(cfg_path); det = load_cfg(str(ROOT / "pipeline" / cc["detector_config"]))
    out_dir = Path(det["base"]["cache_dir"]) / "city"; out_dir.mkdir(exist_ok=True)
    files = sorted(glob.glob(str(Path(cc["raw_root"]) / "*" / "Survey_*" / "Swath*.path")))   # index = position in this list
    size = [Path(f).with_name(Path(f).stem + "_Array02.data").stat().st_size if Path(f).with_name(Path(f).stem + "_Array02.data").exists() else 0 for f in files]
    order = [int(i) for i in np.argsort(size)[::-1]][worker::n_workers]    # longest first, round robin
    if len(sys.argv) > 4:
        order = [int(a) for a in sys.argv[4].split(",")]
    t0 = time.time()
    br = cc.get("branches", ["crossings", "along"])
    for idx in order:
        fj = out_dir / (f"scan_{idx}.json" if "crossings" in br else f"ends_{idx}.json" if "ends" in br else f"along{cc.get('along_tag', '')}_{idx}.json")
        if fj.exists() and not json.load(open(fj)).get("status", "").startswith("error"):
            continue                                 # done (errors are retried, e.g. after a reader fix)
        t1 = time.time(); rng = np.random.default_rng(det["seed"] + 61 + idx)
        try:
            res, frame = scan_swath(files[idx], det, cc, rng)
            if frame and "crossings" in br:
                np.savez(out_dir / f"scan_{idx}_frame.npz", **frame)
        except Exception as e:
            res = dict(file=files[idx], status=f"error: {e}", trace=traceback.format_exc()[-1500:])
        res.update(index=idx, seconds=round(time.time() - t1, 1))
        json.dump(res, open(fj, "w"))
        nA = sum(c["cls"] == "A" for c in res.get("crossings", []))
        print(f"[w{worker}] swath {idx} {Path(files[idx]).parent.name} {Path(files[idx]).stem}: {res['status']}, {res.get('length_m', 0)} m, "
              f"crossings {len(res.get('crossings', []))} (both rows {nA}), tracks {len(res.get('tracks', []))}, {res['seconds']}s, total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[2] == "--workers":                     # one process per shard, BLAS single-threaded per process
        n = int(sys.argv[3]); env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        procs = [subprocess.Popen([sys.executable, __file__, sys.argv[1], str(k), str(n)] + sys.argv[4:], env=env) for k in range(n)]
        sys.exit(max(p.wait() for p in procs))
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
