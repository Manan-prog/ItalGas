"""E42 sideways-motion test on the real along-street tracks of the 12 subset drives (rules: DECISIONS 2026-11-04 + amendments).
Per track: path_u = a0 + a1 s + b y_van + e (dither.slope); ground-fixed -> b about -1 (-0.81 or below with the quadratic
reference), instrument-fixed -> b = 0. Primary: F = fraction of testable tracks with b < -0.5 vs a shift null.
Also: sensitivity to the reference line, and RT vs post-processed agreement of y_van (positioning check).
Run: python dither_test.py configs/dither.yaml -> cache/city/dither_test.json"""
from __future__ import annotations

import copy
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from build_cube import swath_geometry
from build_v2 import load_cfg
from citywide_scan import quality_profiles
from cube import to_su
from dither import pooled_b, shift_null_b, shift_null_fit, sideways, slope, track_test, y_at
from ids_reader import Swath
from run_detection import su_to_map
from run_dither_injection import fixed_fn

ROOT = Path(__file__).resolve().parents[1]


def strip_centre_200(g: dict, s: np.ndarray) -> np.ndarray:
    """Midpoint of the 200 MHz channel footprints' lateral extremes (m, frame u) at frame positions s."""
    S = np.nanmedian(g["s2"], 1); mid = 0.5 * (np.nanmin(g["u2"], 1) + np.nanmax(g["u2"], 1)); ok = np.isfinite(S) & np.isfinite(mid)
    o = np.argsort(S[ok]); return np.interp(s, S[ok][o], mid[ok][o])


def y_per_scan(g: dict, sv: dict, B: float) -> tuple[np.ndarray, np.ndarray]:
    """Array-centre sideways position y (m) for every scan (unsorted, scan order) on sv's reference line; and frame s."""
    S = np.nanmean(g["s2"], 1).astype(float); U = np.nanmean(g["u2"], 1).astype(float); ok = np.isfinite(S) & np.isfinite(U)
    y = np.full(len(S), np.nan)
    X, Y = su_to_map(S[ok] + B, U[ok], g["s_nodes"], g["xn"], g["yn"], float(g["k"]))
    y[ok] = to_su(X, Y, sv["R"][0], sv["R"][1], sv["R"][2], sv["k"])[1]
    y[ok] = np.where((S[ok] >= sv["edge"][0]) & (S[ok] <= sv["edge"][1]), y[ok], np.nan)
    return y, S


def main(cfg_path: str) -> None:
    cc = yaml.safe_load(open(cfg_path)); det = load_cfg(str(ROOT / "pipeline" / cc["detector_config"])); B = cc["along_travel_bias_m"]
    files = sorted(glob.glob(str(Path(cc["raw_root"]) / "*" / "Survey_*" / "Swath*.path"))); cache = Path(det["base"]["cache_dir"]) / "city"
    refs = {"sg2_60": ("sg2", cc["L_ref_m"])} | {f"sg2_{int(L)}": ("sg2", L) for L in cc["L_ref_sens_m"]} | {"mean_60": ("mean", cc["L_ref_m"])}
    base_pp = copy.deepcopy(det["base"]); base_pp["track"]["source"] = "pp"
    tracks, ctx, poscheck, posres = [], {}, {}, {}; t0 = time.time()
    for i in cc["drives"]:
        tr = [t for t in json.load(open(cache / f"along{cc['along_tag']}_{i}.json"))["tracks"] if t["q"] <= 0.05 and t["length_m"] >= 10]
        f = files[i]; sw = Swath.load(str(Path(f).parent), Path(f).stem); g = swath_geometry(sw, det["base"]); q = quality_profiles(sw, g, cc)
        fx = fixed_fn(q, cc["fix_q"]); svs = {name: sideways(g, B, L, method=m) for name, (m, L) in refs.items()}
        ctx[i] = (svs["sg2_60"], fx)
        # positioning check: y_van from the real-time vs the post-processed track, same scans, RTK-fixed scans only
        try:
            gp = swath_geometry(sw, base_pp); svp = sideways(gp, B, cc["L_ref_m"])
            yr, Sr = y_per_scan(g, svs["sg2_60"], B); yp, _ = y_per_scan(gp, svp, B)
            m = np.isfinite(yr) & np.isfinite(yp) & fx(Sr); posres[i] = (yr, yp, Sr, fx)
            d = yr[m] - yp[m]; d = d[np.abs(d) < cc["pp_fail_m"]]
            poscheck[i] = dict(n_scans=int(len(d)), rms_m=float(np.sqrt(np.mean((d - np.median(d)) ** 2))) if len(d) > 100 else None,
                               p95_abs_m=float(np.percentile(np.abs(d - np.median(d)), 95)) if len(d) > 100 else None,
                               sd_y_rt_m=float(np.std(yr[m])) if m.sum() > 100 else None)
        except Exception as e:                                   # no usable post-processed track
            poscheck[i] = dict(error=str(e)[:200])
        for t in tr:
            ps, pu = np.array(t["path_s"]), np.array(t["path_u"])
            off = float(np.median(pu - strip_centre_200(g, ps)))
            rec = dict(drive=i, s_start=t["s_start"], s_end=t["s_end"], length_m=t["length_m"], u_mean=t["u_mean"], offset_m=off,
                       depth_m=t["depth_m"], q=t["q"], cluster=bool(cc["cluster_offset_m"][0] <= off <= cc["cluster_offset_m"][1]),
                       path_s=t["path_s"], path_u=t["path_u"], x=t["x"], y=t["y"])
            for name, sv in svs.items():
                rec[name] = track_test(ps, pu, sv, fx, cc["min_points"], cc["min_sd_d_m"])
            tracks.append(rec)
        print(f"drive {i}: {len(tr)} tracks, testable {sum(r['sg2_60']['testable'] for r in tracks if r['drive'] == i)}, pos {poscheck[i]}, {time.time() - t0:.0f}s", flush=True)
    # primary statistic and shift null
    T = [r for r in tracks if r["sg2_60"]["testable"]]; thr = cc["b_thr"]
    F_obs = float(np.mean([r["sg2_60"]["b"] < thr for r in T])) if T else None
    rng = np.random.default_rng(cc["shift_null"]["seed"]); Fn = []; bnull = []
    for _ in range(cc["shift_null"]["n"]):
        bs = []
        for r in T:
            sv, fx = ctx[r["drive"]]
            bs.append(shift_null_b(np.array(r["path_s"]), np.array(r["path_u"]), sv, fx, cc["min_points"], cc["min_sd_d_m"], tuple(cc["shift_null"]["shift_m"]), rng))
        bs = np.array(bs); bnull.append(bs); ok = np.isfinite(bs)
        Fn.append(float(np.mean(bs[ok] < thr)) if ok.any() else np.nan)
    Fn = np.array(Fn); p = float((np.sum(Fn >= F_obs) + 1) / (np.sum(np.isfinite(Fn)) + 1)) if T else None
    # amendment 2: pooled slope over all tracks with >= min_points RTK-fixed points (no sd_d threshold)
    P = [r for r in tracks if r["sg2_60"]["n_fixed"] >= cc["min_points"]]
    def pooled_block(rows: list[dict], name: str) -> dict:
        fits = [r[name] for r in rows]; b0 = pooled_b(fits)
        if not rows or not np.isfinite(b0):
            return dict(n=len(rows), b_pool=None)
        rb = np.random.default_rng(cc["shift_null"]["seed"] + 1)
        boot = [pooled_b([fits[j] for j in rb.integers(0, len(fits), len(fits))]) for _ in range(cc["mixture_bootstrap"])]
        rn = np.random.default_rng(cc["shift_null"]["seed"] + 2); nulls = []
        for _ in range(cc["shift_null"]["n"]):
            fn = [shift_null_fit(np.array(r["path_s"]), np.array(r["path_u"]), ctx[r["drive"]][0], ctx[r["drive"]][1], cc["min_points"], 0.0,
                                 tuple(cc["shift_null"]["shift_m"]), rn) for r in rows]
            fn = [f for f in fn if f is not None]
            nulls.append(pooled_b(fn) if fn else np.nan)
        nulls = np.array(nulls); okn = np.isfinite(nulls)
        return dict(n=len(rows), b_pool=b0, ci95=[float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))],
                    null_mean=float(np.mean(nulls[okn])), null_q05=float(np.quantile(nulls[okn], 0.05)),
                    p=float((np.sum(nulls[okn] <= b0) + 1) / (okn.sum() + 1)), sum_w=float(sum(f["n"] * f["sd_d"] ** 2 for f in fits)))
    pooled = dict(all=pooled_block(P, "sg2_60"), cluster=pooled_block([r for r in P if r["cluster"]], "sg2_60"),
                  rest=pooled_block([r for r in P if not r["cluster"]], "sg2_60"),
                  sensitivity={name: pooled_b([r[name] for r in P if r[name]["n_fixed"] >= cc["min_points"]]) for name in refs})
    # positioning noise at track scale (30 m windows, trend removed) and attenuation of b
    rp = np.random.default_rng(cc["shift_null"]["seed"] + 3); dsd = []; ysd = []
    for i in cc["drives"]:
        pc = posres.get(i)
        if pc is None:
            continue
        yr, yp, Sr, fx = pc; m = np.isfinite(yr) & np.isfinite(yp)
        for _ in range(200):
            s0 = rp.uniform(Sr[m].min(), Sr[m].max() - 30); ss = np.arange(s0, s0 + 30, 2.0)
            if not np.all(fx(ss)):
                continue
            o = np.argsort(Sr[m]); yri = np.interp(ss, Sr[m][o], yr[m][o]); ypi = np.interp(ss, Sr[m][o], yp[m][o])
            A = np.c_[np.ones(len(ss)), ss - ss.mean()]; Pm = A @ np.linalg.pinv(A)
            d = (yri - ypi) - Pm @ (yri - ypi)
            if np.max(np.abs(d)) < cc["pp_fail_m"]:
                dsd.append(np.std(d)); ysd.append(np.std(yri - Pm @ yri))
    dsd = np.array(dsd); ysd = np.array(ysd)
    var_noise = float(np.median(dsd) ** 2 / 2) if len(dsd) else None
    var_y_tracks = float(np.mean([r["sg2_60"]["sd_d"] ** 2 for r in P])) if P else None
    posnoise = dict(n_windows=int(len(dsd)), sd_rt_minus_pp_30m_median_m=float(np.median(dsd)) if len(dsd) else None,
                    sd_rt_minus_pp_30m_p90_m=float(np.percentile(dsd, 90)) if len(dsd) else None,
                    var_noise_m2=var_noise, mean_var_y_tracks_m2=var_y_tracks,
                    attenuation=float(var_y_tracks / (var_y_tracks + var_noise)) if var_noise is not None and var_y_tracks else None,
                    sd_y_random_30m_median_m=float(np.median(ysd)) if len(ysd) else None)
    # exploratory: do tracks sit where the van moved little? sd_d of tracks vs random windows of the same length
    from scipy.stats import mannwhitneyu
    rr = np.random.default_rng(cc["shift_null"]["seed"] + 4); ctrl = []
    for r in P:
        sv, fx = ctx[r["drive"]]; ps = np.array(r["path_s"])
        for _ in range(20):
            f = shift_null_fit(ps, np.array(r["path_u"]), sv, fx, cc["min_points"], 0.0, (20.0, 2000.0), rr)
            if f is not None:
                ctrl.append(f["sd_d"])
    tsd = [r["sg2_60"]["sd_d"] for r in P]
    explore = dict(track_sd_d_median_m=float(np.median(tsd)) if tsd else None, control_sd_d_median_m=float(np.median(ctrl)) if ctrl else None,
                   p_tracks_smaller=float(mannwhitneyu(tsd, ctrl, alternative="less").pvalue) if tsd and ctrl else None, n_control=len(ctrl))
    summ = dict(n_tracks=len(tracks), n_testable=len(T), F_obs=F_obs, F_null_mean=float(np.nanmean(Fn)) if T else None,
                F_null_q95=float(np.nanquantile(Fn, 0.95)) if T else None, p=p,
                median_b=float(np.median([r["sg2_60"]["b"] for r in T])) if T else None,
                cluster=dict(n=sum(r["cluster"] for r in tracks), n_testable=sum(r["cluster"] for r in T),
                             F=float(np.mean([r["sg2_60"]["b"] < thr for r in T if r["cluster"]])) if any(r["cluster"] for r in T) else None),
                rest=dict(n=sum(not r["cluster"] for r in tracks), n_testable=sum(not r["cluster"] for r in T),
                          F=float(np.mean([r["sg2_60"]["b"] < thr for r in T if not r["cluster"]])) if any(not r["cluster"] for r in T) else None),
                sensitivity={name: dict(n_testable=sum(r[name]["testable"] for r in tracks),
                                        F=float(np.mean([r[name]["b"] < thr for r in tracks if r[name]["testable"]])) if any(r[name]["testable"] for r in tracks) else None,
                                        median_b=float(np.median([r[name]["b"] for r in tracks if r[name]["testable"]])) if any(r[name]["testable"] for r in tracks) else None)
                             for name in refs})
    summ.update(pooled=pooled, positioning_noise=posnoise, explore=explore)
    out = dict(config=cfg_path, summary=summ, positioning=poscheck, tracks=tracks, F_null=Fn.tolist(), b_null=np.array(bnull).tolist())
    json.dump(out, open(cache / "dither_test.json", "w")); print(json.dumps(summ, indent=1)); print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
