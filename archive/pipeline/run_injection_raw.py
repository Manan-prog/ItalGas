"""Background-removal test for along-street pipes (rules in DECISIONS.md 2026-11-03, config configs/injection_raw.yaml):
fake pipes are added to the RAW 200 MHz traces before any processing, then the unchanged along-street detector runs
with each background option. Reports per pipe the fraction of its length covered by matching significant tracks,
the same on the data without injection (baseline), false-track rate per km, and residual ringing of each option.
On the data without injection it also scores municipal storm/sewer records running along the strip (E15 rule: overlap
>= 30 percent within 0.5 m sideways; chance = records shifted 20-200 m along the window, wrapped, 1000 draws).
A variant may override the preprocess section and, under the key 'longitudinal', the longitudinal section.
Run: python run_injection_raw.py <config> <region> -> cache/injection_raw<tag>_<region>.json"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from along_road import inject_raw, longitudinal_records, match
from build_cube import swath_geometry
from build_v2 import instrument_signature, load_cfg, preprocess_array
from citywide_scan import along_tracks, day_files, quality_profiles
from ids_reader import Swath
from processing import dewow, first_break_sample, robust_sigma_t
from run_multipass import longest_run

ROOT = Path(__file__).resolve().parents[1]


def cover(tracks: list[dict], g: dict, m: dict) -> float:
    """Fraction of pipe g's length covered by significant tracks within du / dz of it."""
    iv = sorted((max(t["s_start"], g["s0"]), min(t["s_end"], g["s1"])) for t in tracks
                if t["q"] <= 0.05 and abs(t["u_mean"] - g["u0"]) <= m["du_m"] and abs(t["depth_m"] - g["z"]) <= m["dz_m"]
                and t["s_end"] > g["s0"] and t["s_start"] < g["s1"])
    tot, cur = 0.0, None
    for a, b in iv:
        if cur is None or a > cur[1]:
            tot += 0 if cur is None else cur[1] - cur[0]; cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    tot += 0 if cur is None else cur[1] - cur[0]
    return tot / (g["s1"] - g["s0"])


def records_score(tracks: list[dict], tg: list[dict], w0: float, w1: float, rng: np.random.Generator, n: int = 1000) -> dict:
    """Record hits (E15 rule) and chance with all targets shifted together by +-20-200 m, wrapped inside [w0, w1]."""
    if not tg:
        return dict(n_targets=0)
    hit = int(match(tracks, tg, 0.05, 0.5, None, 0.3).sum()); W = w1 - w0; ch = []
    for o in rng.uniform(20, 200, n) * rng.choice([-1, 1], n):
        sh = [dict(t, s0=w0 + (t["s0"] - w0 + o) % W, s1=w0 + (t["s0"] - w0 + o) % W + (t["s1"] - t["s0"])) for t in tg]
        ch.append(match(tracks, sh, 0.05, 0.5, None, 0.3).sum())
    ch = np.array(ch); return dict(n_targets=len(tg), hits=hit, chance_mean=float(ch.mean()), p=float((np.sum(ch >= hit) + 1) / (n + 1)))


def main(cfg_path: str, region: str) -> None:
    cc = yaml.safe_load(open(cfg_path)); det = load_cfg(str(ROOT / "pipeline" / cc["detector_config"])); v = det["migration"]["velocity_m_ns"]
    sess, surv, swn, w0, w1 = cc["regions"][region]; f = str(Path(cc["raw_root"]) / sess / surv / f"{swn}.path")
    sw = Swath.load(str(Path(f).parent), swn); g = swath_geometry(sw, det["base"]); q = quality_profiles(sw, g, cc)
    sc = np.nanmedian(g["s2"], 1); dt = sw.config.channels_of(2)[0].dt_ns; t0 = time.time()
    centre = float(np.nanmedian(0.5 * (np.nanmin(g["u2"], 1) + np.nanmax(g["u2"], 1))[(sc > w0) & (sc < w1)]))
    W = w1 - w0; amp_of = {r[0]: r[1] for r in cc["runs"]}
    targets = {name: [dict(s0=w0 + p["f0"] * W, s1=w0 + p["f1"] * W, u0=centre + p["du"], z=p["z"], wander_m=p["wander_m"])
                      for p in cc["pipes"][name]] for name in cc["pipes"]}
    u2 = np.nanmedian(g["u2"][(sc > w0) & (sc < w1)], 0)
    rec = [t for t in longitudinal_records([ROOT / "data_public" / "gfi_all_condotte_acque_bianche.json", ROOT / "data_public" / "gfi_all_condotte_acque_nere.json"],
                                           g["s_nodes"], g["xn"], g["yn"], float(g["k"]), (float(u2.min()), float(u2.max()))) if w0 <= t["s0"] and t["s1"] <= w1]
    sig = instrument_signature(day_files(f, cc["raw_root"]), 2, dict(det, preprocess=dict(det["preprocess"], background="signature")))
    print(f"{region}: strip centre u {centre:+.2f} m, signature ready {time.time() - t0:.0f}s", flush=True)
    pieces = [(c0, min(w1, c0 + cc["chunk_m"])) for c0 in np.arange(w0, w1, cc["chunk_m"]) if min(w1, c0 + cc["chunk_m"]) - c0 > 60]
    out = dict(region=region, drive=[sess, surv, swn], window=[w0, w1], centre_u=centre, targets=targets, results={}, ringing={})
    f_mhz = None
    for run, amp in cc["runs"]:
        segs = [dict(tg, amp=amp) for tg in targets.get(run, [])]
        for vname, over in cc["variants"].items():
            over = dict(over); lon = over.pop("longitudinal", {})
            detv = dict(det, preprocess=dict(det["preprocess"], **over), longitudinal=dict(det["longitudinal"], **lon)); tracks = []
            for ip, (c0, c1) in enumerate(pieces):
                s_grid = np.arange(max(0.0, c0 - cc["pad_m"]), c1 + cc["pad_m"], det["grid"]["ds_m"])
                sl = longest_run((sc > s_grid[0] - 15) & (sc < s_grid[-1] + 15))
                raw = np.asarray(sw.data[2][sl], np.float32)
                if segs:
                    dw = dewow(raw.transpose(1, 0, 2), dt, det["base"]["processing"]["dewow_win_ns"][2])            # (n_ch, n, n_t)
                    n0 = np.array([first_break_sample(d.mean(0), det["base"]["processing"]["t0_frac"]) for d in dw])
                    spread = np.stack([robust_sigma_t(d) for d in dw])
                    if f_mhz is None:                    # DERIVED wavelet frequency: peak of the mean spectrum, 3-28 ns after t0
                        i0, i1 = int(n0.mean() + 3 / dt), int(n0.mean() + 28 / dt); sub = dw[:, ::50, i0:i1].reshape(-1, i1 - i0)
                        sp = np.abs(np.fft.rfft(sub * np.hanning(i1 - i0), axis=1)).mean(0); fr = np.fft.rfftfreq(i1 - i0, dt) * 1e3
                        f_mhz = float(fr[1:][np.argmax(sp[1:])]); out["wavelet_mhz"] = f_mhz
                    raw = inject_raw(raw, g["s2"][sl], g["u2"][sl], n0, spread, dt, v, segs, f_mhz)
                rng = np.random.default_rng(det["seed"] + 81 + ip)
                tracks += along_tracks(sw, g, q, sl, s_grid, c0, c1, detv, cc, rng, raw=raw, signature=sig)
                if run == "none" and ip == 0:            # residual of the background option at early times, in noise units
                    X = preprocess_array(sw, 2, detv, scans=sl, raw=raw, signature=sig)["X"]; e0, e1 = int(2 / dt), int(10 / dt)
                    out["ringing"][vname] = float(np.median(np.sqrt((np.median(X[:, :, e0:e1], 0) ** 2).mean(1)))); del X
            res = dict(n_sig_10m=sum(t["q"] <= 0.05 and t["length_m"] >= 10 for t in tracks), km=W / 1e3)
            if run == "none":
                res["records"] = records_score(tracks, rec, w0, w1, np.random.default_rng(det["seed"] + 91))
            res["cover"] = {name: [round(cover(tracks, tg, cc["match"]), 3) for tg in targets[name]] for name in targets}
            out["results"][f"{run}|{vname}"] = res
            print(f"{region} {run} x{amp} {vname}: sig tracks >=10 m {res['n_sig_10m']}, cover {res['cover'].get(run, res['cover'])} {res.get('records', '')} "
                  f"{time.time() - t0:.0f}s", flush=True)
    out["n_records"] = len(rec)
    json.dump(out, open(Path(det["base"]["cache_dir"]) / f"injection_raw{cc.get('tag', '')}_{region}.json", "w")); print("done", region, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
