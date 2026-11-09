"""E42 validity controls (DECISIONS 2026-11-04): straight fake pipes in RAW 200 MHz traces, fixed in the GROUND (G: parallel
to the reference line R at a fixed offset, so they slide across the channels as the van moves sideways) or fixed to the
VAN (V: constant instrument position), at identical places in two separate runs; then the unchanged city along-street
detector (option C) and the slope test of dither.py on every recovered fake.
Run: python run_dither_injection.py configs/dither.yaml <window> -> cache/dither_injection_<window>.json"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from along_road import inject_raw
from build_cube import swath_geometry
from build_v2 import instrument_signature, load_cfg
from citywide_scan import DS_Q, along_det, along_tracks, day_files, quality_profiles
from cube import to_su
from dither import shift_at, sideways, to_ground, track_test, y_at
from ids_reader import Swath
from processing import dewow, first_break_sample, robust_sigma_t
from run_detection import su_to_map
from run_multipass import longest_run

ROOT = Path(__file__).resolve().parents[1]


def fixed_fn(q: dict, fix_q: int):
    """s (m, frame) -> bool array: GGA fix quality == fix_q at the nearest quality-profile sample (DS_Q m)."""
    return lambda s: q["rt_q"][np.clip(np.round(np.asarray(s) / DS_Q).astype(int), 0, len(q["s"]) - 1)] == fix_q


def choose_places(sv: dict, fixed_at, real: list[dict], w0: float, w1: float, F: dict, pieces: list[tuple]) -> list[float]:
    """Start positions (m) of the fakes, from the track alone (pre-registered rule): y_van range >= min_range_m within the
    fake, RTK fixed throughout, >= gap_m between fakes, >= avoid_m from any real significant track, not across a piece edge."""
    L = F["length_m"]; out = []; s = w0 + 20.0; edges = [c0 for c0, _ in pieces[1:]]
    while s + L <= w1 - 20.0 and len(out) < F["max_n"]:
        ss = np.arange(s, s + L + 1e-9, 0.5); yv = y_at(sv, ss)
        ok = (np.all(np.isfinite(yv)) and np.all(fixed_at(ss)) and np.ptp(yv) >= F["min_range_m"]
              and not any(t["s_start"] - F["avoid_m"] < s + L and t["s_end"] + F["avoid_m"] > s for t in real)
              and not any(s - 5.0 < e < s + L + 5.0 for e in edges))
        if ok:
            out.append(float(s)); s += L + F["gap_m"]
        else:
            s += 1.0
    return out


def main(cfg_path: str, win: str) -> None:
    cc = yaml.safe_load(open(cfg_path)); det = load_cfg(str(ROOT / "pipeline" / cc["detector_config"])); detA = along_det(det, cc)
    v = det["migration"]["velocity_m_ns"]; B = cc["along_travel_bias_m"]; F = cc["fakes"]; M = cc["match"]; t0 = time.time()
    files = sorted(glob.glob(str(Path(cc["raw_root"]) / "*" / "Survey_*" / "Swath*.path"))); idx, w0, w1 = cc["windows"][win]; f = files[idx]
    sw = Swath.load(str(Path(f).parent), Path(f).stem); g = swath_geometry(sw, det["base"]); q = quality_profiles(sw, g, cc)
    fixed_at = fixed_fn(q, cc["fix_q"]); sv = sideways(g, B, cc["L_ref_m"]); k = float(g["k"])
    real = [t for t in json.load(open(Path(det["base"]["cache_dir"]) / "city" / f"along{cc['along_tag']}_{idx}.json"))["tracks"]
            if t["q"] <= 0.05 and t["length_m"] >= 10]
    pieces = [(c0, min(w1, c0 + cc["chunk_m"])) for c0 in np.arange(w0, w1, cc["chunk_m"]) if min(w1, c0 + cc["chunk_m"]) - c0 > 60]
    places = choose_places(sv, fixed_at, real, w0, w1, F, pieces)
    sc2 = np.nanmedian(g["s2"], 1); dt = sw.config.channels_of(2)[0].dt_ns
    win_sc = (sc2 > w0) & (sc2 < w1); u_med = np.nanmedian(g["u2"][win_sc], 0); u_c = float(u_med.mean())   # instrument frame
    fakes = []
    for n, s0 in enumerate(places):
        s1 = s0 + F["length_m"]; sm = 0.5 * (s0 + s1); du = F["offsets_m"][n % len(F["offsets_m"])]; z = F["depths_m"][n % len(F["depths_m"])]
        fakes.append(dict(s0=s0, s1=s1, u_off=du, z=z, c_ground=float(y_at(sv, np.array([sm]))[0] + du), u_inst=u_c + du,
                          y_range=float(np.ptp(y_at(sv, np.arange(s0, s1 + 1e-9, 0.5))))))
    sig = instrument_signature(day_files(f, cc["raw_root"]), 2, detA)
    print(f"{win}: drive {idx}, {len(fakes)} fake places {[round(p) for p in places]}, signature ready {time.time() - t0:.0f}s", flush=True)
    out = dict(window=win, drive=idx, file=f, w=[w0, w1], L_ref_m=cc["L_ref_m"], fakes=fakes, runs={})
    f_mhz = None
    frames = cc.get("frames", ["instrument", "ground"])
    for mode in ("G", "V"):
        tracks = {fr: [] for fr in frames}
        for ip, (c0, c1) in enumerate(pieces):
            s_grid = np.arange(max(0.0, c0 - cc["pad_m"]), c1 + cc["pad_m"], det["grid"]["ds_m"])
            sl = longest_run((sc2 > s_grid[0] - 15) & (sc2 < s_grid[-1] + 15))
            raw = np.asarray(sw.data[2][sl], np.float32); S = g["s2"][sl]
            dw = dewow(raw.transpose(1, 0, 2), dt, det["base"]["processing"]["dewow_win_ns"][2])
            n0 = np.array([first_break_sample(d.mean(0), det["base"]["processing"]["t0_frac"]) for d in dw]); spread = np.stack([robust_sigma_t(d) for d in dw])
            if f_mhz is None:                                   # DERIVED wavelet frequency, as run_injection_raw.py
                i0, i1 = int(n0.mean() + 3 / dt), int(n0.mean() + 28 / dt); sub = dw[:, ::50, i0:i1].reshape(-1, i1 - i0)
                sp = np.abs(np.fft.rfft(sub * np.hanning(i1 - i0), axis=1)).mean(0); fr = np.fft.rfftfreq(i1 - i0, dt) * 1e3
                f_mhz = float(fr[1:][np.argmax(sp[1:])]); out["wavelet_mhz"] = f_mhz
            del dw
            if mode == "G":                                     # footprints in ground coordinates across R
                X, Y = su_to_map((S + B).ravel(), g["u2"][sl].ravel(), g["s_nodes"], g["xn"], g["yn"], k)
                U = np.asarray(to_su(X, Y, sv["R"][0], sv["R"][1], sv["R"][2], k)[1], np.float32).reshape(S.shape)
                segs = [dict(s0=d["s0"], s1=d["s1"], u0=d["c_ground"], z=d["z"], amp=F["amp"]) for d in fakes]
            else:                                               # instrument frame: constant channel positions
                U = np.broadcast_to(u_med.astype(np.float32), S.shape)
                segs = [dict(s0=d["s0"], s1=d["s1"], u0=d["u_inst"], z=d["z"], amp=F["amp"]) for d in fakes]
            raw = inject_raw(raw, S, U, n0, spread, dt, v, segs, f_mhz)
            for frame in frames:                                   # E43: same injected data, instrument and ground frame
                rng = np.random.default_rng(det["seed"] + 81 + ip)
                tracks[frame] += along_tracks(sw, g, q, sl, s_grid, c0, c1, detA, cc, rng, raw=raw, signature=sig, sv=sv if frame == "ground" else None)
            del raw
        for frame in frames:
            sig_tr = [t for t in tracks[frame] if t["q"] <= 0.05 and t["length_m"] >= 10]; res = []
            for d in fakes:
                best = None
                for t in sig_tr:
                    ov = min(t["s_end"], d["s1"]) - max(t["s_start"], d["s0"])
                    if ov <= 0 or ov < M["min_overlap"] * t["length_m"]:
                        continue
                    ps, pu = np.array(t["path_s"]), np.array(t["path_u"]); m = (ps >= d["s0"]) & (ps <= d["s1"])
                    if not m.any():
                        continue
                    # offset measured in the fake's own frame: ground offset from R for G, instrument u for V
                    if mode == "G":
                        off = np.median(np.abs(to_ground(sv, np.array(t["x"])[m], np.array(t["y"])[m])[1] - d["c_ground"]))
                    else:
                        pu_inst = pu[m] - shift_at(sv, ps[m]) if frame == "ground" else pu[m]
                        off = np.nanmedian(np.abs(pu_inst - d["u_inst"]))
                    if off <= M["du_m"] and (best is None or ov > best[0]):
                        best = (ov, t, float(off))
                r = dict(found=best is not None)
                if best:
                    t = best[1]; r.update(track=dict(s_start=t["s_start"], s_end=t["s_end"], u_mean=t["u_mean"], depth_m=t["depth_m"], q=t["q"],
                                                     path_s=t["path_s"], path_u=t["path_u"], x=t["x"], y=t["y"]), offset_m=best[2],
                                          test=track_test(np.array(t["path_s"]), np.array(t["path_u"]), sv, fixed_at, cc["min_points"], cc["min_sd_d_m"]) if frame == "instrument" else {})
                res.append(r)
            out["runs"][f"{mode}|{frame}"] = dict(results=res, n_sig_tracks=len(sig_tr))
            found = [r for r in res if r["found"]]
            print(f"{win} {mode} {frame}: found {len(found)}/{len(fakes)}, sig tracks {len(sig_tr)}, {time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(Path(det["base"]["cache_dir"]) / f"{cc.get('out_prefix', 'dither_injection')}_{win}.json", "w")); print("done", win, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
