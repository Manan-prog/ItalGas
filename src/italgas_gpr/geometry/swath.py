"""Process one swath end to end: per-channel trace processing + 2D along-track
migration, georeferencing of every (scan, channel) footprint, and caching of the
envelope traces with (s, u) coordinates. Run: python -m italgas_gpr.geometry.swath config.yaml"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from italgas_gpr.io.paths import config_path, load_yaml

from italgas_gpr.geometry.cube import scan_positions, smooth_track, to_su, track_frame
from italgas_gpr.geometry.frames import lonlat_to_merc, merc_scale, scan_time_offset_s
from italgas_gpr.io.ids_reader import Swath
from italgas_gpr.processing.filters import (align_traces, background_removal, bandpass, dewow, envelope, first_break_sample,
                        kirchhoff_migrate, sec_gain, shift_time_zero)


def cfg_hash(cfg: dict) -> str:
    return hashlib.md5(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()[:10]


def swath_geometry(sw: Swath, cfg: dict, frame: dict | None = None) -> dict:
    """Georeferencing only (no trace processing): per-scan (s, u) of every channel's
    footprint on the swath-local frame (or on frame = dict(s_nodes, xn, yn, k) of another pass),
    plus the frame itself. Same steps as build().
    Returns dict with s1, u1, s2, u2 (n_scans, n_ch) m, s_nodes, xn, yn, k, c_off1/2, corr1/2."""
    from italgas_gpr.geometry.frames import vehicle_to_map
    p = sw.path
    k = float(merc_scale(p["y"]).mean())
    tp = p["t_ms"] / 1e3
    if cfg["track"]["source"] == "pp":
        g = sw.gga_pp; xr, yr = lonlat_to_merc(g["lon"], g["lat"]); tp_src = g["t_s"]
    else:
        xr, yr, tp_src = p["x"], p["y"], tp
    xs_, ys_ = smooth_track(xr, yr, cfg["track"]["smooth_win"])
    if frame is None:                      # own swath-local frame
        s_nodes, xn, yn = track_frame(xs_, ys_, k)
    else:                                  # project onto another pass's frame (multi-pass mosaics)
        s_nodes, xn, yn, k = frame["s_nodes"], frame["xn"], frame["yn"], float(frame["k"])
    out = dict(s_nodes=s_nodes, xn=xn, yn=yn, k=k)
    for arr in (1, 2):
        chans = sw.config.channels_of(arr)
        ix = sw.index[arr]; ts = ix["t_us"] / 1e6
        c_off, corr = scan_time_offset_s(ts, ix["odo_m"], tp_src, xr, yr, k)
        xg, yg, hdg = scan_positions(ts, c_off, tp_src, xs_, ys_)
        S = np.empty((len(ix), len(chans))); U = np.empty_like(S)
        for ch in chans:
            # along-travel position bias (DERIVED from opposite-direction drives, E25; default 0 = off):
            # footprints are moved forward along the direction of travel by this many metres
            bias = float(cfg["track"].get("along_travel_bias_m", 0.0))
            X, Y = vehicle_to_map(xg, yg, hdg, ch.mid_xy[0] + bias, ch.mid_xy[1], sw.config.gps_xyz[:2], k)
            S[:, ch.slot], U[:, ch.slot] = to_su(X, Y, s_nodes, xn, yn, k)
        out[f"s{arr}"] = S.astype(np.float32); out[f"u{arr}"] = U.astype(np.float32)
        out[f"c_off{arr}"] = c_off; out[f"corr{arr}"] = corr; out[f"t_scan{arr}"] = ts + c_off
    return out


def build(cfg: dict, verbose: bool = True) -> Path:
    t_start = time.time()
    sw = Swath.load(cfg["survey_dir"], cfg["swath"])
    out = Path(cfg["cache_dir"]) / f"cube_{cfg['swath']}_{Path(cfg['survey_dir']).parent.name[-19:]}_{cfg_hash(cfg)}.npz"
    if out.exists():
        return out
    p = sw.path
    k = float(merc_scale(p["y"]).mean())
    tp = p["t_ms"] / 1e3
    if cfg["track"]["source"] == "pp":
        g = sw.gga_pp; xr, yr = lonlat_to_merc(g["lon"], g["lat"]); tp_src = g["t_s"]
    else:
        xr, yr, tp_src = p["x"], p["y"], tp
    xs_, ys_ = smooth_track(xr, yr, cfg["track"]["smooth_win"])
    s_nodes, xn, yn = track_frame(xs_, ys_, k)
    pc = cfg["processing"]; v = pc["velocity_m_ns"]; dsmp = pc["t_downsample"]
    result = {}
    for arr in (1, 2):
        chans = sw.config.channels_of(arr)
        ix = sw.index[arr]; ts = ix["t_us"] / 1e6
        c_off, corr = scan_time_offset_s(ts, ix["odo_m"], tp_src, xr, yr, k)
        xg, yg, hdg = scan_positions(ts, c_off, tp_src, xs_, ys_)
        dx = float(np.median(np.diff(ix["odo_m"])))
        dt = chans[0].dt_ns
        n_scan = len(ix)
        env = np.empty((n_scan, len(chans), 512 // dsmp), dtype=np.float32)
        S = np.empty((n_scan, len(chans))); U = np.empty_like(S)
        gps_xy = sw.config.gps_xyz[:2]
        from italgas_gpr.geometry.frames import vehicle_to_map
        for ch in chans:
            x = np.asarray(sw.data[arr][:, ch.slot, :]).astype(np.float32)
            x = dewow(x, dt, pc["dewow_win_ns"][arr])
            n0 = first_break_sample(x.mean(0), pc["t0_frac"])
            if pc.get("align_traces", False):
                x = align_traces(x, x.mean(0), n0)
            x = shift_time_zero(x, n0)
            x = background_removal(x, None)
            lo, hi = pc["bandpass_mhz"][arr]
            x = bandpass(x, dt, lo, hi)
            x = sec_gain(x, dt, 0.0, pc["gain_power"])
            if pc["migration"] == "kirchhoff":
                x = kirchhoff_migrate(x, dx, dt, v, pc["aperture_m"])
            e = envelope(x)
            env[:, ch.slot, :] = e.reshape(n_scan, 512 // dsmp, dsmp).mean(2)
            X, Y = vehicle_to_map(xg, yg, hdg, ch.mid_xy[0], ch.mid_xy[1], gps_xy, k)
            S[:, ch.slot], U[:, ch.slot] = to_su(X, Y, s_nodes, xn, yn, k)
            if verbose:
                print(f"  array {arr} ch {ch.id:2d} t0={n0} done {time.time()-t_start:6.1f}s", flush=True)
        result[f"env{arr}"] = env; result[f"s{arr}"] = S.astype(np.float32); result[f"u{arr}"] = U.astype(np.float32)
        result[f"dt{arr}"] = dt * dsmp; result[f"dx{arr}"] = dx; result[f"c_off{arr}"] = c_off; result[f"corr{arr}"] = corr
        result[f"xg{arr}"] = xg; result[f"yg{arr}"] = yg; result[f"hdg{arr}"] = hdg
    result["s_nodes"] = s_nodes; result["xn"] = xn; result["yn"] = yn; result["k"] = k
    result["mids1"] = np.array([c.mid_xy for c in sw.config.channels_of(1)])
    result["mids2"] = np.array([c.mid_xy for c in sw.config.channels_of(2)])
    np.savez(out, **result)
    if verbose:
        print(f"saved {out} in {time.time()-t_start:.1f}s")
    return out


if __name__ == "__main__":
    cfg = load_yaml(config_path(sys.argv[1]))
    build(cfg)
