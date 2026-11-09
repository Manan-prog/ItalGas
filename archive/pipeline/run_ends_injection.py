"""V1 for the end-point detector (DECISIONS 2026-11-04): polarisation-blind half-crossings (and full crossings as decoys)
injected into the RAW traces of both rows; frozen preprocessing and crossing detector plus terminations.detect_ends.
Run: python run_ends_injection.py configs/ends.yaml <region> -> cache/ends_injection_<region>.json"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from along_road import inject_crossing_raw
from build_cube import swath_geometry
from build_v2 import load_cfg, preprocess_array
from ids_reader import Swath
from processing import dewow, first_break_sample, robust_sigma_t
from run_crossings_v2 import detect_crossings, migrate_grid
from run_multipass import longest_run
from terminations import detect_ends

ROOT = Path(__file__).resolve().parents[1]
PHIS = [0, 15, -15, 30, -30]; DEPTHS = [0.5, 0.8]; ENDS = [0.0, -0.3, 0.3]; SIDES = [1, -1]


def excluded(e: dict, X: list[dict]) -> bool:
    """End point coinciding with a significant full crossing of the same array (pre-registered exclusion)."""
    return any(c["q"] <= 0.05 and abs(c["s0_m"] - e["s0_m"]) <= 0.5 and abs(c["phi_deg"] - e["phi_deg"]) <= 15 and abs(c["t_ns"] - e["t_ns"]) <= 3 for c in X)


def main(cfg_path: str, region: str) -> None:
    cc = yaml.safe_load(open(cfg_path)); det = load_cfg(str(ROOT / "pipeline" / cc["detector_config"])); det["ends"] = cc["ends"]
    v = det["migration"]["velocity_m_ns"]; sess, surv, swn, w0, w1 = cc["regions"][region]
    sw = Swath.load(str(Path(cc["raw_root"]) / sess / surv), swn); g = swath_geometry(sw, det["base"]); t0 = time.time()
    sc = {a: np.nanmedian(g[f"s{a}"], 1) for a in (1, 2)}
    centre = float(np.nanmedian(0.5 * (np.nanmin(g["u2"], 1) + np.nanmax(g["u2"], 1))[(sc[2] > w0) & (sc[2] < w1)]))
    pieces = [(c0, min(w1, c0 + cc["chunk_m"])) for c0 in np.arange(w0, w1, cc["chunk_m"]) if min(w1, c0 + cc["chunk_m"]) - c0 > 60]
    out = dict(region=region, centre_u=centre, runs={})
    real_s = None
    for run, amp in [("none", 0.0)] + [(f"amp{a:g}", a) for a in cc["amps"]]:
        pipes = []
        if amp > 0:
            for n, s0 in enumerate(np.arange(w0 + 20.0, w1 - 20.0, cc["spacing_m"])):
                if np.min(np.abs(real_s - s0)) < 3.0:
                    continue
                p = dict(s0=float(s0), phi_deg=float(PHIS[n % 5]), z=DEPTHS[(n // 5) % 2], amp=amp)
                if n % 5 != 4:                                  # 4 of 5 are half-crossings, 1 of 5 a full decoy
                    p.update(u_end=centre + ENDS[(n // 2) % 3], side=SIDES[n % 2])
                pipes.append(p)
        X = {1: [], 2: []}; E = {1: [], 2: []}
        for ip, (c0, c1) in enumerate(pieces):
            s_grid = np.arange(max(0.0, c0 - cc["pad_m"]), c1 + cc["pad_m"], det["grid"]["ds_m"])
            for a in (1, 2):
                sl = longest_run((sc[a] > s_grid[0] - 15) & (sc[a] < s_grid[-1] + 15)); dt = sw.config.channels_of(a)[0].dt_ns
                raw = np.asarray(sw.data[a][sl], np.float32)
                if pipes:
                    dw = dewow(raw.transpose(1, 0, 2), dt, det["base"]["processing"]["dewow_win_ns"][a])
                    n0 = np.array([first_break_sample(d.mean(0), det["base"]["processing"]["t0_frac"]) for d in dw]); spread = np.stack([robust_sigma_t(d) for d in dw])
                    i0, i1 = int(n0.mean() + 3 / dt), int(n0.mean() + 25 / dt); sub = dw[:, ::50, i0:i1].reshape(-1, i1 - i0)
                    sp = np.abs(np.fft.rfft(sub * np.hanning(i1 - i0), axis=1)).mean(0); fr = np.fft.rfftfreq(i1 - i0, dt) * 1e3
                    raw = inject_crossing_raw(raw, g[f"s{a}"][sl], g[f"u{a}"][sl], n0, spread, dt, v, pipes, float(fr[1:][np.argmax(sp[1:])]))
                pp = preprocess_array(sw, a, det, scans=sl, raw=raw); G = migrate_grid(pp["X"], g[f"s{a}"][sl], s_grid, pp["dx_m"], pp["dt_ns"], det)
                u = np.nanmedian(g[f"u{a}"][sl], 0); dtg = pp["dt_ns"] * det["grid"]["t_down"]
                x, _ = detect_crossings(G, u, s_grid, dtg, pp["snr"], pp["dt_ns"], det, np.random.default_rng(det["seed"] + 121 + ip))
                e, info = detect_ends(G, u, s_grid, dtg, pp["snr"], pp["dt_ns"], det, np.random.default_rng(det["seed"] + 131 + ip))
                X[a] += [c for c in x if c0 <= c["s0_m"] < c1]; E[a] += [c for c in e if c0 <= c["s0_m"] < c1]; del G, pp
        if real_s is None:                                  # significant real crossings and ends only (bug fix 2026-11-04: all candidates blocked every slot)
            real_s = np.array([c["s0_m"] for a in (1, 2) for c in X[a] + E[a] if c["q"] <= 0.05] + [w0 - 1e3])
        Es = {a: [e for e in E[a] if e["q"] <= 0.05 and not excluded(e, X[a])] for a in (1, 2)}
        res = dict(n_ends_sig={a: len(Es[a]) for a in (1, 2)}, n_cross_sig={a: sum(c["q"] <= 0.05 for c in X[a]) for a in (1, 2)},
                   km=(w1 - w0) / 1e3, n_injected=len(pipes), pipes=[])
        for p in pipes:
            r = dict(p)
            for a in (1, 2):
                m = [e for e in Es[a] if abs(e["s0_m"] - p["s0"]) <= 0.3 and abs(e["phi_deg"] - p["phi_deg"]) <= 15 and abs(e["depth_m"] - p["z"]) <= 0.15
                     and ("u_end" not in p or e["side"] == p["side"])]
                r[f"end{a}"] = bool(m); r[f"err{a}"] = None if not m or "u_end" not in p else round(abs(m[0]["u_end_m"] - p["u_end"]), 3)
                near = [e for e in E[a] if abs(e["s0_m"] - p["s0"]) <= 0.3 and abs(e["phi_deg"] - p["phi_deg"]) <= 15]   # diagnostics: before q cut and exclusion
                r[f"near{a}"] = [dict(q=e["q"], score=e["score"], u_end=round(e["u_end_m"], 2), side=e["side"], z=round(e["depth_m"], 2), excl=excluded(e, X[a])) for e in near[:3]]
                r[f"full{a}"] = any(c["q"] <= 0.05 and abs(c["s0_m"] - p["s0"]) <= 0.3 and abs(c["phi_deg"] - p["phi_deg"]) <= 15 for c in X[a])
            res["pipes"].append(r)
        half = [r for r in res["pipes"] if "u_end" in r]; full = [r for r in res["pipes"] if "u_end" not in r]
        summ = {a: dict(half_found=float(np.mean([r[f"end{a}"] for r in half])) if half else None,
                        err_median=float(np.median([r[f"err{a}"] for r in half if r[f"err{a}"] is not None])) if any(r[f"err{a}"] is not None for r in half) else None,
                        full_as_end=float(np.mean([r[f"end{a}"] for r in full])) if full else None,
                        half_as_full=float(np.mean([r[f"full{a}"] for r in half])) if half else None) for a in (1, 2)}
        res["summary"] = summ; out["runs"][run] = res
        print(f"{region} {run}: injected {len(pipes)}, sig ends 600/200 {res['n_ends_sig']}, sig crossings {res['n_cross_sig']}, {summ}, {time.time() - t0:.0f}s", flush=True)
    json.dump(out, open(Path(det["base"]["cache_dir"]) / f"ends_injection_{region}.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
