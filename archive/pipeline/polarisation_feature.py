"""Polarisation contrast feature (E38). The 600 MHz row radiates with its E-field across the street, the 200 MHz row
along it (Survey XML orientations 0 / 90 deg; DERIVED broadside geometry). A metal pipe echoes most when the E-field is
parallel to it; a flat layer does not care. Feature pol = log(R600 / R200), where R_f is the echo strength at the
target relative to its surroundings at the same time (envelope, instrument signature removed, no other background step):
  crossings: mean over channels at the crossing line vs the median along s within +-6 m (excluding +-1 m), same channel/time;
  along-street tracks: the channel nearest the track vs the median across all channels, same s window (+-1 m) and time;
  controls: random (s, u) points at the tracks' times (layer background).
Steel and PE respond in opposite senses (Lualdi & Lombardi 2014), so the test statistic is two-sided:
D = |pol - median pol of controls at that depth|, used only where the 600 MHz signal is above its floor (snr600 >= 1.3).
Predictions (DECISIONS 2026-11-04, E38): D of crossings falls with |phi| towards 45 deg; along-street tracks have larger D
than controls.
Run: python polarisation_feature.py <config> <along_tag> <idx,idx,...> -> cache/city/polfeat<tag>_<idx>.json"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from build_cube import swath_geometry
from build_v2 import grid_Z, instrument_signature, load_cfg, preprocess_array
from citywide_scan import day_files
from ids_reader import Swath
from processing import envelope
from run_multipass import longest_run

ROOT = Path(__file__).resolve().parents[1]; DS = 0.1


def env_grid(sw, g, a, sl, s_grid, det, sig):
    pp = preprocess_array(sw, a, det, scans=sl, signature=sig)
    E = grid_Z(envelope(pp["X"].transpose(1, 0, 2)).transpose(1, 0, 2), g[f"s{a}"][sl], s_grid)     # (n_ch, n_s, n_t)
    return E, np.nanmedian(g[f"u{a}"][sl], 0), pp["dt_ns"], np.median(pp["snr"], 0)


def box(E, ci, si, ti, hs, ht):
    return E[ci, max(0, si - hs): si + hs + 1, max(0, ti - ht): ti + ht + 1].mean(axis=(-1, -2)) if np.ndim(ci) else E[ci, max(0, si - hs): si + hs + 1, max(0, ti - ht): ti + ht + 1].mean()


def main(cfg_path: str, tag: str, idxs: list[int]) -> None:
    cc = yaml.safe_load(open(cfg_path)); det0 = load_cfg(str(ROOT / "pipeline" / cc["detector_config"])); v = det0["migration"]["velocity_m_ns"]
    det = dict(det0, preprocess=dict(det0["preprocess"], background="signature", lateral_background=False))
    cache = Path(det0["base"]["cache_dir"]) / "city"; files = sorted(glob.glob(str(Path(cc["raw_root"]) / "*" / "Survey_*" / "Swath*.path")))
    for idx in idxs:
        f = files[idx]; t0 = time.time(); sw = Swath.load(str(Path(f).parent), Path(f).stem); g = swath_geometry(sw, det["base"]); L = float(g["s_nodes"][-1])
        sig = {a: instrument_signature(day_files(f, cc["raw_root"]), a, det) for a in (1, 2)}
        X = json.load(open(cache / f"scan_{idx}.json"))["crossings"]; fa = cache / f"along{tag}_{idx}.json"
        T = [t for t in (json.load(open(fa))["tracks"] if fa.exists() else []) if t["q"] <= 0.05 and t["length_m"] >= 10]
        rng = np.random.default_rng(det["seed"] + 111 + idx); out = dict(drive=idx, crossings=[], tracks=[], controls=[])
        sc = {a: np.nanmedian(g[f"s{a}"], 1) for a in (1, 2)}
        for c0 in np.arange(0.0, L, cc["chunk_m"]):
            c1 = min(L, c0 + cc["chunk_m"]); s_grid = np.arange(max(0.0, c0 - 10), min(L, c1 + 10), DS); E = {}
            for a in (1, 2):
                sl = longest_run((sc[a] > s_grid[0] - 5) & (sc[a] < s_grid[-1] + 5))
                E[a] = env_grid(sw, g, a, sl, s_grid, det, sig[a]) if sl is not None and sl.stop - sl.start > 200 else None
            if E[1] is None or E[2] is None:
                continue
            si = lambda s: int(np.clip(round((s - s_grid[0]) / DS), 0, len(s_grid) - 1))
            def R_cross(a, o):
                Ea, u, dt, _ = E[a]; ti = int(2 * o["depth_m"] / v / dt); tp = np.tan(np.radians(o["phi_deg"])); vals = []
                for ci in range(len(u)):
                    s_on = o["s0_m"] + u[ci] * tp; on = box(Ea, ci, si(s_on), ti, 2, 2)
                    ring = np.r_[np.arange(s_on - 6, s_on - 1, 0.5), np.arange(s_on + 1, s_on + 6, 0.5)]
                    vals.append(on / (np.median([box(Ea, ci, si(q), ti, 2, 2) for q in ring]) + 1e-9))
                return float(np.mean(vals))
            snr600 = lambda t_ns: float(E[1][3][min(int(t_ns / E[1][2]), len(E[1][3]) - 1)])   # 600 MHz signal / noise floor at that time
            def R_along(a, s, uu, t_ns):
                Ea, u, dt, _ = E[a]; ti = int(t_ns / dt); ci = int(np.argmin(np.abs(u - uu))); k = si(s)
                allc = np.array([box(Ea, c, k, ti, 10, 3) for c in range(len(u))]); return float(allc[ci] / (np.median(allc) + 1e-9))
            for o in X:
                if c0 <= o["s0_m"] < c1 and o["cls"] == "A":
                    r6, r2 = R_cross(1, o), R_cross(2, o)
                    out["crossings"].append(dict(phi=o["phi_deg"], depth=o["depth_m"], R600=r6, R200=r2, pol=float(np.log(r6 / r2)), snr600=snr600(2 * o["depth_m"] / v)))
            for t in T:
                if not (c0 <= t["s_start"] < c1):
                    continue
                pts = [(s, uu, tt) for s, uu, tt in zip(t["path_s"], t["path_u"], t["path_t"]) if s_grid[0] + 2 < s < s_grid[-1] - 2]
                if not pts:
                    continue
                r6 = np.median([R_along(1, *p) for p in pts]); r2 = np.median([R_along(2, *p) for p in pts])
                out["tracks"].append(dict(s0=t["s_start"], s1=t["s_end"], u=t["u_mean"], depth=t["depth_m"], q=t["q"], R600=float(r6), R200=float(r2), pol=float(np.log(r6 / r2)),
                                          snr600=snr600(t["t_ns"]), x=t["x"], y=t["y"]))
                for _ in range(5):                       # bright-spot controls (E38 addendum): brightest 200 MHz channel at a random place
                    for _try in range(20):
                        s = rng.uniform(max(c0, s_grid[0] + 2), min(c1, s_grid[-1] - 2))
                        if s < t["s_start"] - 5 or s > t["s_end"] + 5:
                            break
                    Ea, u2c, dt2, _ = E[2]; k_ = si(s); ti2 = int(t["t_ns"] / dt2)
                    allc = np.array([box(Ea, c, k_, ti2, 10, 3) for c in range(len(u2c))]); uu = float(u2c[int(np.argmax(allc))])
                    r6b, r2b = R_along(1, s, uu, t["t_ns"]), R_along(2, s, uu, t["t_ns"])
                    out.setdefault("bright_controls", []).append(dict(depth=t["depth_m"], R600=r6b, R200=r2b, pol=float(np.log(r6b / r2b)), snr600=snr600(t["t_ns"])))
                for _ in range(5):                       # controls at the same time, random place in the piece and strip
                    s = rng.uniform(max(c0, s_grid[0] + 2), min(c1, s_grid[-1] - 2)); uu = rng.uniform(E[2][1].min(), E[2][1].max())
                    r6c, r2c = R_along(1, s, uu, t["t_ns"]), R_along(2, s, uu, t["t_ns"])
                    out["controls"].append(dict(depth=t["depth_m"], R600=r6c, R200=r2c, pol=float(np.log(r6c / r2c)), snr600=snr600(t["t_ns"])))
            del E
        json.dump(out, open(cache / f"polfeat{tag}_{idx}.json", "w"))
        print(f"drive {idx}: crossings {len(out['crossings'])}, tracks {len(out['tracks'])}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(a) for a in sys.argv[3].split(",")])
