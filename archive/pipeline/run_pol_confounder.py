"""Confounder test for the 600/200 polarisation trend (DECISIONS 2026-11-04): polarisation-blind fake crossing pipes
(same strength relative to each row's raw spread) injected into the RAW traces of both rows of one drive at several
angles; standard crossing detector; does score600/score200 of the detected fake pipes fall with |phi| like the real one?
Run: python run_pol_confounder.py configs/citywide_scan.yaml <swath index> -> cache/pol_confounder_<index>.json"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

from along_road import inject_crossing_raw
from build_cube import swath_geometry
from build_v2 import load_cfg, preprocess_array
from finalize_v2 import merge_crossings
from ids_reader import Swath
from processing import dewow, first_break_sample, robust_sigma_t
from run_crossings_v2 import detect_crossings, migrate_grid
from run_multipass import longest_run

ROOT = Path(__file__).resolve().parents[1]
ANGLES = [0, 15, -15, 30, -30, 45, -45, 60, -60]; DEPTHS = [0.5, 0.8]; AMP = 5.0; SPACING = 30.0


def main(cfg_path: str, idx: int) -> None:
    cc = yaml.safe_load(open(cfg_path)); det = load_cfg(str(ROOT / "pipeline" / cc["detector_config"])); v = det["migration"]["velocity_m_ns"]
    files = sorted(glob.glob(str(Path(cc["raw_root"]) / "*" / "Survey_*" / "Swath*.path"))); f = files[idx]
    sw = Swath.load(str(Path(f).parent), Path(f).stem); g = swath_geometry(sw, det["base"]); L = float(g["s_nodes"][-1])
    real = json.load(open(Path(det["base"]["cache_dir"]) / "city" / f"scan_{idx}.json"))
    real_s = np.array([c["s0_m"] for a in ("1", "2") for c in real["per_array"][a]] + [c["s0_m"] for c in real["crossings"]])
    sc = {a: np.nanmedian(g[f"s{a}"], 1) for a in (1, 2)}; t0 = time.time(); out = dict(drive=idx, runs=[])
    for run, off in enumerate((0.0, SPACING / 2)):
        pipes = []
        for n, s0 in enumerate(np.arange(20.0 + off, L - 20.0, SPACING)):
            if np.min(np.abs(real_s - s0)) < 3.0:
                continue
            pipes.append(dict(s0=float(s0), phi_deg=float(ANGLES[n % len(ANGLES)]), z=DEPTHS[(n // len(ANGLES)) % 2], amp=AMP))
        C = {1: [], 2: []}
        for ip, c0 in enumerate(np.arange(0.0, L, cc["chunk_m"])):
            c1 = min(L, c0 + cc["chunk_m"]); s_grid = np.arange(max(0.0, c0 - cc["pad_m"]), min(L, c1 + cc["pad_m"]), det["grid"]["ds_m"])
            if len(s_grid) * det["grid"]["ds_m"] < 60.0:            # too short for the column-shift null (as in citywide_scan)
                continue
            for a in (1, 2):
                sl = longest_run((sc[a] > s_grid[0] - 15) & (sc[a] < s_grid[-1] + 15))
                if sl is None or sl.stop - sl.start < 200:
                    continue
                dt = sw.config.channels_of(a)[0].dt_ns; raw = np.asarray(sw.data[a][sl], np.float32)
                dw = dewow(raw.transpose(1, 0, 2), dt, det["base"]["processing"]["dewow_win_ns"][a])
                n0 = np.array([first_break_sample(d.mean(0), det["base"]["processing"]["t0_frac"]) for d in dw]); spread = np.stack([robust_sigma_t(d) for d in dw])
                i0, i1 = int(n0.mean() + 3 / dt), int(n0.mean() + 25 / dt); sub = dw[:, ::50, i0:i1].reshape(-1, i1 - i0)
                sp = np.abs(np.fft.rfft(sub * np.hanning(i1 - i0), axis=1)).mean(0); fr = np.fft.rfftfreq(i1 - i0, dt) * 1e3; f_mhz = float(fr[1:][np.argmax(sp[1:])])
                out.setdefault("wavelet_mhz", {})[a] = f_mhz
                raw = inject_crossing_raw(raw, g[f"s{a}"][sl], g[f"u{a}"][sl], n0, spread, dt, v, pipes, f_mhz)
                pp = preprocess_array(sw, a, det, scans=sl, raw=raw); G = migrate_grid(pp["X"], g[f"s{a}"][sl], s_grid, pp["dx_m"], pp["dt_ns"], det)
                rng = np.random.default_rng(det["seed"] + 101 + ip)
                cands, _ = detect_crossings(G, np.nanmedian(g[f"u{a}"][sl], 0), s_grid, pp["dt_ns"] * det["grid"]["t_down"], pp["snr"], pp["dt_ns"], det, rng)
                C[a] += [c for c in cands if c0 <= c["s0_m"] < c1]; del G, pp
        both = [o for o in merge_crossings(C[1], C[2], det["crossing"]["q_max"]) if o["cls"] == "A"]
        rows = []
        for p in pipes:
            m = [o for o in both if abs(o["s0_m"] - p["s0"]) <= 0.3 and abs(o["phi_deg"] - p["phi_deg"]) <= 15]
            d6 = any(abs(c["s0_m"] - p["s0"]) <= 0.3 and abs(c["phi_deg"] - p["phi_deg"]) <= 15 and c["q"] <= 0.05 for c in C[1])
            d2 = any(abs(c["s0_m"] - p["s0"]) <= 0.3 and abs(c["phi_deg"] - p["phi_deg"]) <= 15 and c["q"] <= 0.05 for c in C[2])
            rows.append(dict(p, det600=d6, det200=d2, both=bool(m), ratio=(m[0]["score600"] / m[0]["score200"]) if m else None))
        out["runs"].append(rows); print(f"run {run}: {len(pipes)} fake pipes, both-row detected {sum(r['both'] for r in rows)}, {time.time() - t0:.0f}s", flush=True)
    R = [r for rr in out["runs"] for r in rr]; out["summary"] = {}
    for lo, hi in ((0, 10), (10, 25), (25, 40), (40, 61)):
        b = [r for r in R if lo <= abs(r["phi_deg"]) < hi]; rat = [r["ratio"] for r in b if r["ratio"]]
        out["summary"][f"{lo}-{hi}"] = dict(n=len(b), det600=float(np.mean([r["det600"] for r in b])) if b else None, det200=float(np.mean([r["det200"] for r in b])) if b else None,
                                         both=len(rat), median_ratio=float(np.median(rat)) if rat else None)
    ok = [r for r in R if r["ratio"]]
    if len(ok) > 3:
        rho = spearmanr([abs(r["phi_deg"]) for r in ok], np.log([r["ratio"] for r in ok])); out["spearman"] = [float(rho[0]), float(rho[1])]
    print(json.dumps(out["summary"]), out.get("spearman"))
    json.dump(out, open(Path(det["base"]["cache_dir"]) / f"pol_confounder_{idx}.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
