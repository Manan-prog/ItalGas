"""Injection test of the along-road detector with the arch-shape check (v2.3) on the development
swath, 200 MHz: fake pipes that wander +-0.15 m relative to the van and sit at the centre, off
centre or at the edge of the strip. Compares v2.2 (no shape check) and v2.3 (track must be
pipe-shaped). Run: python run_injection_v23.py configs/dev_swath_v2_2.yaml -> cache/injection_v23.json"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from along_road import inject_longitudinal, match, peak_frequency_mhz, track_shape
from build_v2 import build, load_cfg
from run_longitudinal_v2 import detect_longitudinal

CLASSES = {"centre": (0.0, 0.3), "off-centre": (0.45, 0.7), "edge": (0.8, 1.0)}   # |u0 - array centre| (m)
AMPS = [3.0, 5.0]; DEPTHS = [0.4, 0.7, 1.0, 1.3]; N_SEG = 12


def main(cfg_path: str) -> None:
    cfg = load_cfg(cfg_path); v = cfg["migration"]["velocity_m_ns"]; q_max = cfg["longitudinal"]["q_max"]
    d = np.load(build(cfg_path)[2]); s = d["s_grid"]; u = d["u_ch"]; dt = float(d["dt_full_ns"])
    Z0 = d["Z"].astype(np.float32); f = peak_frequency_mhz(Z0, dt, (3.0, 15.0)); uc = float(np.median(u))
    rng = np.random.default_rng(cfg["seed"] + 23); slots = np.arange(30.0, s[-1] - 50.0, 45.0)
    out = dict(f_mhz=f, array_centre_u=uc, runs=[]); t0 = time.time()
    for amp in AMPS:
        for cls, (a0, a1) in CLASSES.items():
            pick = rng.choice(len(slots), N_SEG, replace=False)
            segs = [dict(s0=float(slots[i]), s1=float(slots[i] + 20.0), u0=float(uc + rng.choice([-1, 1]) * rng.uniform(a0, a1)),
                         z=DEPTHS[n % 4], amp=amp, wander_m=0.15, wander_period_m=15.0) for n, i in enumerate(pick)]
            Zi = inject_longitudinal(Z0, u, s, dt, v, segs, f)
            tracks, info = detect_longitudinal(Zi, u, s, dt, d["snr"], cfg, rng)
            sig = [t for t in tracks if t["q"] <= q_max]
            for t in sig:
                t.update(track_shape(Zi, u, s, dt, v, t))
            shaped = [t for t in sig if t["pipe_like"]]
            h22 = match(sig, segs, q_max, 0.35, 0.15, 0.5); h23 = match(shaped, segs, q_max, 0.35, 0.15, 0.5)
            ov = lambda t: any(min(t["s_end"], g["s1"]) - max(t["s_start"], g["s0"]) > 0 for g in segs)
            fp22 = sum(not ov(t) for t in sig); fp23 = sum(not ov(t) for t in shaped)
            out["runs"].append(dict(amp=amp, cls=cls, segments=segs, recall_v22=h22.tolist(), recall_v23=h23.tolist(),
                                    false_tracks_v22=fp22, false_tracks_v23=fp23))
            print(f"amp {amp} {cls:10s}: v2.2 {h22.sum()}/{N_SEG} (false tracks {fp22}) | v2.3 {h23.sum()}/{N_SEG} (false tracks {fp23})  {time.time()-t0:.0f}s", flush=True)
            del Zi
    # real data without injections: how many significant tracks pass the shape check (mostly layers)
    tracks, _ = detect_longitudinal(Z0, u, s, dt, d["snr"], cfg, rng)
    sig = [t for t in tracks if t["q"] <= q_max]
    for t in sig:
        t.update(track_shape(Z0, u, s, dt, v, t))
    out["real_data"] = dict(n_sig=len(sig), n_pipe_like=sum(t["pipe_like"] for t in sig),
                            tracks=[{k: t[k] for k in ("s_start", "s_end", "u_mean", "depth_m", "q", "pipe_like", "sections")} for t in sig])
    print("real data (no injection): significant tracks", len(sig), "pipe-shaped", out["real_data"]["n_pipe_like"])
    json.dump(out, open(Path(cfg["base"]["cache_dir"]) / "injection_v23.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1])
