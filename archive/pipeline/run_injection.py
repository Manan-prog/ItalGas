"""L1 injection-recovery test of the along-road detector on the development swath
(definition in DECISIONS.md, 3 Nov 2026). Run: python run_injection.py configs/dev_swath_v2_2.yaml
Output: cache/injection_<hash>.json"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from along_road import inject_longitudinal, match, peak_frequency_mhz
from build_v2 import build, cfg_hash, load_cfg
from run_longitudinal_v2 import detect_longitudinal

AMPS = [1.0, 2.0, 3.0, 5.0, 8.0]
DEPTHS = [0.4, 0.7, 1.0, 1.3]
N_SEG = 12


def main(cfg_path: str) -> Path:
    cfg = load_cfg(cfg_path); v = cfg["migration"]["velocity_m_ns"]; q_max = cfg["longitudinal"]["q_max"]
    outs = build(cfg_path); rng = np.random.default_rng(cfg["seed"])
    D = {a: np.load(p) for a, p in outs.items()}
    s = D[1]["s_grid"]
    slots = np.arange(30.0, s[-1] - 50.0, 45.0)
    # calibration: amplitude of real crossing pipes (class A) in noise-equalised units
    fin = sorted(Path(cfg["base"]["cache_dir"]).glob("final_v2_*.json"))[-1]
    A = [o for o in json.load(open(fin))["crossings"] if o["cls"] == "A"]
    real_amp = {}
    for a in (1, 2):
        Z = D[a]["Z"]; u = D[a]["u_ch"]; dt = float(D[a]["dt_full_ns"]); c = int(np.argmin(np.abs(u - 0.17)))
        vals = []
        for o in A:
            i0, i1 = np.searchsorted(s, [o["s0_m"] - 0.3, o["s0_m"] + 0.3]); j = int(2 * o["depth_m"] / v / dt)
            vals.append(float(np.abs(Z[c, i0:i1, max(0, j - 8): j + 8].astype(np.float32)).max()))
        real_amp[a] = dict(median=float(np.median(vals)), q25=float(np.quantile(vals, 0.25)), q75=float(np.quantile(vals, 0.75)))
    f_mhz = {a: peak_frequency_mhz(D[a]["Z"].astype(np.float32), float(D[a]["dt_full_ns"]), (3.0, 15.0)) for a in (1, 2)}
    print("real class-A crossing apex amplitude (x typical level):", real_amp, " wavelet peak MHz:", f_mhz, flush=True)
    res = dict(real_crossing_amp=real_amp, f_mhz=f_mhz, runs=[])
    t0 = time.time()
    for amp in AMPS:
        pick = rng.choice(len(slots), N_SEG, replace=False)
        segs = [dict(s0=float(slots[i]), s1=float(slots[i] + 20.0), u0=float(rng.uniform(-0.45, 0.80)),
                     z=DEPTHS[n % len(DEPTHS)], amp=amp) for n, i in enumerate(pick)]
        run = dict(amp=amp, segments=segs, arrays={})
        for a in (1, 2):
            d = D[a]; dt = float(d["dt_full_ns"])
            Zi = inject_longitudinal(d["Z"].astype(np.float32), d["u_ch"], s, dt, v, segs, f_mhz[a])
            tracks, info = detect_longitudinal(Zi, d["u_ch"], s, dt, d["snr"], cfg, rng)
            del Zi
            hit = match(tracks, segs, q_max, 0.15, 0.15, 0.5)
            hit_any = match(tracks, segs, 1.0, 0.15, 0.15, 0.5)
            sig = [t for t in tracks if t["q"] <= q_max]
            outside = [t for t in sig if not any(min(t["s_end"], g["s1"]) - max(t["s_start"], g["s0"]) > 0 for g in segs)]
            run["arrays"][a] = dict(hit=hit.tolist(), hit_any_q=hit_any.tolist(), n_sig=len(sig), n_sig_outside=len(outside),
                                    t_search_ns=info["t_search_ns"])
            print(f"amp {amp}: array {a} recovered {hit.sum()}/{len(segs)} (any q {hit_any.sum()}), sig tracks {len(sig)}, "
                  f"outside injections {len(outside)}, {time.time()-t0:.0f}s", flush=True)
        res["runs"].append(run)
    out = Path(cfg["base"]["cache_dir"]) / f"injection_{cfg_hash(cfg, keys=('base', 'preprocess', 'longitudinal'))}.json"
    json.dump(res, open(out, "w")); print("saved", out)
    return out


if __name__ == "__main__":
    main(sys.argv[1])
