"""Final v2 product for one swath: crossing objects (two frequencies merged) with a
confidence class, a probability, and a measured plan-position uncertainty; longitudinal
reflector tracks; GeoJSON in WGS84.
Run: python finalize_v2.py configs/dev_swath_v2_2.yaml xing_<hash>.json long_<hash>.json
Classes: A = significant (q <= 0.05) in BOTH frequencies at the same place (top-half
scores of such pairs agree 83 percent of the time, E10); B = significant in one
frequency; candidates with q > 0.05 are not exported.
P = 1 - local fdr from the column-shift null (per frequency); for class A,
P = 1 - (1 - P600)(1 - P200) (ASSUMED independent errors: separate antennas and bands).
Position uncertainty (1 sigma, m): along-track = max(0.3, |PP - RT| along-track disagreement
at that s) (MEASURED lower bound, trajectory_qc); 0.3 m floor = apex picking + 0.05 m grid +
channel tolerance (ASSUMED)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from build_v2 import load_cfg
from geometry import merc_to_lonlat
from ids_reader import Swath
from run_detection import su_to_map
from trajectory_qc import along_cross_disagreement


def merge_crossings(c6: list[dict], c2: list[dict], qmax: float) -> list[dict]:
    a = [c for c in c6 if c["q"] <= qmax]; b = [c for c in c2 if c["q"] <= qmax]
    used = set(); out = []
    for c in a:
        j = next((i for i, d in enumerate(b) if i not in used and abs(c["s0_m"] - d["s0_m"]) <= 0.3
                  and abs(c["phi_deg"] - d["phi_deg"]) <= 15 and abs(c["depth_m"] - d["depth_m"]) <= 0.2), None)
        if j is None:
            out.append(dict(cls="B", freq="600", s0_m=c["s0_m"], phi_deg=c["phi_deg"], depth_m=c["depth_m"], P=c["p_real"],
                            score600=c["score"], q600=c["q"]))
        else:
            d = b[j]; used.add(j)
            out.append(dict(cls="A", freq="600+200", s0_m=round((c["s0_m"] + d["s0_m"]) / 2, 2), phi_deg=c["phi_deg"],
                            depth_m=round((c["depth_m"] + d["depth_m"]) / 2, 2),
                            P=round(1 - (1 - c["p_real"]) * (1 - d["p_real"]), 3),
                            score600=c["score"], q600=c["q"], score200=d["score"], q200=d["q"]))
    for i, d in enumerate(b):
        if i not in used:
            out.append(dict(cls="B", freq="200", s0_m=d["s0_m"], phi_deg=d["phi_deg"], depth_m=d["depth_m"], P=d["p_real"],
                            score200=d["score"], q200=d["q"]))
    return sorted(out, key=lambda o: o["s0_m"])


def main(cfg_path: str, xing: str, long_: str) -> Path:
    cfg = load_cfg(cfg_path); cache = Path(cfg["base"]["cache_dir"])
    cube = np.load(cache / cfg["cube_file"]); k = float(cube["k"])
    X = json.load(open(cache / xing)); Lg = json.load(open(cache / long_))
    sw = Swath.load(cfg["base"]["survey_dir"], cfg["base"]["swath"])
    dq = along_cross_disagreement(sw.path, sw.gga_pp, k)
    along = np.abs(median_filter(dq["along"], 51, mode="nearest"))
    objs = merge_crossings(X["arrays"]["1"]["candidates"], X["arrays"]["2"]["candidates"], cfg["crossing"]["q_max"])
    u_lo, u_hi = -0.58, 0.93                                    # outermost channel footprints (m)
    feats = []
    for o in objs:
        o["sigma_along_m"] = round(float(max(0.3, np.interp(o["s0_m"], dq["s_rt"], along))), 2)
        uu = np.array([u_lo, u_hi]); ss = o["s0_m"] + uu * np.tan(np.radians(o["phi_deg"]))
        mx, my = su_to_map(ss, uu, cube["s_nodes"], cube["xn"], cube["yn"], k)
        lon, lat = merc_to_lonlat(mx, my)
        feats.append(dict(type="Feature", geometry=dict(type="LineString", coordinates=np.c_[lon, lat].round(7).tolist()),
                          properties=dict(kind="crossing", **o)))
    tracks = []
    for a, lab in (("1", "600"), ("2", "200")):
        for t in Lg["arrays"][a]["tracks"]:
            if t["length_m"] >= 20:
                tracks.append(dict(freq=lab, **{kk: t[kk] for kk in ("s_start", "s_end", "length_m", "u_mean", "depth_m", "q")},
                                   path_s=t["path_s"], path_u=t["path_u"]))
                mx, my = su_to_map(np.array(t["path_s"]), np.array(t["path_u"]), cube["s_nodes"], cube["xn"], cube["yn"], k)
                lon, lat = merc_to_lonlat(mx, my)
                feats.append(dict(type="Feature", geometry=dict(type="LineString", coordinates=np.c_[lon, lat].round(7).tolist()),
                                  properties=dict(kind="longitudinal_reflector", freq=lab, length_m=t["length_m"],
                                                  depth_m=t["depth_m"], q=t["q"],
                                                  note="wide convex-up reflector, probable sewer collector or its trench, not a compact pipe")))
    out = cache / f"final_v2_{Path(xing).stem}_{Path(long_).stem}"
    json.dump(dict(type="FeatureCollection", features=feats), open(str(out) + ".geojson", "w"))
    json.dump(dict(crossings=objs, tracks=tracks, along_disagreement=dict(s=dq["s_rt"][::10].round(1).tolist(),
                                                                           along=dq["along"][::10].round(2).tolist(),
                                                                           cross=dq["cross"][::10].round(2).tolist())),
              open(str(out) + ".json", "w"))
    A = [o for o in objs if o["cls"] == "A"]; B = [o for o in objs if o["cls"] == "B"]
    print(f"crossings: A {len(A)}, B {len(B)} (600 only {sum(o['freq']=='600' for o in B)}, 200 only {sum(o['freq']=='200' for o in B)}); "
          f"long tracks >= 20 m: {len(tracks)}")
    print("P quantiles A", np.round(np.quantile([o['P'] for o in A], [0, .5, 1]), 2) if A else None,
          " B", np.round(np.quantile([o['P'] for o in B], [0, .5, 1]), 2) if B else None)
    print("sigma_along quantiles", np.round(np.quantile([o['sigma_along_m'] for o in objs], [0, .5, .9, 1]), 2))
    print("saved", out)
    return out


if __name__ == "__main__":
    main(*sys.argv[1:4])
