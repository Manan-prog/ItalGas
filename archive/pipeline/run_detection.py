"""End-to-end detection for one swath cube: clean -> line evidence -> plan-view
probability -> line segments in map coordinates with uncertainty.
Usage: python run_detection.py cube.npz tag [pi0]
Outputs $ITALGAS_DATA/cache/det_<tag>.npz and det_<tag>.geojson"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from skimage.transform import probabilistic_hough_line

from detect import lfdr_probability, remove_layers, resample_cube
from detect2 import evidence_v2
from geometry import merc_to_lonlat
from paths import CACHE

DS, DU = 0.10, 0.08
V_M_NS = 0.10          # DERIVED velocity (see configs/dev_swath.yaml); depth = t * v / 2


def su_to_map(s: np.ndarray, u: np.ndarray, s_nodes: np.ndarray, xn: np.ndarray, yn: np.ndarray, k: float
              ) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of cube.to_su for points: position on the reference polyline at arc
    length s, offset u metres to the LEFT of the local tangent."""
    x = np.interp(s, s_nodes, xn); y = np.interp(s, s_nodes, yn)
    tx = np.interp(s, s_nodes, np.gradient(xn)); ty = np.interp(s, s_nodes, np.gradient(yn))
    n = np.hypot(tx, ty) + 1e-12; tx /= n; ty /= n
    return x + (-ty) * u / k, y + tx * u / k


def run(cube_file: str, tag: str, pi0: float = 0.95) -> dict:
    t0 = time.time()
    z = np.load(cube_file)
    out = {}
    plans = {}
    s_max = min(float(np.nanmax(z["s1"])), float(np.nanmax(z["s2"])))
    for arr in (1, 2):
        env = z[f"env{arr}"]; dt = float(z[f"dt{arr}"])
        f = max(1, int(round(2.0 / dt))); nt = env.shape[2] // f
        env = env[:, :, :nt * f].reshape(env.shape[0], env.shape[1], nt, f).mean(3)
        cube, sg, ug = resample_cube(env, z[f"s{arr}"], z[f"u{arr}"], DS, DU, (-0.58, 0.93), s_max=s_max)
        cl = remove_layers(cube, DS, 3.0)
        dtv = dt * f; jmax = int((45 if arr == 1 else 70) / dtv); j0 = int(4 / dtv); cl = cl[:, :, j0:jmax]
        res = evidence_v2(cl, DS, 2.0)
        ev = res["evidence"]
        plan_ev = ev.max(2); jbest = ev.argmax(2)
        prob, info = lfdr_probability(plan_ev, pi0)
        t_best = (jbest + j0) * dtv                                   # ns two-way
        ang = np.take_along_axis(res["angle"], jbest[:, :, None], 2)[:, :, 0]
        out[f"prob{arr}"] = prob; out[f"evidence{arr}"] = plan_ev; out[f"t_ns{arr}"] = t_best.astype(np.float32)
        out[f"angle{arr}"] = ang; out[f"clean{arr}"] = cl.astype(np.float16); out[f"dt{arr}"] = dtv; out[f"t0_ns{arr}"] = j0 * dtv
        out[f"calib_centres{arr}"] = info["centres"]; out[f"calib_hist{arr}"] = info["hist"]; out[f"calib_f0{arr}"] = info["f0"]; out[f"calib_prob{arr}"] = info["prob_curve"]
        plans[arr] = prob
        print(f"{tag} array{arr}: plan {prob.shape}; frac p>0.5 {(prob>0.5).mean():.4f} p>0.9 {(prob>0.9).mean():.4f}; {time.time()-t0:.1f}s", flush=True)
    out["sg"] = sg; out["ug"] = ug
    # combined probability: the two rows are independent measurements of the same ground.
    # Fisher-free rule: noisy-OR would overstate; we report the per-frequency maps AND
    # the agreement map p_both = p1 * p2 (probability both detect, lower bound of joint).
    p1, p2 = plans[1], plans[2]
    out["prob_both"] = p1 * p2; out["prob_any"] = 1 - (1 - p1) * (1 - p2)
    # --- line extraction on each map
    s_nodes, xn, yn, k = z["s_nodes"], z["xn"], z["yn"], float(z["k"])
    feats = []
    for name, pm in (("600", p1), ("200", p2), ("both", out["prob_both"])):
        img = (pm > 0.9).astype(np.uint8) if name != "both" else (pm > 0.8).astype(np.uint8)
        segs = probabilistic_hough_line(img, threshold=5, line_length=int(1.5 / DS), line_gap=int(0.4 / DS), seed=0)
        for (c0, r0), (c1, r1) in segs:   # skimage returns (col=u index, row=s index)
            s_a, s_b = sg[0] + r0 * DS, sg[0] + r1 * DS; u_a, u_b = ug[0] + c0 * DU, ug[0] + c1 * DU
            ss = np.linspace(s_a, s_b, 20); uu = np.linspace(u_a, u_b, 20)
            ii = np.clip(((ss - sg[0]) / DS).round().astype(int), 0, len(sg) - 1); jj = np.clip(((uu - ug[0]) / DU).round().astype(int), 0, len(ug) - 1)
            pmean = float(pm[ii, jj].mean())
            tarr = out["t_ns1"] if name == "600" else out["t_ns2"]
            depth = float(np.median(tarr[ii, jj]) * V_M_NS / 2)
            X, Y = su_to_map(np.array([s_a, s_b]), np.array([u_a, u_b]), s_nodes, xn, yn, k)
            lon, lat = merc_to_lonlat(X, Y)
            length = float(np.hypot(s_b - s_a, u_b - u_a))
            feats.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[float(lon[0]), float(lat[0])], [float(lon[1]), float(lat[1])]]},
                          "properties": {"source": name, "p_mean": round(pmean, 3), "depth_m_at_v0.10": round(depth, 2), "length_m": round(length, 2),
                                         "s_m": [round(s_a, 2), round(s_b, 2)], "u_m": [round(u_a, 2), round(u_b, 2)],
                                         "angle_to_track_deg": round(float(np.degrees(np.arctan2(u_b - u_a, s_b - s_a))), 1)}})
    out["n_segments"] = np.array([sum(f["properties"]["source"] == n for f in feats) for n in ("600", "200", "both")])
    np.savez(CACHE / f"det_{tag}.npz", **out)
    json.dump({"type": "FeatureCollection", "features": feats}, open(CACHE / f"det_{tag}.geojson", "w"))
    print(f"{tag}: segments 600/200/both = {out['n_segments']}; total {time.time()-t0:.1f}s")
    return out


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.95)
