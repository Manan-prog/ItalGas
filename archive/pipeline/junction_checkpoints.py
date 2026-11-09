"""Step 2 (3 Nov 2026, INTERNAL): at places where two drives cross at >= 45 deg, each drive passes
ACROSS the other street, so that street's gas main should appear as a CROSSING (which we detect
reliably). Detect crossings (both radar rows) in +-25 m of the junction and compare with where the
town-plan sketch's gas line crosses the drive's strip; chance from random positions in the window.
Run: python junction_checkpoints.py configs/citywide.yaml -> cache/city/junctions.json"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp
from scipy.spatial import cKDTree

from build_cube import swath_geometry
from build_v2 import grid_Z, load_cfg, preprocess_array
from geometry import merc_scale
from ids_reader import Swath
from puc_overlay import network_su
from run_crossings_v2 import detect_crossings, migrate_grid
from run_multipass import longest_run
from finalize_v2 import merge_crossings


def junction_places(tracks, min_angle=45.0):
    k = float(merc_scale(tracks[0][:, 1]).mean()); P, H, I = [], [], []
    for i, tr in enumerate(tracks):
        if len(tr) < 10:
            continue
        L = np.r_[0, np.cumsum(np.hypot(*np.diff(tr, axis=0).T) * k)]; g = np.arange(0, L[-1], 2.0)
        if len(g) < 3:
            continue
        x = np.interp(g, L, tr[:, 0]); y = np.interp(g, L, tr[:, 1]); P.append(np.c_[x, y]); H.append(np.arctan2(np.gradient(y), np.gradient(x))); I.append(np.full(len(g), i))
    P = np.vstack(P); H = np.concatenate(H); I = np.concatenate(I)
    pr = cKDTree(P).query_pairs(1.5 / k, output_type="ndarray"); a, b = pr[:, 0], pr[:, 1]
    ang = np.degrees(np.abs(np.angle(np.exp(1j * (H[a] - H[b]))))); ang = np.minimum(ang, 180 - ang)
    m = (I[a] != I[b]) & (ang >= min_angle); a, b, ang = a[m], b[m], ang[m]
    pts = (P[a] + P[b]) / 2; pp = cKDTree(pts).query_pairs(10 / k, output_type="ndarray")
    A = sp.coo_matrix((np.ones(len(pp)), (pp[:, 0], pp[:, 1])), shape=(len(pts), len(pts))); nc, lab = connected_components(A, directed=False)
    out = []
    for c in range(nc):
        idx = np.nonzero(lab == c)[0]; j = idx[0]
        prs = {}
        for q in idx:                                   # EVERY distinct pair of drives crossing at this place
            key = tuple(sorted((int(I[a[q]]), int(I[b[q]]))))
            if key not in prs:
                prs[key] = (float(pts[q, 0]), float(pts[q, 1]), float(ang[q]))
        out.append(dict(x=float(pts[j, 0]), y=float(pts[j, 1]), drive_a=int(I[a[j]]), drive_b=int(I[b[j]]), angle=float(ang[j]),
                        pairs=[dict(drive_a=k_[0], drive_b=k_[1], x=v_[0], y=v_[1], angle=v_[2]) for k_, v_ in prs.items()]))
    return out


def main(cfg_path: str) -> None:
    cc = yaml.safe_load(open(cfg_path)); det = load_cfg(str(Path(__file__).resolve().parents[1] / "pipeline" / cc["detector_config"]))
    cache = Path(det["base"]["cache_dir"]) / "city"; rng = np.random.default_rng(det["seed"] + 41)
    files = sorted(glob.glob(str(Path(cc["raw_root"]) / "*" / "Survey_*" / "Swath*.path")))
    z = np.load(cc["swath_tracks"], allow_pickle=True); tracks = [z[f"s{i}"] for i in range(len(z["days"]))]
    J = junction_places(tracks); rows = []
    for n, jp in enumerate(J):
        for drv in (jp["drive_a"], jp["drive_b"]):
            r = dict(junction=n, drive=drv, angle=jp["angle"])
            try:
                f = files[drv]; sw = Swath.load(str(Path(f).parent), Path(f).stem)
                g = swath_geometry(sw, det["base"]); fr = {k: g[k] for k in ("s_nodes", "xn", "yn", "k")}
                i0 = int(np.argmin(np.hypot(fr["xn"] - jp["x"], fr["yn"] - jp["y"]))); sj = float(fr["s_nodes"][i0])
                s_grid = np.arange(max(0.0, sj - 25), min(float(fr["s_nodes"][-1]), sj + 25), det["grid"]["ds_m"])
                C = {}
                for arr in (1, 2):
                    S = g[f"s{arr}"]; sc = np.nanmedian(S, 1); sl = longest_run((sc > s_grid[0] - 15) & (sc < s_grid[-1] + 15))
                    if sl is None or sl.stop - sl.start < 200:
                        break
                    pp = preprocess_array(sw, arr, det, scans=sl)
                    G = migrate_grid(pp["X"], S[sl], s_grid, pp["dx_m"], pp["dt_ns"], det); u = np.nanmedian(g[f"u{arr}"][sl], 0)
                    C[arr], _ = detect_crossings(G, u, s_grid, pp["dt_ns"] * det["grid"]["t_down"], pp["snr"], pp["dt_ns"], det, rng)
                if len(C) < 2:
                    r["status"] = "window too short"; rows.append(r); continue
                X = merge_crossings(C[1], C[2], 0.05); det_s = np.array([o["s0_m"] for o in X])
                gs, gu = network_su(fr, "gas", u_max=1.5); m = (gs > s_grid[0]) & (gs < s_grid[-1]) & (np.abs(gu) <= 0.9)
                r.update(status="ok", s_junction=sj, n_detections=len(X), n_class_A=sum(o["cls"] == "A" for o in X), gas_crosses_strip=bool(m.sum() >= 2))
                if m.sum() >= 2:
                    s_gas = float(np.median(gs[m])); r["s_gas"] = s_gas
                    r["nearest_detection_m"] = float(np.min(np.abs(det_s - s_gas))) if len(det_s) else None
                    rs = rng.uniform(s_grid[0] + 2, s_grid[-1] - 2, 500)
                    r["chance_within_2m"] = float(np.mean([np.min(np.abs(det_s - x)) <= 2.0 for x in rs])) if len(det_s) else 0.0
            except Exception as e:
                r["status"] = f"error: {e}"
            rows.append(r); print(r, flush=True)
    ok = [r for r in rows if r.get("gas_crosses_strip")]
    hits = [r for r in ok if r.get("nearest_detection_m") is not None and r["nearest_detection_m"] <= 2.0]
    summ = dict(n_junctions=len(J), n_drive_windows=len(rows), n_with_drawn_gas_crossing=len(ok), n_hits_within_2m=len(hits),
                expected_by_chance=float(np.sum([r["chance_within_2m"] for r in ok])))
    print("SUMMARY", summ)
    json.dump(dict(summary=summ, rows=rows, junctions=J), open(cache / "junctions.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1])
