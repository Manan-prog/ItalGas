"""Step 3 data (3 Nov 2026): single-drive along-road tracks on all drives (200 MHz), with per-track
measurements and the distance to the town-plan sketch's gas and water lines (INTERNAL use).
Drives are processed in ~1.5 km chunks (statistics per chunk). Run:
python citywide_tracks.py configs/citywide.yaml -> cache/city/tracks_<swath index>.json"""
from __future__ import annotations

import glob
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml
from pyproj import Transformer
from scipy.spatial import cKDTree

from along_road import track_shape
from build_cube import swath_geometry
from build_v2 import grid_Z, load_cfg, preprocess_array
from ids_reader import Swath
from pipe_features import reference_sign, reflection_shape
from puc_overlay import MASK
from run_detection import su_to_map
from run_longitudinal_v2 import detect_longitudinal

CHUNK_M = 1500.0


def sketch_trees():
    import rasterio
    with rasterio.open(MASK) as r:
        a = r.read(1); T = r.transform
    out = {}
    for name, k in (("gas", 1), ("water", 2), ("sewer", 3)):
        rr, cc = np.nonzero(a == k); xs, ys = rasterio.transform.xy(T, rr, cc)
        out[name] = cKDTree(np.c_[xs, ys])
    with rasterio.open(MASK) as r:
        b = r.bounds
    return out, b


def track_measures(t: dict, Z: np.ndarray, u: np.ndarray, s: np.ndarray, dt: float, v: float, ref: np.ndarray) -> dict:
    sh = track_shape(Z, u, s, dt, v, t)
    ps, pu, pt = np.array(t["path_s"]), np.array(t["path_u"]), np.array(t["path_t"])
    pol, asy, amp = [], [], []
    for a, b, c in zip(ps[::3], pu[::3], pt[::3]):
        ch = int(np.argmin(np.abs(u - b))); i = int(np.searchsorted(s, a)); j = int(c / dt); h = int(6 / dt)
        w = Z[ch, max(0, i - 2): i + 3, max(0, j - h): j + h + 1].astype(np.float32).mean(0)
        if len(w) > 8:
            r = reflection_shape(w, dt, 4.0); pol.append(r["sign"] * ref[ch]); asy.append(r["asymmetry"]); amp.append(r["main"])
    d2 = np.diff(pt, 2) if len(pt) > 3 else np.array([0.0])
    return dict(n_pipe_sections=sh["n_pipe_sections"], polarity=float(np.mean(pol)) if pol else 0.0, asymmetry=float(np.median(asy)) if asy else 0.0,
                amplitude=float(np.median(amp)) if amp else 0.0, depth_roughness_ns=float(np.sqrt(np.mean(d2 ** 2))))


def main(cfg_path: str) -> None:
    cc = yaml.safe_load(open(cfg_path)); det = load_cfg(str(Path(__file__).resolve().parents[1] / "pipeline" / cc["detector_config"]))
    cache = Path(det["base"]["cache_dir"]) / "city"; cache.mkdir(exist_ok=True); v = det["migration"]["velocity_m_ns"]
    files = sorted(glob.glob(str(Path(cc["raw_root"]) / "*" / "Survey_*" / "Swath*.path")))
    trees, bounds = sketch_trees(); to3003 = Transformer.from_crs("EPSG:3857", "EPSG:3003", always_xy=True)
    rng = np.random.default_rng(det["seed"] + 31); t0 = time.time()
    for idx, f in enumerate(files):
        out = cache / f"tracks_{idx}.json"
        if out.exists():
            continue
        res = dict(index=idx, file=f, tracks=[])
        try:
            sw = Swath.load(str(Path(f).parent), Path(f).stem)
            if len(sw.index[2]) < 2000:
                res["status"] = "too short"; json.dump(res, open(out, "w")); continue
            g = swath_geometry(sw, det["base"]); frame = {k: g[k] for k in ("s_nodes", "xn", "yn", "k")}
            S = g["s2"]; sc = np.nanmedian(S, 1); L = float(np.nanmax(sc))
            chans = sw.config.channels_of(2)
            ref = np.array([reference_sign(np.asarray(sw.data[2][::20, ch.slot, :]), chans[0].dt_ns, det["base"]["processing"]["dewow_win_ns"][2]) for ch in chans])
            for c0 in np.arange(0.0, L, CHUNK_M):
                c1 = min(L, c0 + CHUNK_M + 30.0)
                m = (sc >= c0) & (sc < c1)
                if m.sum() < 1500:
                    continue
                ii = np.nonzero(m)[0]; sl = slice(int(ii[0]), int(ii[-1]) + 1)
                s_grid = np.arange(c0, c1, det["grid"]["ds_m"])
                pp = preprocess_array(sw, 2, det, scans=sl); Z = grid_Z(pp["X"], S[sl], s_grid); u = np.nanmedian(g["u2"][sl], 0); dt = pp["dt_ns"]
                tracks, info = detect_longitudinal(Z, u, s_grid, dt, pp["snr"], det, rng)
                for t in tracks:
                    if t["length_m"] < 10 or t["s_start"] < c0 + 5 or t["s_start"] > c0 + CHUNK_M:
                        continue
                    t.update(track_measures(t, Z, u, s_grid, dt, v, ref))
                    mx, my = su_to_map(np.array(t["path_s"][::4]), np.array(t["path_u"][::4]), frame["s_nodes"], frame["xn"], frame["yn"], float(frame["k"]))
                    X, Y = to3003.transform(mx, my)
                    inside = (X > bounds.left) & (X < bounds.right) & (Y > bounds.bottom) & (Y < bounds.top)
                    t["in_sketch"] = bool(inside.mean() > 0.9)
                    for name, tr in trees.items():
                        t[f"d_{name}_m"] = float(np.median(tr.query(np.c_[X, Y])[0])) if t["in_sketch"] else None
                    t["x"] = float(np.median(mx)); t["y"] = float(np.median(my))
                    for kk in ("path_s", "path_u", "path_t"):
                        t[kk] = t[kk][::4]
                    res["tracks"].append(t)
                del Z, pp
            res["status"] = "ok"
        except Exception as e:
            res["status"] = f"error: {e}"; res["trace"] = traceback.format_exc()[-1500:]
        json.dump(res, open(out, "w"))
        print(f"[{idx+1}/{len(files)}] {Path(f).parent.name} {Path(f).stem}: {res['status']}, tracks {len(res['tracks'])}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
