"""Step 1 of the along-road plan (3 Nov 2026): the two-drive apex test on EVERY pair of drives whose
footprints overlap, processed in memory (200 MHz only; only results are saved).
Per pair: overlap window in the first drive's frame -> v2 preprocessing, migration, crossings for both
drives -> registration (across: overlap correlation; along: crossing votes with the across offset
fixed, else overlap) -> repeated along-road features -> apex agreement vs random places.
Run: python citywide_pairs.py configs/citywide.yaml  -> cache/city/pair_<i>_<j>.json"""
from __future__ import annotations

import glob
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

from apex_agreement import apex_pair, repeated_features
from build_cube import swath_geometry
from build_v2 import grid_Z, load_cfg, preprocess_array
from cube import smooth_track, track_frame
from geometry import merc_scale
from ids_reader import Swath
from register_multipass import overlap_shift, pairs as tie_pairs, place_interp, slice_map, vote_A
from run_crossings_v2 import detect_crossings, migrate_grid
from run_multipass import interp_u, longest_run
from run_detection import su_to_map


def overlap_windows(tracks: list[np.ndarray], lat_max: float = 1.2, min_len: float = 50.0) -> list[dict]:
    """Pairs of drives running parallel with lateral offset < lat_max (footprints overlap) for >= min_len m;
    window = longest contiguous overlapping run (gaps <= 20 m) in the first drive's frame (s, m)."""
    out = []
    frames = []
    for tr in tracks:
        if len(tr) < 20:
            frames.append(None); continue
        k = float(merc_scale(tr[:, 1]).mean()); xs, ys = smooth_track(tr[:, 0], tr[:, 1], 11); sn, xn, yn = track_frame(xs, ys, k)
        frames.append((sn, xn, yn, k, cKDTree(np.c_[xn, yn]), np.arctan2(np.gradient(yn), np.gradient(xn))))
    for i, fi in enumerate(frames):
        if fi is None:
            continue
        sn, xn, yn, k, tree, hd = fi
        for j, B in enumerate(tracks):
            if j <= i or frames[j] is None:
                continue
            d, idx = tree.query(B, distance_upper_bound=4 / k); ok = np.isfinite(d)
            if ok.sum() < 20:
                continue
            ii = idx[ok]; dx = (B[ok, 0] - xn[ii]) * k; dy = (B[ok, 1] - yn[ii]) * k
            lat = -dx * np.sin(hd[ii]) + dy * np.cos(hd[ii])
            hb = np.arctan2(np.gradient(B[:, 1]), np.gradient(B[:, 0]))[ok]
            m = (np.abs(np.cos(hb - hd[ii])) > 0.97) & (np.abs(lat) < lat_max)
            if m.sum() < 20:
                continue
            s = np.sort(sn[ii[m]]); br = np.nonzero(np.diff(s) > 20)[0]
            runs = np.split(s, br + 1); r = max(runs, key=lambda x: x[-1] - x[0])
            if r[-1] - r[0] >= min_len:
                out.append(dict(i=i, j=j, s0=float(r[0] - 10), s1=float(r[-1] + 10), lat=float(np.median(lat[m])),
                                same=bool(np.mean(np.cos(hb[m] - hd[ii[m]]) > 0) > 0.5)))
    return out


def process_pass(sw: Swath, frame: dict, det: dict, w0: float, w1: float, s_grid: np.ndarray, rng) -> dict | None:
    geo = swath_geometry(sw, det["base"], frame)
    S, U = geo["s2"], geo["u2"]; sc, uc = np.nanmedian(S, 1), np.nanmedian(U, 1)
    sl = longest_run((sc > w0 - 15) & (sc < w1 + 15) & (np.abs(uc) < 8.0))
    if sl is None or sl.stop - sl.start < 200:
        return None
    pp = preprocess_array(sw, 2, det, scans=sl); Ss, Us = S[sl], U[sl]
    G = migrate_grid(pp["X"], Ss, s_grid, pp["dx_m"], pp["dt_ns"], det)
    Z = grid_Z(pp["X"], Ss, s_grid).astype(np.float16); Ug = interp_u(Us, Ss, s_grid); u_ch = np.nanmedian(Us, 0)
    dtg = pp["dt_ns"] * det["grid"]["t_down"]
    cands, info = detect_crossings(G, u_ch, s_grid, dtg, pp["snr"], pp["dt_ns"], det, rng)
    return dict(G=G, Z=Z, Ug=Ug, u_ch=u_ch, dt=pp["dt_ns"], dtg=dtg, crossings=cands,
                direction=int(np.sign(np.nanmedian(np.diff(np.nanmedian(Ss, 1))))), u_median=float(np.nanmedian(Us)))


def register_pair(Da: dict, Db: dict, s_grid, u_grid, SL, v, tie) -> dict:
    """Offsets of drive b relative to drive a: across (B) from overlap correlation of depth maps, along (A)
    from crossing votes with B fixed (else from overlap). ok False if neither is reliable."""
    ds, du = float(s_grid[1] - s_grid[0]), float(u_grid[1] - u_grid[0])
    mean = lambda D, zi: place_interp(slice_map(D["G"][0], D["dtg"], v, *SL[zi]), D["Ug"], s_grid, 0.0, 0.0, u_grid)
    ov = [o for o in (overlap_shift(mean(Da, zi), mean(Db, zi), ds, du) for zi in (1, 2, 3)) if o]
    o = None
    if ov:
        o = dict(dA=float(np.median([x["dA"] for x in ov])), dB=float(np.median([x["dB"] for x in ov])), r_best=float(np.median([x["r_best"] for x in ov])),
                 r_zero=ov[0]["r_zero"], spread_A=float(np.ptp([x["dA"] for x in ov])), spread_B=float(np.ptp([x["dB"] for x in ov])))
    B = o["dB"] if (o and o["spread_B"] <= 0.25) else 0.0
    sig = lambda L: [c for c in L if c["q"] <= 0.05]
    P = tie_pairs(sig(Da["crossings"]), sig(Db["crossings"]), tie)
    A_t, peak, far = vote_A(P, B) if len(P) else (0.0, 0, 0)
    if peak >= 6 and peak >= 1.5 * max(far, 1):
        A, how = A_t, "crossings"
    elif o and o["spread_A"] <= 0.5 and o["r_best"] >= 0.5 and abs(o["dA"]) < 2.7:
        A, how = o["dA"], "overlap"
    else:
        return dict(ok=False, overlap=o, tie_peak=peak, tie_far=far)
    return dict(ok=True, A=A, B=B, how=how, overlap=o, tie_peak=peak, tie_far=far, B_known=bool(o and o["spread_B"] <= 0.25))


def main(cfg_path: str) -> None:
    cc = yaml.safe_load(open(cfg_path)); det = load_cfg(str(Path(__file__).resolve().parents[1] / "pipeline" / cc["detector_config"]))
    cache = Path(det["base"]["cache_dir"]) / "city"; cache.mkdir(exist_ok=True)
    v = det["migration"]["velocity_m_ns"]; SL = cc["depth_slices_m"]
    files = sorted(glob.glob(str(Path(cc["raw_root"]) / "*" / "Survey_*" / "Swath*.path")))
    z = np.load(cc["swath_tracks"], allow_pickle=True); tracks = [z[f"s{i}"] for i in range(len(z["days"]))]
    plist = overlap_windows(tracks)
    if cc.get("only_pairs"):
        plist = [p for p in plist if [p["i"], p["j"]] in cc["only_pairs"]]
    json.dump(plist, open(cache / "pair_list.json", "w"))
    print(f"{len(plist)} overlapping pairs; total overlap {sum(p['s1'] - p['s0'] for p in plist) / 1000:.1f} km", flush=True)
    t0 = time.time()
    for n_, p in enumerate(plist):
        out = cache / f"pair_{p['i']}_{p['j']}.json"
        if out.exists() and not json.load(open(out)).get("status", "").startswith("error"):
            continue                                   # done; pairs that ended in an error are retried
        rng = np.random.default_rng([det["seed"], p["i"], p["j"]])
        res = dict(pair=p, file_i=files[p["i"]], file_j=files[p["j"]])
        try:
            swi = Swath.load(str(Path(files[p["i"]]).parent), Path(files[p["i"]]).stem)
            swj = Swath.load(str(Path(files[p["j"]]).parent), Path(files[p["j"]]).stem)
            g = swath_geometry(swi, det["base"]); frame = {k: g[k] for k in ("s_nodes", "xn", "yn", "k")}
            s_grid = np.arange(max(p["s0"], 0.0), min(p["s1"], float(frame["s_nodes"][-1])), det["grid"]["ds_m"])
            Da = process_pass(swi, frame, det, s_grid[0], s_grid[-1], s_grid, rng); Db = process_pass(swj, frame, det, s_grid[0], s_grid[-1], s_grid, rng)
            if Da is None or Db is None:
                res["status"] = "no scans in window"
            else:
                lo = np.nanmin([np.nanmin(Da["Ug"]), np.nanmin(Db["Ug"])]) - 0.3; hi = np.nanmax([np.nanmax(Da["Ug"]), np.nanmax(Db["Ug"])]) + 0.3
                u_grid = np.arange(np.floor(lo / 0.08) * 0.08, hi, 0.08)
                reg = register_pair(Da, Db, s_grid, u_grid, SL, v, cc["tie"])
                res["registration"] = reg; res["directions"] = (Da["direction"], Db["direction"])
                if not reg["ok"]:
                    res["status"] = "registration failed"
                else:
                    REG = {"a": dict(A=0.0, B=0.0), "b": dict(A=reg["A"], B=reg["B"])}
                    D = {"a": Da, "b": Db}
                    feats = repeated_features(D, REG, ["a", "b"], s_grid, u_grid, SL, v)
                    rows = []
                    for f in feats:
                        sm = 0.5 * (f["s0"] + f["s1"]); r = apex_pair(D, REG, "a", "b", s_grid, u_grid, sm, f["u"], float(np.mean(SL[f["zi"]])), v)
                        ucs = [q for q in r["uc"] if q is not None]
                        x, y = su_to_map(np.array([sm]), np.array([float(np.mean(ucs)) if ucs else f["u"]]), frame["s_nodes"], frame["xn"], frame["yn"], float(frame["k"]))
                        rows.append(dict({k: f[k] for k in ("zi", "s0", "s1", "u")}, **r, x=float(x[0]), y=float(y[0])))
                    ctrl = []
                    if feats:
                        cov = np.isfinite(place_interp(slice_map(Da["G"][0], Da["dtg"], v, *SL[2]), Da["Ug"], s_grid, 0, 0, u_grid)) & \
                              np.isfinite(place_interp(slice_map(Db["G"][0], Db["dtg"], v, *SL[2]), Db["Ug"], s_grid, reg["A"], reg["B"], u_grid))
                        ii, jj = np.nonzero(cov)
                        for _ in range(cc["n_controls_per_pair"] if len(ii) else 0):
                            kk = rng.integers(len(ii)); zi = int(rng.choice([f["zi"] for f in feats]))
                            r = apex_pair(D, REG, "a", "b", s_grid, u_grid, float(s_grid[ii[kk]]), float(u_grid[jj[kk]]), float(np.mean(SL[zi])), v)
                            x, y = su_to_map(np.array([s_grid[ii[kk]]]), np.array([u_grid[jj[kk]]]), frame["s_nodes"], frame["xn"], frame["yn"], float(frame["k"]))
                            ctrl.append(dict(agree=r["agree"], both_pipe=r["both_pipe"], x=float(x[0]), y=float(y[0])))
                    res.update(status="ok", features=rows, controls=ctrl, n_feat=len(rows), n_agree=sum(r["agree"] for r in rows))
        except Exception as e:  # keep going; record the failure
            res["status"] = f"error: {e}"; res["trace"] = traceback.format_exc()[-1500:]
        json.dump(res, open(out, "w"))
        extra = f", features {res['n_feat']}, agree {res['n_agree']}" if res.get("status") == "ok" else ""
        print(f"[{n_+1}/{len(plist)}] pair {p['i']}-{p['j']} ({p['s1'] - p['s0']:.0f} m, {'same' if p['same'] else 'opposite'} dir): {res['status']}{extra}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
