"""Stage A of the multi-pass mosaic: every pass is placed directly on the reference pass's
(s, u) frame; only its scans inside the window are processed (v2 preprocessing, migration,
crossing detection). Run: python run_multipass.py configs/multipass_aurelia.yaml
Output (local cache): mp_<pass>_<arr>.npz (G, Z, per-channel u along s, ...) and mp_<pass>.json
(crossing candidates, scan range, direction, post-processed-track displacement)."""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
from paths import load_yaml

from build_cube import swath_geometry
from build_v2 import grid_Z, load_cfg, preprocess_array
from ids_reader import Swath
from run_crossings_v2 import detect_crossings, migrate_grid

ROOT = Path(__file__).resolve().parents[1]


def longest_run(mask: np.ndarray) -> slice | None:
    """Longest contiguous run of True in a 1-D mask, as a slice."""
    if not mask.any():
        return None
    d = np.diff(np.r_[0, mask.astype(int), 0]); st = np.nonzero(d == 1)[0]; en = np.nonzero(d == -1)[0]
    i = int(np.argmax(en - st)); return slice(int(st[i]), int(en[i]))


def interp_u(Usub: np.ndarray, Ssub: np.ndarray, s_grid: np.ndarray) -> np.ndarray:
    """Per-channel lateral position (m) at each s grid node (nan outside the pass). -> (n_ch, n_s)."""
    out = np.full((Usub.shape[1], len(s_grid)), np.nan, np.float32)
    for c in range(Usub.shape[1]):
        o = np.argsort(Ssub[:, c]); ss = Ssub[o, c]; uu = Usub[o, c]
        out[c] = np.interp(s_grid, ss, uu, left=np.nan, right=np.nan)
    return out


def pass_spec(mc: dict, p) -> tuple[str, str, str]:
    """(survey_dir, swath, label) for a pass given as 'SwathNNN' (config session/survey) or [session, survey, swath]."""
    if isinstance(p, str):
        return str(Path(mc["raw_root"]) / mc["session"] / mc["survey"]), p, p
    return str(Path(mc["raw_root"]) / p[0] / p[1]), p[2], f"{p[1][-8:]}_{p[2]}"


def main(cfg_path: str) -> None:
    mc = load_yaml(cfg_path); det = load_cfg(str(ROOT / "pipeline" / mc["detector_config"]))
    cache = Path(det["base"]["cache_dir"]); rng = np.random.default_rng(det["seed"]); t0 = time.time()
    tag = mc.get("tag", "")
    rdir, rsw, _ = pass_spec(mc, mc["reference"])
    ref = Swath.load(rdir, rsw); g_ref = swath_geometry(ref, det["base"])
    frame = {k: g_ref[k] for k in ("s_nodes", "xn", "yn", "k")}
    np.savez(cache / f"mp_{tag}frame.npz", **frame)
    w0, w1 = mc["window_s_m"]; s_grid = np.arange(w0, w1, det["grid"]["ds_m"])
    base_pp = copy.deepcopy(det["base"]); base_pp["track"] = dict(det["base"]["track"], source="pp")
    for spec in mc["passes"]:
        sdir, swname, name = pass_spec(mc, spec)
        fj = cache / f"mp_{tag}{name}.json"
        if fj.exists():
            continue
        sw = Swath.load(sdir, swname)
        geo = swath_geometry(sw, det["base"], frame)
        try:
            geo_pp = swath_geometry(sw, base_pp, frame)
        except Exception as e:                       # post-processed track missing or unreadable
            geo_pp = None; print(name, "no post-processed track:", e)
        res = dict(name=name, arrays={})
        for a in (1, 2):
            S, U = geo[f"s{a}"], geo[f"u{a}"]
            sc, uc = np.nanmedian(S, 1), np.nanmedian(U, 1)
            sl = longest_run((sc > w0 - 15) & (sc < w1 + 15) & (np.abs(uc) < mc["u_max_m"]))
            if sl is None or sl.stop - sl.start < 200:
                res["arrays"][a] = None; continue
            pp = preprocess_array(sw, a, det, scans=sl)
            Ssub, Usub = S[sl], U[sl]
            G = migrate_grid(pp["X"], Ssub, s_grid, pp["dx_m"], pp["dt_ns"], det)
            Z = grid_Z(pp["X"], Ssub, s_grid).astype(np.float16)
            Ug = interp_u(Usub, Ssub, s_grid); u_ch = np.nanmedian(Usub, 0)
            dtg = pp["dt_ns"] * det["grid"]["t_down"]
            cands, info = detect_crossings(G, u_ch, s_grid, dtg, pp["snr"], pp["dt_ns"], det, rng)
            disp = None
            if geo_pp is not None:                   # where the post-processed track would put the same scans
                ds_pp = np.nanmedian(geo_pp[f"s{a}"][sl] - Ssub, 1); du_pp = np.nanmedian(geo_pp[f"u{a}"][sl] - Usub, 1)
                disp = dict(s=np.round(np.nanmedian(Ssub, 1)[::20], 2).tolist(), ds=np.round(ds_pp[::20], 3).tolist(), du=np.round(du_pp[::20], 3).tolist())
            direction = int(np.sign(np.nanmedian(np.diff(np.nanmedian(Ssub, 1)))))
            np.savez(cache / f"mp_{tag}{name}_{a}.npz", G=G, Z=Z, Ug=Ug, u_ch=u_ch, s_grid=s_grid, snr=pp["snr"], dt=pp["dt_ns"], dtg=dtg)
            res["arrays"][a] = dict(scans=[sl.start, sl.stop], direction=direction, u_median=float(np.nanmedian(Usub)),
                                    s_cover=[float(np.nanmin(Ssub)), float(np.nanmax(Ssub))], crossings=cands, info=info, pp_disp=disp,
                                    time_offset_corr=float(geo.get(f"corr{a}", np.nan)))
            print(f"{name} array {a}: scans {sl.start}-{sl.stop}, dir {direction:+d}, u {np.nanmedian(Usub):+.2f} m, "
                  f"crossings q<=0.05: {info['n_sig']}, {time.time()-t0:.0f}s", flush=True)
            del pp, G, Z
        json.dump(res, open(fj, "w"))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main(sys.argv[1])
