"""Stage 3: pipes running along the street. Run after processing/preprocess.py (build):
    python -m italgas_gpr.along.tracks configs/detector.yaml (then dev_swath_v2_1.yaml, archived)
Output: long_<hash>.json (tracks with q-values) and long_<hash>.npz (score images)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from italgas_gpr.processing.preprocess import build, cfg_hash, load_cfg
from italgas_gpr.crossing.score import empirical_fdr
from italgas_gpr.along.focus import lateral_contrast, lateral_migrate, link_tracks, point_vs_line_contrast, stack_along_s, window_peaks
from italgas_gpr.crossing.detect import search_window


def tracks_of(score, thr, L, du, dt, hop, t_rng):
    pk = window_peaks(score, thr, max(1, int(L["sep_u_m"] / du)), max(1, int(L["sep_ns"] / dt)), t_rng)
    tr = link_tracks(pk, score, max(1, int(L["gate_u_m"] / du)), max(1, int(L["gate_ns"] / dt)), L["max_gap"])
    keep = [t for t in tr if (t[-1, 0] - t[0, 0]) * hop >= L["min_len_m"]]
    return keep, np.array([score[t[:, 0], t[:, 1], t[:, 2]].sum() for t in keep])


def to_ground_frame(sc: np.ndarray, shift_cells: np.ndarray, n_pad: int) -> np.ndarray:
    """Shift each window's across-strip axis by shift_cells[w] (integer cells, = round(y_van / du)) onto a common
    ground grid of n_u + 2 n_pad cells (ground u = instrument u + y_van). Cells outside the strip are 0.
    sc (n_win, n_u, n_t) -> (n_win, n_u + 2 n_pad, n_t)."""
    n_win, n_u, n_t = sc.shape; out = np.zeros((n_win, n_u + 2 * n_pad, n_t), sc.dtype)
    for w in range(n_win):                                           # loop over windows (hundreds), not cells
        k = int(shift_cells[w]) + n_pad
        lo, hi = max(0, k), min(n_u + 2 * n_pad, k + n_u)
        if hi > lo:
            out[w, lo:hi] = sc[w, lo - k:hi - k]
    return out


def detect_longitudinal(Z: np.ndarray, u: np.ndarray, s: np.ndarray, dt: float, snr: np.ndarray, cfg: dict,
                        rng: np.random.Generator, y_win=None) -> tuple[list[dict], dict]:
    """Longitudinal-pipe detection on one array. Z (n_ch, n_s, n_t) noise-equalised traces on
    the s grid (m), u (n_ch,) channel lateral positions (m), dt (ns), snr (n_ch, n_t_full).
    y_win: optional callable s (m) -> van sideways position (m); if given, evidence and nulls are re-registered to the
    ground frame (u_ground = u_instrument + y_van) before linking (E43). Track u is then a ground offset.
    Returns (tracks sorted by score, info) where each track has s_start/s_end/length (m),
    u_mean/u_std (m), t_ns, depth_m, score, q, p_real and its path; info holds the threshold,
    counts and a plan-view map (max over search depths of the score, (n_win, n_u)) + centres."""
    L = cfg["longitudinal"]; xc = cfg["crossing"]; v = cfg["migration"]["velocity_m_ns"]
    ds = float(s[1] - s[0])
    i0g, i1g, t_hi = search_window(snr, dt, dt, xc["t_min_ns"], xc["snr_min"])
    X, centres = stack_along_s(Z, ds, L["win_m"], L["hop_m"])
    ug = np.arange(u.min() + 0.06, u.max() - 0.06 + 1e-9, L["du_m"])
    contrast = point_vs_line_contrast if L.get("contrast") == "point_vs_line" else lateral_contrast
    sc = contrast(lateral_migrate(X, u, ug, dt, v))
    nulls = [contrast(lateral_migrate(X[:, rng.permutation(len(u)), :], u, ug, dt, v)) for _ in range(L["n_surrogates"])]
    thr = float(np.quantile(np.stack([n[:, :, i0g:i1g] for n in nulls]), L["peak_null_quantile"]))   # before any re-registration (padding is 0)
    if y_win is not None:                                            # E43 ground frame: y_win(s) -> shift at each window centre
        yw = np.asarray(y_win(s[centres]), float); yw = np.where(np.isfinite(yw), yw, 0.0)
        kc = np.round(yw / L["du_m"]).astype(int); n_pad = int(max(1, np.max(np.abs(kc))))
        sc = to_ground_frame(sc, kc, n_pad); nulls = [to_ground_frame(n, kc, n_pad) for n in nulls]
        ug = np.concatenate([ug[0] - L["du_m"] * np.arange(n_pad, 0, -1), ug, ug[-1] + L["du_m"] * np.arange(1, n_pad + 1)])
    tr, ts = tracks_of(sc, thr, L, L["du_m"], dt, L["hop_m"], (i0g, i1g))
    null_ts = np.concatenate([tracks_of(n, thr, L, L["du_m"], dt, L["hop_m"], (i0g, i1g))[1] for n in nulls])
    if len(ts) and len(null_ts):
        q, p_real = empirical_fdr(ts, null_ts, L["n_surrogates"])
    else:                                    # no data tracks, or no null track at all (then nothing can be a false discovery)
        q, p_real = np.zeros(len(ts)), np.ones(len(ts))
    out = []
    for t, sco, qq, pp in zip(tr, ts, q, p_real):
        sc_m = s[centres[t[:, 0]]]; uu = ug[t[:, 1]]; tt = t[:, 2] * dt
        out.append(dict(s_start=round(float(sc_m.min()), 2), s_end=round(float(sc_m.max()), 2),
                        length_m=round(float(np.ptp(sc_m)), 2), u_mean=round(float(uu.mean()), 3), u_std=round(float(uu.std()), 3),
                        t_ns=round(float(np.median(tt)), 2), depth_m=round(float(v * np.median(tt) / 2), 2),
                        score=round(float(sco), 2), q=round(float(qq), 4), p_real=round(float(pp), 3),
                        path_s=np.round(sc_m, 2).tolist(), path_u=np.round(uu, 3).tolist(), path_t=np.round(tt, 2).tolist()))
    out.sort(key=lambda x: -x["score"])
    info = dict(thr=thr, t_search_ns=[round(i0g * dt, 2), round(t_hi, 2)], n_tracks=len(out),
                n_null_tracks_per_rep=len(null_ts) / max(1, L["n_surrogates"]), n_sig=sum(o["q"] <= L["q_max"] for o in out),
                plan=sc[:, :, i0g:i1g].max(2).astype(np.float16), plan_t=(sc[:, :, i0g:i1g].argmax(2) + i0g).astype(np.int16),
                centres=centres, ug=ug)
    return out, info


def run(cfg_path: str) -> Path:
    cfg = load_cfg(cfg_path); L = cfg["longitudinal"]
    outs = build(cfg_path); rng = np.random.default_rng(cfg["seed"])
    res = {"arrays": {}}; npz = {}; t00 = time.time()
    for arr, path in outs.items():
        d = np.load(path)
        tracks, info = detect_longitudinal(d["Z"].astype(np.float32), d["u_ch"], d["s_grid"], float(d["dt_full_ns"]), d["snr"], cfg, rng)
        for k_ in ("plan", "plan_t", "centres", "ug"):
            npz[f"{k_}{arr}"] = info.pop(k_)
        res["arrays"][arr] = dict(**info, tracks=tracks)
        print(f"array {arr}: thr {info['thr']:.3f}, tracks {info['n_tracks']} (null/rep {info['n_null_tracks_per_rep']:.1f}), "
              f"q<={L['q_max']}: {info['n_sig']}, {time.time()-t00:.0f}s", flush=True)
    h = cfg_hash(cfg, keys=("base", "preprocess", "migration", "grid", "cube_file", "crossing", "longitudinal"))
    cache = Path(cfg["base"]["cache_dir"])
    np.savez(cache / f"long_{h}.npz", **npz)
    o = cache / f"long_{h}.json"; json.dump(res, open(o, "w")); print("saved", o)
    return o


if __name__ == "__main__":
    run(sys.argv[1])
