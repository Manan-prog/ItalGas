"""Stage 2 of the v2 detector: crossing-pipe candidates with an empirical false
discovery rate from channel-shift surrogates. Run after processing/preprocess.py (build):
    python -m italgas_gpr.crossing.detect configs/detector.yaml (then dev_swath_v2.yaml, archived)
Output (local cache): xing_<hash>.npz (score maps, candidates) and xing_<hash>.json."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from italgas_gpr.processing.preprocess import build, cfg_hash, load_cfg
from italgas_gpr.crossing.score import crossing_score, empirical_fdr, local_maxima, phi_velocity_index


def search_window(snr: np.ndarray, dt_full: float, dt_g: float, t_min_ns: float, snr_min: float) -> tuple[int, int, float]:
    """Time-index window of G to search: from t_min_ns to the first time after it where the
    median over channels of sigma(t)/floor drops below snr_min (MEASURED per swath)."""
    med = np.median(snr, 0)
    t = np.arange(len(med)) * dt_full
    after = np.nonzero((t > t_min_ns) & (med < snr_min))[0]
    t_hi = t[after[0]] if len(after) else t[-1]
    return int(t_min_ns / dt_g), int(t_hi / dt_g), float(t_hi)


def migrate_grid(X: np.ndarray, S: np.ndarray, s_grid: np.ndarray, dx: float, dt: float, cfg: dict) -> np.ndarray:
    """Along-track Kirchhoff migration of every channel at each apparent velocity, envelope,
    t_down averaging, robust z, gridding. X (n_scans, n_ch, n_t) -> G (n_v, n_ch, n_s, n_td) float16."""
    from italgas_gpr.processing.filters import envelope, kirchhoff_migrate, robust_z
    from italgas_gpr.crossing.score import to_s_grid
    m = cfg["migration"]; td = cfg["grid"]["t_down"]; n_td = X.shape[2] // td
    G = np.zeros((len(m["v_apparent_m_ns"]), X.shape[1], len(s_grid), n_td), np.float16)
    for c in range(X.shape[1]):
        for k, v in enumerate(m["v_apparent_m_ns"]):
            mig = kirchhoff_migrate(X[:, c], dx, dt, v, m["aperture_m"])
            e = envelope(mig)[:, : n_td * td].reshape(len(mig), n_td, td).mean(2)
            G[k, c] = to_s_grid(robust_z(e), S[:, c], s_grid)
    return G


def detect_crossings(G: np.ndarray, u: np.ndarray, s_grid: np.ndarray, dtg: float, snr: np.ndarray, dt_full: float,
                     cfg: dict, rng: np.random.Generator) -> tuple[list[dict], dict]:
    """Crossing candidates for one array with q-values from the column-shift null.
    G (n_v, n_ch, n_s, n_td) migrated robust-z envelopes; u (n_ch,) m. Returns (candidates, info)."""
    xc = cfg["crossing"]
    phis = np.arange(xc["phi_deg"][0], xc["phi_deg"][1] + 1e-9, xc["phi_deg"][2], dtype=float)
    v_idx = phi_velocity_index(phis, cfg["migration"]["velocity_m_ns"], cfg["migration"]["v_apparent_m_ns"])
    ds = float(s_grid[1] - s_grid[0])
    i0, i1, t_hi = search_window(snr, dt_full, dtg, xc["t_min_ns"], xc["snr_min"])
    Gl = [G[k][:, :, : i1 + 4].astype(np.float32) for k in range(G.shape[0])]
    n_ch, n_s = Gl[0].shape[:2]
    kw = dict(flank_m=tuple(xc["flank_m"])) if xc.get("flank_m") else {}
    sc, ph = crossing_score(Gl, u, ds, phis, v_idx, xc["keep_frac"], tuple(xc["tol_cells"]), **kw)
    edge = int(2.0 / ds); sc[:edge] = 0; sc[-edge:] = 0
    i_s, i_t = local_maxima(sc, ds, dtg, xc["sep_m"], xc["sep_ns"], (i0, i1), thr=0.0)
    data = sc[i_s, i_t]; null = []; lo = int(xc["min_shift_m"] / ds)
    for _ in range(xc["n_surrogates"]):
        if xc.get("null_group_m"):
            grp = np.round((u - np.median(u)) / xc["null_group_m"]).astype(int)
            shifts = rng.integers(lo, n_s - lo, grp.max() - grp.min() + 1)[grp - grp.min()]
        else:
            shifts = rng.integers(lo, n_s - lo, n_ch)
        sn, _ = crossing_score(Gl, u, ds, phis, v_idx, xc["keep_frac"], tuple(xc["tol_cells"]), shifts=shifts, **kw)
        sn[:edge] = 0; sn[-edge:] = 0
        a, b = local_maxima(sn, ds, dtg, xc["sep_m"], xc["sep_ns"], (i0, i1), thr=0.0)
        null.append(sn[a, b])
    null = np.concatenate(null)
    q, p_real = empirical_fdr(data, null, xc["n_surrogates"]) if len(data) and len(null) else (np.zeros(len(data)), np.ones(len(data)))
    v = cfg["migration"]["velocity_m_ns"]
    cands = sorted([dict(s0_m=round(float(s_grid[i]), 2), t_ns=round(float(j * dtg), 2), depth_m=round(float(v * j * dtg / 2), 2),
                         phi_deg=float(phis[ph[i, j]]), score=round(float(sc[i, j]), 3), q=round(float(qq), 4), p_real=round(float(pp), 3))
                    for i, j, qq, pp in zip(i_s, i_t, q, p_real)], key=lambda c: -c["score"])
    return cands, dict(t_search_ns=[round(i0 * dtg, 2), round(t_hi, 2)], n_candidates=len(cands),
                       n_sig=int(sum(c["q"] <= xc["q_max"] for c in cands)))


def run(cfg_path: str) -> Path:
    cfg = load_cfg(cfg_path); xc = cfg["crossing"]
    outs = build(cfg_path)
    rng = np.random.default_rng(cfg["seed"])
    phis = np.arange(xc["phi_deg"][0], xc["phi_deg"][1] + 1e-9, xc["phi_deg"][2], dtype=float)
    v_idx = phi_velocity_index(phis, cfg["migration"]["velocity_m_ns"], cfg["migration"]["v_apparent_m_ns"])
    res = {"phis": phis.tolist(), "arrays": {}}
    npz = {}
    t0 = time.time()
    for arr, path in outs.items():
        d = np.load(path)
        ds = float(d["s_grid"][1] - d["s_grid"][0]); dtg = float(d["dt_ns"])
        i0, i1, t_hi = search_window(d["snr"], float(d["dt_full_ns"]), dtg, xc["t_min_ns"], xc["snr_min"])
        G = [d["G"][k][:, :, : i1 + 4].astype(np.float32) for k in range(d["G"].shape[0])]
        u = d["u_ch"]; n_ch, n_s = G[0].shape[:2]
        sc, ph = crossing_score(G, u, ds, phis, v_idx, xc["keep_frac"], tuple(xc["tol_cells"]), flank_m=tuple(xc["flank_m"]))
        edge = int(2.0 / ds)
        sc[:edge] = 0; sc[-edge:] = 0
        i_s, i_t = local_maxima(sc, ds, dtg, xc["sep_m"], xc["sep_ns"], (i0, i1), thr=0.0)
        data = sc[i_s, i_t]
        null = []
        lo = int(xc["min_shift_m"] / ds)
        for r in range(xc["n_surrogates"]):
            if xc.get("null_group_m"):
                grp = np.round((u - np.median(u)) / xc["null_group_m"]).astype(int)
                g_shift = rng.integers(lo, n_s - lo, grp.max() - grp.min() + 1)
                shifts = g_shift[grp - grp.min()]
            else:
                shifts = rng.integers(lo, n_s - lo, n_ch)
            sn, _ = crossing_score(G, u, ds, phis, v_idx, xc["keep_frac"], tuple(xc["tol_cells"]), shifts=shifts, flank_m=tuple(xc["flank_m"]))
            sn[:edge] = 0; sn[-edge:] = 0
            a, b = local_maxima(sn, ds, dtg, xc["sep_m"], xc["sep_ns"], (i0, i1), thr=0.0)
            null.append(sn[a, b])
            print(f"array {arr} surrogate {r+1}/{xc['n_surrogates']} {time.time()-t0:.0f}s", flush=True)
        null = np.concatenate(null)
        q, p_real = empirical_fdr(data, null, xc["n_surrogates"])
        v = cfg["migration"]["velocity_m_ns"]
        cands = [dict(s0_m=round(float(i * ds), 2), t_ns=round(float(j * dtg), 2),
                      depth_m=round(float(v * j * dtg / 2), 2), phi_deg=float(phis[ph[i, j]]),
                      score=round(float(sc[i, j]), 3), q=round(float(qq), 4), p_real=round(float(pp), 3))
                 for i, j, qq, pp in zip(i_s, i_t, q, p_real)]
        cands.sort(key=lambda c: -c["score"])
        n_sig = int((q <= xc["q_max"]).sum())
        res["arrays"][arr] = dict(t_search_ns=[round(i0 * dtg, 2), round(t_hi, 2)], n_candidates=len(cands),
                                  n_sig=n_sig, candidates=cands)
        npz[f"score{arr}"] = sc.astype(np.float16); npz[f"phi{arr}"] = ph
        npz[f"null_scores{arr}"] = null; npz[f"data_scores{arr}"] = data
        print(f"array {arr}: search {i0*dtg:.1f}-{t_hi:.1f} ns, candidates {len(cands)}, q<={xc['q_max']}: {n_sig}", flush=True)
    h = cfg_hash(cfg, keys=("base", "preprocess", "migration", "grid", "cube_file", "crossing"))
    cache = Path(cfg["base"]["cache_dir"])
    np.savez(cache / f"xing_{h}.npz", **npz)
    out = cache / f"xing_{h}.json"
    json.dump(res, open(out, "w"))
    print("saved", out)
    return out


if __name__ == "__main__":
    run(sys.argv[1])
