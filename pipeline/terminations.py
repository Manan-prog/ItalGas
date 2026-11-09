"""End points of crossing pipes inside the strip (partial crossings): a service pipe that ends at a main running along
the street lines up across only part of the antenna columns. The crossing score (trimmed mean over 75 percent of the
channels) suppresses such pipes by design; this module scores them (DECISIONS 2026-11-04).
Score per (s0, t): channels ordered across the strip, each flank-subtracted along s (a pipe is a thin line); for each
split with enough channels on both sides, mean on the 'present' side minus max(0, mean on the 'absent' side);
maximised over split, side and phi. Significance: the antenna-column-shift null and target-decoy FDR of the crossings."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter

from crossings import empirical_fdr, flank_level, local_maxima, phi_velocity_index
from run_crossings_v2 import search_window


def end_score(G_list: list[np.ndarray], u_ch: np.ndarray, ds: float, phis_deg: np.ndarray, v_idx: np.ndarray,
              min_side: int, tol: tuple[int, int] = (3, 3), shifts: np.ndarray | None = None,
              flank_m: tuple[float, float] | None = (0.6, 1.5)) -> tuple[np.ndarray, ...]:
    """G_list[k]: (n_ch, n_s, n_t) migrated robust-z envelopes at velocity k; u_ch (n_ch,) m; min_side: channels needed
    on each side of the end. Returns score (n_s, n_t), phi index, split index m (end between the m-th and (m+1)-th
    channel in u order), side (+1: pipe on the high-u side of the end, -1: on the low-u side)."""
    order = np.argsort(u_ch); n = len(u_ch); uo = u_ch[order]
    Gm = []
    for G in G_list:
        H = maximum_filter(G, size=(1, tol[0], tol[1]))
        if flank_m is not None:
            H = H - np.stack([flank_level(H[c], ds, flank_m) for c in range(n)])
        Gm.append(H)
    best = None
    for j, phi in enumerate(phis_deg):
        G = Gm[v_idx[j]]; k = np.round(u_ch * np.tan(np.radians(phi)) / ds).astype(int)
        if shifts is not None:
            k = k + shifts
        cs = np.cumsum(np.stack([np.roll(G[c], -k[c], axis=0) for c in order]), axis=0); tot = cs[-1]
        for m in range(min_side, n - min_side + 1):
            lo = cs[m - 1] / m; hi = (tot - cs[m - 1]) / (n - m)
            for side, sc in ((1, hi - np.maximum(lo, 0)), (-1, lo - np.maximum(hi, 0))):
                if best is None:
                    best = sc.copy(); arg = np.zeros(sc.shape, np.int16); split = np.full(sc.shape, m, np.int16); sd = np.full(sc.shape, side, np.int8)
                else:
                    w = sc > best; best[w] = sc[w]; arg[w] = j; split[w] = m; sd[w] = side
    return best, arg, split, sd, uo


def detect_ends(G: np.ndarray, u: np.ndarray, s_grid: np.ndarray, dtg: float, snr: np.ndarray, dt_full: float,
                cfg: dict, rng: np.random.Generator) -> tuple[list[dict], dict]:
    """End-point candidates for one array with q-values from the column-shift null (as detect_crossings).
    Returns (candidates with s0_m (line at u = 0), phi_deg, t_ns, depth_m, u_end_m, side, score, q, p_real), info)."""
    xc = cfg["crossing"]; ec = cfg["ends"]
    phis = np.arange(ec["phi_deg"][0], ec["phi_deg"][1] + 1e-9, ec["phi_deg"][2], dtype=float)
    v_idx = phi_velocity_index(phis, cfg["migration"]["velocity_m_ns"], cfg["migration"]["v_apparent_m_ns"])
    ds = float(s_grid[1] - s_grid[0]); pitch = float(np.median(np.diff(np.sort(u))))
    min_side = max(2, int(round(ec["min_side_m"] / max(pitch, 1e-3))))
    i0, i1, _ = search_window(snr, dt_full, dtg, xc["t_min_ns"], xc["snr_min"])
    Gl = [G[k][:, :, : i1 + 4].astype(np.float32) for k in range(G.shape[0])]
    n_ch, n_s = Gl[0].shape[:2]
    sc, ph, sp, sd, uo = end_score(Gl, u, ds, phis, v_idx, min_side, tuple(xc["tol_cells"]), flank_m=tuple(xc["flank_m"]))
    edge = int(2.0 / ds); sc[:edge] = 0; sc[-edge:] = 0
    i_s, i_t = local_maxima(sc, ds, dtg, xc["sep_m"], xc["sep_ns"], (i0, i1), thr=0.0)
    data = sc[i_s, i_t]; null = []; lo = int(xc["min_shift_m"] / ds)
    for _ in range(xc["n_surrogates"]):
        grp = np.round((u - np.median(u)) / xc["null_group_m"]).astype(int)
        shifts = rng.integers(lo, n_s - lo, grp.max() - grp.min() + 1)[grp - grp.min()]
        sn = end_score(Gl, u, ds, phis, v_idx, min_side, tuple(xc["tol_cells"]), shifts=shifts, flank_m=tuple(xc["flank_m"]))[0]
        sn[:edge] = 0; sn[-edge:] = 0
        a, b = local_maxima(sn, ds, dtg, xc["sep_m"], xc["sep_ns"], (i0, i1), thr=0.0); null.append(sn[a, b])
    null = np.concatenate(null)
    q, p_real = empirical_fdr(data, null, xc["n_surrogates"]) if len(data) and len(null) else (np.zeros(len(data)), np.ones(len(data)))
    v = cfg["migration"]["velocity_m_ns"]; out = []
    for i, j, qq, pp in zip(i_s, i_t, q, p_real):
        m = int(sp[i, j]); out.append(dict(s0_m=round(float(s_grid[i]), 2), t_ns=round(float(j * dtg), 2), depth_m=round(float(v * j * dtg / 2), 2),
                                           phi_deg=float(phis[ph[i, j]]), u_end_m=round(float(0.5 * (uo[m - 1] + uo[m])), 3), side=int(sd[i, j]),
                                           score=round(float(sc[i, j]), 3), q=round(float(qq), 4), p_real=round(float(pp), 3)))
    out.sort(key=lambda c: -c["score"])
    return out, dict(min_side=min_side, n_candidates=len(out), n_sig=int(sum(c["q"] <= xc["q_max"] for c in out)))
