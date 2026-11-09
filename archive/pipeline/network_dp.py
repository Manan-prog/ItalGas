"""Road-network reasoning on a registered multi-pass mosaic (3 Nov 2026):
(1) crossing continuity: does a crossing seen by one group of drives continue under the other group?
(2) along-road path linking: a hidden-Markov / Viterbi path across the street position u, along s,
    through the along-road line evidence, with a 'no pipe' state, smooth lateral drift and a
    chance level from drives shifted sideways at random.
Run: python network_dp.py configs/multipass_aurelia.yaml -> cache/mp_<tag>network.json"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

from build_v2 import load_cfg
from register_multipass import line_evidence, place_interp, slice_map
from run_multipass import pass_spec

ROOT = Path(__file__).resolve().parents[1]


def viterbi_paths(llr: np.ndarray, sigma_cells: float = 1.0, max_step: int = 3, c_on: float = 4.0, c_off: float = 4.0):
    """Best path through llr (n_s, n_u) (log-likelihood ratio pipe vs no pipe per cell; 0 = no data)
    with states u (pipe) or 'none'. Lateral move of k cells per step costs k^2 / (2 sigma^2), up to
    max_step; entering / leaving the pipe state costs c_on / c_off. Returns list of segments
    (i0, i1, u indices, score). Loops over offsets and steps along s, not over traces."""
    n_s, n_u = llr.shape
    P = np.full(n_u, -np.inf); N = 0.0
    bp_p = np.zeros((n_s, n_u), np.int16); bp_n = np.zeros(n_s, np.int16)       # backpointers (-1 = from none)
    from_none = np.zeros((n_s, n_u), bool); none_from = np.full(n_s, -1, np.int32)
    for t in range(n_s):
        best = np.full(n_u, -np.inf); arg = np.zeros(n_u, np.int16)
        for k in range(-max_step, max_step + 1):
            sh = np.roll(P, k)
            if k > 0:
                sh[:k] = -np.inf
            elif k < 0:
                sh[k:] = -np.inf
            cand = sh - k * k / (2 * sigma_cells ** 2)
            m = cand > best; best[m] = cand[m]; arg[m] = (np.arange(n_u) - k)[m]
        enter = N - c_on
        fn = enter > best
        newP = np.where(fn, enter, best) + llr[t]
        from_none[t] = fn; bp_p[t] = arg
        j = int(np.argmax(P)); leave = P[j] - c_off
        if leave > N:
            newN = leave; none_from[t] = j
        else:
            newN = N; none_from[t] = -1
        P, N = newP, newN
    # backtrack
    state = ("none", -1) if N >= P.max() else ("pipe", int(np.argmax(P)))
    path = np.full(n_s, -1, int)
    for t in range(n_s - 1, -1, -1):
        if state[0] == "pipe":
            u = state[1]; path[t] = u
            state = ("none", -1) if from_none[t, u] else ("pipe", int(bp_p[t, u]))
        else:
            j = none_from[t]
            state = ("pipe", int(j)) if j >= 0 else ("none", -1)
    segs = []; t = 0
    while t < n_s:
        if path[t] >= 0:
            t0 = t
            while t < n_s and path[t] >= 0:
                t += 1
            uu = path[t0:t]; segs.append((t0, t, uu, float(llr[np.arange(t0, t), uu].sum())))
        else:
            t += 1
    return segs


def main(cfg_path: str, n_null: int = 40) -> None:
    mc = yaml.safe_load(open(cfg_path)); det = load_cfg(str(ROOT / "pipeline" / mc["detector_config"]))
    cache = Path(det["base"]["cache_dir"]); tag = mc.get("tag", ""); v = det["migration"]["velocity_m_ns"]
    rng = np.random.default_rng(det["seed"] + 7)
    REG = json.load(open(cache / f"mp_{tag}registration.json"))["registration"]
    labels = [n for n in (pass_spec(mc, p)[2] for p in mc["passes"]) if n in REG]
    out = {}
    # ---- (1) crossing continuity between the two direction groups (lines in true coordinates)
    groups = {d: [n for n in labels if REG[n]["direction"] == d] for d in (1, -1)}
    if groups[1] and groups[-1]:
        def lines(names):
            L = []
            for n in names:
                R = json.load(open(cache / f"mp_{tag}{n}.json"))
                for a in ("1", "2"):
                    for c in (R["arrays"].get(a) or {}).get("crossings", []):
                        if c["q"] <= 0.05:
                            L.append((c["s0_m"] - REG[n]["A"] - REG[n]["B"] * np.tan(np.radians(c["phi_deg"])), c["phi_deg"], n))
            return L
        LA, LB = lines(groups[1]), lines(groups[-1])
        SB = np.array([x[0] for x in LB]); PB = np.array([x[1] for x in LB])
        def frac(shift=0.0):
            return float(np.mean([np.any((np.abs(SB + shift - s0) <= 1.0) & (np.abs(PB - p) <= 15)) for s0, p, _ in LA]))
        obs = frac(); ch = np.array([frac(o) for o in rng.uniform(20, 200, 300) * rng.choice([-1, 1], 300)])
        out["crossing_continuity"] = dict(n_A=len(LA), n_B=len(LB), frac_continuing=obs, chance=float(ch.mean()), p=float((ch >= obs).mean()))
        print("crossing continuity:", out["crossing_continuity"], flush=True)
    # ---- (2) along-road path linking on the combined line evidence (0.5-1.1 m slices)
    D = {n: np.load(cache / f"mp_{tag}{n}_2.npz") for n in labels}
    s = D[labels[0]]["s_grid"]; u_grid = np.arange(mc["mosaic"]["u_range_m"][0], mc["mosaic"]["u_range_m"][1] + 1e-9, mc["mosaic"]["du_m"])
    SL = mc["mosaic"]["depth_slices_m"]; ds, du = float(s[1] - s[0]), float(u_grid[1] - u_grid[0])
    per = {zi: {n: place_interp(slice_map(D[n]["G"][0], float(D[n]["dtg"]), v, *SL[zi]), D[n]["Ug"], s, REG[n]["A"], REG[n]["B"], u_grid)
                for n in labels} for zi in (1, 2, 3)}

    def evidence(shifts: dict | None = None):
        Lm = None; cov = None
        for zi in (1, 2, 3):
            maps = []
            for n in labels:
                m = per[zi][n]
                if shifts:
                    k = shifts[n]; m = np.roll(m, k, axis=1)
                    if k > 0:
                        m[:, :k] = np.nan
                    elif k < 0:
                        m[:, k:] = np.nan
                maps.append(m)
            st = np.stack(maps); cnt = np.isfinite(st).sum(0)
            with np.errstate(all="ignore"):
                M = np.nanmean(st, 0)
            L = line_evidence(np.nan_to_num(M), (cnt > 0).astype(np.float32), ds, du)
            Lm = L if Lm is None else np.fmax(Lm, L); cov = cnt > 0 if cov is None else (cov | (cnt > 0))
        step = int(round(0.5 / ds)); n_s = Lm.shape[0] // step
        E = np.nanmean(Lm[: n_s * step].reshape(n_s, step, -1), 1)
        return E

    E = evidence()
    med = np.nanmedian(E); mad = 1.4826 * np.nanmedian(np.abs(E - med))
    to_llr = lambda X: np.where(np.isfinite(X), np.clip((X - med) / mad - 2.0, -2.0, 4.0), 0.0)
    segs = [x for x in viterbi_paths(to_llr(E)) if (x[1] - x[0]) * 0.5 >= 10]
    null_best = []
    for _ in range(n_null):
        sh = {n: int(rng.integers(4, 13)) * int(rng.choice([-1, 1])) for n in labels}
        sn = [x[3] for x in viterbi_paths(to_llr(evidence(sh))) if (x[1] - x[0]) * 0.5 >= 10]
        null_best.append(max(sn) if sn else 0.0)
    null_best = np.array(null_best); thr = float(np.quantile(null_best, 0.95))
    rows = []
    for t0, t1, uu, sc in sorted(segs, key=lambda x: -x[3]):
        rows.append(dict(s0=float(s[0] + t0 * 0.5), s1=float(s[0] + t1 * 0.5), u_mean=float(u_grid[uu].mean()), u_std=float(u_grid[uu].std()),
                         score=sc, p=float((null_best >= sc).mean()), significant=bool(sc > thr),
                         path_s=(s[0] + np.arange(t0, t1) * 0.5).round(2).tolist(), path_u=u_grid[uu].round(3).tolist()))
    out["dp"] = dict(n_paths=len(rows), null_best_q95=thr, null_best_median=float(np.median(null_best)), paths=rows)
    print("DP paths >= 10 m:", len(rows), " null best-path score q95 %.1f" % thr)
    for r in rows[:8]:
        print("   s %.0f-%.0f  u %.2f (sd %.2f)  score %.1f  p %.3f  %s" % (r["s0"], r["s1"], r["u_mean"], r["u_std"], r["score"], r["p"], "SIGNIFICANT" if r["significant"] else ""))
    json.dump(out, open(cache / f"mp_{tag}network.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1])


def viterbi_anchored(llr: np.ndarray, u0: float, sigma0: float, max_step: int = 3, sigma_step: float = 1.0,
                     end: tuple[float, float] | None = None) -> np.ndarray:
    """Pipe path that STARTS at a pinned sideways position: llr (n_s, n_u) evidence per step and offset
    cell; prior at step 0 = Gaussian around u0 (cells) with sigma0 (cells); the path stays in the pipe
    state; lateral move of k cells per step costs k^2 / (2 sigma_step^2), up to max_step. Optional end =
    (u_end, sigma_end) adds a Gaussian prior at the last step (path pinned at both ends).
    Returns the path (n_s,) of offset-cell indices."""
    n_s, n_u = llr.shape; uu = np.arange(n_u)
    P = -0.5 * ((uu - u0) / sigma0) ** 2 + llr[0]
    bp = np.zeros((n_s, n_u), np.int32)
    for t in range(1, n_s):
        best = np.full(n_u, -np.inf); arg = np.zeros(n_u, np.int32)
        for k in range(-max_step, max_step + 1):
            sh = np.roll(P, k)
            if k > 0:
                sh[:k] = -np.inf
            elif k < 0:
                sh[k:] = -np.inf
            cand = sh - k * k / (2 * sigma_step ** 2)
            m = cand > best; best[m] = cand[m]; arg[m] = (uu - k)[m]
        P = best + llr[t]; bp[t] = arg
    if end is not None:
        P = P - 0.5 * ((uu - end[0]) / end[1]) ** 2
    path = np.zeros(n_s, int); path[-1] = int(np.argmax(P))
    for t in range(n_s - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def viterbi_uz(llr: np.ndarray, c_on: float = 4.0, c_off: float = 4.0, max_du: int = 2, max_dz: int = 1,
               sigma_u: float = 1.0, sigma_z: float = 0.7):
    """Best path through llr (n_s, n_u, n_z) over states (sideways offset, depth) plus 'no pipe'. A move of
    (du, dz) cells per step costs du^2 / (2 sigma_u^2) + dz^2 / (2 sigma_z^2); entering / leaving the pipe
    costs c_on / c_off. Returns segments [(t0, t1, iu (array), iz (array), score)]. Loops over steps and
    neighbour offsets only."""
    n_s, n_u, n_z = llr.shape
    P = np.full((n_u, n_z), -np.inf); N = 0.0
    bdu = np.zeros((n_s, n_u, n_z), np.int8); bdz = np.zeros((n_s, n_u, n_z), np.int8)
    fromN = np.zeros((n_s, n_u, n_z), bool); nfrom = np.full(n_s, -1, np.int64)
    offs = [(a, b) for a in range(-max_du, max_du + 1) for b in range(-max_dz, max_dz + 1)]
    for t in range(n_s):
        best = np.full((n_u, n_z), -np.inf); au = np.zeros((n_u, n_z), np.int8); az = np.zeros((n_u, n_z), np.int8)
        for a, b in offs:
            sh = np.roll(np.roll(P, a, 0), b, 1)
            if a > 0: sh[:a] = -np.inf
            if a < 0: sh[a:] = -np.inf
            if b > 0: sh[:, :b] = -np.inf
            if b < 0: sh[:, b:] = -np.inf
            cand = sh - a * a / (2 * sigma_u ** 2) - b * b / (2 * sigma_z ** 2)
            m = cand > best; best[m] = cand[m]; au[m] = a; az[m] = b
        enter = N - c_on; fn = enter > best
        newP = np.where(fn, enter, best) + llr[t]
        fromN[t] = fn; bdu[t] = au; bdz[t] = az
        j = int(np.argmax(P)); leave = P.flat[j] - c_off if np.isfinite(P.flat[j]) else -np.inf
        if leave > N:
            N = leave; nfrom[t] = j
        P = newP
    path_u = np.full(n_s, -1); path_z = np.full(n_s, -1)
    if P.max() > N:
        iu, iz = np.unravel_index(int(np.argmax(P)), P.shape); state = "pipe"
    else:
        state = "none"
    for t in range(n_s - 1, -1, -1):
        if state == "pipe":
            path_u[t], path_z[t] = iu, iz
            if fromN[t, iu, iz]:
                state = "none"
            else:
                iu, iz = iu - bdu[t, iu, iz], iz - bdz[t, iu, iz]
        else:
            if nfrom[t] >= 0:
                iu, iz = np.unravel_index(int(nfrom[t]), (n_u, n_z)); state = "pipe"
    segs = []; t = 0
    while t < n_s:
        if path_u[t] >= 0:
            t0 = t
            while t < n_s and path_u[t] >= 0:
                t += 1
            uu, zz = path_u[t0:t], path_z[t0:t]; segs.append((t0, t, uu, zz, float(llr[np.arange(t0, t), uu, zz].sum())))
        else:
            t += 1
    return segs


def viterbi_uz_multi(llr: np.ndarray, k_paths: int = 8, min_steps: int = 20, sup_u: int = 4, sup_z: int = 2, **kw):
    """Several pipes: take the best segment, suppress the evidence around it, repeat up to k_paths times."""
    L = llr.copy(); out = []
    for _ in range(k_paths):
        segs = [s for s in viterbi_uz(L, **kw) if s[1] - s[0] >= min_steps]
        if not segs:
            break
        s = max(segs, key=lambda q: q[4]); out.append(s)
        for t, iu, iz in zip(range(s[0], s[1]), s[2], s[3]):
            L[t, max(0, iu - sup_u): iu + sup_u + 1, max(0, iz - sup_z): iz + sup_z + 1] = -2.0
    return out
