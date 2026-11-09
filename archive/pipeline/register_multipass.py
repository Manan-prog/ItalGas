"""Stage B of the multi-pass mosaic (3 Nov 2026): register every pass to the reference with
crossing pipes as tie points, check the registration, build the corrected depth-slice
mosaic and test along-road features for repeatability between overlapping passes.
Run: python register_multipass.py configs/multipass_aurelia.yaml
Physics of the tie model: a crossing is the line s = S0 + u tan(phi) in the map. A pass placed
with error (ds, du) sees it at s0' = S0 + ds - du tan(phi) (at its u = 0). So between a pass and
the reference: r = s0_pass - s0_ref = A + A1 (s - s_mid) - B tan(phi), with A, A1 the relative
along-track offset and drift and B the relative cross-track offset (m)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import uniform_filter1d

from build_v2 import load_cfg

ROOT = Path(__file__).resolve().parents[1]


def pairs(ref: list[dict], oth: list[dict], tie: dict) -> np.ndarray:
    """Candidate tie pairs (nearest in s with compatible angle and depth). Returns (n, 4):
    s_ref, r = s_oth - s_ref, tan(phi_ref), depth_ref."""
    out = []
    so = np.array([d["s0_m"] for d in oth]) if oth else np.empty(0)
    for c in ref:
        if not len(so):
            break
        j = np.nonzero(np.abs(so - c["s0_m"]) <= tie["max_ds_m"])[0]
        j = [k for k in j if abs(oth[k]["phi_deg"] - c["phi_deg"]) <= tie["max_dphi_deg"] and abs(oth[k]["depth_m"] - c["depth_m"]) <= tie["max_dz_m"]]
        for k in j:                                   # keep all compatible pairs: voting resolves ambiguity
            out.append((c["s0_m"], oth[k]["s0_m"] - c["s0_m"], np.tan(np.radians(c["phi_deg"])), c["depth_m"]))
    return np.array(out).reshape(-1, 4)


def vote(P: np.ndarray, tol: float = 0.2) -> tuple[float, float, int, float]:
    """Hough-style vote over (A, B): count pairs with |r - (A - B tan phi)| <= tol. Returns
    (A, B, peak count, median count over the grid) - the contrast shows if the peak is real."""
    Ag = np.arange(-3, 3.001, 0.05); Bg = np.arange(-2, 2.001, 0.05)
    pred = Ag[:, None, None] - Bg[None, :, None] * P[None, None, :, 2]
    cnt = (np.abs(P[None, None, :, 1] - pred) <= tol).sum(2)
    i, j = np.unravel_index(np.argmax(cnt), cnt.shape)
    return float(Ag[i]), float(Bg[j]), int(cnt[i, j]), float(np.median(cnt))


def fit(P: np.ndarray, A0: float, B0: float, s_mid: float, tol: float = 0.4, n_iter: int = 5) -> dict:
    """Least squares on inliers of r = A + A1 (s - s_mid) - B tan(phi), starting from the vote.
    Returns A, A1 (m per km), B, inlier mask, residual RMS and leave-one-out RMS (m)."""
    X = np.c_[np.ones(len(P)), (P[:, 0] - s_mid) / 1000.0, -P[:, 2]]
    beta = np.array([A0, 0.0, B0])
    for _ in range(n_iter):
        inl = np.abs(P[:, 1] - X @ beta) <= tol
        if inl.sum() < 5:
            break
        beta = np.linalg.lstsq(X[inl], P[inl, 1], rcond=None)[0]
    res = P[inl, 1] - X[inl] @ beta
    loo = []
    Xi, yi = X[inl], P[inl, 1]
    for k in range(len(yi)):
        m = np.ones(len(yi), bool); m[k] = False
        b = np.linalg.lstsq(Xi[m], yi[m], rcond=None)[0]; loo.append(yi[k] - Xi[k] @ b)
    before = P[inl, 1] - np.median(P[inl, 1])
    return dict(A=float(beta[0]), A1_per_km=float(beta[1]), B=float(beta[2]), n_inliers=int(inl.sum()),
                rms_m=float(np.sqrt(np.mean(res ** 2))), loo_rms_m=float(np.sqrt(np.mean(np.square(loo)))),
                rms_constant_only_m=float(np.sqrt(np.mean(before ** 2))), inliers=inl.tolist())


def slice_map(G0: np.ndarray, dtg: float, v: float, z0: float, z1: float) -> np.ndarray:
    """Max of the migrated robust-z envelope over the depth slice z0..z1 (m). (n_ch, n_s, n_td) -> (n_ch, n_s)."""
    i0, i1 = int(2 * z0 / v / dtg), int(np.ceil(2 * z1 / v / dtg))
    return G0[:, :, i0:i1].astype(np.float32).max(2)


def place(vals: np.ndarray, Ug: np.ndarray, s_grid: np.ndarray, A: float, A1: float, B: float, s_mid: float,
          u_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Put one pass's per-channel values (n_ch, n_s) onto the common (s, u) grid after correction:
    s_true = s - (A + A1 (s - s_mid)/1000), u_true = u - B. Returns (sum, count) (n_s, n_u)."""
    shift = A + A1 * (s_grid - s_mid) / 1000.0
    ds = s_grid[1] - s_grid[0]
    acc = np.zeros((len(s_grid), len(u_grid)), np.float32); cnt = np.zeros_like(acc)
    for c in range(vals.shape[0]):
        st = s_grid - shift; uu = Ug[c] - B
        ok = np.isfinite(uu)
        iu = np.round((uu[ok] - u_grid[0]) / (u_grid[1] - u_grid[0])).astype(int)
        is_ = np.round((st[ok] - s_grid[0]) / ds).astype(int)
        g = (iu >= 0) & (iu < len(u_grid)) & (is_ >= 0) & (is_ < len(s_grid))
        np.add.at(acc, (is_[g], iu[g]), vals[c, ok][g]); np.add.at(cnt, (is_[g], iu[g]), 1)
    return acc, cnt


def place_interp(vals: np.ndarray, Ug: np.ndarray, s_grid: np.ndarray, A: float, B: float, u_grid: np.ndarray) -> np.ndarray:
    """One pass's per-channel values (n_ch, n_s) onto the common (s, u) grid with linear
    interpolation ACROSS the pass's channels (no holes between channels; nan outside the pass)
    after correction s_true = s - A, u_true = u - B. Vectorised over s and u. -> (n_s, n_u)."""
    ds = s_grid[1] - s_grid[0]; k = int(round(A / ds))
    V = np.roll(vals, -k, axis=1); U = np.roll(Ug, -k, axis=1) - B          # value at true s
    if k > 0:
        V[:, -k:] = np.nan; U[:, -k:] = np.nan
    elif k < 0:
        V[:, :-k] = np.nan; U[:, :-k] = np.nan
    o = np.argsort(np.nanmedian(U, 1)); U = U[o]; V = V[o]                  # channels ordered by u
    lo = (U[:, :, None] <= u_grid[None, None, :]).sum(0) - 1                 # (n_s, n_u) index of channel below
    n = U.shape[0]; ok = (lo >= 0) & (lo < n - 1)
    lo_c = np.clip(lo, 0, n - 2); si = np.arange(U.shape[1])[:, None]
    u0 = U[lo_c, si]; u1 = U[lo_c + 1, si]; v0 = V[lo_c, si]; v1 = V[lo_c + 1, si]
    w = np.clip((u_grid[None, :] - u0) / np.maximum(u1 - u0, 1e-6), 0, 1)
    out = (1 - w) * v0 + w * v1
    out[~ok | ~np.isfinite(u0) | ~np.isfinite(u1) | (u1 - u0 > 0.3)] = np.nan
    return out.astype(np.float32)


def line_evidence(M: np.ndarray, cnt: np.ndarray, ds: float, du: float, along_m: float = 5.0, flank_m=(0.24, 0.48)) -> np.ndarray:
    """Narrow along-road line contrast on a plan map M (n_s, n_u): mean over along_m along s minus
    the brighter flank mean at u +- flank_m. Cells without coverage (cnt = 0) are nan."""
    Mm = np.where(cnt > 0, M / np.maximum(cnt, 1), np.nan)
    w = max(1, int(along_m / ds))
    valid = np.isfinite(Mm).astype(np.float32)
    S = uniform_filter1d(np.nan_to_num(Mm), w, axis=0, mode="nearest"); V = uniform_filter1d(valid, w, axis=0, mode="nearest")
    mean_s = np.where(V > 0.5, S / np.maximum(V, 1e-6), np.nan)
    k0, k1 = int(round(flank_m[0] / du)), int(round(flank_m[1] / du))
    fl = []
    for sgn in (-1, 1):
        acc = np.zeros_like(mean_s); n = np.zeros_like(mean_s)
        for k in range(k0, k1 + 1):
            sh = np.roll(mean_s, -sgn * k, axis=1)
            if sgn < 0:
                sh[:, :k] = np.nan
            else:
                sh[:, -k:] = np.nan
            ok = np.isfinite(sh); acc[ok] += sh[ok]; n[ok] += 1
        fl.append(np.where(n > 0, acc / np.maximum(n, 1), np.nan))
    return mean_s - np.fmax(fl[0], fl[1])


def vote_A(P: np.ndarray, B: float, tol: float = 0.2) -> tuple[float, int, int]:
    """1-D vote for the along-track offset with the cross-track offset fixed. Returns
    (A, peak count, best count more than 0.6 m away from the peak)."""
    Ag = np.arange(-4, 4.001, 0.05)
    cnt = np.array([(np.abs(P[:, 1] - (A - B * P[:, 2])) <= tol).sum() for A in Ag]) if len(P) else np.zeros(len(Ag), int)
    i = int(np.argmax(cnt)); far = cnt[np.abs(Ag - Ag[i]) > 0.6]
    return float(Ag[i]), int(cnt[i]), int(far.max()) if len(far) else 0


def overlap_shift(Ma: np.ndarray, Mb: np.ndarray, ds: float, du: float, min_cells: int = 20000) -> dict | None:
    """Shift of map b (rolled along s, across u) that best correlates with map a over cells covered
    by both, searched over +-3 m along (0.2 m) and +-2 m across (one cell). Returns dict(dA = A_b - A_a,
    dB = B_b - B_a, r_best, r_zero, n_cells) or None if they never overlap enough."""
    best = None; r0 = None
    dec = int(round(0.2 / ds)); Ma = Ma[::dec]; Mb = Mb[::dec]; min_cells = min_cells // dec; ds = ds * dec   # 0.2 m along-s steps
    for ks in range(-int(3 / ds), int(3 / ds) + 1):
        bs = np.roll(Mb, ks, axis=0)
        for ku in range(-int(2 / du), int(2 / du) + 1):
            bb = np.roll(bs, ku, axis=1)
            if ku > 0:
                bb[:, :ku] = np.nan
            elif ku < 0:
                bb[:, ku:] = np.nan
            m = np.isfinite(Ma) & np.isfinite(bb)
            if m.sum() < min_cells:
                continue
            r = float(np.corrcoef(Ma[m], bb[m])[0, 1])
            if ks == 0 and ku == 0:
                r0 = r
            if best is None or r > best[0]:
                best = (r, ks, ku, int(m.sum()))
    if best is None:
        return None
    return dict(dA=-best[1] * ds, dB=-best[2] * du, r_best=best[0], r_zero=r0, n_cells=best[3])


def solve(names: list[str], ref: str, meas: list[tuple], dirs: dict | None = None) -> tuple[dict, np.ndarray]:
    """Weighted least squares for per-pass offsets X_i (X_ref = 0) from pairwise measurements
    (i, j, value = X_j - X_i, sigma). If dirs is given (pass -> +1/-1 travel direction along s), one
    extra unknown eps is estimated with X_i = A_i + eps * dir_i, i.e. a position bias along the
    direction of travel (time latency or forward lever-arm error), which only opposite-direction
    pairs can see. Returns ({name: X}, residuals)."""
    idx = {n: k for k, n in enumerate([n for n in names if n != ref])}
    n_u = len(idx) + (1 if dirs else 0)
    rows, y, w = [], [], []
    for i, j, val, sig in meas:
        r = np.zeros(n_u)
        if j != ref:
            r[idx[j]] += 1
        if i != ref:
            r[idx[i]] -= 1
        if dirs:
            r[-1] = dirs[j] - dirs[i]
        rows.append(r); y.append(val); w.append(1 / sig)
    if not rows:
        return {n: 0.0 for n in names}, np.empty(0)
    Aw = np.array(rows) * np.array(w)[:, None]; yw = np.array(y) * np.array(w)
    x = np.linalg.lstsq(Aw, yw, rcond=None)[0]
    out = {ref: 0.0, **{n: float(x[k]) for n, k in idx.items()}}
    if dirs:
        out["eps_along_travel_m"] = float(x[-1])
    return out, (np.array(rows) @ x - np.array(y))


def main(cfg_path: str) -> None:
    mc = yaml.safe_load(open(cfg_path)); det = load_cfg(str(ROOT / "pipeline" / mc["detector_config"]))
    cache = Path(det["base"]["cache_dir"]); v = det["migration"]["velocity_m_ns"]; tie = mc["tie"]
    rng = np.random.default_rng(det["seed"])
    w0, w1 = mc["window_s_m"]; s_mid = 0.5 * (w0 + w1)
    from run_multipass import pass_spec
    tag = mc.get("tag", "")
    labels = [pass_spec(mc, p)[2] for p in mc["passes"]]
    R = {n: json.load(open(cache / f"mp_{tag}{n}.json")) for n in labels if (cache / f"mp_{tag}{n}.json").exists()}
    names = [n for n in R if R[n]["arrays"].get("2")]; ref = pass_spec(mc, mc["reference"])[2]
    dirs = {n: R[n]["arrays"]["2"]["direction"] for n in names}
    mo = mc["mosaic"]; u_grid = np.arange(mo["u_range_m"][0], mo["u_range_m"][1] + 1e-9, mo["du_m"])
    # raw (unregistered) 200 MHz depth-slice maps of every pass, for overlap correlation and mosaics
    raw = {}; s_grid = None; G0 = {}
    for n in names:
        d = np.load(cache / f"mp_{tag}{n}_2.npz"); s_grid = d["s_grid"]; G0[n] = d
        for zi, (z0, z1) in enumerate(mo["depth_slices_m"]):
            acc, cnt = place(slice_map(d["G"][0], float(d["dtg"]), v, z0, z1), d["Ug"], s_grid, 0, 0, 0, s_mid, u_grid)
            raw[(n, zi)] = (acc, cnt)
    ds = float(s_grid[1] - s_grid[0]); du = mo["du_m"]
    mean = lambda n, zi: np.where(raw[(n, zi)][1] > 0, raw[(n, zi)][0] / np.maximum(raw[(n, zi)][1], 1), np.nan)
    sig = lambda L: [c for c in L if c["q"] <= tie["q_max"]]
    pair_log = []; measA = []; measB = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            # overlap correlation (median over the middle depth slices)
            near = abs(R[a]["arrays"]["2"]["u_median"] - R[b]["arrays"]["2"]["u_median"]) < 2.6
            ov = [overlap_shift(mean(a, zi), mean(b, zi), ds, du) for zi in (1, 2, 3)] if near else []
            ov = [o for o in ov if o]
            o = None
            if ov:
                o = dict(dA=float(np.median([x["dA"] for x in ov])), dB=float(np.median([x["dB"] for x in ov])),
                         r_best=float(np.median([x["r_best"] for x in ov])), r_zero=ov[0]["r_zero"], n_cells=ov[0]["n_cells"],
                         spread_A=float(np.ptp([x["dA"] for x in ov])), spread_B=float(np.ptp([x["dB"] for x in ov])))
                if o["spread_B"] <= 0.25:
                    measB.append((a, b, o["dB"], 0.1))
                if o["spread_A"] <= 0.5:
                    measA.append((a, b, o["dA"], 0.25))
            # crossing ties (both arrays), A voted with B fixed (overlap estimate if available, else 0)
            P = [pairs(sig(R[a]["arrays"][x]["crossings"]), sig(R[b]["arrays"][x]["crossings"]), tie)
                 for x in ("1", "2") if R[a]["arrays"].get(x) and R[b]["arrays"].get(x)]
            P = np.vstack(P) if P else np.empty((0, 4))
            Bfix = o["dB"] if (o and o["spread_B"] <= 0.25) else 0.0
            A_t, peak, far = vote_A(P, Bfix)
            ok = peak >= 6 and peak >= 1.5 * max(far, 1)
            if ok:
                measA.append((a, b, A_t, 0.15))
            pair_log.append(dict(a=a, b=b, dir_a=dirs[a], dir_b=dirs[b], n_pairs=int(len(P)), A_tie=A_t, tie_peak=peak, tie_far=far,
                                 tie_used=bool(ok), overlap=o))
            print(f"{a}-{b}: dirs {dirs[a]:+d}/{dirs[b]:+d}, ties A {A_t:+.2f} (peak {peak}, far {far}, used {ok}); overlap {o and {k: round(x_, 3) for k, x_ in o.items()}}", flush=True)
    A_sol, resA = solve(names, ref, measA, dirs)
    B_sol, resB = solve(names, ref, measB)
    A_nodir, resA0 = solve(names, ref, measA)
    connB = {n for m in measB for n in m[:2]} | {ref}
    reg = {}
    for n in names:
        reg[n] = dict(A=A_sol[n] + A_sol["eps_along_travel_m"] * dirs[n], A_rel=A_sol[n], B=B_sol[n] if n in connB else 0.0,
                      B_known=n in connB, direction=dirs[n], u_median=R[n]["arrays"]["2"]["u_median"], A1_per_km=0.0)
    out = dict(pairs=pair_log, solution=dict(A=A_sol, B=B_sol, eps_along_travel_m=A_sol["eps_along_travel_m"],
                                             resid_A_rms=float(np.sqrt(np.mean(resA ** 2))) if len(resA) else None,
                                             resid_A_rms_without_eps=float(np.sqrt(np.mean(resA0 ** 2))) if len(resA0) else None,
                                             resid_B_rms=float(np.sqrt(np.mean(resB ** 2))) if len(resB) else None,
                                             n_measA=len(measA), n_measB=len(measB)), registration=reg)
    print("SOLUTION", json.dumps(out["solution"], indent=None))
    # ---- registered mosaic per pass and slice
    per = {}
    for n in names:
        d = G0[n]; g = reg[n]
        for zi, (z0, z1) in enumerate(mo["depth_slices_m"]):
            vals = slice_map(d["G"][0], float(d["dtg"]), v, z0, z1)
            per[(n, zi, "reg")] = place(vals, d["Ug"], s_grid, g["A"], 0.0, g["B"], s_mid, u_grid)
            per[(n, zi, "raw")] = raw[(n, zi)]
    np.savez_compressed(cache / f"mp_{tag}mosaic.npz", s_grid=s_grid, u_grid=u_grid,
                        **{f"{n}|{zi}|{t}|{k}": (v_[0] if k == "sum" else v_[1]).astype(np.float16)
                           for (n, zi, t), v_ in per.items() for k in ("sum", "cnt")})
    # ---- repeatability of along-road line evidence between overlapping passes (lateral-shift null)
    out["repeatability"] = {}
    for zi in range(len(mo["depth_slices_m"])):
        L = {n: line_evidence(*per[(n, zi, "reg")], ds, du) for n in names}
        rows = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                both = np.isfinite(L[a]) & np.isfinite(L[b])
                if both.sum() < 2000:
                    continue
                r_obs = float(np.corrcoef(L[a][both], L[b][both])[0, 1]); nulls = []
                for _ in range(200):
                    k = int(rng.integers(4, 13)) * int(rng.choice([-1, 1]))
                    Lb = np.roll(L[b], k, axis=1)
                    if k > 0:
                        Lb[:, :k] = np.nan
                    else:
                        Lb[:, k:] = np.nan
                    bb = np.isfinite(L[a]) & np.isfinite(Lb)
                    if bb.sum() > 1000:
                        nulls.append(np.corrcoef(L[a][bb], Lb[bb])[0, 1])
                nulls = np.array(nulls)
                rows.append(dict(a=a, b=b, n_cells=int(both.sum()), r=r_obs, null_mean=float(nulls.mean()) if len(nulls) else None,
                                 null_q95=float(np.quantile(nulls, 0.95)) if len(nulls) else None, p=float((nulls >= r_obs).mean()) if len(nulls) else None))
        out["repeatability"][str(mo["depth_slices_m"][zi])] = rows
        print("slice", mo["depth_slices_m"][zi], [(x["a"][-3:], x["b"][-3:], round(x["r"], 3), x["null_q95"] and round(x["null_q95"], 3), x["p"]) for x in rows], flush=True)
    json.dump(out, open(cache / f"mp_{tag}registration.json", "w"))
    print("saved", cache / f"mp_{tag}registration.json")


if __name__ == "__main__":
    main(sys.argv[1])


def cross_section(D: dict, reg: dict, s_grid: np.ndarray, u_grid: np.ndarray, s_true: float, v: float,
                  half_len_m: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Across-street envelope section at true position s_true from every registered pass:
    coherent 2*half_len_m stack of each pass's noise-equalised traces, Hilbert envelope, placed at the
    channels' registered u and averaged over passes. Returns (X (n_t, n_u), depth (n_t,) m)."""
    from scipy.signal import hilbert
    cols = []; dt = None
    for n, d in D.items():
        g = reg[n]; dt = float(d["dt"])
        ws = (s_grid - g["A"] > s_true - half_len_m) & (s_grid - g["A"] < s_true + half_len_m)
        if not ws.any():
            continue
        Uc = np.nanmedian(d["Ug"][:, ws], 1)
        if not np.isfinite(Uc).all():
            continue
        E = np.abs(hilbert(d["Z"][:, ws].astype(np.float32).mean(1), axis=1))
        cols.append(place_interp(E, np.repeat(Uc[:, None], E.shape[1], 1), np.arange(E.shape[1]) * 1.0, 0.0, g["B"], u_grid))
    if not cols:
        return np.full((1, len(u_grid)), np.nan), np.zeros(1)
    with np.errstate(all="ignore"):
        X = np.nanmean(np.stack(cols), 0)
    return X, np.arange(X.shape[0]) * dt * v / 2
