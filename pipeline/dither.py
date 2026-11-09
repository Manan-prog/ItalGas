"""Sideways-motion ('dither') test: does an along-street track move with the ground or with the van?
Rules in DECISIONS.md 2026-11-04 (E42). Units: metres. (s, u) follow cube.to_su: s along the reference
polyline, u positive to the LEFT of travel.

A pipe fixed in the ground appears at instrument position u_inst(s) = c - y_van(s) (slope b = -1 on the van's
sideways position y_van); an instrument artefact stays at a constant u_inst (b = 0)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

from cube import to_su, track_frame
from run_detection import su_to_map


def centre_ground(g: dict, B: float, arr: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-scan ground position of the array centre = mean channel footprint (S, U) of array arr in the drive frame,
    moved to the map with the along-travel bias B (m, as the tracks are: su_to_map(s + B, u)).
    g: swath_geometry output. Returns (S (n,) frame s m, X (n,), Y (n,) EPSG:3857), sorted by S, finite only."""
    S = np.nanmean(g[f"s{arr}"], 1).astype(float); U = np.nanmean(g[f"u{arr}"], 1).astype(float)
    ok = np.isfinite(S) & np.isfinite(U); o = np.argsort(S[ok]); S, U = S[ok][o], U[ok][o]
    X, Y = su_to_map(S + B, U, g["s_nodes"], g["xn"], g["yn"], float(g["k"]))
    return S, X, Y, U


def reference_line(S: np.ndarray, X: np.ndarray, Y: np.ndarray, k: float, L_ref: float, step: float = 1.0,
                   method: str = "sg2") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference line R of the ground path (X, Y) resampled every step m of S, smoothed over L_ref m.
    method 'sg2' (adopted, amendment 2026-11-04): local quadratic (Savitzky-Golay order 2), which follows a
    constant-radius bend exactly; 'mean': running mean, which cuts a bend of radius r by L_ref^2 / (24 r).
    Returns (s_nodes m, x_nodes, y_nodes) of R (EPSG:3857); k = cos(lat) Mercator scale."""
    u_s, i = np.unique(S, return_index=True)
    grid = np.arange(u_s[0], u_s[-1], step)
    Xi = np.interp(grid, u_s, X[i]); Yi = np.interp(grid, u_s, Y[i])
    n = max(1, int(round(L_ref / step)))
    if method == "sg2":
        n = n + 1 - n % 2                                         # odd window
        n = min(n, len(grid) - 1 + len(grid) % 2)
        Xr = savgol_filter(Xi, n, 2, mode="interp"); Yr = savgol_filter(Yi, n, 2, mode="interp")
    else:
        Xr = uniform_filter1d(Xi, n, mode="nearest"); Yr = uniform_filter1d(Yi, n, mode="nearest")
    sr, _, _ = track_frame(Xr, Yr, k)
    return sr, Xr, Yr


def sideways(g: dict, B: float, L_ref: float, arr: int = 2, method: str = "sg2") -> dict:
    """Van sideways position y_van (m, left-positive) of the array centre from R (reference_line over L_ref m).
    Returns dict(S (n,) frame s sorted, y (n,), R=(s_nodes, x, y), k, edge (s_min, s_max) of the usable range,
    i.e. excluding L_ref / 2 at both drive ends where the smoother is one-sided)."""
    k = float(g["k"]); S, X, Y, U = centre_ground(g, B, arr)
    R = reference_line(S, X, Y, k, L_ref, method=method)
    sR, y = to_su(X, Y, R[0], R[1], R[2], k)
    return dict(S=S, y=np.asarray(y, float), uc=U, sR=np.asarray(sR, float), R=R, k=k, edge=(S[0] + L_ref / 2, S[-1] - L_ref / 2))


def uc_at(sv: dict, s: np.ndarray) -> np.ndarray:
    """Frame u (m) of the array centre at frame positions s: the frame's reference polyline is the GPS antenna path,
    so a track's frame u must be taken relative to the array centre before adding y_van (correction 2026-11-04)."""
    return np.interp(s, sv["S"], sv["uc"])


def shift_at(sv: dict, s: np.ndarray) -> np.ndarray:
    """Instrument -> ground shift (m): u_ground = u_frame + shift, shift = y_van - uc; nan outside the usable range."""
    return y_at(sv, s) - uc_at(sv, s)


def ground_to_map(sv: dict, s: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frame position s (m, along the drive) and ground offset U (m, left of R) -> map (EPSG:3857) via R.
    The along-travel bias is already inside sv (centre_ground adds it)."""
    sR = np.interp(s, sv["S"], sv["sR"]); return su_to_map(sR, U, sv["R"][0], sv["R"][1], sv["R"][2], sv["k"])


def y_at(sv: dict, s: np.ndarray) -> np.ndarray:
    """y_van (m) at frame positions s (m); nan outside the usable range."""
    y = np.interp(s, sv["S"], sv["y"])
    return np.where((s >= sv["edge"][0]) & (s <= sv["edge"][1]), y, np.nan)


def to_ground(sv: dict, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map points (EPSG:3857) -> (s_R, u_R) on the reference line R (m)."""
    sR, xR, yR = sv["R"]
    return to_su(np.asarray(x, float), np.asarray(y, float), sR, xR, yR, sv["k"])


def slope(ps: np.ndarray, pu: np.ndarray, yv: np.ndarray) -> dict:
    """Fit pu = a0 + a1 (ps - mean ps) + b yv (OLS, Frisch-Waugh). ps, pu, yv (n,) m.
    Returns dict(b, sd_d = sd of yv after removing a0 + a1 s (m), sd_e = residual sd (m), n,
    rss_ground = RSS with b fixed at -1, rss_van = RSS with b fixed at 0)."""
    ps, pu, yv = (np.asarray(a, float) for a in (ps, pu, yv))
    A = np.c_[np.ones_like(ps), ps - ps.mean()]
    P = A @ np.linalg.pinv(A)                                     # projection on the nuisance (offset + trend)
    ry = yv - P @ yv; ru = pu - P @ pu
    sdd = float(np.sqrt(np.mean(ry ** 2))); den = float(ry @ ry)
    b = float(ry @ ru / den) if den > 1e-12 else np.nan
    e = ru - (b if np.isfinite(b) else 0.0) * ry
    return dict(b=b, sd_d=sdd, sd_e=float(np.sqrt(np.mean(e ** 2))), n=int(len(ps)),
                rss_ground=float(np.sum((ru + ry) ** 2)), rss_van=float(np.sum(ru ** 2)))


def track_test(ps: np.ndarray, pu: np.ndarray, sv: dict, fixed_at, min_points: int, min_sd: float) -> dict:
    """Slope test of one track. ps, pu (n,) frame s and instrument u of its points (m); fixed_at(s) -> bool array
    (RTK fixed); only fixed points inside the usable range are used. Adds 'testable' and the points used."""
    yv = y_at(sv, ps); m = np.isfinite(yv) & fixed_at(ps); pu = pu - uc_at(sv, ps)     # relative to the array centre
    r = dict(n_fixed=int(m.sum()), testable=False)
    if m.sum() < min_points:
        return r
    r.update(slope(ps[m], pu[m], yv[m])); r["testable"] = bool(r["sd_d"] >= min_sd)
    return r


def shift_null_fit(ps: np.ndarray, pu: np.ndarray, sv: dict, fixed_at, min_points: int, min_sd: float,
                   shift_m: tuple[float, float], rng: np.random.Generator, max_tries: int = 400) -> dict | None:
    """One null draw for a track: y_van read at ps + D (|D| uniform in shift_m, random sign), accepted only if all
    shifted points are RTK fixed, inside the usable range, and sd_d >= min_sd (min_sd = 0: any shift, pooled rule).
    Returns the slope() dict or None."""
    if len(ps) < min_points:
        return None
    for _ in range(max_tries):
        D = rng.uniform(*shift_m) * rng.choice([-1.0, 1.0]); s2 = ps + D
        yv = y_at(sv, s2)
        if not np.all(np.isfinite(yv)) or not np.all(fixed_at(s2)):
            continue
        r = slope(ps, pu - uc_at(sv, ps), yv)
        if r["sd_d"] >= min_sd:
            return r
    return None


def shift_null_b(*a, **k) -> float:
    """b of shift_null_fit, nan if no acceptable shift."""
    r = shift_null_fit(*a, **k); return np.nan if r is None else r["b"]


def pooled_b(fits: list[dict]) -> float:
    """Pooled slope over tracks: sum(ry . ru) / sum(ry . ry), each track with its own offset and trend removed.
    From slope() outputs: ry . ry = n sd_d^2, ry . ru = b n sd_d^2. nan if nothing to pool."""
    w = np.array([f["n"] * f["sd_d"] ** 2 for f in fits]); b = np.array([f["b"] for f in fits])
    ok = np.isfinite(b) & (w > 0)
    return float(np.sum(w[ok] * b[ok]) / np.sum(w[ok])) if ok.any() and np.sum(w[ok]) > 0 else np.nan
