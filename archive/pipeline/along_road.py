"""Tests for along-the-road (longitudinal) pipe detection (3 Nov 2026).

1. Injection-recovery: add synthetic pipes running along the track to the real
   noise-equalised traces (real clutter, layers and noise stay) and count how many the
   detector recovers, as a function of depth and strength.
2. Records: municipal storm-drain / sewer pipes that run inside the array footprint.
Units: m, ns. Z arrays are (n_ch, n_s, n_t) on the swath s grid."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from cube import to_su
from geometry import lonlat_to_merc


def ricker(t: np.ndarray, f_mhz: float) -> np.ndarray:
    """Ricker wavelet, unit peak, centre frequency f (MHz), t in ns (0 = peak)."""
    a = (np.pi * f_mhz * 1e-3 * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def peak_frequency_mhz(Z: np.ndarray, dt: float, t_win_ns: tuple[float, float]) -> float:
    """DERIVED wavelet centre frequency: peak of the mean amplitude spectrum of Z in a time window."""
    i0, i1 = int(t_win_ns[0] / dt), int(t_win_ns[1] / dt)
    sub = Z[:, ::50, i0:i1].reshape(-1, i1 - i0)
    spec = np.abs(np.fft.rfft(sub * np.hanning(i1 - i0), axis=1)).mean(0)
    f = np.fft.rfftfreq(i1 - i0, dt) * 1e3
    return float(f[1:][np.argmax(spec[1:])])


def inject_longitudinal(Z: np.ndarray, u_ch: np.ndarray, s_grid: np.ndarray, dt: float, v: float,
                        segs: list[dict], f_mhz: float, taper_m: float = 1.0) -> np.ndarray:
    """Add pipes parallel to the track. seg: dict(s0, s1, u0, z, amp). Channel c sees the pipe
    at two-way time t_c = 2 r / v, r = sqrt(z^2 + (u_c - u0)^2), amplitude amp * z / r (ASSUMED
    obliquity falloff), wavelet ricker(f_mhz). amp is in noise-equalised units (times the
    typical level at that depth). Cosine taper of taper_m at both ends. Returns a copy."""
    Z = Z.copy(); n_t = Z.shape[2]; t = np.arange(n_t) * dt; ds = s_grid[1] - s_grid[0]
    for g in segs:
        i0, i1 = np.searchsorted(s_grid, [g["s0"], g["s1"]])
        n = i1 - i0; nt_ = max(1, int(taper_m / ds))
        w = np.ones(n); ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, min(nt_, n // 2)))
        w[: len(ramp)] = ramp; w[n - len(ramp):] = ramp[::-1]
        R = g.get("radius_m", 0.0)               # pipe radius: path to the pipe surface = sqrt((z + R)^2 + d^2) - R
        if g.get("wander_m", 0.0) == 0.0:
            rr = np.sqrt((g["z"] + R) ** 2 + (u_ch - g["u0"]) ** 2); r = rr - R
            trace = g["amp"] * ((g["z"] + R) / rr)[:, None] * ricker(t[None, :] - 2 * r[:, None] / v, f_mhz)   # (n_ch, n_t)
            Z[:, i0:i1, :] += (w[None, :, None] * trace[:, None, :]).astype(Z.dtype)
        else:                                   # pipe position relative to the van wanders: u0(s) = u0 + A sin(2 pi s / P)
            uu = g["u0"] + g["wander_m"] * np.sin(2 * np.pi * s_grid[i0:i1] / g.get("wander_period_m", 15.0))   # (n,)
            rr = np.sqrt((g["z"] + R) ** 2 + (u_ch[:, None] - uu[None, :]) ** 2); r = rr - R                  # (n_ch, n)
            Z[:, i0:i1, :] += (w[None, :, None] * g["amp"] * ((g["z"] + R) / rr)[:, :, None] *
                               ricker(t[None, None, :] - 2 * r[:, :, None] / v, f_mhz)).astype(Z.dtype)
    return Z


def inject_raw(raw: np.ndarray, S: np.ndarray, U: np.ndarray, n0: np.ndarray, spread: np.ndarray, dt: float, v: float,
               segs: list[dict], f_mhz: float, taper_m: float = 1.0) -> np.ndarray:
    """Add pipes parallel to the track to RAW traces, before any processing. raw (n_scans, n_ch, n_t) float32 (modified in
    place and returned); S, U (n_scans, n_ch) footprint positions (m); n0 (n_ch,) time-zero sample of each channel;
    spread (n_ch, n_t) raw trace-to-trace spread (robust sigma) at each sample. seg: dict(s0, s1, u0, z, amp[, wander_m,
    wander_period_m, radius_m]); amp in multiples of the raw spread at the echo time; obliquity z / r (ASSUMED);
    cosine taper of taper_m at the ends; wavelet ricker(f_mhz) arriving at n0 + 2 r / v."""
    n_t = raw.shape[2]; t = np.arange(n_t) * dt
    for g in segs:
        R = g.get("radius_m", 0.0)
        for c in range(raw.shape[1]):
            j = np.nonzero((S[:, c] >= g["s0"]) & (S[:, c] <= g["s1"]))[0]
            if not len(j):
                continue
            s = S[j, c]; w = np.clip(np.minimum(s - g["s0"], g["s1"] - s) / taper_m, 0, 1); w = 0.5 - 0.5 * np.cos(np.pi * w)
            uu = g["u0"] + g.get("wander_m", 0.0) * np.sin(2 * np.pi * s / g.get("wander_period_m", 15.0))
            rr = np.sqrt((g["z"] + R) ** 2 + (U[j, c] - uu) ** 2); tc = n0[c] * dt + 2 * (rr - R) / v
            a = g["amp"] * spread[c, np.clip(np.round(tc / dt).astype(int), 0, n_t - 1)] * (g["z"] + R) / rr * w
            raw[j, c, :] += (a[:, None] * ricker(t[None, :] - tc[:, None], f_mhz)).astype(raw.dtype)
    return raw


@lru_cache(maxsize=64)
def _end_taper_table(z: float, f_mhz: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    """Kirchhoff (physical-optics) peak amplitude of a half-line at depth z (m) seen by a monostatic antenna at signed
    distance delta (m) beyond its end (delta < 0: above the pipe), relative to the full line; point diffractors every 5 mm,
    two-way spreading 1/r^2, obliquity z/r, Ricker at f_mhz, velocity v (m/ns). Isotropic antenna (ASSUMED).
    Returns (delta grid -1.5..1.5 m step 0.02, amplitude ratio); 0.5 at delta = 0 (Fresnel half-zone)."""
    dt = 0.02; t = np.arange(-8.0, 2 * 3.5 / v, dt); w = np.arange(-4.0, 4.0, 0.005); r = np.sqrt(z ** 2 + w ** 2)
    kern = np.zeros_like(t)                                    # summed response of each 5 mm element, binned on the time grid
    np.add.at(kern, np.clip(np.round((2 * r / v + 8.0) / dt).astype(int), 0, len(t) - 1), (z / r) / r ** 2 * 0.005)
    wav = ricker(np.arange(-8.0, 8.0 + dt, dt), f_mhz)
    deltas = np.round(np.arange(-1.5, 1.5001, 0.02), 3); out = np.empty(len(deltas))
    full = np.abs(np.convolve(kern, wav, "same")).max()
    for k, d in enumerate(deltas):                               # 151 small convolutions, once per (z, f)
        kk = np.zeros_like(t); m = w >= d
        np.add.at(kk, np.clip(np.round((2 * r[m] / v + 8.0) / dt).astype(int), 0, len(t) - 1), (z / r[m]) / r[m] ** 2 * 0.005)
        out[k] = np.abs(np.convolve(kk, wav, "same")).max() / full
    return deltas, out


def end_taper(delta: np.ndarray, z: float, f_mhz: float, v: float) -> np.ndarray:
    """Amplitude ratio (half-line / full line) at signed distance delta (m) beyond the end; see _end_taper_table."""
    dg, a = _end_taper_table(round(float(z), 3), round(float(f_mhz), 0), round(float(v), 4))
    return np.interp(delta, dg, a, left=a[0], right=0.0)


def inject_crossing_raw(raw: np.ndarray, S: np.ndarray, U: np.ndarray, n0: np.ndarray, spread: np.ndarray, dt: float, v: float,
                        pipes: list[dict], f_mhz: float) -> np.ndarray:
    """Add straight pipes CROSSING the drive to RAW traces (polarisation-blind: no dependence on antenna orientation).
    pipe: dict(s0, phi_deg, z, amp[, u_end, side]): plan line s = s0 + u tan(phi); horizontal distance of a footprint
    (S, U) to it d = |S - s0 - U tan(phi)| cos(phi); echo at n0 + 2 sqrt(z^2 + d^2) / v with amplitude amp * spread * z / r.
    With u_end and side (+1: pipe only where u > u_end, -1: only where u < u_end) the pipe is a half-line and footprints
    beyond its end see the end point (distance to the nearest point of the half-line); the amplitude of a half-line is
    tapered by end_taper (Kirchhoff, 0.5 at the end; fix 2026-11-04: before, beyond-end echoes kept z/r, 0.86-0.94 at 0.3 m).
    raw (n_scans, n_ch, n_t) float32, modified in place; S, U (n_scans, n_ch) m; n0 (n_ch,); spread (n_ch, n_t)."""
    n_t = raw.shape[2]; t = np.arange(n_t) * dt
    for g in pipes:
        tp = np.tan(np.radians(g["phi_deg"])); cp = np.cos(np.radians(g["phi_deg"]))
        for c in range(raw.shape[1]):
            d = np.abs(S[:, c] - g["s0"] - U[:, c] * tp) * cp
            taper = np.ones_like(d)
            if "u_end" in g:                            # along-pipe coordinate w of the foot vs the end at w_end = u_end / cos(phi)
                w = (S[:, c] - g["s0"]) * np.sin(np.radians(g["phi_deg"])) + U[:, c] * cp; w_end = g["u_end"] / cp
                delta = (w_end - w) * g["side"]          # signed distance beyond the end (m), < 0 above the pipe
                taper = end_taper(delta, g["z"], f_mhz, v)
                rz = np.sqrt(g["z"] ** 2 + d ** 2)      # z/r of the perpendicular foot, kept as for a full line
                d = np.where(delta > 0, np.hypot(d, delta), d)
                taper = taper * np.sqrt(g["z"] ** 2 + d ** 2) / rz   # cancel the extra z/r below: decay beyond the end is in taper
            j = np.nonzero(d < 3.0)[0]
            if not len(j):
                continue
            r = np.sqrt(g["z"] ** 2 + d[j] ** 2); tc = n0[c] * dt + 2 * r / v
            a = g["amp"] * spread[c, np.clip(np.round(tc / dt).astype(int), 0, n_t - 1)] * g["z"] / r * taper[j]
            raw[j, c, :] += (a[:, None] * ricker(t[None, :] - tc[:, None], f_mhz)).astype(raw.dtype)
    return raw


def match(tracks: list[dict], targets: list[dict], q_max: float, du: float, dz: float | None,
          min_overlap: float) -> np.ndarray:
    """For each target (s0, s1, u0, z) True if a track with q <= q_max overlaps it along s by
    >= min_overlap of the target length, |u_mean - u0| <= du and (if dz) |depth - z| <= dz."""
    hit = np.zeros(len(targets), bool)
    for i, g in enumerate(targets):
        for t in tracks:
            if t["q"] > q_max:
                continue
            ov = min(t["s_end"], g["s1"]) - max(t["s_start"], g["s0"])
            if ov >= min_overlap * (g["s1"] - g["s0"]) and abs(t["u_mean"] - g["u0"]) <= du and \
                    (dz is None or g.get("z") is None or abs(t["depth_m"] - g["z"]) <= dz):
                hit[i] = True; break
    return hit


def longitudinal_records(geojsons: list[Path], s_nodes, xn, yn, k: float, u_lim: tuple[float, float],
                         min_len: float = 10.0, max_angle_deg: float = 20.0, depth_m: tuple[float, float] = (0.2, 1.5),
                         step_m: float = 0.25) -> list[dict]:
    """Municipal record pipes that run inside the strip u_lim (m) for >= min_len m at <= max_angle
    to the track, with recorded depth in depth_m (0.0 and blank treated as missing and excluded).
    Returns dict(s0, s1, u0 (mean u inside), z (recorded depth, m), len_in, angle, diam, material,
    street, kind, id). Record positional accuracy is not stated by the publisher."""
    out = []
    for gj in geojsons:
        kind = "storm" if "bianche" in str(gj) else "sewer"
        for f in json.load(open(gj))["features"]:
            p = f["properties"]
            dep = [abs(float(x)) / 100 for x in (p.get("prof_iniz"), p.get("prof_fin")) if x not in ("", None) and float(x) != 0]
            if not dep or not (depth_m[0] <= min(dep) <= depth_m[1]):
                continue
            g = f["geometry"]; lines = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
            for ln in lines:
                ln = np.asarray(ln, float); X, Y = lonlat_to_merc(ln[:, 0], ln[:, 1])
                L = np.hypot(np.diff(X), np.diff(Y)) * k
                if L.sum() < min_len:
                    continue
                n = np.maximum(2, np.ceil(L / step_m).astype(int))
                xs = np.concatenate([np.linspace(X[i], X[i + 1], n[i]) for i in range(len(L))])
                ys = np.concatenate([np.linspace(Y[i], Y[i + 1], n[i]) for i in range(len(L))])
                s, u = to_su(xs, ys, s_nodes, xn, yn, k)
                ins = (u >= u_lim[0]) & (u <= u_lim[1]) & (s > s_nodes[1]) & (s < s_nodes[-2])
                if ins.sum() * step_m < min_len:
                    continue
                si, ui = s[ins], u[ins]
                ang = np.degrees(np.arctan(np.polyfit(si, ui, 1)[0])) if np.ptp(si) > 1 else 90.0
                if abs(ang) > max_angle_deg:
                    continue
                out.append(dict(s0=float(si.min()), s1=float(si.max()), u0=float(ui.mean()), z=min(dep),
                                len_in=float(ins.sum() * step_m), angle=float(ang), diam=p.get("diametro"),
                                material=p.get("materiale"), street=p.get("via"), kind=kind, id=f.get("id")))
    return out


def shifted_targets(targets: list[dict], off: float, lo: float, hi: float) -> list[dict]:
    """Shift targets along s by off, wrapped into [lo, hi) (chance level)."""
    out = []
    for g in targets:
        L = g["s1"] - g["s0"]; s0 = lo + np.mod(g["s0"] + off - lo, hi - lo)
        out.append(dict(g, s0=s0, s1=s0 + L))
    return out


def pipe_shape_test(X: np.ndarray, u_grid: np.ndarray, depth: np.ndarray, u0: float, z_hint: float,
                    half_w: float = 0.8, allow_outside_m: float = 0.0) -> dict:
    """Is an along-road feature's cross-section pipe-shaped? X (n_t, n_u) envelope across the street,
    depth (n_t,) m at the migration velocity. Top edge per u column (first depth where X exceeds
    half of the window maximum, searched z_hint - 0.35 .. z_hint + 0.45 m) within u0 +- half_w.
    A buried pipe gives a top edge z(u) = sqrt(z0^2 + (u - uc)^2) (zero-offset diffraction, same
    velocity as the depth axis); a layer gives a tilted line z = a + b u. allow_outside_m > 0 lets the
    apex lie up to that far beyond the covered columns (a half arch at the strip edge must still curve
    like a pipe to win). Returns both RMS misfits
    (m), the fitted (uc, z0), whether the apex is inside the window, and the verdict
    'pipe-like' if the hyperbola fits at least 1.5 x better than the plane with the apex inside."""
    cu = (u_grid >= u0 - half_w) & (u_grid <= u0 + half_w)
    cz = (depth >= z_hint - 0.35) & (depth <= z_hint + 0.45)
    if cz.sum() < 3 or cu.sum() < 3 or X.shape[0] != len(depth):
        return dict(verdict="insufficient coverage")
    W = X[np.ix_(cz, cu)]; uu = u_grid[cu]; zz = depth[cz]
    ok_col = np.isfinite(W).all(0)
    if ok_col.sum() < 8:
        return dict(verdict="insufficient coverage")
    W = W[:, ok_col]; uu = uu[ok_col]
    thr = 0.5 * np.nanmax(W)
    above = W >= thr
    has = above.any(0)
    edge = np.where(has, zz[np.argmax(above, 0)], np.nan)
    m = np.isfinite(edge)
    if m.sum() < 8:
        return dict(verdict="no clear reflector")
    ue, ze = uu[m], edge[m]
    best = None
    for uc in np.arange(u0 - 0.5 - allow_outside_m, u0 + 0.5 + allow_outside_m + 1e-9, 0.02):
        for z0 in np.arange(max(0.15, z_hint - 0.35), z_hint + 0.45, 0.01):
            r = np.sqrt(np.mean((np.sqrt(z0 ** 2 + (ue - uc) ** 2) - ze) ** 2))
            if best is None or r < best[0]:
                best = (r, uc, z0)
    A = np.c_[np.ones_like(ue), ue]; coef = np.linalg.lstsq(A, ze, rcond=None)[0]
    r_plane = float(np.sqrt(np.mean((A @ coef - ze) ** 2)))
    inside = (ue.min() + 0.15) < best[1] < (ue.max() - 0.15)
    near = (ue.min() - allow_outside_m) <= best[1] <= (ue.max() + allow_outside_m)
    verdict = "pipe-like" if (best[0] * 1.5 <= r_plane and (inside or (allow_outside_m > 0 and near))) else "layer-like"
    return dict(verdict=verdict, rms_hyperbola_m=float(best[0]), rms_plane_m=r_plane, uc=float(best[1]), z0=float(best[2]),
                apex_inside=bool(inside), n_cols=int(m.sum()), edge_u=ue.round(3).tolist(), edge_z=ze.round(3).tolist())



def section_single_pass(Z: np.ndarray, u_ch: np.ndarray, s_grid: np.ndarray, s_mid: float, u_grid: np.ndarray,
                        half_len_m: float = 1.0) -> np.ndarray:
    """Across-street envelope section of ONE pass at s_mid: coherent stack of Z (n_ch, n_s, n_t) over
    +-half_len_m, Hilbert envelope, interpolated across the channels onto u_grid. -> (n_t, n_u), nan outside."""
    from scipy.signal import hilbert
    from register_multipass import place_interp
    w = (s_grid > s_mid - half_len_m) & (s_grid < s_mid + half_len_m)
    E = np.abs(hilbert(Z[:, w].astype(np.float32).mean(1), axis=1))
    return place_interp(E, np.repeat(u_ch[:, None], E.shape[1], 1), np.arange(E.shape[1]) * 1.0, 0.0, 0.0, u_grid)


def track_shape(Z: np.ndarray, u_ch: np.ndarray, s_grid: np.ndarray, dt: float, v: float, track: dict,
                fracs=(0.25, 0.5, 0.75), allow_outside_m: float = 0.3) -> dict:
    """Pipe-or-layer verdict for a longitudinal track of one pass: shape test at several positions
    along it (half arches allowed). Pipe-like if at least 2 of 3 sections are pipe-like."""
    u_grid = np.arange(u_ch.min() - 0.04, u_ch.max() + 0.041, 0.04)
    depth = np.arange(Z.shape[2]) * dt * v / 2
    res = []
    for f in fracs:
        sm = track["s_start"] + f * (track["s_end"] - track["s_start"])
        X = section_single_pass(Z, u_ch, s_grid, sm, u_grid)
        res.append(pipe_shape_test(X, u_grid, depth, track["u_mean"], track["depth_m"], half_w=0.8, allow_outside_m=allow_outside_m)["verdict"])
    k = sum(r == "pipe-like" for r in res)
    return dict(pipe_like=k >= 2, n_pipe_sections=k, sections=res)


RADII_M = (0.0, 0.025, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6)


def pipe_radius_fit(X: np.ndarray, u_grid: np.ndarray, depth: np.ndarray, u0: float, z_hint: float,
                    half_w: float = 0.8, radii=RADII_M) -> dict:
    """Across-street shape scored against pipes of several radii (user tip, 3 Nov 2026; Sham & Lai 2016,
    Ristic 2009). Top edge of the echo z(x) for a pipe of radius R with its top at z_top under offset uc:
    z(x) = sqrt((z_top + R)^2 + (x - uc)^2) - R (R = 0: point-like arch; large R: flat top, like a layer).
    Velocity is fixed (depth axis), since velocity and radius are not jointly identifiable (Giannakis 2022).
    Also measures the echo's across-street width (columns above half maximum at the apex depth) and
    whether it reaches the edge of the covered width. Verdict: 'pipe' (best R < 0.4 m and clearly better
    than a plane), 'large pipe or confined layer' (best R >= 0.4 m or within 15 percent of the plane),
    'layer' (plane clearly best)."""
    cu = (u_grid >= u0 - half_w) & (u_grid <= u0 + half_w); cz = (depth >= z_hint - 0.35) & (depth <= z_hint + 0.45)
    if cz.sum() < 3 or cu.sum() < 3 or X.shape[0] != len(depth):
        return dict(verdict="insufficient coverage")
    W = X[np.ix_(cz, cu)]; uu = u_grid[cu]; zz = depth[cz]; okc = np.isfinite(W).all(0)
    if okc.sum() < 8:
        return dict(verdict="insufficient coverage")
    W = W[:, okc]; uu = uu[okc]; thr = 0.5 * np.nanmax(W); above = W >= thr; has = above.any(0)
    edge = np.where(has, zz[np.argmax(above, 0)], np.nan); m = np.isfinite(edge)
    if m.sum() < 8:
        return dict(verdict="no clear reflector")
    ue, ze = uu[m], edge[m]
    res = {}
    for R in radii:
        best = None
        for uc in np.arange(u0 - 0.6, u0 + 0.6001, 0.02):
            zt = np.arange(max(0.1, z_hint - 0.35), z_hint + 0.45, 0.01)
            mod = np.sqrt((zt[:, None] + R) ** 2 + (ue[None, :] - uc) ** 2) - R
            r = np.sqrt(np.mean((mod - ze[None, :]) ** 2, 1)); j = int(np.argmin(r))
            if best is None or r[j] < best[0]:
                best = (float(r[j]), float(uc), float(zt[j]))
        res[R] = best
    A = np.c_[np.ones_like(ue), ue]; coef = np.linalg.lstsq(A, ze, rcond=None)[0]; r_plane = float(np.sqrt(np.mean((A @ coef - ze) ** 2)))
    Rb = min(res, key=lambda R: res[R][0]); rb, ucb, ztb = res[Rb]
    jz = int(np.argmin(np.abs(zz - ztb))); band = W[max(0, jz - 2): jz + 3].max(0); wide = band >= 0.5 * band.max()
    width = float(wide.sum() * (u_grid[1] - u_grid[0])); touches = bool(wide[0] or wide[-1])
    if r_plane < rb / 1.5:
        verdict = "layer"
    elif Rb >= 0.4 or rb > r_plane / 1.15:
        verdict = "large pipe or confined layer"
    else:
        verdict = "pipe"
    return dict(verdict=verdict, best_radius_m=Rb, diameter_m=2 * Rb, rms_m=rb, rms_plane_m=r_plane, uc=ucb, z_top=ztb,
                width_m=width, width_reaches_edge=touches, rms_by_radius={str(R): round(v[0], 4) for R, v in res.items()})
