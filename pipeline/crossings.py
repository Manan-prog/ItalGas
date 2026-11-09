"""Crossing-pipe detection by cross-column stacking (3 Nov 2026).

Physics: a pipe that crosses under the array is a straight line in plan view. Every
channel passes over it, so every channel's migrated B-scan has a focused point at
the same depth, at along-track position s_c = s0 + u_c * tan(phi), where u_c is the
channel's lateral position (m, + left) and phi the pipe's deviation from
perpendicular to the track. Stacking channels along that line keeps the pipe and
averages out what is not shared. A trimmed mean (lowest 75 percent of channels)
stops one or two hot channels from making a 'pipe'.

Null: shift every channel independently along s by a random distance. This keeps
every channel's own artefacts, depth structure and noise, and breaks only the
cross-channel alignment, which is the physical signature of a crossing object.

Arrays: G is (n_channels, n_s, n_t), robust z units, on a common s grid (step ds m).
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import maximum_filter, uniform_filter1d


def to_s_grid(e: np.ndarray, s: np.ndarray, s_grid: np.ndarray) -> np.ndarray:
    """Interpolate one channel's traces (n_scans, n_t) at along-track positions s (m)
    onto s_grid (m). Repeated s (vehicle stopped) are collapsed by sorting + jitter."""
    o = np.argsort(s, kind="stable")
    ss = s[o] + np.arange(len(s)) * 1e-7
    f = interp1d(ss, e[o], axis=0, bounds_error=False, fill_value=0.0, assume_sorted=True)
    return f(s_grid).astype(np.float32)


def phi_velocity_index(phis_deg: np.ndarray, v: float, v_list: list[float]) -> np.ndarray:
    """For a pipe at deviation phi from perpendicular, the along-track hyperbola has
    apparent velocity v / cos(phi). Return the index of the closest migration velocity."""
    v_app = v / np.cos(np.radians(phis_deg))
    return np.argmin(np.abs(np.asarray(v_list)[None, :] - v_app[:, None]), axis=1)


def crossing_score(G_list: list[np.ndarray], u_ch: np.ndarray, ds: float, phis_deg: np.ndarray,
                   v_idx: np.ndarray, keep_frac: float = 0.75, tol: tuple[int, int] = (3, 3),
                   shifts: np.ndarray | None = None, flank_m: tuple[float, float] | None = (0.6, 1.5)
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Score(s0, t) = max over phi of the trimmed mean over channels of G_c(s0 + u_c tan phi, t).
    G_list[k]: (n_ch, n_s, n_t) migrated at velocity k. tol: (s, t) local-max tolerance in
    grid cells applied per channel (absorbs 0.1 m track jitter and 1 sample of t0 jitter).
    shifts: optional (n_ch,) integer circular shifts along s (surrogate null).
    flank_m: a crossing pipe is a THIN line in plan view, so the stack at s0 is compared
    with the brighter of its two flanks (mean over flank_m[0]..flank_m[1] m either side);
    layers that vary along s are bright on at least one flank and score about 0.
    Returns (score (n_s, n_t), phi index (n_s, n_t))."""
    n_ch = G_list[0].shape[0]
    n_keep = max(1, int(round(keep_frac * n_ch)))
    Gm = [maximum_filter(G, size=(1, tol[0], tol[1])) for G in G_list]
    best = None; arg = None
    for j, phi in enumerate(phis_deg):
        G = Gm[v_idx[j]]
        k = np.round(u_ch * np.tan(np.radians(phi)) / ds).astype(int)
        if shifts is not None:
            k = k + shifts
        st = np.stack([np.roll(G[c], -k[c], axis=0) for c in range(n_ch)])   # (n_ch, n_s, n_t)
        st.sort(axis=0)
        sc = st[:n_keep].mean(0)
        if flank_m is not None:
            sc = sc - flank_level(sc, ds, flank_m)
        if best is None:
            best = sc; arg = np.zeros(sc.shape, np.int16)
        else:
            m = sc > best
            best[m] = sc[m]; arg[m] = j
    return best, arg


def flank_level(sc: np.ndarray, ds: float, flank_m: tuple[float, float]) -> np.ndarray:
    """Brighter of the two flank means of sc (n_s, n_t) along s, flanks from flank_m[0]
    to flank_m[1] m either side of each s."""
    n_w = max(1, int(round((flank_m[1] - flank_m[0]) / ds)))
    off = int(round((flank_m[0] + flank_m[1]) / 2 / ds))
    m = uniform_filter1d(sc, n_w, axis=0, mode="nearest")
    return np.maximum(np.roll(m, off, axis=0), np.roll(m, -off, axis=0))


def nms(i_s: np.ndarray, i_t: np.ndarray, score: np.ndarray, sep_s: int, sep_t: int) -> np.ndarray:
    """Greedy non-maximum suppression over candidate peaks (loop over candidates, not
    traces): keep the strongest, drop any other within sep_s cells in s AND sep_t in t.
    Returns indices of kept candidates."""
    order = np.argsort(-score); keep = []
    taken_s = np.empty(0, int); taken_t = np.empty(0, int)
    for k in order:
        if np.any((np.abs(taken_s - i_s[k]) <= sep_s) & (np.abs(taken_t - i_t[k]) <= sep_t)):
            continue
        keep.append(k); taken_s = np.r_[taken_s, i_s[k]]; taken_t = np.r_[taken_t, i_t[k]]
    return np.array(keep, int)


def local_maxima(score: np.ndarray, ds: float, dt_ns: float, sep_m: float = 1.0, sep_ns: float = 6.0,
                 t_range: tuple[int, int] | None = None, thr: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Candidate peaks: local maxima of score within +-sep_m along s and +-sep_ns in time,
    above thr, optionally restricted to time indices t_range. Returns (i_s, i_t)."""
    size = (2 * int(sep_m / ds) + 1, 2 * int(sep_ns / dt_ns) + 1)
    mx = maximum_filter(score, size=size, mode="nearest")
    ok = (score == mx) & (score > thr)
    if t_range is not None:
        ok[:, : t_range[0]] = False; ok[:, t_range[1]:] = False
    i_s, i_t = np.nonzero(ok)
    k = nms(i_s, i_t, score[i_s, i_t], size[0] // 2, size[1] // 2)
    return i_s[k], i_t[k]


def empirical_fdr(data: np.ndarray, null: np.ndarray, n_null_reps: int) -> tuple[np.ndarray, np.ndarray]:
    """Target-decoy style estimate. data: candidate scores from the real data; null:
    candidate scores pooled over n_null_reps surrogates of the same length.
    q(x) = min_{y <= x} [#null >= y / n_reps] / [#data >= y]; P_real(x) = 1 - local fdr,
    local fdr from the ratio of null to data densities in score bins, made monotone.
    Returns (q, p_real) aligned with data."""
    srt = np.sort(null)
    n_null_ge = (len(srt) - np.searchsorted(srt, data, side="left")) / n_null_reps
    dsrt = np.sort(data)
    n_data_ge = len(dsrt) - np.searchsorted(dsrt, data, side="left")
    fdr = np.minimum(1.0, n_null_ge / np.maximum(n_data_ge, 1))
    o = np.argsort(data)
    q = np.empty_like(fdr); q[o] = np.minimum.accumulate(fdr[o])   # min over thresholds t <= x
    edges = np.quantile(data, np.linspace(0, 1, 21))
    edges = np.unique(np.r_[edges, data.max() + 1e-9])
    hd, _ = np.histogram(data, edges); hn, _ = np.histogram(null, edges)
    lfdr = np.minimum(1.0, (hn / n_null_reps) / np.maximum(hd, 1))
    lfdr = np.minimum.accumulate(lfdr)           # non-increasing with score
    idx = np.clip(np.searchsorted(edges, data, side="right") - 1, 0, len(lfdr) - 1)
    return q, 1.0 - lfdr[idx]
