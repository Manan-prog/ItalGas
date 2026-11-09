"""Detection of pipes running ALONG the street under the array (3 Nov 2026).

Physics: a pipe parallel to the track at lateral position u0 and depth z is the same
at every s. Across the street it is a diffraction hyperbola over the channels,
t_c = sqrt(t0^2 + (2 (u_c - u0) / v)^2). So: (1) average the noise-equalised traces
coherently along s over a window (noise falls as 1/sqrt(n), the pipe stays);
(2) migrate the across-street section over the channels' real u positions;
(3) a pipe focuses to a compact spot, a flat layer stays flat and is removed by the
median across u; (4) keep spots that persist along s (tracks).
Null: permute the ORDER of the channels (same permutation for the whole swath).
Every channel keeps its own content, artefacts and layers; only the hyperbolic
moveout across the array, the signature of a real object, is destroyed.
Shapes: Z (n_ch, n_s, n_t); sections X (n_win, n_ch, n_t); images (n_win, n_u, n_t)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.signal import hilbert


def stack_along_s(Z: np.ndarray, ds: float, win_m: float, hop_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Coherent (signed) mean of Z over windows of win_m every hop_m along s.
    Returns (X (n_win, n_ch, n_t), window-centre index into s)."""
    w = int(round(win_m / ds)); h = int(round(hop_m / ds))
    cs = np.cumsum(np.concatenate([np.zeros_like(Z[:, :1]), Z], 1, dtype=np.float32), axis=1, dtype=np.float64)
    starts = np.arange(0, Z.shape[1] - w, h)
    X = (cs[:, starts + w] - cs[:, starts]) / w                    # (n_ch, n_win, n_t)
    return np.transpose(X, (1, 0, 2)).astype(np.float32), starts + w // 2


def lateral_migrate(X: np.ndarray, u_ch: np.ndarray, u_grid: np.ndarray, dt_ns: float, v: float) -> np.ndarray:
    """Kirchhoff migration across the channels at their true lateral positions (m).
    X (n_win, n_ch, n_t) -> envelope image (n_win, n_u, n_t). Obliquity weight t0/t_c."""
    n_win, n_ch, n_t = X.shape
    t0 = np.arange(n_t) * dt_ns
    out = np.zeros((n_win, len(u_grid), n_t), np.float32); wsum = np.zeros((len(u_grid), n_t), np.float32)
    for c in range(n_ch):                                           # loop over channels, not traces
        tc = np.sqrt(t0[None, :] ** 2 + (2 * (u_ch[c] - u_grid[:, None]) / v) ** 2)   # (n_u, n_t)
        idx = tc / dt_ns; i0 = np.floor(idx).astype(int); fr = (idx - i0).astype(np.float32)
        ok = i0 < n_t - 1; i0 = np.clip(i0, 0, n_t - 2)
        wgt = np.where(tc > 0, t0[None, :] / np.maximum(tc, 1e-6), 1.0).astype(np.float32) * ok
        xc = X[:, c, :]
        out += ((1 - fr) * xc[:, i0] + fr * xc[:, i0 + 1]) * wgt
        wsum += wgt
    img = out / np.maximum(wsum, 1e-6)
    return np.abs(hilbert(img, axis=2)).astype(np.float32)


def lateral_contrast(env: np.ndarray) -> np.ndarray:
    """Spot minus the median across u at the same (window, time): flat layers -> about 0."""
    return env - np.median(env, axis=1, keepdims=True)


def window_peaks(score: np.ndarray, thr: float, sep_u: int, sep_t: int, t_range: tuple[int, int]) -> np.ndarray:
    """Local maxima per window in (u, t) above thr. Returns (k, 3) int array (window, iu, it)."""
    mx = maximum_filter(score, size=(1, 2 * sep_u + 1, 2 * sep_t + 1), mode="nearest")
    ok = (score == mx) & (score > thr)
    ok[:, :, : t_range[0]] = False; ok[:, :, t_range[1]:] = False
    return np.argwhere(ok)


def link_tracks(peaks: np.ndarray, score: np.ndarray, max_du: int, max_dt: int, max_gap: int = 1) -> list[np.ndarray]:
    """Greedy marching linker (Dou et al. 2016, simplified): walk windows in order and
    extend the open track whose last peak is closest in (u, t) within the gates; a track
    closes after max_gap empty windows. Returns a list of (n_i, 3) peak arrays."""
    tracks: list[list] = []; open_: list[int] = []
    by_w = {}
    for p in peaks:
        by_w.setdefault(int(p[0]), []).append(p)
    for w in sorted(by_w):
        still = []
        for ti in open_:
            if w - tracks[ti][-1][0] <= max_gap + 1:
                still.append(ti)
        open_ = still
        used = set()
        for p in sorted(by_w[w], key=lambda q: -score[q[0], q[1], q[2]]):
            best, bd = None, None
            for ti in open_:
                if ti in used:
                    continue
                q = tracks[ti][-1]
                du, dtt = abs(int(p[1]) - int(q[1])), abs(int(p[2]) - int(q[2]))
                if du <= max_du and dtt <= max_dt and (bd is None or du + dtt < bd):
                    best, bd = ti, du + dtt
            if best is None:
                tracks.append([p]); open_.append(len(tracks) - 1); used.add(len(tracks) - 1)
            else:
                tracks[best].append(p); used.add(best)
    return [np.array(t) for t in tracks]


def point_vs_line_contrast(env: np.ndarray, k_range: tuple[int, int] = (3, 8), slopes: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3)
                           ) -> np.ndarray:
    """A migrated pipe is a compact SPOT in the across-street section; a (dipping) layer is
    a LINE through the spot and a layer edge is a half-line. Score = env minus the
    brightest continuation to either side: for side in (left, right) and dip slope p
    (time samples per u cell), the mean of env at (u + side*k, t + p*k), k in k_range.
    Off-section samples are ignored. env (n_win, n_u, n_t) -> (n_win, n_u, n_t)."""
    n_w, n_u, n_t = env.shape
    best = np.full(env.shape, -np.inf, np.float32)
    ks = np.arange(k_range[0], k_range[1] + 1)
    for side in (-1, 1):
        for p in slopes:
            acc = np.zeros(env.shape, np.float32); cnt = np.zeros((n_u, n_t), np.float32)
            for k in ks:
                sh = np.roll(np.roll(env, -side * k, axis=1), -p * k, axis=2)
                uu = np.arange(n_u) + side * k; tt = np.arange(n_t) + p * k
                ok = ((uu >= 0) & (uu < n_u))[:, None] & ((tt >= 0) & (tt < n_t))[None, :]
                acc += sh * ok; cnt += ok
            m = np.where(cnt >= len(ks) / 2, acc / np.maximum(cnt, 1), -np.inf)
            best = np.maximum(best, m)
    best = np.where(np.isfinite(best), best, 0.0)
    return (env - best).astype(np.float32)


def lateral_migrate_radius(X: np.ndarray, u_ch: np.ndarray, u_grid: np.ndarray, dt_ns: float, v: float, radius_m: float) -> np.ndarray:
    """Across-street Kirchhoff focusing for a pipe of radius R (user tip, 3 Nov 2026): for an output point
    (u0, t0) the pipe top is at z = v t0 / 2 and the echo reaches channel c at
    t_c = 2 (sqrt((z + R)^2 + (u_c - u0)^2) - R) / v. R = 0 is lateral_migrate. X (n_win, n_ch, n_t) ->
    envelope image (n_win, n_u, n_t)."""
    n_win, n_ch, n_t = X.shape
    z = np.arange(n_t) * dt_ns * v / 2
    out = np.zeros((n_win, len(u_grid), n_t), np.float32); wsum = np.zeros((len(u_grid), n_t), np.float32)
    for c in range(n_ch):
        d = u_ch[c] - u_grid[:, None]; rr = np.sqrt((z[None, :] + radius_m) ** 2 + d ** 2)
        tc = 2 * (rr - radius_m) / v; idx = tc / dt_ns; i0 = np.floor(idx).astype(int); fr = (idx - i0).astype(np.float32)
        ok = i0 < n_t - 1; i0 = np.clip(i0, 0, n_t - 2)
        wgt = ((z[None, :] + radius_m) / np.maximum(rr, 1e-6)).astype(np.float32) * ok
        xc = X[:, c, :]; out += ((1 - fr) * xc[:, i0] + fr * xc[:, i0 + 1]) * wgt; wsum += wgt
    return np.abs(hilbert(out / np.maximum(wsum, 1e-6), axis=2)).astype(np.float32)
