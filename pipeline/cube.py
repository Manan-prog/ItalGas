"""Swath-local coordinates and gridding of channel traces into a (s, u, t) cube.

s = along-track arc length (m) of the smoothed GPS track, u = signed cross-track
offset (m, positive LEFT), both derived from the measured track and the measured
channel positions in the vehicle frame (Survey XML). The cube is the 'C-scan'
container: cube[s_bin, u_bin, sample]."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from geometry import heading_rad, interp_track, merc_scale, vehicle_to_map


def smooth_track(x: np.ndarray, y: np.ndarray, win: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Moving-average smoothing of the 10 Hz track (win samples). Suppresses
    sub-metre GNSS jitter; the lane-level bias is NOT removed (see DECISIONS.md)."""
    return (uniform_filter1d(x, win, mode="nearest"), uniform_filter1d(y, win, mode="nearest"))


def scan_positions(t_scan_s: np.ndarray, c_offset_s: float, t_path_s: np.ndarray,
                   x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPS-antenna map position (3857) and heading (rad) at each scan time."""
    tq = t_scan_s + c_offset_s
    xs, ys = interp_track(tq, t_path_s, x, y)
    hdg_track = heading_rad(x, y, smooth=11)
    # interpolate heading via unit vectors to avoid wrap problems
    cx = np.interp(tq, t_path_s, np.cos(hdg_track)); sy = np.interp(tq, t_path_s, np.sin(hdg_track))
    return xs, ys, np.arctan2(sy, cx)


def channel_footprints(xs: np.ndarray, ys: np.ndarray, hdg: np.ndarray, mids: np.ndarray,
                       gps_xy: tuple[float, float], k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map position (3857) of every (scan, channel) footprint.
    mids: (n_ch, 2) vehicle-frame midpoints. Returns (n_scans, n_ch) arrays X, Y."""
    X = np.empty((len(xs), len(mids))); Y = np.empty_like(X)
    for j, (fx, fy) in enumerate(mids):
        X[:, j], Y[:, j] = vehicle_to_map(xs, ys, hdg, fx, fy, gps_xy, k)
    return X, Y


def track_frame(x: np.ndarray, y: np.ndarray, k: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference polyline for (s,u): returns (s_nodes m, x_nodes, y_nodes)."""
    d = np.hypot(np.diff(x), np.diff(y)) * k
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s, x, y


def to_su(X: np.ndarray, Y: np.ndarray, s_nodes: np.ndarray, xn: np.ndarray, yn: np.ndarray,
          k: float) -> tuple[np.ndarray, np.ndarray]:
    """Project map points onto the reference polyline: s = arc length at the foot of
    the perpendicular, u = signed cross-track distance (m, positive LEFT of travel).
    Vectorised with a KD-tree on the nodes; exact projection on the two adjacent segments."""
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[xn, yn])
    _, i = tree.query(np.c_[X.ravel(), Y.ravel()])
    i = np.clip(i, 1, len(xn) - 2)
    best_s = np.full(X.size, np.nan); best_u = np.full(X.size, np.nan); best_d = np.full(X.size, np.inf)
    px, py = X.ravel(), Y.ravel()
    for a, b in ((i - 1, i), (i, i + 1)):
        ax, ay, bx, by = xn[a], yn[a], xn[b], yn[b]
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy
        tt = np.clip(((px - ax) * vx + (py - ay) * vy) / np.maximum(L2, 1e-12), 0, 1)
        fx, fy = ax + tt * vx, ay + tt * vy
        d = np.hypot(px - fx, py - fy)
        cross = (vx * (py - ay) - vy * (px - ax)) / np.sqrt(np.maximum(L2, 1e-12))  # + = left
        better = d < best_d
        best_d[better] = d[better]
        best_s[better] = (s_nodes[a] + tt * (s_nodes[b] - s_nodes[a]))[better]
        best_u[better] = (cross * k)[better]
    return best_s.reshape(X.shape), best_u.reshape(X.shape)


def grid_cube(values: np.ndarray, s: np.ndarray, u: np.ndarray, ds: float, du: float,
              s_range: tuple[float, float], u_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Bin traces (n_traces, n_samples) with coordinates (s, u) into a cube
    (n_s, n_u, n_samples) by averaging. Returns (cube, counts)."""
    ns = int(np.ceil((s_range[1] - s_range[0]) / ds)); nu = int(np.ceil((u_range[1] - u_range[0]) / du))
    si = np.floor((s - s_range[0]) / ds).astype(int); ui = np.floor((u - u_range[0]) / du).astype(int)
    ok = (si >= 0) & (si < ns) & (ui >= 0) & (ui < nu)
    flat = si[ok] * nu + ui[ok]
    cube = np.zeros((ns * nu, values.shape[1]), dtype=np.float32)
    np.add.at(cube, flat, values[ok])
    cnt = np.bincount(flat, minlength=ns * nu).astype(np.float32)
    cube /= np.maximum(cnt, 1)[:, None]
    return cube.reshape(ns, nu, -1), cnt.reshape(ns, nu)
