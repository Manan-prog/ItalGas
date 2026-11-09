"""Disagreement between the two trajectory solutions of a swath (real-time RTK .path
vs post-processed _Ext.nmea), resolved into along-track and cross-track parts as a
function of s. Neither solution is truth: the disagreement is a MEASURED lower bound
on plan-position uncertainty, and the post-processed solution is an independent
alternative placement for tests."""
from __future__ import annotations

import numpy as np

from italgas_gpr.geometry.frames import lonlat_to_merc


def _speed(t, x, y, k):
    return np.hypot(np.gradient(x), np.gradient(y)) * k / np.gradient(t)


def clock_offset_s(t_a, x_a, y_a, t_b, x_b, y_b, k, search_s=5.0, step_s=0.02) -> tuple[float, float]:
    """Offset c (s) such that track b at time t + c matches the motion of track a at t,
    from the speed cross-correlation (same idea as geometry.scan_time_offset_s)."""
    grid = np.arange(t_a[0], t_a[-1], 0.05)
    va = np.interp(grid, t_a, _speed(t_a, x_a, y_a, k)); vb = _speed(t_b, x_b, y_b, k)
    best = (0.0, -1.0)
    for c in np.arange(-search_s, search_s + 1e-9, step_s):
        v = np.interp(grid + c, t_b, vb, left=np.nan, right=np.nan); m = ~np.isnan(v)
        if m.mean() > 0.8:
            r = np.corrcoef(va[m], v[m])[0, 1]
            if r > best[1]:
                best = (float(c), float(r))
    return best


def along_cross_disagreement(path: np.ndarray, gga_pp: np.ndarray, k: float) -> dict:
    """Return dict(s_rt (m, arc length of the RT track), along (m, PP - RT along heading),
    cross (m, PP - RT, + left), clock_offset_s, corr)."""
    t, x, y = path["t_ms"] / 1e3, path["x"], path["y"]
    xg, yg = lonlat_to_merc(gga_pp["lon"], gga_pp["lat"])
    c, r = clock_offset_s(t, x, y, gga_pp["t_s"], xg, yg, k)
    xi = np.interp(t + c, gga_pp["t_s"], xg); yi = np.interp(t + c, gga_pp["t_s"], yg)
    hd = np.arctan2(np.gradient(y), np.gradient(x))
    dx, dy = (xi - x) * k, (yi - y) * k
    s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)) * k)]
    return dict(s_rt=s, along=dx * np.cos(hd) + dy * np.sin(hd), cross=-dx * np.sin(hd) + dy * np.cos(hd),
                clock_offset_s=c, corr=r)
