"""Vehicle-frame to map-frame geometry for the IDS Stream UP.

Conventions (verified against the IDS KML channel lines, DECISIONS.md 2026-11-02):
  vehicle frame: x forward, y positive LEFT of travel, metres.
  map frame: EPSG:3857 (Web Mercator). Distances in 3857 must be divided by
  k = cos(lat) to be metres on the ground; vehicle offsets are divided by k
  before being added in 3857.
"""
from __future__ import annotations

import numpy as np

R_EARTH = 6378137.0


def merc_to_lonlat(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.degrees(x / R_EARTH)
    lat = np.degrees(np.arctan(np.sinh(y / R_EARTH)))
    return lon, lat


def lonlat_to_merc(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = R_EARTH * np.radians(lon)
    y = R_EARTH * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def merc_scale(y: np.ndarray) -> np.ndarray:
    """k = cos(lat): ground metres = 3857 units * k."""
    return np.cos(np.arctan(np.sinh(y / R_EARTH)))


def scan_time_offset_s(t_scan_s: np.ndarray, odo_m: np.ndarray,
                       t_path_s: np.ndarray, x: np.ndarray, y: np.ndarray,
                       k: float, search_s: float = 10.0, step_s: float = 0.01) -> tuple[float, float]:
    """Find the constant offset c such that path_time = t_scan + c, by cross-correlating
    wheel-odometer speed with GNSS speed on a common grid.

    Returns (c_seconds, peak_correlation). Both inputs are measured; c is derived.
    """
    g = np.arange(0.0, t_scan_s[-1] - t_scan_s[0], step_s)
    v_odo = np.gradient(np.interp(g, t_scan_s - t_scan_s[0], odo_m), step_s)
    dist = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)) * k)])
    gp = np.arange(0.0, t_path_s[-1] - t_path_s[0], step_s)
    v_gps = np.gradient(np.interp(gp, t_path_s - t_path_s[0], dist), step_s)
    lags = np.arange(-int(search_s / step_s), int(search_s / step_s) + 1)
    best = (0, -np.inf)
    a0 = v_odo - v_odo.mean()
    for lag in lags:
        if lag >= 0:
            a = a0[: len(a0) - lag]; b = v_gps[lag: lag + len(a)]
        else:
            a = a0[-lag:]; b = v_gps[: len(a)]
        m = min(len(a), len(b))
        if m < 100:
            continue
        a = a[:m]; b = b[:m] - b[:m].mean()
        c = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
        if c > best[1]:
            best = (lag, c)
    # path_time(scan i) = (t_scan_i - t_scan_0) + t_path_0 + lag*step
    c_s = best[0] * step_s + t_path_s[0] - t_scan_s[0]
    return c_s, best[1]


def interp_track(t_query: np.ndarray, t_track: np.ndarray, x: np.ndarray, y: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    return np.interp(t_query, t_track, x), np.interp(t_query, t_track, y)


def heading_rad(x: np.ndarray, y: np.ndarray, smooth: int = 21) -> np.ndarray:
    """Heading (radians, map frame, from +x axis counter-clockwise) of a track,
    from a moving-average-smoothed gradient. Positions in 3857 (direction only)."""
    kern = np.ones(smooth) / smooth
    dx = np.convolve(np.gradient(x), kern, mode="same")
    dy = np.convolve(np.gradient(y), kern, mode="same")
    return np.arctan2(dy, dx)


def vehicle_to_map(x_gps: np.ndarray, y_gps: np.ndarray, hdg: np.ndarray,
                   fx: float, fy: float, gps_xy: tuple[float, float], k: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Map position (3857) of a point at vehicle-frame (fx, fy) given the GPS antenna
    track and heading. gps_xy is the GPS antenna's vehicle-frame position.
    Offsets in metres are converted to 3857 units by dividing by k."""
    dx = (fx - gps_xy[0]) / k
    dy = (fy - gps_xy[1]) / k
    c, s = np.cos(hdg), np.sin(hdg)
    # vehicle y is LEFT of travel: left of heading h is direction h + 90 deg
    return x_gps + dx * c - dy * s, y_gps + dx * s + dy * c
