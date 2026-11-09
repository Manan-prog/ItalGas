"""Town-plan network sketch (Comune di Savona PUC DF S.2.1, May 2011, 1:10,000) as an internal,
provisional overlay (user approval 3 Nov 2026; not redistributed, not a scored ground truth).
Mask georeferenced to EPSG:3003 by matching its sewer layer to the SIT sewer vectors (median 0.00 m,
p90 1.69 m); classes 1 = gas, 2 = water, 3 = sewer. Lines are schematic: placed to several metres."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pyproj import Transformer

from cube import to_su

MASK = Path("/private/tmp/claude-501/-Users-manansarupria-Library-CloudStorage-GoogleDrive-manansarupria599-gmail-com-My-Drive-Smartauger-ItalGas/4dd8e26b-56a0-4360-81df-80e7b8167035/scratchpad/derived_gt/puc_DFS21_network_mask_300dpi_EPSG3003.tif")
CLASSES = {"gas": 1, "water": 2, "sewer": 3}


def network_points(xn: np.ndarray, yn: np.ndarray, k: float, cls: str, pad_m: float = 25.0, mask: Path = MASK) -> np.ndarray:
    """Map positions (EPSG:3857) of the sketch's pixels of one network class near a track (xn, yn in 3857)."""
    import rasterio
    to3003 = Transformer.from_crs("EPSG:3857", "EPSG:3003", always_xy=True)
    to3857 = Transformer.from_crs("EPSG:3003", "EPSG:3857", always_xy=True)
    X, Y = to3003.transform(xn, yn)
    with rasterio.open(mask) as r:
        win = rasterio.windows.from_bounds(X.min() - pad_m, Y.min() - pad_m, X.max() + pad_m, Y.max() + pad_m, r.transform)
        a = r.read(1, window=win); T = r.window_transform(win)
    rr, cc = np.nonzero(a == CLASSES[cls])
    xs, ys = rasterio.transform.xy(T, rr, cc)
    mx, my = to3857.transform(np.asarray(xs), np.asarray(ys))
    return np.c_[mx, my]


def network_su(frame: dict, cls: str, u_max: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """Sketch pixels of one class in a swath frame (s along, u across, m), within |u| <= u_max."""
    P = network_points(frame["xn"], frame["yn"], float(frame["k"]), cls)
    if not len(P):
        return np.empty(0), np.empty(0)
    s, u = to_su(P[:, 0], P[:, 1], frame["s_nodes"], frame["xn"], frame["yn"], float(frame["k"]))
    ok = np.abs(u) <= u_max
    return s[ok], u[ok]


def nearest_line_offset(s_pts: np.ndarray, u_pts: np.ndarray, s0: float, s1: float, u_target: float) -> float:
    """Median over 1 m bins in s0..s1 of the smallest |u_line - u_target| (m); nan if no pixel in range."""
    out = []
    for a in np.arange(s0, s1, 1.0):
        m = (s_pts >= a) & (s_pts < a + 1.0)
        if m.any():
            out.append(np.min(np.abs(u_pts[m] - u_target)))
    return float(np.median(out)) if out else float("nan")
