"""Shared basemap helpers for figures (single implementation of the street-aligned,
rotated Google satellite mosaic; rotation sign verified on a synthetic line, see
plot_pipe_probability.py history)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from contextily import bounds2img
from scipy.ndimage import rotate as nd_rotate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from run_detection import su_to_map  # noqa: E402

GSAT = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"


def street_aligned_tiles(s0: float, s1: float, half_w: float, s_nodes, xn, yn, k: float, zoom: int = 21):
    """Satellite image of the strip s0..s1 (m) x +-half_w (m), rotated so the track runs
    left to right (+s to the right, +u = left of travel = up). Returns (img, extent)
    with extent = [s0, s1, -half_w, half_w] for imshow(origin='upper')."""
    sc_mid = 0.5 * (s0 + s1); half_len = 0.5 * (s1 - s0)
    xc, yc = su_to_map(np.array([sc_mid]), np.array([0.0]), s_nodes, xn, yn, k); xc, yc = float(xc[0]), float(yc[0])
    tx = np.interp(sc_mid, s_nodes, np.gradient(xn)); ty = np.interp(sc_mid, s_nodes, np.gradient(yn)); hdg = np.arctan2(ty, tx)
    R = (half_len + half_w) / k
    img, ext = bounds2img(xc - R, yc - R, xc + R, yc + R, zoom=zoom, source=GSAT, ll=False)
    px = (ext[1] - ext[0]) / img.shape[1]
    rot = nd_rotate(img, -np.degrees(hdg), reshape=False, order=1, mode="constant", cval=0)
    cx_px = (xc - ext[0]) / px; cy_px = (ext[3] - yc) / px
    cxa, cya = (img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2
    dxp, dyp = cx_px - cxa, cy_px - cya
    rx = cxa + dxp * np.cos(hdg) - dyp * np.sin(hdg); ry = cya + dxp * np.sin(hdg) + dyp * np.cos(hdg)
    mpp = px * k
    a0 = int(rx - half_len / mpp); a1 = int(rx + half_len / mpp); b0 = int(ry - half_w / mpp); b1 = int(ry + half_w / mpp)
    return rot[b0:b1, a0:a1], [s0, s1, -half_w, half_w], float(np.degrees(hdg)), (xc, yc)
