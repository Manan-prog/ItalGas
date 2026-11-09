"""Line evidence v2: contrast-based oriented line detector with depth locality.

For each depth slice E(s,u) of the layer-removed envelope cube:
  R_theta = mean of E along a segment of length L through the pixel at angle theta
            / mean of E in the parallel band (|offset| in [w1, w2]) on both sides
  R = max over theta (Tupin et al. 1998 line-ratio detector, re-derived for additive
  GPR clutter: the ratio of means is used, no speckle model).
Depth locality: D = R_j / mean_j'(R_j') over all depths at the same (s,u); a scan-wide
time-shift artefact gives R ~ equal at all depths (D ~ 1), a pipe gives D >> 1.
Final evidence = R * D, calibrated to a probability by the two-groups lfdr model."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve


def _line_kernel(half: int, theta: float, offset: float = 0.0) -> np.ndarray:
    k = np.zeros((2 * half + 1, 2 * half + 1), dtype=np.float32)
    tt = np.linspace(-half, half, 4 * half + 1)
    ii = np.round(half + tt * np.cos(theta) - offset * np.sin(theta)).astype(int)
    jj = np.round(half + tt * np.sin(theta) + offset * np.cos(theta)).astype(int)
    ok = (ii >= 0) & (ii <= 2 * half) & (jj >= 0) & (jj <= 2 * half)
    k[ii[ok], jj[ok]] = 1.0
    return k / max(k.sum(), 1)


def ratio_line_detector(img: np.ndarray, length_px: int, band: tuple[int, int] = (2, 4),
                        n_angles: int = 24, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Returns (ratio, angle) per pixel. img >= 0, shape (n_s, n_u). Angle from the s axis."""
    half = length_px // 2
    best = np.zeros(img.shape, dtype=np.float32); ang = np.zeros(img.shape, dtype=np.float32)
    glob = img.mean() + eps
    for th in np.linspace(0, np.pi, n_angles, endpoint=False):
        kl = _line_kernel(half, th)
        kb = sum(_line_kernel(half, th, o) for o in list(range(-band[1], -band[0] + 1)) + list(range(band[0], band[1] + 1)))
        kb /= kb.sum()
        line = convolve(img, kl, mode="nearest")
        back = convolve(img, kb, mode="nearest")
        r = line / (back + 0.2 * glob)               # floor keeps ratio bounded in dark areas
        upd = r > best
        best[upd] = r[upd]; ang[upd] = th
    return best, ang


def evidence_v2(clean: np.ndarray, ds: float, line_len_m: float = 2.0, n_angles: int = 24
                ) -> dict[str, np.ndarray]:
    L = int(round(line_len_m / ds)) | 1
    n_s, n_u, n_t = clean.shape
    R = np.empty_like(clean); A = np.empty_like(clean)
    for j in range(n_t):
        R[:, :, j], A[:, :, j] = ratio_line_detector(clean[:, :, j], L, n_angles=n_angles)
    D = R / (R.mean(axis=2, keepdims=True) + 1e-6)
    return {"ratio": R, "angle": A, "depth_contrast": D, "evidence": (R * D).astype(np.float32)}
