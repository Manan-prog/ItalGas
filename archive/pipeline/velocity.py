"""Velocity estimation from diffraction hyperbolas by hyperbola-summation
(semblance-like) analysis. Pure NumPy, vectorised over apex candidates.

Model (zero-offset, homogeneous half space): a point diffractor at along-track
position x0 and two-way time t0 produces t(x) = sqrt(t0^2 + (2 (x - x0) / v)^2).
TX-RX separation (0.08 m) is neglected: ASSUMED negligible for t0 > 5 ns."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter


def apex_candidates(env: np.ndarray, dt_ns: float, dx_m: float, t_min_ns: float, t_max_ns: float,
                    win_m: float = 0.5, win_ns: float = 3.0, n_max: int = 400) -> np.ndarray:
    """Local maxima of the envelope (n_traces, n_samples) inside the time window.
    Returns array of (trace_index, sample_index, amplitude), strongest first."""
    wx = max(3, int(win_m / dx_m) | 1); wt = max(3, int(win_ns / dt_ns) | 1)
    mf = maximum_filter(env, size=(wx, wt), mode="nearest")
    j0, j1 = int(t_min_ns / dt_ns), int(t_max_ns / dt_ns)
    cand = np.argwhere((env == mf) & (env > 0))
    cand = cand[(cand[:, 1] >= j0) & (cand[:, 1] < j1)]
    amp = env[cand[:, 0], cand[:, 1]]
    order = np.argsort(-amp)[:n_max]
    return np.c_[cand[order], amp[order]]


def hyperbola_stack(env: np.ndarray, cands: np.ndarray, dt_ns: float, dx_m: float,
                    v_grid: np.ndarray, half_aperture_m: float = 0.8) -> np.ndarray:
    """For each candidate apex (i0, j0) and each velocity, the mean envelope along the
    model hyperbola, normalised by the apex value (so a flat reflector scores ~1 at
    every v and a true hyperbola peaks at the correct v).
    Returns (n_cand, n_v)."""
    nh = int(half_aperture_m / dx_m)
    h = (np.arange(-nh, nh + 1) * dx_m)[None, None, :]            # (1,1,nh2)
    i0 = cands[:, 0].astype(int)[:, None, None]
    t0 = (cands[:, 1] * dt_ns)[:, None, None]
    v = v_grid[None, :, None]
    th = np.sqrt(t0 ** 2 + (2 * h / v) ** 2)                      # (nc, nv, nh2)
    jj = th / dt_ns
    ii = i0 + np.arange(-nh, nh + 1)[None, None, :]
    ok = (ii >= 0) & (ii < env.shape[0]) & (jj < env.shape[1] - 1)
    ii = np.clip(ii, 0, env.shape[0] - 1); j0 = np.clip(np.floor(jj).astype(int), 0, env.shape[1] - 2)
    fr = jj - j0
    val = (1 - fr) * env[ii, j0] + fr * env[ii, j0 + 1]
    val = np.where(ok, val, 0.0)
    apex = env[cands[:, 0].astype(int), cands[:, 1].astype(int)][:, None, None]
    return (val / np.maximum(apex, 1e-9)).mean(axis=2)


def velocity_distribution(stack: np.ndarray, v_grid: np.ndarray, min_contrast: float = 0.15
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Keep candidates whose best-velocity stack exceeds the flat (max-v) level by
    min_contrast (i.e. true hyperbolas, not layers). Returns (v_best per kept
    candidate, contrast)."""
    best = stack.max(axis=1)
    contrast = best - stack[:, -1]
    keep = contrast > min_contrast
    return v_grid[np.argmax(stack[keep], axis=1)], contrast[keep]
