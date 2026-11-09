"""Linear-feature (pipe) evidence from a georeferenced envelope cube.

Steps (all vectorised):
 1. resample every channel onto a common along-track grid s (linear interp) and
    across-track grid u (linear interp between channels) -> cube (n_s, n_u, n_t)
 2. per-channel robust normalisation (median absolute level over the swath)
 3. layer removal: subtract, per (s, t), the median over u, after smoothing it
    along s (window win_s); layers are uniform in u, pipes are not
 4. depth normalisation: divide each t-slice by its robust scale (depth AGC)
 5. tubular-structure evidence: multi-scale Hessian 'vesselness' (Frangi 1998,
    3D, bright tubes) on the cube resampled to near-isotropic voxels
 6. null calibration: the same evidence computed on a cube whose traces are
    randomly permuted along s (continuity destroyed) gives the null distribution;
    the reported probability is 1 - P_null(evidence >= observed).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter1d
from skimage.filters import frangi


def resample_cube(env: np.ndarray, s: np.ndarray, u: np.ndarray, ds: float, du: float,
                  u_range: tuple[float, float], s_max: float | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """env (n_scans, n_ch, n_t); s,u (n_scans, n_ch). Returns (cube, s_grid, u_grid)."""
    n, nc, nt = env.shape
    s_max = float(np.nanmax(s)) if s_max is None else s_max
    sg = np.arange(0, s_max, ds); ug = np.arange(u_range[0], u_range[1] + 1e-9, du)
    per_ch = np.empty((len(sg), nc, nt), dtype=np.float32)
    for c in range(nc):
        sc = s[:, c]; ok = np.isfinite(sc)
        order = np.argsort(sc[ok]); sc_sorted = sc[ok][order]; e = env[ok][order, c, :]
        # normalise the channel by its own median level AT EACH TIME SAMPLE, so that
        # channel-specific gain and depth profiles cannot masquerade as along-track lines
        e = e / (np.median(e, axis=0, keepdims=True) + 1e-9)
        idx = np.clip(np.searchsorted(sc_sorted, sg) - 1, 0, len(sc_sorted) - 2)
        w = np.clip((sg - sc_sorted[idx]) / np.maximum(sc_sorted[idx + 1] - sc_sorted[idx], 1e-6), 0, 1)
        per_ch[:, c, :] = (1 - w)[:, None] * e[idx] + w[:, None] * e[idx + 1]
    uc = np.nanmedian(u, axis=0)                    # channel lateral positions (sorted descending)
    order = np.argsort(uc); uc = uc[order]; per_ch = per_ch[:, order, :]
    # linear interpolation across u (vectorised through index/weight arrays)
    j = np.clip(np.searchsorted(uc, ug) - 1, 0, len(uc) - 2)
    w = np.clip((ug - uc[j]) / (uc[j + 1] - uc[j]), 0, 1).astype(np.float32)
    cube = (1 - w)[None, :, None] * per_ch[:, j, :] + w[None, :, None] * per_ch[:, j + 1, :]
    return cube, sg, ug


def remove_layers(cube: np.ndarray, ds: float, win_s_m: float = 3.0) -> np.ndarray:
    """Subtract the across-u median (smoothed along s) from every (s,t)."""
    bg = np.median(cube, axis=1)                                   # (n_s, n_t)
    bg = uniform_filter1d(bg, max(3, int(win_s_m / ds) | 1), axis=0, mode="nearest")
    out = cube - bg[:, None, :]
    # depth normalisation by the robust scale of each t slice
    scale = np.median(np.abs(out), axis=(0, 1)) + 1e-6
    out = np.maximum(out / scale[None, None, :], 0.0).astype(np.float32)
    # burst rejection: scans whose total energy (all u, all t) exceeds 3x the local
    # (10 m) median are interference bursts (they span every depth); zero them
    tot = out.mean(axis=(1, 2))
    loc = median_filter(tot, size=max(3, int(10.0 / ds) | 1), mode="nearest")
    burst = tot > 3.0 * loc
    out[burst] = 0.0
    # physical minimum width along s: a migrated point response is >= 0.3 m wide,
    # so a grey-level opening with a 3-pixel (0.3 m) line along s removes thinner stripes
    from scipy.ndimage import grey_opening
    out = grey_opening(out, size=(3, 1, 1), mode="nearest")
    return out


def tube_evidence(cube: np.ndarray, sigmas: tuple[float, ...] = (1.0, 1.5, 2.0),
                  chunk: int = 4000) -> np.ndarray:
    """Frangi vesselness for bright tubular structures in a 3D array with
    near-isotropic voxels. Processed in overlapping chunks along axis 0."""
    pad = int(4 * max(sigmas))
    out = np.empty_like(cube, dtype=np.float32)
    n = cube.shape[0]
    for a in range(0, n, chunk):
        a0, b0 = max(0, a - pad), min(n, a + chunk + pad)
        v = frangi(cube[a0:b0].astype(np.float64), sigmas=sigmas, black_ridges=False,
                   alpha=0.5, beta=0.5, gamma=None)
        out[a: min(n, a + chunk)] = v[a - a0: a - a0 + min(chunk, n - a)]
    return out


def null_calibrate(evidence: np.ndarray, evidence_null: np.ndarray, n_bins: int = 2000
                   ) -> np.ndarray:
    """Map evidence to 1 - P_null(E >= e) using the empirical null CDF."""
    nullv = np.sort(evidence_null.ravel())
    ranks = np.searchsorted(nullv, evidence.ravel(), side="left")
    return (ranks / len(nullv)).reshape(evidence.shape).astype(np.float32)


def block_shuffle(slice2d: np.ndarray, rng: np.random.Generator, bs: int = 5, bu: int = 3) -> np.ndarray:
    """Null surrogate of a 2D (s, u) slice: the slice is cut into bs x bu pixel blocks
    (0.5 m x 0.24 m at our grid) which are randomly permuted and flipped. The
    amplitude distribution and the within-block texture are preserved; coherent
    structure longer than one block is destroyed. (The earlier Fourier phase
    surrogate was rejected: it Gaussianises the amplitudes, see EXPERIMENTS.md.)"""
    ns, nu = slice2d.shape
    ns2, nu2 = ns - ns % bs, nu - nu % bu
    blocks = slice2d[:ns2, :nu2].reshape(ns2 // bs, bs, nu2 // bu, bu).transpose(0, 2, 1, 3).reshape(-1, bs, bu)
    perm = rng.permutation(len(blocks))
    blocks = blocks[perm]
    flip_s = rng.random(len(blocks)) < 0.5; flip_u = rng.random(len(blocks)) < 0.5
    blocks[flip_s] = blocks[flip_s][:, ::-1, :]; blocks[flip_u] = blocks[flip_u][:, :, ::-1]
    out = slice2d.copy()
    out[:ns2, :nu2] = blocks.reshape(ns2 // bs, nu2 // bu, bs, bu).transpose(0, 2, 1, 3).reshape(ns2, nu2)
    return out


def line_filter(img: np.ndarray, length_px: int, n_angles: int = 36) -> tuple[np.ndarray, np.ndarray]:
    """Oriented line-integral filter: for each pixel, the maximum over orientations of
    the mean of img along a straight segment of `length_px` centred on the pixel.
    Returns (response, best_angle_rad) with angle measured from the s axis (axis 0)."""
    from scipy.ndimage import convolve
    best = np.full(img.shape, -np.inf, dtype=np.float32); ang = np.zeros(img.shape, dtype=np.float32)
    half = length_px // 2
    for th in np.linspace(0, np.pi, n_angles, endpoint=False):
        kern = np.zeros((2 * half + 1, 2 * half + 1), dtype=np.float32)
        tt = np.linspace(-half, half, 4 * half + 1)
        ii = np.round(half + tt * np.cos(th)).astype(int); jj = np.round(half + tt * np.sin(th)).astype(int)
        kern[ii, jj] = 1.0
        kern /= kern.sum()
        r = convolve(img, kern, mode="nearest")
        upd = r > best
        best[upd] = r[upd]; ang[upd] = th
    return best, ang


def evidence_pipeline(clean: np.ndarray, ds: float, du: float, line_len_m: float = 2.0,
                      sigmas: tuple[float, ...] = (1.0, 1.5), seed: int = 0
                      ) -> dict[str, np.ndarray]:
    """clean: layer-removed cube (n_s, n_u, n_t). Returns per-voxel line evidence,
    its surrogate-null counterpart, and the calibrated probability cube."""
    rng = np.random.default_rng(seed)
    tube = tube_evidence(clean, sigmas)
    sur = np.stack([block_shuffle(clean[:, :, j], rng) for j in range(clean.shape[2])], axis=2)
    tube_null = tube_evidence(sur, sigmas)
    L = int(round(line_len_m / ds)) | 1
    ev = np.empty_like(tube); ev_null = np.empty_like(tube); ang = np.empty_like(tube)
    for j in range(clean.shape[2]):
        ev[:, :, j], ang[:, :, j] = line_filter(tube[:, :, j], L)
        ev_null[:, :, j], _ = line_filter(tube_null[:, :, j], L)
    prob = null_calibrate(ev, ev_null)
    return {"tube": tube, "evidence": ev, "evidence_null": ev_null, "angle": ang, "prob": prob}


def lfdr_probability(evidence: np.ndarray, pi0: float = 0.95, eps: float = 1e-9
                     ) -> tuple[np.ndarray, dict]:
    """Empirical-Bayes two-groups calibration (Efron 2004, local false discovery rate).
    z = log(evidence). The null density f0 is a normal fitted to the central bulk
    (median and MAD of z); the mixture density f is a histogram KDE of all z.
    P(structure | z) = 1 - min(1, pi0 f0(z) / f(z)), made monotone in z.
    ASSUMED: pipes occupy < 5 percent of voxels (pi0 = 0.95) and the null bulk is
    log-normal; both are checked against the fitted histogram (returned in info)."""
    z = np.log(evidence + eps).ravel()
    med = np.median(z)
    # null scale from the LOWER half only (the signal lives in the upper tail):
    # for a normal, median - q25 = 0.6745 sigma
    mad = (med - np.quantile(z, 0.25)) / 0.6745
    edges = np.linspace(z.min(), z.max(), 400); centres = 0.5 * (edges[1:] + edges[:-1])
    hist, _ = np.histogram(z, bins=edges, density=True)
    hist = np.convolve(hist, np.ones(5) / 5, mode="same")
    f0 = np.exp(-0.5 * ((centres - med) / mad) ** 2) / (mad * np.sqrt(2 * np.pi))
    lfdr = np.minimum(1.0, pi0 * f0 / np.maximum(hist, 1e-12))
    # monotone non-increasing lfdr for z above the median (signal is in the upper tail)
    above = centres >= med
    lfdr[above] = np.minimum.accumulate(lfdr[above])
    lfdr[~above] = 1.0
    prob_c = 1.0 - lfdr
    prob = np.interp(z, centres, prob_c).reshape(evidence.shape).astype(np.float32)
    return prob, {"median_z": med, "mad_z": mad, "centres": centres, "hist": hist, "f0": pi0 * f0, "prob_curve": prob_c}
