"""Learned along-road evidence (3 Nov 2026): a classifier trained on realistic fake pipes (wandering,
radius 0-0.25 m, strength 2-8, any position across the strip) hidden in REAL data of one street, used as
the emission of the depth-aware Viterbi. Features per cell (window along s, offset u, depth bin), each
normalised per depth bin within the drive (median / MAD) so the model can transfer between soils."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from longitudinal import lateral_contrast, lateral_migrate_radius, point_vs_line_contrast

RADII = (0.0, 0.1, 0.25); Z0, Z1, DZ = 0.2, 1.5, 0.06
FEATURES = ["e_r0", "e_r10", "e_r25", "point_vs_line", "persistence", "width", "depth_contrast", "depth", "edge_dist"]


def _bin(E, dt, v):
    zb = np.arange(Z0, Z1, DZ); depth = np.arange(E.shape[2]) * dt * v / 2
    return np.stack([E[:, :, (depth >= a) & (depth < a + DZ)].max(2) for a in zb], 2), zb


def _norm(A):
    med = np.median(A, axis=(0, 1), keepdims=True); mad = 1.4826 * np.median(np.abs(A - med), axis=(0, 1), keepdims=True) + 1e-9
    return (A - med) / mad


def cell_features(X, u, ug, dt, v):
    """X (n_win, n_ch, n_t) 2 m stacks -> F (n_win, n_u, n_z, n_feat), depth bins zb."""
    imgs = [lateral_migrate_radius(X, u, ug, dt, v, R) for R in RADII]
    E = [_bin(lateral_contrast(im), dt, v)[0] for im in imgs]; pvl, zb = _bin(point_vs_line_contrast(imgs[0]), dt, v)
    emax = np.maximum.reduce(E); pers = uniform_filter1d(emax, 11, axis=0, mode="nearest")
    width = np.zeros_like(emax)
    for k in range(-12, 13):
        sh = np.roll(emax, k, axis=1)
        width += (sh >= 0.5 * emax) & (emax > 0)
    width *= ug[1] - ug[0]
    dcon = emax / (np.abs(emax).mean(2, keepdims=True) + 1e-6)
    n_w, n_u, n_z = emax.shape
    depth = np.broadcast_to(zb[None, None, :], emax.shape); edge = np.minimum(ug - u.min(), u.max() - ug)
    edge = np.broadcast_to(edge[None, :, None], emax.shape)
    F = np.stack([_norm(E[0]), _norm(E[1]), _norm(E[2]), _norm(pvl), _norm(pers), width, dcon, depth, edge], -1).astype(np.float32)
    return F, zb


def labels_from_segments(segs, s_cen, ug, zb):
    """Positive cells: within 1 offset cell and the depth bin of each fake pipe's top; exclusion zone around them."""
    pos = np.zeros((len(s_cen), len(ug), len(zb)), bool); excl = np.zeros_like(pos)
    for g in segs:
        w = (s_cen >= g["s0"]) & (s_cen <= g["s1"])
        uu = g["u0"] + g.get("wander_m", 0.0) * np.sin(2 * np.pi * s_cen[w] / g.get("wander_period_m", 15.0))
        iu = np.round((uu - ug[0]) / (ug[1] - ug[0])).astype(int); iz = int(np.clip(np.floor((g["z"] - zb[0]) / DZ), 0, len(zb) - 1))
        for t, j in zip(np.nonzero(w)[0], iu):
            pos[t, max(0, j - 1): j + 2, iz] = True
            excl[max(0, t - 4): t + 5, max(0, j - 10): j + 11, max(0, iz - 4): iz + 5] = True
    return pos, excl


def to_llr(p, prior):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.clip(np.log(p / (1 - p)) - np.log(prior / (1 - prior)), -4.0, 4.0)
