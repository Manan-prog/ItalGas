"""Known-answer tests for the cross-column crossing detector."""
import sys
from pathlib import Path

import numpy as np

from italgas_gpr.crossing.score import crossing_score, empirical_fdr, local_maxima, phi_velocity_index  # noqa: E402

U = np.array([0.93, 0.77, 0.59, 0.43, 0.25, 0.09, -0.08, -0.24, -0.42, -0.58])   # 600 MHz channel u (m)
DS, DT = 0.05, 0.47


def _synthetic(phi_deg=30.0, s0=50.0, it=20, amp=4.0, seed=0):
    rng = np.random.default_rng(seed)
    n_s, n_t = 2000, 60
    G = rng.normal(0, 1, (len(U), n_s, n_t)).astype(np.float32)
    for c, u in enumerate(U):
        i = int(round((s0 + u * np.tan(np.radians(phi_deg))) / DS))
        G[c, i - 1: i + 2, it - 1: it + 2] += amp
    return G


def test_oblique_pipe_found_at_right_place_and_angle():
    G = _synthetic()
    phis = np.arange(-60, 61, 5.0)
    sc, arg = crossing_score([G], U, DS, phis, np.zeros(len(phis), int))
    i, j = np.unravel_index(np.argmax(sc), sc.shape)
    assert abs(i * DS - 50.0) <= 0.1 and abs(j - 20) <= 1
    assert phis[arg[i, j]] == 30.0


def test_single_channel_stripe_is_not_a_crossing():
    rng = np.random.default_rng(1)
    G = rng.normal(0, 1, (len(U), 2000, 60)).astype(np.float32)
    G[3, :, 30] += 10.0                     # artefact locked to one channel, all along s
    phis = np.arange(-60, 61, 5.0)
    sc, _ = crossing_score([G], U, DS, phis, np.zeros(len(phis), int))
    # trimmed mean ignores the hot channel: the stripe row scores like pure-noise rows
    assert sc[:, 29:32].mean() - sc[:, 5:25].mean() < 0.3


def test_null_shift_destroys_the_pipe():
    G = _synthetic(amp=6.0)
    phis = np.arange(-60, 61, 5.0)
    sc, _ = crossing_score([G], U, DS, phis, np.zeros(len(phis), int))
    shifts = np.random.default_rng(2).integers(100, 1900, len(U))
    sn, _ = crossing_score([G], U, DS, phis, np.zeros(len(phis), int), shifts=shifts)
    assert sc.max() > sn.max() + 1.0


def test_velocity_index_and_fdr():
    assert list(phi_velocity_index(np.array([0, 30, 45, 60.0]), 0.10, [0.10, 0.115, 0.14, 0.20])) == [0, 1, 2, 3]
    data = np.r_[np.random.default_rng(3).normal(0, 1, 1000), np.full(20, 8.0)]
    null = np.random.default_rng(4).normal(0, 1, 3000)
    q, p = empirical_fdr(data, null, 3)
    assert q[-1] < 0.01 and p[-1] > 0.95 and np.median(p[:1000]) < 0.2
    assert np.median(q[:1000]) > 0.5            # noise candidates must NOT inherit the strong ones' q


def test_flank_contrast_rejects_broad_layers():
    rng = np.random.default_rng(5)
    G = rng.normal(0, 1, (len(U), 2000, 60)).astype(np.float32)
    G[:, 800:1100, 40] += 5.0               # 15 m long laterally uniform layer (not a pipe)
    G[:, 1500:1503, 20] += 5.0              # thin perpendicular pipe
    phis = np.array([0.0])
    sc, _ = crossing_score([G], U, DS, phis, np.zeros(1, int))
    assert sc[1500:1503, 20].max() > 3.0 and sc[850:1050, 40].max() < 2.0
