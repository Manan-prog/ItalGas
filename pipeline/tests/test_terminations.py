"""End-point detector: a half-line present on channels with u > u_end is found with the right side and end position."""
import numpy as np

from terminations import end_score


def test_half_line_end_found():
    rng = np.random.default_rng(0); u = np.arange(19) * 0.08 - 0.6; n_s, n_t, ds = 400, 40, 0.05
    G = rng.normal(0, 0.3, (19, n_s, n_t)).astype(np.float32)
    G[u > 0.1, 200, 20] += 6.0                                   # pipe square across, present only for u > 0.1 m
    sc, ph, sp, sd, uo = end_score([G], u, ds, np.array([0.0]), np.array([0]), min_side=4, tol=(1, 1), flank_m=(0.6, 1.5))
    i, j = np.unravel_index(np.argmax(sc), sc.shape)
    assert (i, j) == (200, 20) and sd[i, j] == 1
    m = sp[i, j]; assert abs(0.5 * (uo[m - 1] + uo[m]) - 0.1) <= 0.08


def test_full_line_scores_low():
    u = np.arange(19) * 0.08 - 0.6; G = np.zeros((19, 400, 40), np.float32); G[:, 200, 20] = 6.0   # full crossing
    sc = end_score([G], u, 0.05, np.array([0.0]), np.array([0]), min_side=4, tol=(1, 1), flank_m=(0.6, 1.5))[0]
    assert sc[200, 20] <= 1e-6
