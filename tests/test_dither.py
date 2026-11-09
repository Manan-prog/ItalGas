"""Known-answer tests for the sideways-motion test (geometry/dither.py)."""
import numpy as np

from italgas_gpr.geometry.cube import to_su, track_frame
from italgas_gpr.geometry.dither import reference_line, slope


def test_slope_ground_and_van_fixed():
    rng = np.random.default_rng(0)
    s = np.arange(0, 40, 2.0); y = 0.3 * np.sin(2 * np.pi * s / 25)
    ground = 0.4 + 0.002 * s - y + rng.normal(0, 0.01, len(s))      # pipe fixed in the ground, slightly oblique
    van = 0.55 + rng.normal(0, 0.01, len(s))                          # artefact fixed to the instrument
    assert abs(slope(s, ground, y)["b"] + 1) < 0.1
    assert abs(slope(s, van, y)["b"]) < 0.1


def test_reference_line_recovers_lane_weave():
    """Straight east-going path with a 0.25 m, 30 m-period weave (left-positive = north, k = 1). The 60 m running mean
    averages two full periods, so the offset from R equals the weave; the quadratic smoother has a negative sidelobe
    there and returns 1.2 x the weave (same phase), so a ground-fixed pipe gives b about -1 / 1.2 = -0.83, still < -0.5."""
    s = np.arange(0, 600, 0.5); w = 0.25 * np.sin(2 * np.pi * s / 30)
    X, Y = s.copy(), w.copy(); m = (s > 60) & (s < 540)
    R = reference_line(s, X, Y, 1.0, 60.0, method="mean"); _, u = to_su(X, Y, R[0], R[1], R[2], 1.0)
    assert np.max(np.abs(u[m] - w[m])) < 0.01
    R = reference_line(s, X, Y, 1.0, 60.0, method="sg2"); _, u = to_su(X, Y, R[0], R[1], R[2], 1.0)
    gain = float(u[m] @ w[m] / (w[m] @ w[m]))
    assert 1.0 <= gain < 1.3 and np.corrcoef(u[m], w[m])[0, 1] > 0.99


def test_reference_line_follows_bend():
    """Constant-radius bend (r = 50 m, k = 1): the quadratic smoother follows it (offset < 0.05 m); the running mean
    cuts it by about L^2 / (24 r) = 3.0 m for L = 60 m."""
    r = 50.0; s = np.arange(0, 120, 0.25); X, Y = r * np.sin(s / r), r * (1 - np.cos(s / r))
    m = (s > 35) & (s < 85)
    R = reference_line(s, X, Y, 1.0, 60.0, method="sg2"); _, u = to_su(X, Y, R[0], R[1], R[2], 1.0)
    assert np.max(np.abs(u[m])) < 0.05
    R = reference_line(s, X, Y, 1.0, 60.0, method="mean"); _, u = to_su(X, Y, R[0], R[1], R[2], 1.0)
    assert abs(np.median(np.abs(u[m])) - 60.0 ** 2 / (24 * r)) < 0.3


def test_to_ground_frame_straightens_a_sliding_line():
    """A line at ground offset 0 under a van weaving +-5 cells: in the instrument frame it wanders, after the shift it is straight."""
    from italgas_gpr.along.tracks import to_ground_frame
    n_win, n_u, n_t = 40, 31, 3; y = np.round(5 * np.sin(np.arange(n_win) / 4)).astype(int)
    sc = np.zeros((n_win, n_u, n_t)); mid = n_u // 2
    for w in range(n_win): sc[w, mid - y[w], :] = 1.0           # instrument u = ground u - y_van
    g = to_ground_frame(sc, y, 6)
    assert np.all(g[:, mid + 6, :] == 1.0) and g.sum() == sc.sum()
