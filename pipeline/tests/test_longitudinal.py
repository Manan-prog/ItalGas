"""Known-answer tests for the longitudinal detector."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longitudinal import lateral_contrast, lateral_migrate, link_tracks, stack_along_s, window_peaks  # noqa: E402

U = np.linspace(0.93, -0.58, 19)
DT, V = 0.196, 0.10


def _ricker(t, t0, f=0.3):
    a = (np.pi * f * (t - t0)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def test_pipe_along_track_focuses_at_its_u_and_depth():
    rng = np.random.default_rng(0)
    n_s, n_t = 400, 200
    t = np.arange(n_t) * DT
    Z = rng.normal(0, 1, (len(U), n_s, n_t)).astype(np.float32)
    u0, z = 0.30, 1.0
    for c, u in enumerate(U):
        Z[c, 100:300] += 3 * _ricker(t, 2 * np.hypot(z, u - u0) / V)[None, :]
    X, centres = stack_along_s(Z, 0.05, 2.0, 0.5)
    ug = np.arange(-0.5, 0.86, 0.04)
    sc = lateral_contrast(lateral_migrate(X, U, ug, DT, V))
    w = np.argmin(np.abs(centres - 200))
    iu, it = np.unravel_index(np.argmax(sc[w]), sc[w].shape)
    assert abs(ug[iu] - u0) <= 0.08 and abs(it * DT - 2 * z / V) <= 1.0
    pk = window_peaks(sc, np.percentile(sc, 99.9), 3, 10, (5, n_t - 5))
    tr = link_tracks(pk, sc, 3, 8)
    longest = max(tr, key=len)
    assert len(longest) >= 10                      # 10 m of pipe, hop 0.5 m


def test_flat_layer_is_removed():
    n_t = 200; t = np.arange(n_t) * DT
    Z = np.zeros((len(U), 100, n_t), np.float32) + 3 * _ricker(t, 15.0)[None, None, :]
    X, _ = stack_along_s(Z, 0.05, 2.0, 0.5)
    ug = np.arange(-0.5, 0.86, 0.04)
    sc = lateral_contrast(lateral_migrate(X, U, ug, DT, V))
    assert np.abs(sc[:, 5:-5]).max() < 0.2 * lateral_migrate(X, U, ug, DT, V).max()


def test_dipping_layer_scores_below_point_diffractor():
    from longitudinal import point_vs_line_contrast
    env = np.zeros((1, 36, 150), np.float32)
    for iu in range(36):                     # dipping layer: t index = 40 + iu
        env[0, iu, 40 + iu] = 5.0
    env[0, 18, 120] = 5.0                    # compact spot (pipe)
    sc = point_vs_line_contrast(env)
    assert sc[0, 18, 120] > 4.0 and sc[0, 18, 58] < 1.0














def test_lateral_migrate_radius_zero_matches_point_model():
    from longitudinal import lateral_migrate, lateral_migrate_radius
    rng = np.random.default_rng(3); X = rng.normal(0, 1, (5, len(U), 120)).astype(np.float32); ug = np.arange(-0.5, 0.86, 0.04)
    a = lateral_migrate(X, U, ug, DT, V); b = lateral_migrate_radius(X, U, ug, DT, V, 0.0)
    assert np.allclose(a, b, atol=1e-4)
