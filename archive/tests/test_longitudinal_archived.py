"""Tests of modules moved to archive/pipeline (9 Nov 2026): run with archive/pipeline on sys.path."""
"""Known-answer tests for the longitudinal detector."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from longitudinal import lateral_contrast, lateral_migrate, link_tracks, stack_along_s, window_peaks  # noqa: E402

U = np.linspace(0.93, -0.58, 19)
DT, V = 0.196, 0.10


def test_injected_pipe_is_recovered_by_match():
    from along_road import inject_longitudinal, match
    rng = np.random.default_rng(3)
    s = np.arange(0, 60, 0.05); t = np.arange(200) * DT
    Z = rng.normal(0, 1, (len(U), len(s), 200)).astype(np.float32)
    seg = dict(s0=20.0, s1=40.0, u0=0.3, z=1.0, amp=3.0)
    Z = inject_longitudinal(Z, U, s, DT, V, [seg], 300.0)
    X, centres = stack_along_s(Z, 0.05, 2.0, 0.5)
    ug = np.arange(-0.5, 0.86, 0.04)
    sc = lateral_contrast(lateral_migrate(X, U, ug, DT, V))
    pk = window_peaks(sc, np.percentile(sc, 99.5), 3, 10, (5, 195))
    tr = link_tracks(pk, sc, 3, 8)
    tracks = [dict(s_start=float(s[centres[x[:, 0]]].min()), s_end=float(s[centres[x[:, 0]]].max()),
                   u_mean=float(ug[x[:, 1]].mean()), depth_m=float(V * np.median(x[:, 2] * DT) / 2), q=0.0) for x in tr if len(x) > 5]
    assert match(tracks, [dict(seg, z=1.0)], 0.05, 0.15, 0.15, 0.5)[0]


def test_pipe_shape_test_separates_hyperbola_from_plane():
    from along_road import pipe_shape_test
    u = np.arange(-1.5, 1.5, 0.08); z = np.arange(0, 2, 0.01)
    Xp = np.exp(-((z[:, None] - np.sqrt(0.7 ** 2 + (u[None, :] - 0.1) ** 2)) / 0.04) ** 2)   # pipe at u 0.1, 0.7 m
    Xl = np.exp(-((z[:, None] - (0.7 + 0.2 * u[None, :])) / 0.04) ** 2)                      # dipping layer
    assert pipe_shape_test(Xp, u, z, 0.0, 0.75)["verdict"] == "pipe-like"
    assert pipe_shape_test(Xl, u, z, 0.0, 0.75)["verdict"] == "layer-like"


def test_viterbi_follows_a_drifting_line_and_ignores_noise():
    from network_dp import viterbi_paths
    rng = np.random.default_rng(0)
    llr = rng.normal(-1, 0.5, (200, 40))
    uu = (20 + 3 * np.sin(np.arange(60, 160) / 15)).astype(int)
    llr[np.arange(60, 160), uu] = 3.0
    segs = viterbi_paths(llr)
    best = max(segs, key=lambda x: x[3])
    assert abs(best[0] - 60) <= 3 and abs(best[1] - 160) <= 3 and np.mean(np.abs(best[2] - uu[best[0] - 60: best[1] - 60])) < 1.0


def test_viterbi_anchored_follows_from_the_pin():
    from network_dp import viterbi_anchored
    rng = np.random.default_rng(1)
    llr = rng.normal(-0.5, 0.5, (120, 40)); uu = (10 + np.arange(120) / 10).astype(int)   # pipe drifting from cell 10 to 21
    llr[np.arange(120), uu] += 2.5
    llr[:, 30] += 1.2                                                                        # competing layer at cell 30
    p = viterbi_anchored(llr, 10, 2.0)
    assert abs(p[-1] - uu[-1]) <= 1


def test_pipe_radius_fit_recovers_radius_and_flags_layers():
    from along_road import pipe_radius_fit
    u = np.arange(-1.5, 1.5, 0.04); z = np.arange(0, 2, 0.005)
    def pipe(R, zt=0.7, uc=0.1):
        return np.exp(-((z[:, None] - (np.sqrt((zt + R) ** 2 + (u[None, :] - uc) ** 2) - R)) / 0.03) ** 2)
    r1 = pipe_radius_fit(pipe(0.1), u, z, 0.0, 0.72)
    assert r1["verdict"] == "pipe" and abs(r1["best_radius_m"] - 0.1) <= 0.05 and abs(r1["uc"] - 0.1) <= 0.06
    lay = np.exp(-((z[:, None] - (0.7 + 0.15 * u[None, :])) / 0.03) ** 2)
    assert pipe_radius_fit(lay, u, z, 0.0, 0.72)["verdict"] in ("layer", "large pipe or confined layer")


def test_viterbi_uz_finds_two_pipes_at_different_depths():
    from network_dp import viterbi_uz_multi
    rng = np.random.default_rng(2)
    llr = rng.normal(-1.0, 0.5, (200, 30, 15))
    llr[20:120, 8, 4] += 4.0; llr[90:190, 22, 10] += 4.0
    segs = viterbi_uz_multi(llr, k_paths=3)
    found = {(int(np.median(s[2])), int(np.median(s[3]))) for s in segs if s[1] - s[0] > 60}
    assert (8, 4) in found and (22, 10) in found
