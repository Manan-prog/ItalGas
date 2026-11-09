"""Known-answer tests for the monotone local-FDR curve of along/evidence.py (correction of 4 Nov 2026)."""
import sys
from pathlib import Path

import numpy as np

from italgas_gpr.along.evidence import isotonic_decreasing, lfdr_curve, search_samples  # noqa: E402


def test_isotonic_already_monotone_unchanged():
    y = np.array([1.0, 0.8, 0.5, 0.5, 0.1]); w = np.ones(5)
    assert np.allclose(isotonic_decreasing(y, w), y)


def test_isotonic_pools_violators_by_weight():
    # a light bin (weight 1) above a heavy bin (weight 9): the pooled value is the weighted mean 0.19, not the max
    y = np.array([0.9, 0.1, 1.0]); w = np.array([1.0, 9.0, 1.0])
    out = isotonic_decreasing(y, w)
    assert np.all(np.diff(out) <= 1e-12)
    assert np.isclose(out[0], 0.9) and np.isclose(out[1], (0.1 * 9 + 1.0 * 1) / 10) and np.isclose(out[2], out[1])


def test_lfdr_not_killed_by_sparse_extreme_null_tail():
    """Observed scores have a real excess at moderate scores; the null has a few cells far above every observed cell.
    The old running-max rule gave lfdr = 1 everywhere; the isotonic rule keeps lfdr about f0/f at the excess."""
    rng = np.random.default_rng(0)
    obs = np.r_[rng.normal(0, 1, 200000), rng.normal(3, 0.3, 20000)]          # excess at 3
    null = np.r_[rng.normal(0, 1, 4 * 200000), rng.uniform(8, 9, 20)]          # 4 shuffles, 20 extreme cells
    edges, lf = lfdr_curve(obs, null, 4)
    c = 0.5 * (edges[1:] + edges[:-1])
    assert np.all(np.diff(lf) <= 1e-12)                                        # non-increasing
    at3 = lf[np.searchsorted(c, 3.0)]
    assert at3 < 0.2, at3                                                      # excess cells are recognised
    assert lf[np.searchsorted(c, 0.0)] > 0.9                                   # bulk is chance-level


def test_search_samples_applies_floor_and_piece_window():
    """Piece window 3.9-20 ns at dt 0.2 ns; floor 6.5 ns -> first kept sample is the first t >= 6.5 (index 33), last t <= 20."""
    t = np.arange(120) * 0.2; d = dict(t_ns=t, t_search=np.array([3.9, 20.0]))
    it = search_samples(d, 6.5)
    assert it[0] == 33 and t[it[0]] >= 6.5 and t[it[0] - 1] < 6.5 and t[it[-1]] <= 20.0 and t[it[-1] + 1] > 20.0
    assert search_samples(d, 0.0)[0] == 20                                                 # no floor: the piece's own window start
