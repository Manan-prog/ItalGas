"""Known-answer tests for the raster line rule (along/lines.py): linking, block permutation, chance reference."""
import sys
from pathlib import Path

import numpy as np

from italgas_gpr.along import lines as rl  # noqa: E402


def peaks(cells, p=0.95):
    keys = np.array(cells, int); o = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    return dict(keys=keys[o], vals=np.full(len(keys), p)[o], dt=0.2, n_win=int(keys[:, 0].max()) + 1, ug=np.arange(35) * 0.04)


def test_straight_run_is_one_line_and_short_run_is_dropped():
    pk = peaks([(w, 10, 50) for w in range(30)] + [(w, 25, 80) for w in range(40, 50)])   # 14.5 m run and a 4.5 m run
    runs = rl.link_runs(pk, 10.0)
    assert len(runs) == 1 and runs[0][2] == 14.5 and len(runs[0][0]) == 30


def test_gate_breaks_a_run_that_jumps_sideways():
    pk = peaks([(w, 10 if w < 15 else 20, 50) for w in range(30)])                         # 0.4 m jump at window 15 > 0.12 m gate
    runs = rl.link_runs(pk, 10.0)
    assert len(runs) == 0                                                                  # two 7 m halves, both below 10 m


def test_depth_gate_allows_slow_drift_and_blocks_a_jump():
    slow = peaks([(w, 10, 50 + w // 2) for w in range(30)])                                # 1 sample per metre = 0.02 m/m
    jump = peaks([(w, 10, 50 if w < 15 else 80) for w in range(30)])                       # 30 samples = 6 ns jump
    assert len(rl.link_runs(slow, 10.0)) == 1 and len(rl.link_runs(jump, 10.0)) == 0


def test_block_perm_is_a_permutation_that_keeps_blocks():
    rng = np.random.default_rng(0); perm = rl.block_perm(37, 8, rng)
    assert sorted(perm) == list(range(37))
    for b in range(0, 32, 8):                                                              # whole blocks stay contiguous and in order
        assert np.all(np.diff(perm[b:b + 8]) == 1)


def test_permuted_windows_break_a_drifting_run():
    """A 24-window run drifting 1 cell per window (within the 3-cell gate) in a 200-window drive: real = one 11.5 m line;
    after block shuffling the blocks land apart or out of order (an 8-cell jump breaks the gate), so no 10 m line remains."""
    pk = peaks([(w, 10 + (w - 100), 50) for w in range(100, 124)] + [(199, 30, 60)]); rng = np.random.default_rng(1)
    strict = dict(gate_u_m=0.12, max_gap=1)                                                # the 4 Nov rule; the config now allows 8 gaps
    real = rl.link_runs(pk, 10.0, rule=strict); assert len(real) == 1 and real[0][2] == 11.5
    nulls = [rl.link_runs(pk, 10.0, rl.block_perm(pk["n_win"], 8, rng), rule=strict) for _ in range(20)]
    assert sum(len(n) for n in nulls) == 0


def test_merge_runs_joins_depth_stacked_runs_and_keeps_separate_ones():
    """Three runs: A at lateral 10 and time 50, B at lateral 11 and time 70 on the same windows (a multiple of A),
    C at lateral 20 (0.4 m away). A and B merge into one line drawn at the shallower time; C stays separate."""
    A = (np.array([(w, 10, 50) for w in range(30)]), np.full(30, 0.6), 14.5)
    B = (np.array([(w, 11, 70) for w in range(5, 35)]), np.full(30, 0.9), 14.5)
    C = (np.array([(w, 20, 50) for w in range(30)]), np.full(30, 0.7), 14.5)
    m = rl.merge_runs([A, B, C], 2, 0.5)
    assert len(m) == 2
    ab = [x for x in m if len(x[3]) == 2][0]; cells, P, length, members = ab
    assert length == 17.0 and cells[:, 0].tolist() == list(range(35))                   # union of windows 0..34
    assert (cells[5:30, 2] == 50).all() and (cells[30:, 2] == 70).all()                  # shallowest time where both exist
    assert P[5:30].max() == 0.9 and P[:5].max() == 0.6 and [mm[0] for mm in members] == [50.0, 70.0]
    assert rl.merge_runs([], 2, 0.5) == []
