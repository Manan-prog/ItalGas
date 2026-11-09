"""Tests of modules moved to archive/pipeline (9 Nov 2026): run with archive/pipeline on sys.path."""
"""End-point detector: a half-line present on channels with u > u_end is found with the right side and end position."""
import numpy as np

from terminations import end_score


def test_end_taper_fresnel_half_zone():
    """Kirchhoff half-line: 0.5 at the end, ~1 well above the pipe, <= 0.2 at 0.3 m beyond (220 MHz, 0.5 m, 0.10 m/ns)."""
    from along_road import end_taper
    a = end_taper(np.array([-1.0, 0.0, 0.3, 1.0]), 0.5, 220.0, 0.10)
    assert abs(a[1] - 0.5) < 0.03 and 0.9 < a[0] < 1.15 and a[2] < 0.2 and a[3] < 0.05
