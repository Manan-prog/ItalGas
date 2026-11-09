"""Physics features that may separate a pipe from a soil layer (3 Nov 2026, ideas from the user):
polarity of the reflection relative to the channel's direct wave, side-lobe asymmetry (single
reflection vs top/bottom pair) and surface following (does a track's depth copy the shallow road
layers?). All on noise-equalised SIGNED traces (zero-phase filters keep polarity).
Reflection coefficient r = (sqrt(e1) - sqrt(e2)) / (sqrt(e1) + sqrt(e2)): soil (e about 9) to air-filled
pipe +0.5, to water about -0.5, to metal -1; a layer can have either sign."""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from processing import dewow, first_break_sample


def reference_sign(raw: np.ndarray, dt: float, dewow_ns: float, t0_frac: float = 0.2, win_ns: float = 3.0) -> int:
    """Sign of the largest lobe of the channel's median trace in the first win_ns after time zero
    (direct wave + surface reflection): the polarity reference of that channel."""
    x = dewow(raw.astype(np.float32), dt, dewow_ns); m = np.median(x, 0)
    n0 = first_break_sample(x.mean(0), t0_frac); seg = m[n0: n0 + max(2, int(win_ns / dt))]
    return int(np.sign(seg[np.argmax(np.abs(seg))]))


def reflection_shape(w: np.ndarray, dt: float, period_ns: float) -> dict:
    """Main-lobe sign and side-lobe asymmetry of a signed waveform w (n_t,) around its envelope peak.
    asymmetry = (|larger side lobe| - |smaller|) / (|larger| + |smaller|): about 0 for one reflection,
    larger for a top/bottom pair or overlapping reflectors."""
    e = np.abs(hilbert(w)); p = int(np.argmax(e)); h = max(1, int(0.5 * period_ns / dt))
    seg = w[max(0, p - h): p + h + 1]; m = int(np.argmax(np.abs(seg))) + max(0, p - h); sgn = np.sign(w[m])
    L = w[max(0, m - 2 * h): m]; R = w[m + 1: m + 2 * h + 1]
    lo = np.abs(L[np.sign(L) == -sgn]).max() if np.any(np.sign(L) == -sgn) else 0.0
    ro = np.abs(R[np.sign(R) == -sgn]).max() if np.any(np.sign(R) == -sgn) else 0.0
    a = abs(lo - ro) / max(lo + ro, 1e-9)
    return dict(sign=int(sgn), peak_t_ns=float(m * dt), asymmetry=float(a), main=float(abs(w[m])))


def surface_following(Zc: np.ndarray, i_s: np.ndarray, t_track_ns: np.ndarray, dt: float, shallow_ns=(3.0, 10.0)) -> float:
    """Correlation between a track's two-way time profile and the time of the strongest shallow echo
    (pavement/base, shallow_ns) above it, both linearly detrended. Zc (n_s, n_t) one channel."""
    i0, i1 = int(shallow_ns[0] / dt), int(shallow_ns[1] / dt)
    E = np.abs(hilbert(Zc[i_s][:, i0:i1], axis=1)); p = (np.argmax(E, 1) + i0) * dt
    if np.ptp(t_track_ns) < 1e-6 or np.ptp(p) < 1e-6 or len(i_s) < 10:
        return float("nan")
    x = np.arange(len(i_s)); d1 = t_track_ns - np.polyval(np.polyfit(x, t_track_ns, 1), x); d2 = p - np.polyval(np.polyfit(x, p, 1), x)
    return float(np.corrcoef(d1, d2)[0, 1])
