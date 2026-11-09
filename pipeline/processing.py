"""Trace-domain processing for IDS Stream UP B-scans. All functions are pure,
vectorised over (n_scans, n_samples) float32 arrays; sample axis is last.
Units: time in ns, distance in m, velocity in m/ns."""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d


def dewow(x: np.ndarray, dt_ns: float, win_ns: float = 10.0) -> np.ndarray:
    """Subtract a running mean along time (removes low-frequency 'wow' and DC)."""
    w = max(3, int(round(win_ns / dt_ns)) | 1)
    return x - uniform_filter1d(x, w, axis=-1, mode="nearest")


def first_break_sample(mean_trace: np.ndarray, frac: float = 0.2) -> int:
    """Time-zero estimate: first sample where |mean trace| exceeds frac of its max.
    Derived quantity; the direct wave onset is used as t = 0 (ASSUMED: onset of the
    TX-RX direct wave, offset 0.08 m, is within one sample of true t0)."""
    a = np.abs(mean_trace)
    return int(np.argmax(a > frac * a.max()))


def shift_time_zero(x: np.ndarray, n0: int) -> np.ndarray:
    """Drop the first n0 samples (time zero alignment) and zero-pad at the end."""
    out = np.zeros_like(x)
    out[..., : x.shape[-1] - n0] = x[..., n0:]
    return out


def background_removal(x: np.ndarray, win_traces: int | None = None) -> np.ndarray:
    """Subtract the mean trace (global if win_traces is None, else a moving window
    along the scan axis). Removes direct wave, ringing and flat layers."""
    if win_traces is None:
        return x - x.mean(axis=0, keepdims=True)
    w = int(win_traces) | 1
    return x - uniform_filter1d(x, w, axis=0, mode="nearest")


def bandpass(x: np.ndarray, dt_ns: float, f_lo_mhz: float, f_hi_mhz: float, order: int = 4) -> np.ndarray:
    fs = 1e3 / dt_ns  # MHz
    sos = signal.butter(order, [f_lo_mhz, f_hi_mhz], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def sec_gain(x: np.ndarray, dt_ns: float, alpha_db_per_ns: float = 0.0, power: float = 1.0) -> np.ndarray:
    """Spreading-and-exponential-compensation gain g(t) = t^power * 10^(alpha t / 20)."""
    t = np.arange(x.shape[-1]) * dt_ns
    g = (t ** power) * 10 ** (alpha_db_per_ns * t / 20)
    g = g / g[g > 0].mean()
    return x * g.astype(np.float32)


def agc(x: np.ndarray, dt_ns: float, win_ns: float = 10.0, eps: float = 1e-3) -> np.ndarray:
    """Automatic gain control: divide by the running RMS over win_ns."""
    w = max(3, int(round(win_ns / dt_ns)) | 1)
    rms = np.sqrt(uniform_filter1d(x.astype(np.float32) ** 2, w, axis=-1, mode="nearest"))
    return x / (rms + eps * rms.mean())


def envelope(x: np.ndarray) -> np.ndarray:
    """Instantaneous amplitude (Hilbert envelope) along the time axis."""
    return np.abs(signal.hilbert(x, axis=-1)).astype(np.float32)


def stolt_migrate(x: np.ndarray, dx_m: float, dt_ns: float, v_m_ns: float, pad: int = 2) -> np.ndarray:
    """2D zero-offset Stolt (f-k) migration of a B-scan x (n_traces, n_samples)
    for constant velocity v (m/ns). Returns the migrated image, same shape.
    Uses the exploding-reflector model with half velocity v/2 (two-way time)."""
    nx, nt = x.shape
    nxp, ntp = pad * nx, pad * nt
    X = np.fft.fft2(x, s=(nxp, ntp))
    kx = 2 * np.pi * np.fft.fftfreq(nxp, d=dx_m)          # rad/m
    w = 2 * np.pi * np.fft.fftfreq(ntp, d=dt_ns)          # rad/ns
    vh = v_m_ns / 2.0
    KX, W = np.meshgrid(kx, w, indexing="ij")
    # Stolt mapping: for output (kx, kz) with kz = sqrt((w/vh)^2 - kx^2); here we
    # interpolate the input spectrum at w' = vh*sqrt(kz^2 + kx^2) where kz = w/vh.
    kz = W / vh
    w_src = vh * np.sign(kz) * np.sqrt(kz ** 2 + KX ** 2)
    # Jacobian |dw'/dkz| = vh * kz / sqrt(kz^2+kx^2)
    jac = np.where(np.abs(w_src) > 0, vh * np.abs(kz) / np.maximum(np.abs(w_src / vh), 1e-12), 0.0)
    out = np.zeros_like(X)
    dw = w[1] - w[0]
    idx = w_src / dw                                      # fractional index into w axis
    idx = np.mod(idx, ntp)
    i0 = np.floor(idx).astype(int); fr = idx - i0; i1 = (i0 + 1) % ntp
    rows = np.arange(nxp)[:, None]
    out = (1 - fr) * X[rows, i0] + fr * X[rows, i1]
    out *= jac
    out[np.abs(w_src) > np.abs(w).max()] = 0
    img = np.fft.ifft2(out)[:nx, :nt].real
    return img.astype(np.float32)


def kirchhoff_migrate(x: np.ndarray, dx_m: float, dt_ns: float, v_m_ns: float,
                      aperture_m: float = 1.5, t0_ns: float = 0.0) -> np.ndarray:
    """2D zero-offset Kirchhoff (diffraction-stack) migration, vectorised over
    traces: for each lateral offset h within the aperture, shift every trace's
    samples from t_h = sqrt(t^2 + (2h/v)^2) to t and sum. Returns (n_traces, n_samples)."""
    nx, nt = x.shape
    t = np.arange(nt) * dt_ns - t0_ns
    t = np.maximum(t, 0)
    nh = int(aperture_m / dx_m)
    out = np.zeros_like(x, dtype=np.float32)
    cnt = np.zeros(nt, dtype=np.float32)
    for h in range(-nh, nh + 1):
        th = np.sqrt(t ** 2 + (2 * h * dx_m / v_m_ns) ** 2) + t0_ns
        idx = th / dt_ns
        i0 = np.floor(idx).astype(int)
        valid = i0 < nt - 1
        i0 = np.clip(i0, 0, nt - 2); fr = (idx - i0).astype(np.float32)
        # obliquity / amplitude weight: cos(theta) = t / th
        wgt = np.where(th > 0, (t + t0_ns) / np.maximum(th, 1e-6), 1.0).astype(np.float32) * valid
        shifted = np.roll(x, -h, axis=0)                   # trace at offset h
        samp = (1 - fr) * shifted[:, i0] + fr * shifted[:, i0 + 1]
        out += samp * wgt
        cnt += wgt
    return out / np.maximum(cnt, 1e-6)


def focus_metric(img: np.ndarray) -> float:
    """Migration focusing measure: normalised kurtosis of the envelope (higher = more focused).
    Reference: velocity analysis by migration focusing (see REFERENCES.md)."""
    e = envelope(img)
    e = e - e.mean()
    return float((e ** 4).mean() / (e ** 2).mean() ** 2)


def align_traces(x: np.ndarray, ref: np.ndarray, n0: int, max_lag: int = 6, win: int = 60) -> np.ndarray:
    """Per-trace integer time shift that maximises the cross-correlation of the early
    window [n0 - 10, n0 + win) of each trace with the reference (mean) trace.
    Compensates antenna-height variation (vehicle bounce) that otherwise leaves
    scan-wide residuals after background removal. Returns the aligned array."""
    a0 = max(n0 - 10, max_lag)
    r = ref[a0: a0 + win]
    scores = np.stack([(x[:, a0 + lag: a0 + lag + win] * r).sum(1) for lag in range(-max_lag, max_lag + 1)], axis=1)
    lag = scores.argmax(1) - max_lag                                      # (n_traces,)
    idx = np.arange(x.shape[1])[None, :] + lag[:, None]
    idx = np.clip(idx, 0, x.shape[1] - 1)
    return np.take_along_axis(x, idx, axis=1)


# ---------------------------------------------------------------------------
# v2 preprocessing (3 Nov 2026): statistics taken over ALL traces of a channel
# ---------------------------------------------------------------------------

def median_background(x: np.ndarray) -> np.ndarray:
    """Subtract the median trace over all traces (n_traces, n_samples). Removes the
    antenna ringing and direct wave like the mean trace, but a strong reflector in a
    minority of traces does not leak into the estimate (Fathom horseshoe DSP choice)."""
    return x - np.median(x, axis=0, keepdims=True)


def robust_sigma_t(x: np.ndarray) -> np.ndarray:
    """Robust standard deviation across traces at every time sample: 1.4826 * MAD.
    x: (n_traces, n_samples) -> (n_samples,). Same units as x."""
    med = np.median(x, axis=0, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=0)


def noise_floor(sigma_t: np.ndarray, late_frac: tuple[float, float] = (0.80, 0.96)) -> float:
    """Receiver noise floor = median of sigma(t) in the late part of the record, where
    sigma(t) has stopped decaying (MEASURED on the dev swath: flat beyond about 35 ns at
    600 MHz and 45 ns at 200 MHz). The pre-time-zero samples are NOT quiet on this
    instrument (2 to 33 times the late level), so they are not used."""
    n = len(sigma_t)
    return float(np.median(sigma_t[int(late_frac[0] * n): int(late_frac[1] * n)]))


def spectral_wiener_weights(x: np.ndarray, sig_win: tuple[int, int], noise_win: tuple[int, int],
                            n_fft: int) -> np.ndarray:
    """Zero-phase Wiener weights W(f) = max(S - N, 0) / S from the mean power spectra of
    an early (signal + clutter) and a late (receiver noise) window, averaged over ALL
    traces. x: (n_traces, n_samples); windows in samples (PSD normalised by taper energy). Returns
    (n_fft // 2 + 1,) weights in [0, 1] for np.fft.rfft(., n_fft)."""
    def psd(a, b):
        w = np.hanning(b - a)
        return (np.abs(np.fft.rfft(x[:, a:b] * w, n_fft, axis=1)) ** 2).mean(0) / (w ** 2).sum()
    S = psd(*sig_win)
    N = psd(*noise_win)
    return np.clip((S - N) / np.maximum(S, 1e-12), 0.0, 1.0)


def apply_spectral_weights(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Filter every trace with real (zero-phase) spectral weights from spectral_wiener_weights
    after interpolating them to this trace length."""
    n = x.shape[-1]
    nf = 2 * (len(weights) - 1)
    f_w = np.fft.rfftfreq(nf); f_x = np.fft.rfftfreq(n)
    wx = np.interp(f_x, f_w, weights)
    return np.fft.irfft(np.fft.rfft(x, axis=-1) * wx, n, axis=-1).astype(np.float32)


def noise_equalising_gain(x: np.ndarray, dt_ns: float, smooth_ns: float = 2.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Data-driven depth gain: divide each time sample by the robust spread of all
    traces at that time (smoothed over smooth_ns). Output is dimensionless: 'times the
    typical level at this depth', so one detection threshold means the same at every
    depth. Returns (z, sigma_t, floor) with sigma_t in input units and
    floor = receiver noise floor (input units). sigma_t / floor is the measured
    signal-to-noise depth profile of the channel."""
    from scipy.ndimage import gaussian_filter1d
    s = robust_sigma_t(x)
    s = gaussian_filter1d(s, max(1.0, smooth_ns / dt_ns), mode="nearest")
    fl = noise_floor(s)
    s = np.maximum(s, 0.5 * fl)
    return (x / s[None, :]).astype(np.float32), s, fl


def robust_z(e: np.ndarray, axis: int = 0) -> np.ndarray:
    """(e - median) / (1.4826 MAD) along axis (default: across traces, per time sample)."""
    med = np.median(e, axis=axis, keepdims=True)
    mad = 1.4826 * np.median(np.abs(e - med), axis=axis, keepdims=True)
    return ((e - med) / np.maximum(mad, 1e-9)).astype(np.float32)
