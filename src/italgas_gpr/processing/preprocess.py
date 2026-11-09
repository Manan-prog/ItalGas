"""Stage 1 of the v2 detector: per-channel v2 preprocessing, along-track Kirchhoff
migration at several apparent velocities, envelope, robust z, and gridding of every
channel onto one along-track axis. Run: python -m italgas_gpr.processing.preprocess configs/detector.yaml (then dev_swath_v2.yaml, archived)
Output (local cache): v2grid_<arr>_<hash>.npz with
  G (n_v, n_ch, n_s, n_t) float16  migrated envelope, robust z per time sample
  Z (n_ch, n_s, n_t_full) float16  unmigrated noise-equalised signed traces
  s_grid (m), u_ch (m), dt_ns (of G), dt_full_ns, snr (n_ch, n_t_full) = sigma(t)/floor,
  wiener (n_f,), burst_frac."""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from italgas_gpr.io.paths import config_path, load_yaml
from scipy.ndimage import median_filter

from italgas_gpr.crossing.score import to_s_grid
from italgas_gpr.io.ids_reader import Swath
from italgas_gpr.processing.filters import (align_traces, apply_spectral_weights, dewow, envelope, first_break_sample,
                        kirchhoff_migrate, median_background, noise_equalising_gain, robust_z,
                        shift_time_zero, spectral_wiener_weights)


def load_cfg(path: str) -> dict:
    cfg = load_yaml(config_path(path))
    base = load_yaml(config_path(cfg["base_config"]))
    cfg["base"] = base
    return cfg


def cfg_hash(cfg: dict, keys=("base", "preprocess", "migration", "grid", "cube_file")) -> str:
    sub = {k: cfg[k] for k in keys}
    return hashlib.md5(yaml.safe_dump(sub, sort_keys=True).encode()).hexdigest()[:10]


def preprocess_channel(x: np.ndarray, dt: float, arr: int, cfg: dict, signature: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Raw traces (n_scans, 512) -> time-zeroed, background-removed traces (float32, full 512 samples). Returns (x, n0).
    Background (cfg preprocess.background): 'median' = median trace of these scans (removes the antenna ringing and
    flat layers, but also any echo constant along the drive, e.g. a pipe running along it); 'signature' = subtract
    the given instrument signature (512,) of this channel (median over a whole day of driving, instrument_signature);
    'none' = keep."""
    b = cfg["base"]["processing"]
    x = dewow(x.astype(np.float32), dt, b["dewow_win_ns"][arr])
    n0 = first_break_sample(x.mean(0), b["t0_frac"])
    x = align_traces(x, x.mean(0), n0)
    x = shift_time_zero(x, n0)
    mode = cfg["preprocess"].get("background", "median")
    if mode == "median":
        x = median_background(x)
    elif mode == "signature":
        x = x - signature[None, :].astype(np.float32)
    return x, n0


def instrument_signature(files: list[str], arr: int, cfg: dict, step: int = 100) -> np.ndarray:
    """Per-channel instrument signature (n_ch, 512): median of time-zeroed, aligned traces over every step-th scan of
    all the given drives (one survey day). A buried object cannot sit under the same channel at the same time for
    most of a day's driving across many streets, so this keeps antenna ringing and direct wave only. Cached."""
    cache = Path(cfg["base"]["cache_dir"])
    h = hashlib.md5(yaml.safe_dump(dict(files=sorted(files), arr=arr, step=step, p=cfg["base"]["processing"]), sort_keys=True).encode()).hexdigest()[:10]
    f = cache / f"signature_{arr}_{h}.npz"
    if f.exists():
        return np.load(f)["sig"]
    acc = []
    for path in files:
        sw = Swath.load(str(Path(path).parent), Path(path).stem)
        if arr not in sw.data or len(sw.index[arr]) < 20 * step:
            continue
        chans = sw.config.channels_of(arr); dt = chans[0].dt_ns
        no_bg = dict(cfg, preprocess=dict(cfg["preprocess"], background="none"))
        acc.append(np.stack([preprocess_channel(np.asarray(sw.data[arr][::step, ch.slot, :]), dt, arr, no_bg)[0] for ch in chans], 1))
    sig = np.median(np.concatenate(acc, 0), axis=0)
    np.savez(f, sig=sig, n_scans=sum(len(a) for a in acc), files=np.array(files)); return sig


def geometry_for(cfg: dict, sw: Swath) -> dict:
    """Per-scan (s, u) of every channel: from the v1 cube file if configured and present,
    else computed with geometry.swath.swath_geometry and cached (small npz)."""
    cache = Path(cfg["base"]["cache_dir"])
    if cfg.get("cube_file") and (cache / cfg["cube_file"]).exists():
        c = np.load(cache / cfg["cube_file"])
        return {k: c[k] for k in ("s1", "u1", "s2", "u2", "s_nodes", "xn", "yn", "k")}
    h = hashlib.md5(yaml.safe_dump({k: cfg["base"][k] for k in ("survey_dir", "swath", "track")}, sort_keys=True).encode()).hexdigest()[:10]
    f = cache / f"geom_{cfg['base']['swath']}_{Path(cfg['base']['survey_dir']).parent.name[-19:]}_{h}.npz"
    if f.exists():
        return dict(np.load(f))
    from italgas_gpr.geometry.swath import swath_geometry
    g = swath_geometry(sw, cfg["base"]); np.savez(f, **g)
    return g


def preprocess_array(sw: Swath, arr: int, cfg: dict, scans: slice | None = None, raw: np.ndarray | None = None,
                     signature: np.ndarray | None = None) -> dict:
    """v2 preprocessing of all channels of one array. Returns dict(X (n_scans, n_ch, n_t)
    noise-equalised, burst-blanked, cropped to t_max; snr (n_ch, 512) = sigma(t)/floor;
    wiener; burst (n_scans,) bool; dt_ns; dx_m). scans: optional contiguous subset (statistics
    such as the median background and the gain are then taken over that subset). raw: optional replacement for the
    recorded traces of those scans (n_scans, n_ch, 512), e.g. with injected test pipes. signature: (n_ch, 512) for
    preprocess.background = 'signature'. preprocess.lateral_background = true also subtracts, at every scan and time,
    the median across the array's channels (removes echoes that are uniform across the 1.5 m strip, i.e. flat layers,
    but not a pipe, which covers only a few channels at any one time)."""
    p = cfg["preprocess"]
    chans = sw.config.channels_of(arr); dt = chans[0].dt_ns
    n_t = int(p["t_max_ns"][arr] / dt)
    sl = scans if scans is not None else slice(None)
    src = sw.data[arr] if raw is None else None
    X = np.stack([preprocess_channel(np.asarray(src[sl, ch.slot, :]) if raw is None else raw[:, c], dt, arr, cfg,
                                     None if signature is None else signature[c])[0] for c, ch in enumerate(chans)], 1)
    if p.get("lateral_background", False):
        X -= np.median(X, axis=1, keepdims=True)
    sw_, nw_ = (np.array(p["wiener_signal_ns"][arr]) / dt).astype(int), (np.array(p["wiener_noise_ns"][arr]) / dt).astype(int)
    W = spectral_wiener_weights(X[::7].reshape(-1, X.shape[2]), tuple(sw_), tuple(nw_), 256)
    X = apply_spectral_weights(X, W)
    snr = np.zeros((len(chans), X.shape[2]), np.float32)
    for c in range(len(chans)):
        z, sig, fl = noise_equalising_gain(X[:, c], dt, p["gain_smooth_ns"])
        X[:, c] = z; snr[c] = sig / fl
    # burst rejection on the LATE window only (receiver-noise zone, no reflections
    # survive there): interference raises it, a strong real reflector does not.
    en = (X[:, :, nw_[0]:nw_[1]] ** 2).mean((1, 2))
    X = X[:, :, :n_t]
    dx = float(np.median(np.diff(sw.index[arr]["odo_m"])))
    loc = median_filter(en, size=int(10 / dx) | 1, mode="nearest")
    burst = en > p["burst_factor"] * loc
    X[burst] = 0.0
    return dict(X=X, snr=snr, wiener=W, burst=burst, dt_ns=dt, dx_m=dx)


def grid_Z(X: np.ndarray, S: np.ndarray, s_grid: np.ndarray) -> np.ndarray:
    """(n_scans, n_ch, n_t) at per-scan positions S (n_scans, n_ch) -> (n_ch, n_s, n_t) float32."""
    return np.stack([to_s_grid(X[:, c], S[:, c], s_grid) for c in range(X.shape[1])])


def build(cfg_path: str) -> dict[int, Path]:
    cfg = load_cfg(cfg_path); b = cfg["base"]; m = cfg["migration"]; g = cfg["grid"]
    h = cfg_hash(cfg)
    cache = Path(b["cache_dir"])
    outs = {arr: cache / f"v2grid_{arr}_{h}.npz" for arr in (1, 2)}
    if all(o.exists() for o in outs.values()):
        return outs
    t0 = time.time()
    sw = Swath.load(b["survey_dir"], b["swath"])
    geo = geometry_for(cfg, sw)
    s_max = float(min(np.nanmax(geo["s1"]), np.nanmax(geo["s2"])))
    s_grid = np.arange(0.0, s_max, g["ds_m"])
    for arr in (1, 2):
        if outs[arr].exists():
            continue
        S = geo[f"s{arr}"]; u_ch = np.nanmedian(geo[f"u{arr}"], 0)
        pp = preprocess_array(sw, arr, cfg); X = pp["X"]; dt = pp["dt_ns"]; dx = pp["dx_m"]
        n_ch, n_t = X.shape[1], X.shape[2]
        print(f"array {arr}: preprocessed {n_ch} ch, burst scans {pp['burst'].mean()*100:.2f} %, {time.time()-t0:.0f}s", flush=True)
        td = g["t_down"]; n_td = n_t // td
        G = np.zeros((len(m["v_apparent_m_ns"]), n_ch, len(s_grid), n_td), np.float16)
        Zg = grid_Z(X, S, s_grid).astype(np.float16)
        for c in range(n_ch):
            for k, v in enumerate(m["v_apparent_m_ns"]):
                mig = kirchhoff_migrate(X[:, c], dx, dt, v, m["aperture_m"])
                e = envelope(mig)[:, : n_td * td].reshape(len(mig), n_td, td).mean(2)
                G[k, c] = to_s_grid(robust_z(e), S[:, c], s_grid)
            print(f"  array {arr} ch {c+1}/{n_ch} migrated x{len(m['v_apparent_m_ns'])}, {time.time()-t0:.0f}s", flush=True)
        np.savez(outs[arr], G=G, Z=Zg, s_grid=s_grid, u_ch=u_ch, dt_ns=dt * td, dt_full_ns=dt,
                 snr=pp["snr"], wiener=pp["wiener"], burst_frac=pp["burst"].mean())
        del X, G, Zg
    print(f"saved {outs} in {time.time()-t0:.0f}s")
    return outs


if __name__ == "__main__":
    build(sys.argv[1])
