"""E28 (3 Nov 2026): is the across-array arch of an along-road pipe measurable as a DELAY?
Idea (user): an image cannot resolve a 0.1 m bend at 200 MHz (wavelength 0.5 m), but the time
shift of one reflector between two channels can be timed to a small fraction of a wavelength by
cross-correlation. So: stack along the street, gate the target depth, cross-correlate every
channel with a reference channel (sub-sample), convert the delays to depth, and fit a pipe
hyperbola z(u) = sqrt(z0^2 + (u - uc)^2) against a line z = a + b u.
Three estimator variants were tried (3 Nov 2026, all judged on the positive control = confirmed
crossing pipes timed ALONG s with the same trace pitch): (a) every trace against one reference in a
fixed gate: cycle skipping beyond 0.3 m; (b) chained adjacent-trace timing in a gate that follows
the pick; (c) as (b) after subtracting the median trace across positions (flat-layer removal).
Final method = (c). Per-channel static delays are DERIVED as the median delay of each channel over
the random control windows of the same drive and removed.
Run: python arch_delay.py configs/arch_delay.yaml -> cache/arch_delay_vianizza.json
Shapes: Z (n_ch, n_s, n_t) signed noise-equalised traces (float16 cache), dt in ns."""
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import hilbert, resample, windows

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_v2 import load_cfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def gated_stack(Z: np.ndarray, ws: np.ndarray, t_hint_ns: float, dt: float, half_ns: float) -> tuple[np.ndarray, int]:
    """Coherent mean over the s window of every channel, tapered gate around t_hint.
    Z (n_ch, n_s, n_t) -> (n_ch, n_gate) float32 and the gate start sample."""
    X = Z[:, ws, :].astype(np.float32).mean(1)
    i0 = max(0, int(round((t_hint_ns - half_ns) / dt))); i1 = min(X.shape[1], int(round((t_hint_ns + half_ns) / dt)))
    return X[:, i0:i1] * windows.tukey(i1 - i0, 0.5)[None, :], i0


def delays(X: np.ndarray, ref: int, dt: float, max_lag_ns: float, up: int) -> tuple[np.ndarray, np.ndarray]:
    """Delay (ns) of every channel relative to channel ref by cross-correlation of the gated traces,
    resampled up x, peak refined by parabolic interpolation. Returns (tau (n_ch,), peak corr (n_ch,))."""
    n_ch, n = X.shape; Xu = resample(X, n * up, axis=1); dtu = dt / up
    r = Xu[ref]; L = int(round(max_lag_ns / dtu)); lags = np.arange(-L, L + 1)
    num = np.stack([np.sum(Xu * np.roll(r, k)[None, :], 1) for k in lags], 1)          # (n_ch, n_lag)
    den = np.sqrt(np.sum(Xu ** 2, 1) * np.sum(r ** 2))[:, None] + 1e-9
    cc = num / den
    j = np.argmax(cc, 1); jc = np.clip(j, 1, len(lags) - 2)
    y0, y1, y2 = cc[np.arange(n_ch), jc - 1], cc[np.arange(n_ch), jc], cc[np.arange(n_ch), jc + 1]
    frac = 0.5 * (y0 - y2) / np.maximum(y0 - 2 * y1 + y2, 1e-9)
    tau = (lags[jc] + np.clip(frac, -1, 1)) * dtu                                          # trace c = ref delayed by tau
    return tau.astype(float), cc[np.arange(n_ch), j].astype(float)


def pair_delay(a: np.ndarray, b: np.ndarray, dt: float, max_lag_ns: float, up: int) -> tuple[float, float]:
    """Delay (ns) of trace b relative to trace a (b = a delayed by tau), sub-sample. Returns (tau, peak corr)."""
    n = len(a); A = resample(a, n * up); B = resample(b, n * up); dtu = dt / up
    L = int(round(max_lag_ns / dtu)); lags = np.arange(-L, L + 1)
    cc = np.array([np.sum(B * np.roll(A, k)) for k in lags]) / (np.sqrt(np.sum(A ** 2) * np.sum(B ** 2)) + 1e-9)
    j = int(np.argmax(cc)); jc = int(np.clip(j, 1, len(lags) - 2)); y0, y1, y2 = cc[jc - 1], cc[jc], cc[jc + 1]
    fr = 0.5 * (y0 - y2) / max(y0 - 2 * y1 + y2, 1e-9)
    return float((lags[jc] + np.clip(fr, -1, 1)) * dtu), float(cc[j])


def chained_times(T: np.ndarray, t_ref_ns: float, dt: float, ref: int, gate_half_ns: float, max_lag_ns: float, up: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Reflector time (ns) at every position of T (n_pos, n_t): from ref outward, each trace is timed
    against its neighbour inside a tapered gate centred on the neighbour's pick (no cycle skipping as
    long as the step delay is below max_lag_ns). Returns (t (n_pos,), corr (n_pos,))."""
    n = T.shape[0]; t = np.full(n, np.nan); r = np.zeros(n); t[ref] = t_ref_ns; r[ref] = 1.0
    for side in (1, -1):
        for i in range(ref + side, n if side > 0 else -1, side):
            j = i - side; tc = t[j]; i0 = int(round((tc - gate_half_ns) / dt)); i1 = int(round((tc + gate_half_ns) / dt))
            if i0 < 0 or i1 > T.shape[1]:
                break
            w = windows.tukey(i1 - i0, 0.5); d, c = pair_delay(T[j, i0:i1] * w, T[i, i0:i1] * w, dt, max_lag_ns, up)
            t[i] = tc + d; r[i] = c
    return t, r


def time_section(T: np.ndarray, z_hint: float, dt: float, v: float, ref: int, cfg: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Final estimator (variant c): subtract the median trace across positions, find the reference
    reflector in a gate around z_hint, then chained timing. Returns (t (n_pos,) ns, corr, t_ref)."""
    T = T - np.median(T, 0, keepdims=True)
    th = 2 * z_hint / v; g = slice(max(0, int((th - cfg["gate_half_ns"]) / dt)), int((th + cfg["gate_half_ns"]) / dt))
    e = np.abs(hilbert(T[ref]))[g]; t_ref = (g.start + int(np.argmax(e))) * dt
    t, r = chained_times(T, t_ref, dt, ref, cfg["chain_gate_half_ns"], cfg["chain_max_lag_ns"], cfg["upsample"])
    return t, r, t_ref


def fit_shapes(u: np.ndarray, z: np.ndarray, allow_outside_m: float = 0.3) -> dict:
    """Hyperbola z = sqrt(z0^2 + (u - uc)^2) (grid search) vs line z = a + b u (least squares)."""
    best = None
    for uc in np.arange(u.min() - allow_outside_m, u.max() + allow_outside_m + 1e-9, 0.01):
        d2 = (u - uc) ** 2
        for z0 in np.arange(max(0.1, z.min() - 0.3), z.max() + 1e-9, 0.005):
            r = np.sqrt(np.mean((np.sqrt(z0 ** 2 + d2) - z) ** 2))
            if best is None or r < best[0]:
                best = (r, uc, z0)
    A = np.c_[np.ones_like(u), u]; coef = np.linalg.lstsq(A, z, rcond=None)[0]
    r_line = float(np.sqrt(np.mean((A @ coef - z) ** 2)))
    return dict(rms_hyperbola_m=float(best[0]), uc=float(best[1]), z0=float(best[2]), rms_line_m=r_line,
                slope=float(coef[1]), apex_inside=bool(u.min() + 0.15 < best[1] < u.max() - 0.15))


def measure(Z, Ug, s_true, dt, v, s0, s1, u0, z_hint, cfg, static_ns=None, sub_fit=True) -> dict:
    """One window across the array: reflector time per channel (final estimator), depths, fits,
    and sub-window scatter. tau = t - t_ref (ns)."""
    ws = (s_true >= s0) & (s_true <= s1)
    if ws.sum() < 20:
        return dict(status="window too short")
    u = np.nanmedian(Ug[:, ws], 1)
    if not np.isfinite(u).all():
        return dict(status="no geometry")
    ref = int(np.argmin(np.abs(u - u0)))
    T = Z[:, ws, :].astype(np.float32).mean(1)
    t, r, t_ref = time_section(T, z_hint, dt, v, ref, cfg); tau = t - t_ref
    if static_ns is not None:
        tau = tau - static_ns
    z = (t_ref + tau) * v / 2
    ok = np.isfinite(z) & (r >= cfg["min_corr"])
    out = dict(status="ok", ref=ref, u=u.round(3).tolist(), tau_ns=np.nan_to_num(tau).round(3).tolist(), corr=r.round(3).tolist(),
               z=np.nan_to_num(z).round(4).tolist(), n_used=int(ok.sum()), t_ref_ns=float(t_ref))
    if ok.sum() >= 8:
        out.update(fit_shapes(u[ok], z[ok]))
        pr = cfg["pipe_rule"]
        out["pipe_like"] = bool(out["rms_hyperbola_m"] * pr["rms_ratio"] <= out["rms_line_m"] and out["apex_inside"])
    if sub_fit:
        edges = np.arange(s0, s1 - cfg["sub_win_m"] + 1e-9, cfg["sub_win_m"]); taus, apex = [], []
        for e in edges:
            w2 = (s_true >= e) & (s_true < e + cfg["sub_win_m"])
            if w2.sum() < 10:
                continue
            t2, r2, tr2 = time_section(Z[:, w2, :].astype(np.float32).mean(1), z_hint, dt, v, ref, cfg); tau2 = t2 - tr2
            if static_ns is not None:
                tau2 = tau2 - static_ns
            taus.append(np.nan_to_num(tau2)); ok2 = np.isfinite(t2) & (r2 >= cfg["min_corr"])
            if ok2.sum() >= 8:
                apex.append(fit_shapes(u[ok2], (tr2 + tau2[ok2]) * v / 2)["uc"])
        if len(taus) >= 2:
            Tt = np.array(taus); out["pick_sd_ns"] = float(np.median(Tt.std(0))); out["pick_sd_m"] = float(np.median(Tt.std(0)) * v / 2); out["n_sub"] = len(taus)
        if len(apex) >= 2:
            out["apex_sd_m"] = float(np.std(apex)); out["apex_sub"] = [round(a, 3) for a in apex]
    return out


def along_s_control(Z, s_grid, dt, v, s0, z_hint, cfg, n_half: int = 9, step_m: float = 0.10) -> dict:
    """Positive control: the same estimator applied ALONG s over a confirmed crossing pipe (centre
    5 channels averaged), traces every step_m like the array pitch. 'u' here is s - s0 (m)."""
    ds = s_grid[1] - s_grid[0]; k = int(round(step_m / ds)); i_c = int(np.argmin(np.abs(s_grid - s0)))
    idx = i_c + k * np.arange(-n_half, n_half + 1); cen = Z.shape[0] // 2
    T = Z[cen - 2: cen + 3][:, idx, :].astype(np.float32).mean(0)
    t, r, t_ref = time_section(T, z_hint, dt, v, n_half, cfg)
    u = (idx - i_c) * ds; z = t * v / 2; ok = np.isfinite(z) & (r >= cfg["min_corr"])
    out = dict(status="ok", u=u.round(3).tolist(), tau_ns=np.nan_to_num(t - t_ref).round(3).tolist(), corr=r.round(3).tolist(),
               z=np.nan_to_num(z).round(4).tolist(), n_used=int(ok.sum()), expected_dz=(np.sqrt(z_hint ** 2 + u ** 2) - z_hint).round(4).tolist())
    if ok.sum() >= 8:
        out.update(fit_shapes(u[ok], z[ok])); pr = cfg["pipe_rule"]
        out["pipe_like"] = bool(out["rms_hyperbola_m"] * pr["rms_ratio"] <= out["rms_line_m"] and out["apex_inside"])
        zz = z - z[n_half]; out["measured_over_expected_at_edge"] = float(np.nanmean(zz[[0, 1, -2, -1]]) / np.mean(np.array(out["expected_dz"])[[0, 1, -2, -1]]))
    return out


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path)); mc = yaml.safe_load(open(ROOT / "pipeline" / cfg["multipass_config"]))
    det = load_cfg(str(ROOT / "pipeline" / mc["detector_config"])); cache = Path(det["base"]["cache_dir"])
    v = cfg["velocity_m_ns"]; rng = np.random.default_rng(cfg["seed"])
    reg = json.load(open(cache / "mp_registration.json"))["registration"]
    names = sorted({d for t in cfg["targets"] for d in t["drives"]} | {p["drive"] for p in cfg["positive_control"]})
    D = {n: np.load(cache / f"mp_{n}_2.npz") for n in names}
    res = dict(config=cfg, controls={}, statics_ns={}, targets=[], positive=[])
    # 1. controls per drive (also give the per-channel statics)
    cc = cfg["controls"]; z_lo, z_hi = cc["z_range_m"]
    for n in names:
        z = D[n]; Z = z["Z"]; dt = float(z["dt"]); s_true = z["s_grid"] - reg[n]["A"]; Ug = z["Ug"] - reg[n]["B"]
        lo, hi = mc["window_s_m"]; rows = []
        cen = Z.shape[0] // 2; it = (np.arange(Z.shape[2]) * dt * v / 2)
        for _ in range(cc["n_per_drive"]):
            s0 = float(rng.uniform(lo + 5, hi - cc["win_m"] - 5)); s1 = s0 + cc["win_m"]
            ws = (s_true >= s0) & (s_true <= s1)
            if ws.sum() < 20 or not np.isfinite(Ug[:, ws]).all():
                continue
            e = np.abs(hilbert(Z[cen, ws, :].astype(np.float32).mean(0)))
            m = (it >= z_lo) & (it <= z_hi); zh = float(it[m][np.argmax(e[m])])
            u0 = float(np.nanmedian(Ug[cen, ws]))
            r = measure(Z, Ug, s_true, dt, v, s0, s1, u0, zh, cfg, sub_fit=False)
            if r["status"] == "ok":
                r.update(s0=s0, s1=s1, z_hint=zh); rows.append(r)
        T = np.array([r["tau_ns"] for r in rows]); R = np.array([r["corr"] for r in rows])
        stat = np.nanmedian(np.where(R >= cfg["min_corr"], T, np.nan), 0); stat = np.nan_to_num(stat)
        res["statics_ns"][n] = stat.round(3).tolist()
        # re-fit controls with statics removed
        rows2 = []
        for r in rows:
            tau = np.array(r["tau_ns"]) - stat; ok = np.array(r["corr"]) >= cfg["min_corr"]; u = np.array(r["u"])
            zz = (r["t_ref_ns"] + tau) * v / 2
            if ok.sum() >= 8:
                f = fit_shapes(u[ok], zz[ok]); pr = cfg["pipe_rule"]
                f["pipe_like"] = bool(f["rms_hyperbola_m"] * pr["rms_ratio"] <= f["rms_line_m"] and f["apex_inside"])
                rows2.append(dict(s0=r["s0"], z_hint=r["z_hint"], n_used=int(ok.sum()), **f))
        res["controls"][n] = rows2
        print(n, "controls", len(rows2), "pipe-like", sum(r["pipe_like"] for r in rows2), "static sd ns", round(float(np.std(stat)), 3), flush=True)
    # 2. targets
    for t in cfg["targets"]:
        for n in t["drives"]:
            z = D[n]; dt = float(z["dt"]); s_true = z["s_grid"] - reg[n]["A"]; Ug = z["Ug"] - reg[n]["B"]
            r = measure(z["Z"], Ug, s_true, dt, v, t["s0"], t["s1"], t["u0"], t["z_hint"], cfg, static_ns=np.array(res["statics_ns"][n]))
            r.update(name=t["name"], drive=n, expect=t["expect"]); res["targets"].append(r)
            print(t["name"], n, {k: r.get(k) for k in ("n_used", "rms_hyperbola_m", "rms_line_m", "uc", "z0", "slope", "pipe_like", "pick_sd_m", "apex_sd_m")}, flush=True)
    # 3. positive controls along s
    for p in cfg["positive_control"]:
        z = D[p["drive"]]; r = along_s_control(z["Z"], z["s_grid"], float(z["dt"]), v, p["s0_drive"], p["z_hint"], cfg)
        r.update(p); res["positive"].append(r)
        print("positive", p["s0_drive"], {k: r.get(k) for k in ("n_used", "rms_hyperbola_m", "rms_line_m", "uc", "z0", "pipe_like")}, flush=True)
    summ = {n: dict(n=len(rs), pipe_like=int(sum(r["pipe_like"] for r in rs)), frac=float(np.mean([r["pipe_like"] for r in rs])) if rs else None)
            for n, rs in res["controls"].items()}
    res["summary"] = summ; print("SUMMARY", summ)
    json.dump(res, open(cache / "arch_delay_vianizza.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1])
