"""Per-cell pipe probability on the 200 MHz B-scans of one piece of one drive (display product, DECISIONS 2026-11-04 E42).
P(s, u, t) = 1 - lfdr(score), lfdr = local false discovery rate of the along-street evidence score against its
channel-shuffle null (Efron et al. 2001): lfdr(x) = pi0 f0(x) / f(x), pi0 = 1 (conservative), f0 from the null scores,
f from the observed scores, both as histogram densities on the same bins, lfdr made non-increasing in x.
Meaning: 'fraction of cells with this evidence that chance alone would produce', NOT 'this is a pipe'.
Also saves the preprocessed B-scans (option C) of the 5 single-column channels for display.
Run: python -m italgas_gpr.along.evidence configs/along_raster.yaml <drive index> <piece start m> -> cache/prob_raster_<i>_<s0>.npz"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

from italgas_gpr.geometry.swath import swath_geometry
from italgas_gpr.processing.preprocess import grid_Z, instrument_signature, load_cfg, preprocess_array
from italgas_gpr.crossing.scan import along_det, day_files
from italgas_gpr.io.ids_reader import Swath
from italgas_gpr.io.paths import config_path, load_yaml
from italgas_gpr.along.focus import lateral_migrate, point_vs_line_contrast, stack_along_s
from italgas_gpr.crossing.detect import search_window
from italgas_gpr.lanes.multipass import longest_run



def search_samples(d, t_min_ns: float) -> np.ndarray:
    """Sample indices of the evaluated time band of one raster piece d (an opened prob_raster npz): the piece's own
    search window t_search (from the SNR of the data) cut at the near-surface floor t_min_ns (configs/along_lines.yaml,
    DECISIONS 2026-11-04). Single rule for the line derivation, the repeat-drive test and the page raster."""
    t = d["t_ns"]; return np.nonzero((t >= max(float(d["t_search"][0]), t_min_ns)) & (t <= float(d["t_search"][1])))[0]


def search_columns(d, u_min_m: float, u_max_m: float) -> np.ndarray:
    """Lateral column indices of one raster piece d that are evaluated: u_min_m <= ug <= u_max_m (configs/along_lines.yaml;
    DECISIONS 2026-11-05 lateral exclusion of the outer columns). Single rule for lines, repeat-drive test and page raster."""
    ug = d["ug"]; return np.nonzero((ug >= u_min_m) & (ug <= u_max_m))[0]


def isotonic_decreasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least-squares fit of a non-increasing sequence to y (pool-adjacent-violators). Loop over bins (hundreds),
    not cells. Returns the fitted values, same shape as y."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y.astype(float), w.astype(float)):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:                     # violation: pool the last two blocks
            wt = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / wt if wt > 0 else 0.5 * (vals[-2] + vals[-1])
            vals[-2:] = [v]; wts[-2:] = [wt]; cnt[-2:] = [cnt[-2] + cnt[-1]]
    return np.repeat(vals, cnt)


def lfdr_curve(obs: np.ndarray, null: np.ndarray, n_null_reps: int, n_bins: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Monotone local FDR as a function of score. obs: observed scores (all cells); null: pooled null scores from
    n_null_reps shuffles. Returns (bin edges, lfdr per bin) with lfdr = min(1, f0 / f) made non-increasing by a
    count-weighted isotonic fit (weights = observed cells per bin). Correction of 4 Nov 2026 (DECISIONS): the earlier
    rule took the running maximum from the top bin, so a sparse extreme tail of the null (a few cells) set lfdr = 1
    for the whole chunk; 51 of 195 chunks had no cell above P = 0.5 for that reason."""
    hi = float(np.quantile(np.r_[obs, null], 0.9999)); edges = np.linspace(min(obs.min(), null.min()), hi, n_bins + 1)
    f = np.histogram(obs, edges)[0].astype(float); f0 = np.histogram(null, edges)[0].astype(float) / n_null_reps
    m = f > 0                                                         # empty observed bins carry no information
    r = np.minimum(1.0, f0[m] / f[m]); fit = np.minimum(1.0, isotonic_decreasing(r, f[m]))
    full = np.ones(len(f)); full[m] = fit
    last = np.maximum.accumulate(np.where(m, np.arange(len(f)), -1))   # empty bins take the value of the last filled bin below
    out = np.where(last >= 0, full[np.maximum(last, 0)], 1.0)
    return edges, out


def main(cfg_path: str, idx: int, c0: float) -> Path:
    cc = load_yaml(config_path(cfg_path)); det = load_cfg(config_path(cc["detector_config"])); detA = along_det(det, cc)
    L = detA["longitudinal"]; xc = detA["crossing"]; v = detA["migration"]["velocity_m_ns"]
    files = sorted(glob.glob(str(Path(cc["raw_root"]) / "*" / "Survey_*" / "Swath*.path"))); f = files[idx]
    sw = Swath.load(str(Path(f).parent), Path(f).stem); g = swath_geometry(sw, det["base"]); sc2 = np.nanmedian(g["s2"], 1)
    c1 = min(float(g["s_nodes"][-1]), c0 + cc["chunk_m"]); s_grid = np.arange(max(0.0, c0 - cc["pad_m"]), c1 + cc["pad_m"], det["grid"]["ds_m"])
    sl = longest_run((sc2 > s_grid[0] - 15) & (sc2 < s_grid[-1] + 15))
    sig = instrument_signature(day_files(f, cc["raw_root"]), 2, detA)
    pp = preprocess_array(sw, 2, detA, scans=sl, signature=sig); dt = pp["dt_ns"]; u = np.nanmedian(g["u2"][sl], 0)
    Z = grid_Z(pp["X"], g["s2"][sl], s_grid)                                           # (n_ch, n_s, n_t)
    ds = float(s_grid[1] - s_grid[0]); i0, i1, t_hi = search_window(pp["snr"], dt, dt, xc["t_min_ns"], xc["snr_min"])
    X, centres = stack_along_s(Z, ds, L["win_m"], L["hop_m"]); ug = np.arange(u.min() + 0.06, u.max() - 0.06 + 1e-9, L["du_m"])
    sc = point_vs_line_contrast(lateral_migrate(X, u, ug, dt, v))                      # (n_win, n_u, n_t)
    rng = np.random.default_rng(det["seed"] + 81 + int(round((c0 - 0) / cc["chunk_m"])))  # same seed rule as the city scan piece
    nulls = [point_vs_line_contrast(lateral_migrate(X[:, rng.permutation(len(u)), :], u, ug, dt, v)) for _ in range(L["n_surrogates"])]
    obs = sc[:, :, i0:i1].ravel(); nul = np.concatenate([n[:, :, i0:i1].ravel() for n in nulls])   # ONE curve for all columns. A curve per
    edges, lf = lfdr_curve(obs, nul, L["n_surrogates"])                               # column was tried on 5 Nov 2026 and reverted (DECISIONS):
    P = np.zeros_like(sc); b = np.clip(np.digitize(sc[:, :, i0:i1], edges) - 1, 0, len(lf) - 1); P[:, :, i0:i1] = 1 - lf[b]   # the outer columns
    # got worse (234-353 vs 12-150 cells >= 0.5 per 10 000); they are excluded downstream instead (search_columns, raster_lines.yaml u_min_m/u_max_m)
    cols = [int(np.argmin(np.abs(u - y))) for y in (0.671, 0.335, 0.0, -0.335, -0.671)]  # single-column channels (Survey XML y)
    keep = (s_grid >= c0) & (s_grid < c1)
    out = Path(det["base"]["cache_dir"]) / f"prob_raster_{idx}_{int(c0)}.npz"
    np.savez_compressed(out, s=s_grid[keep], t_ns=np.arange(Z.shape[2]) * dt, u_ch=u, cols=np.array(cols), B=Z[cols][:, keep, :].astype(np.float16),
                        centres_s=s_grid[centres], ug=ug, P=P.astype(np.float16), score=sc.astype(np.float16), t_search=np.array([i0 * dt, t_hi]),
                        lfdr_edges=edges, lfdr=lf, frac_P50=float(np.mean(P[:, :, i0:i1] >= 0.5)), frac_P90=float(np.mean(P[:, :, i0:i1] >= 0.9)),
                        drive=idx, c0=c0, c1=c1, path=f)
    print("saved", out, "search", round(i0 * dt, 1), round(t_hi, 1), "ns; cells P>=0.5:", round(float(np.mean(P[:, :, i0:i1] >= 0.5)), 4), "P>=0.9:", round(float(np.mean(P[:, :, i0:i1] >= 0.9)), 5))
    return out


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
