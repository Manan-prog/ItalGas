"""Descriptive: how far the van moves sideways relative to its own 60 m running-mean path within 20 m (lane changes, weave).
Run: python sweep_stats.py (configs/subset_merge.yaml drives). Track = real-time .path (EPSG:3857, x cos(lat)), 11-sample smoothing as in the pipeline; RTK-fixed epochs only; turns excluded."""
import glob, sys
from pathlib import Path
import numpy as np, yaml
from scipy.ndimage import uniform_filter1d
sys.path.insert(0, '.')
from ids_reader import Swath
from cube import smooth_track, track_frame, to_su
from geometry import merc_scale, lonlat_to_merc
cc = yaml.safe_load(open('configs/citywide_scan.yaml')); raw = '/Users/manansarupria/ItalGas/Savona'
files = sorted(glob.glob(raw + '/*/Survey_*/Swath*.path'))
drives = yaml.safe_load(open('configs/subset_merge.yaml'))['drives']
REF_M, WIN_M, TURN_DEG = 60, 20, 20
tot = dict(L=0.0, l3=0.0, l5=0.0, l67=0.0, ev5=0, fixed=0.0)
for i in drives:
    f = files[i]; sw = Swath.load(str(Path(f).parent), Path(f).stem); p = sw.path
    k = float(merc_scale(p["y"]).mean()); xs, ys = smooth_track(p["x"], p["y"], 11); s, _, _ = track_frame(xs, ys, k)
    m = np.r_[True, np.diff(s) > 1e-3]; s, xs, ys = s[m], xs[m], ys[m]
    g = np.arange(0, s[-1], 1.0); X = np.interp(g, s, xs); Y = np.interp(g, s, ys)
    Xr = uniform_filter1d(X, REF_M, mode='nearest'); Yr = uniform_filter1d(Y, REF_M, mode='nearest')
    sr, _, _ = track_frame(Xr, Yr, k); _, u = to_su(X, Y, sr, Xr, Yr, k)
    hd = np.unwrap(np.arctan2(np.gradient(Yr), np.gradient(Xr)))
    turn = np.abs(hd[np.clip(np.arange(len(g)) + WIN_M // 2, 0, len(g) - 1)] - hd[np.clip(np.arange(len(g)) - WIN_M // 2, 0, len(g) - 1)]) > np.radians(TURN_DEG)
    q = sw.gga_rt
    if q is not None and len(q):
        qx, qy = lonlat_to_merc(q["lon"], q["lat"]); qs, qu = to_su(qx, qy, s, xs, ys, k); o = np.argsort(qs)
        fixed = np.asarray(q["q"])[o][np.clip(np.searchsorted(qs[o], g), 0, len(qs) - 1)] == 4
    else:
        fixed = np.zeros(len(g), bool)
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    rng20 = maximum_filter1d(u, WIN_M) - minimum_filter1d(u, WIN_M)
    ok = fixed & ~turn & (g > REF_M) & (g < g[-1] - REF_M)
    ev = np.diff(np.r_[0, (rng20 >= 0.5) & ok, 0].astype(int)); n_ev = int((ev == 1).sum())
    L = ok.sum(); tot['L'] += L; tot['fixed'] += fixed.sum()
    for key, th in (('l3', 0.3), ('l5', 0.5), ('l67', 0.67)): tot[key] += ((rng20 >= th) & ok).sum()
    tot['ev5'] += n_ev
    print(f"drive {i:3d} {len(g)/1e3:.2f} km, usable {L/1e3:.2f} km (RTK fixed {fixed.mean():.2f}), sideways range in 20 m: median {np.median(rng20[ok]) if L else np.nan:.2f} m, "
          f">=0.3 m {((rng20>=0.3)&ok).sum()/max(L,1):.2f}, >=0.5 m {((rng20>=0.5)&ok).sum()/max(L,1):.2f}, >=0.67 m {((rng20>=0.67)&ok).sum()/max(L,1):.2f}, events>=0.5 m {n_ev}")
L = tot['L']
print(f"POOLED usable {L/1e3:.1f} km: >=0.3 m {tot['l3']/L:.3f}, >=0.5 m {tot['l5']/L:.3f}, >=0.67 m {tot['l67']/L:.3f} of length; events >=0.5 m: {tot['ev5']} ({tot['ev5']/(L/1e3):.1f} per km)")
