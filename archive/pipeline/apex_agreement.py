"""Idea 3 (user, 3 Nov 2026): a pipe stays at the same map position when seen from a different lane;
a layer edge appears wherever each drive's strip ends. For every along-road feature repeated by two
registered drives, fit the pipe apex separately in each drive's own cross-section and compare the
two apex positions; base rate from random places covered by the same two drives.
Run: python apex_agreement.py configs/multipass_<cluster>.yaml -> cache/mp_<tag>apex.json"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

from along_road import pipe_shape_test
from build_v2 import load_cfg
from register_multipass import cross_section, line_evidence, place_interp, slice_map
from run_multipass import pass_spec

ROOT = Path(__file__).resolve().parents[1]
MIN_SEP = 0.5        # m, independent-lane rule (DECISIONS 2026-11-04): strip centres at least this far apart


def lane_sep(D, reg, a, b, s_true: float) -> float:
    """Distance (m) between the strip centres of drives a and b at true position s_true (registered frame:
    s_true = s - A, u_true = u - B, as in register_multipass.place_interp)."""
    c = []
    for n in (a, b):
        sg = D[n]["s_grid"]; ds = float(sg[1] - sg[0]); i = int(np.argmin(np.abs(sg - s_true))) + int(round(reg[n]["A"] / ds))
        i = int(np.clip(i, 20, len(sg) - 21)); c.append(float(np.nanmedian(D[n]["Ug"][:, i - 20:i + 20])) - reg[n]["B"])
    return abs(c[0] - c[1])


def apex_pair(D, reg, a, b, s_grid, u_grid, s_true, u0, z_hint, v):
    out = {}
    for n in (a, b):
        X, z = cross_section({n: D[n]}, reg, s_grid, u_grid, s_true, v)
        out[n] = pipe_shape_test(X, u_grid, z, u0, z_hint, half_w=0.8, allow_outside_m=0.3)
    ra, rb = out[a], out[b]
    both = ra.get("verdict") == "pipe-like" and rb.get("verdict") == "pipe-like"
    d = abs(ra["uc"] - rb["uc"]) if both else None
    return dict(both_pipe=both, du=d, agree=bool(both and d <= 0.2), uc=(ra.get("uc"), rb.get("uc")), z0=(ra.get("z0"), rb.get("z0")))


def repeated_features(D, reg, names, s_grid, u_grid, SL, v):
    from scipy.ndimage import label
    feats = []; ds, du = float(s_grid[1] - s_grid[0]), float(u_grid[1] - u_grid[0])
    for zi in range(len(SL)):
        per = {n: place_interp(slice_map(D[n]["G"][0], float(D[n]["dtg"]), v, *SL[zi]), D[n]["Ug"], s_grid, reg[n]["A"], reg[n]["B"], u_grid) for n in names}
        L = {n: line_evidence(np.nan_to_num(per[n]), np.isfinite(per[n]).astype(np.float32), ds, du) for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                both = np.isfinite(L[a]) & np.isfinite(L[b])
                if both.sum() < 2000:
                    continue
                lab, nl = label((L[a] > 1.0) & (L[b] > 1.0) & both)
                for j in range(1, nl + 1):
                    ii, jj = np.nonzero(lab == j)
                    if (ii.max() - ii.min()) * ds >= 10 and (jj.max() - jj.min() + 1) * du <= 0.6:
                        feats.append(dict(a=a, b=b, zi=zi, s0=float(s_grid[ii.min()]), s1=float(s_grid[ii.max()]), u=float(u_grid[jj].mean()),
                                          cover=both))
    return feats


def main(cfg_path: str, n_ctrl: int = 300) -> None:
    mc = yaml.safe_load(open(cfg_path)); det = load_cfg(str(ROOT / "pipeline" / mc["detector_config"]))
    cache = Path(det["base"]["cache_dir"]); tag = mc.get("tag", ""); v = det["migration"]["velocity_m_ns"]
    rng = np.random.default_rng(det["seed"] + 3)
    reg = json.load(open(cache / f"mp_{tag}registration.json"))["registration"]
    names = [n for n in (pass_spec(mc, p)[2] for p in mc["passes"]) if n in reg]
    D = {n: np.load(cache / f"mp_{tag}{n}_2.npz") for n in names}
    s_grid = D[names[0]]["s_grid"]; u_grid = np.arange(mc["mosaic"]["u_range_m"][0], mc["mosaic"]["u_range_m"][1] + 1e-9, mc["mosaic"]["du_m"])
    SL = mc["mosaic"]["depth_slices_m"]
    feats = repeated_features(D, reg, names, s_grid, u_grid, SL, v)
    rows = []
    for f in feats:
        sm = 0.5 * (f["s0"] + f["s1"]); r = apex_pair(D, reg, f["a"], f["b"], s_grid, u_grid, sm, f["u"], float(np.mean(SL[f["zi"]])), v)
        rows.append(dict({k: f[k] for k in ("a", "b", "zi", "s0", "s1", "u")}, **r, lane_sep=lane_sep(D, reg, f["a"], f["b"], sm)))
    # controls: random places covered by the same pair of drives, same depth slices
    ctrl = []; pairs_ = [(f["a"], f["b"], f["cover"], f["zi"]) for f in feats]
    tries = 0
    while (len(ctrl) < n_ctrl or sum(c["lane_sep"] >= MIN_SEP for c in ctrl) < n_ctrl) and pairs_ and tries < 20 * n_ctrl:
        tries += 1
        a, b, cov, zi = pairs_[rng.integers(len(pairs_))]
        ii, jj = np.nonzero(cov); k = rng.integers(len(ii))
        r = apex_pair(D, reg, a, b, s_grid, u_grid, float(s_grid[ii[k]]), float(u_grid[jj[k]]), float(np.mean(SL[zi])), v)
        ctrl.append(dict(r, lane_sep=lane_sep(D, reg, a, b, float(s_grid[ii[k]]))))
    c300 = ctrl[:n_ctrl]                               # the original controls (same seed), for the any-lane summary
    fa = np.mean([r["agree"] for r in rows]) if rows else float("nan"); fc = np.mean([r["agree"] for r in c300]) if c300 else float("nan")
    from scipy.stats import binomtest
    k = int(sum(r["agree"] for r in rows)); p = binomtest(k, len(rows), max(fc, 1e-3), alternative="greater").pvalue if rows else None
    summ = dict(n_features=len(rows), n_agree=k, frac_agree=float(fa), n_controls=len(c300), frac_agree_controls=float(fc), p=p)
    ri = [r for r in rows if r["lane_sep"] >= MIN_SEP]; ci = [r for r in ctrl if r["lane_sep"] >= MIN_SEP]   # independent lanes only
    ki = int(sum(r["agree"] for r in ri)); fci = float(np.mean([r["agree"] for r in ci])) if ci else float("nan")
    summ["independent_lanes"] = dict(min_sep_m=MIN_SEP, n_features=len(ri), n_agree=ki, frac_agree=ki / len(ri) if ri else float("nan"),
                                     n_controls=len(ci), frac_agree_controls=fci,
                                     p=binomtest(ki, len(ri), max(fci, 1e-3), alternative="greater").pvalue if ri and ci else None)
    print(tag or "vianizza_", summ)
    for r in sorted([r for r in rows if r["agree"]], key=lambda r: r["s0"]):
        print("   agree: s %.0f-%.0f u %.2f  apex u %.2f / %.2f  depth %.2f / %.2f  (%s + %s, lanes %.2f m apart)" % (r["s0"], r["s1"], r["u"], r["uc"][0], r["uc"][1], r["z0"][0], r["z0"][1], r["a"], r["b"], r["lane_sep"]))
    json.dump(dict(summary=summ, features=rows), open(cache / f"mp_{tag}apex.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1])
