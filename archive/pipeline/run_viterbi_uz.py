"""Viterbi along-road test (DECISIONS.md 3 Nov 2026) on one drive: radius-aware across-street focusing
(R = 0, 0.1, 0.25 m), lateral median contrast, depth-aware multi-path Viterbi, chance from channel-order
permutations, scored against the municipal records inside the strip (same targets as E15).
Run: python run_viterbi_uz.py <survey_dir> <swath> <along_json> -> cache/city/viterbi_<swath>.json"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from along_road import match, shifted_targets
from build_cube import swath_geometry
from build_v2 import grid_Z, load_cfg, preprocess_array
from ids_reader import Swath
from longitudinal import lateral_contrast, lateral_migrate_radius, stack_along_s
from network_dp import viterbi_uz_multi

ROOT = Path(__file__).resolve().parents[1]
RADII = (0.0, 0.1, 0.25); Z0, Z1, DZ = 0.2, 1.5, 0.06


def evidence(X, u, ug, dt, v, perm=None):
    Xp = X if perm is None else X[:, perm, :]
    E = None
    for R in RADII:
        e = lateral_contrast(lateral_migrate_radius(Xp, u, ug, dt, v, R)); E = e if E is None else np.maximum(E, e)
    zb = np.arange(Z0, Z1, DZ); depth = np.arange(E.shape[2]) * dt * v / 2
    out = np.stack([E[:, :, (depth >= a) & (depth < a + DZ)].max(2) for a in zb], 2)    # (n_win, n_u, n_z)
    return out, zb


def main(survey_dir: str, swath: str, along_json: str) -> None:
    det = load_cfg(str(ROOT / "pipeline/configs/dev_swath_v2_2.yaml")); v = det["migration"]["velocity_m_ns"]; rng = np.random.default_rng(det["seed"] + 61)
    sw = Swath.load(survey_dir, swath); g = swath_geometry(sw, det["base"]); S = g["s2"]
    pp = preprocess_array(sw, 2, det); L = float(np.nanmax(np.nanmedian(S, 1))); s = np.arange(0.0, L, det["grid"]["ds_m"])
    Z = grid_Z(pp["X"], S, s); u = np.nanmedian(g["u2"], 0); dt = pp["dt_ns"]; del pp
    X, cen = stack_along_s(Z, det["grid"]["ds_m"], 2.0, 0.5); del Z
    ug = np.arange(u.min() - 0.3, u.max() + 0.3 + 1e-9, 0.04)
    E, zb = evidence(X, u, ug, dt, v)
    nulls = [evidence(X, u, ug, dt, v, rng.permutation(len(u)))[0] for _ in range(8)]
    Nst = np.stack(nulls); mu = Nst.mean((0, 1, 2)); sd = Nst.std((0, 1, 2)) + 1e-9
    to_llr = lambda A: np.clip((A - mu) / sd - 2.0, -2.0, 4.0)
    paths = viterbi_uz_multi(to_llr(E), k_paths=8, min_steps=20)
    null_best = []
    for Nn in nulls:
        pn = viterbi_uz_multi(to_llr(Nn), k_paths=1, min_steps=20); null_best.append(pn[0][4] if pn else 0.0)
    thr = float(np.quantile(null_best, 0.95))
    sc = s[cen]; tracks = []
    for t0, t1, iu, iz, score in paths:
        tracks.append(dict(s_start=float(sc[t0]), s_end=float(sc[t1 - 1]), length_m=float(sc[t1 - 1] - sc[t0]), u_mean=float(ug[iu].mean()),
                           u_std=float(ug[iu].std()), depth_m=float(np.median(zb[iz] + DZ / 2)), score=float(score), significant=bool(score > thr),
                           q=0.0 if score > thr else 1.0, path_s=sc[t0:t1:4].round(2).tolist(), path_u=ug[iu][::4].round(3).tolist()))
    tg = json.load(open(along_json))["targets"]
    hit = match(tracks, tg, 0.05, 0.5, None, 0.3)
    offs = rng.uniform(20, 200, 1000) * rng.choice([-1, 1], 1000)
    ch = np.array([match(tracks, shifted_targets(tg, o, 0.0, L), 0.05, 0.5, None, 0.3).sum() for o in offs])
    near = []
    for t in tracks:                                    # nearest record to each path (for reading the result)
        best = None
        for r in tg:
            ov = min(t["s_end"], r["s1"]) - max(t["s_start"], r["s0"])
            if ov > 0:
                d = abs(t["u_mean"] - r["u0"])
                if best is None or d < best[0]:
                    best = (d, r)
        near.append(None if best is None else dict(du=round(best[0], 2), record=f"{best[1]['kind']} {best[1]['diam']} mm {best[1]['material']} {best[1]['z']} m",
                                                    rec_s=[round(best[1]['s0']), round(best[1]['s1'])]))
    res = dict(swath=swath, length_m=L, null_best=null_best, threshold=thr, tracks=tracks, nearest_record=near, n_targets=len(tg),
               hits=int(hit.sum()), chance=float(ch.mean()), p=float((ch >= hit.sum()).mean()), hit_targets=[tg[i] for i in np.nonzero(hit)[0]])
    print(f"threshold {thr:.1f} (null best scores {np.round(null_best, 1).tolist()})")
    for t, n_ in zip(tracks, near):
        print(f"  path s {t['s_start']:.0f}-{t['s_end']:.0f} u {t['u_mean']:+.2f} (sd {t['u_std']:.2f}) depth {t['depth_m']:.2f} score {t['score']:.1f} "
              f"{'SIGNIFICANT' if t['significant'] else ''}  nearest record: {n_}")
    print(f"RESULT: {res['hits']} of {len(tg)} recorded pipes hit vs {res['chance']:.2f} by chance (p = {res['p']:.3f})")
    out = Path(det["base"]["cache_dir"]) / "city" / f"viterbi_{swath}_{Path(survey_dir).name[-8:]}.json"; json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
