"""Analysis of the citywide runs (3 Nov 2026), INTERNAL (uses the town-plan sketch).
(1) step 1: are two-drive apex candidates closer to drawn gas lines than random places covered by the same drives?
(2) step 3: do single-drive track measurements predict closeness to a drawn gas line on held-out days?
Run: python analyse_citywide.py -> cache/city/analysis.json"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

from citywide_tracks import sketch_trees

CITY = Path("/Users/manansarupria/ItalGas/cache/city")
NEAR = 2.0                                  # m: "near a drawn line" (sketch placed to several metres; georeference p90 1.7 m)


def dist(trees, bounds, x, y):
    to3003 = Transformer.from_crs("EPSG:3857", "EPSG:3003", always_xy=True); X, Y = to3003.transform(np.asarray(x), np.asarray(y))
    inside = (X > bounds.left) & (X < bounds.right) & (Y > bounds.bottom) & (Y < bounds.top)
    return {n: np.where(inside, t.query(np.c_[X, Y])[0], np.nan) for n, t in trees.items()}, inside


def step1(trees, bounds) -> dict:
    C = []; K = []; NF = []
    for f in glob.glob(str(CITY / "pair_*.json")):
        r = json.load(open(f))
        if r.get("status") != "ok":
            continue
        for x in r["features"]:
            (C if x["agree"] else NF).append((x["x"], x["y"], x["s1"] - x["s0"]))
        K += [(c["x"], c["y"]) for c in r["controls"]]
    out = dict(n_candidates_raw=len(C), n_repeated_not_agreeing=len(NF), n_controls=len(K))
    if not C:
        return out
    P = np.array([c[:2] for c in C]); tr = cKDTree(P); grp = -np.ones(len(P), int); g = 0
    for i in range(len(P)):                          # merge candidates within 3 m (same place seen by several pairs)
        if grp[i] < 0:
            for j in tr.query_ball_point(P[i], 3.0 / np.cos(np.radians(44.3))):
                grp[j] = g
            g += 1
    Pm = np.array([P[grp == k].mean(0) for k in range(g)])
    dc, ic = dist(trees, bounds, Pm[:, 0], Pm[:, 1]); dk, ik = dist(trees, bounds, *np.array(K).T)
    dn, inn = dist(trees, bounds, *np.array([c[:2] for c in NF]).T) if NF else ({n: np.array([]) for n in trees}, np.array([]))
    out["n_candidates_distinct"] = int(g); out["n_candidates_inside_sketch"] = int(ic.sum())
    for n in ("gas", "water", "sewer"):
        pc = float(np.nanmean(dc[n][ic] <= NEAR)) if ic.any() else None; pk = float(np.nanmean(dk[n][ik] <= NEAR))
        pn = float(np.nanmean(dn[n][inn] <= NEAR)) if len(inn) and inn.any() else None
        rng = np.random.default_rng(0); kk = dk[n][ik]
        boot = [np.mean(rng.choice(kk, ic.sum()) <= NEAR) for _ in range(2000)] if ic.any() else []
        out[n] = dict(near_candidates=pc, near_controls=pk, near_repeated_not_agreeing=pn,
                      p_value=float(np.mean(np.array(boot) >= pc)) if boot else None,
                      excess_fraction=(pc - pk) / (1 - pk) if pc is not None else None)
    out["candidates"] = [dict(x=float(a), y=float(b), d_gas=float(dc["gas"][i]) if ic[i] else None, d_water=float(dc["water"][i]) if ic[i] else None)
                         for i, (a, b) in enumerate(Pm)]
    return out


def step3() -> dict:
    rows = []
    for f in glob.glob(str(CITY / "tracks_*.json")):
        r = json.load(open(f))
        day = next((d for d in ("2025.10.13", "2025.10.14", "2025.10.15", "2025.10.17") if d in r["file"]), None)
        for t in r.get("tracks", []):
            if t.get("in_sketch") and t.get("d_gas_m") is not None:
                rows.append((day, t))
    feats = ["length_m", "depth_m", "u_std", "score", "q", "n_pipe_sections", "polarity", "asymmetry", "amplitude", "depth_roughness_ns"]
    X = np.array([[t[k] for k in feats] for _, t in rows], float); dgas = np.array([t["d_gas_m"] for _, t in rows]); dw = np.array([t["d_water_m"] for _, t in rows])
    day = np.array([d for d, _ in rows])
    lab = np.where(dgas <= NEAR, 1, np.where(dgas >= 6.0, 0, -1)); keep = lab >= 0
    out = dict(n_tracks=len(rows), n_near=int((lab == 1).sum()), n_far=int((lab == 0).sum()), features=feats)
    if keep.sum() < 50:
        return out
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    tr = keep & np.isin(day, ["2025.10.13", "2025.10.14"]); te = keep & np.isin(day, ["2025.10.15", "2025.10.17"])
    clf = GradientBoostingClassifier(random_state=0, max_depth=3, n_estimators=200, learning_rate=0.05).fit(X[tr], lab[tr])
    auc = roc_auc_score(lab[te], clf.predict_proba(X[te])[:, 1])
    rng = np.random.default_rng(0); null = []
    for _ in range(100):
        perm = lab[tr].copy(); rng.shuffle(perm)
        c2 = GradientBoostingClassifier(random_state=0, max_depth=3, n_estimators=200, learning_rate=0.05).fit(X[tr], perm)
        null.append(roc_auc_score(lab[te], c2.predict_proba(X[te])[:, 1]))
    out.update(n_train=int(tr.sum()), n_test=int(te.sum()), auc_test=float(auc), auc_null_mean=float(np.mean(null)), auc_null_q95=float(np.quantile(null, 0.95)),
               p=float(np.mean(np.array(null) >= auc)), importances=dict(zip(feats, np.round(clf.feature_importances_, 3).tolist())))
    # single-feature check: which raw measurements differ between near and far tracks (test days)
    out["per_feature_auc_test"] = {k: float(roc_auc_score(lab[te], X[te, i])) for i, k in enumerate(feats)}
    return out


if __name__ == "__main__":
    trees, bounds = sketch_trees()
    res = dict(step1=step1(trees, bounds), step3=step3())
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "candidates"} for k, v in res.items()}, indent=1))
    json.dump(res, open(CITY / "analysis.json", "w"))
