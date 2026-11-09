"""Diagnostic (NOT pre-registered) for E32: is the learned evidence better than the hand-built one, independent of
the path-significance threshold? On Via Fontanassa with 24 new fake pipes: (1) cell-level ROC AUC of each evidence,
fake-pipe cells vs all cells away from them; (2) fraction of fake pipes matched by any of the top 30 paths (no
significance cut); (3) the best-path score a 20 m pipe can reach vs the permutation threshold.
Run: python diag_learned_emission.py -> cache/city/learned_diag.json"""
from __future__ import annotations

import json
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score

import run_learned_viterbi as R
from along_road import inject_longitudinal, peak_frequency_mhz
from build_cube import swath_geometry
from build_v2 import grid_Z, preprocess_array
from ids_reader import Swath
from learned_emission import FEATURES, cell_features, labels_from_segments, to_llr
from network_dp import viterbi_uz_multi
from run_viterbi_uz import evidence as handbuilt_evidence

R.rng = np.random.default_rng(R.DET["seed"] + 97)
M = pickle.load(open(R.CACHE / "city" / "learned_emission_model.pkl", "rb")); clf, prior = M["clf"], M["prior"]
sw = Swath.load("/Users/manansarupria/ItalGas/Savona/savona 1410 matt_2025.10.15_09.07.28/Survey_2025-10-15_09-11-09", "Swath016")
g = swath_geometry(sw, R.DET["base"]); S = g["s2"]; pp = preprocess_array(sw, 2, R.DET); LF = float(np.nanmax(np.nanmedian(S, 1)))
s = np.arange(0.0, LF, R.DET["grid"]["ds_m"]); Z = grid_Z(pp["X"], S, s); u = np.nanmedian(g["u2"], 0); dt = pp["dt_ns"]; del pp
segs = R.fake_pipes(30, LF - 30, u, 24); f = peak_frequency_mhz(Z, dt, (3.0, 15.0))
X, sc, ug = R.stacks(inject_longitudinal(Z, u, s, dt, R.V, segs, f), s, u)
F, zb = cell_features(X, u, ug, dt, R.V); pos, excl = labels_from_segments(segs, sc, ug, zb)
L_learn = to_llr(clf.predict_proba(F.reshape(-1, len(FEATURES)))[:, 1].reshape(F.shape[:3]), prior)
E = handbuilt_evidence(X, u, ug, dt, R.V)[0]; L_hand = np.clip((E - E.mean((0, 1))) / (E.std((0, 1)) + 1e-9) - 2.0, -2.0, 4.0)
keep = pos | ~excl; y = pos[keep]
out = dict(n_fake=len(segs))
for name, L, raw in (("learned", L_learn, L_learn), ("hand-built", L_hand, E)):
    auc = roc_auc_score(y, raw[keep]); paths = viterbi_uz_multi(L, k_paths=30, min_steps=20)
    tr = R.tracks_from(paths, sc, ug, zb, -np.inf); hit, _ = R.recovery(tr, segs)
    on = [float(np.clip(L[pos[:, :, :]].mean(), -9, 9))]
    best20 = float(np.sort(L.reshape(len(sc), -1).max(1))[::-1][:40].sum())
    by = {f"strength {a:g}": f"{hit[[i for i, q in enumerate(segs) if q['amp'] == a]].sum()}/{sum(q['amp'] == a for q in segs)}" for a in (2.0, 3.0, 5.0, 8.0)}
    out[name] = dict(cell_auc=float(auc), top30_recovered=int(hit.sum()), by_strength=by, mean_llr_on_pipe=on[0],
                     top_path_scores=[round(float(p[4]), 1) for p in paths[:10]], top_path_lengths_m=[round(float(sc[p[1] - 1] - sc[p[0]]), 1) for p in paths[:10]])
    print(f"{name}: cell AUC {auc:.3f}; fake pipes matched by any of the top 30 paths {hit.sum()}/{len(segs)} {by}; "
          f"mean emission on fake-pipe cells {on[0]:+.2f}; top path scores {out[name]['top_path_scores'][:6]} lengths {out[name]['top_path_lengths_m'][:6]}", flush=True)
json.dump(out, open(R.CACHE / "city" / "learned_diag.json", "w"))
