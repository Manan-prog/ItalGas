"""Train the learned along-road evidence on Corso Ricci (fake pipes in real data), test on Via Fontanassa:
(a) fake-pipe recovery (new fake pipes, other street) vs the hand-built evidence of E31;
(b) the municipal-record test of E31 on the real data.
Run: python run_learned_viterbi.py -> cache/city/learned_viterbi.json"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from along_road import inject_longitudinal, match, peak_frequency_mhz, shifted_targets
from build_cube import swath_geometry
from build_v2 import cfg_hash, grid_Z, load_cfg, preprocess_array
from ids_reader import Swath
from learned_emission import FEATURES, cell_features, labels_from_segments, to_llr
from longitudinal import stack_along_s
from network_dp import viterbi_uz_multi
from run_viterbi_uz import evidence as handbuilt_evidence

ROOT = Path(__file__).resolve().parents[1]; DET = load_cfg(str(ROOT / "pipeline/configs/dev_swath_v2_2.yaml")); V = DET["migration"]["velocity_m_ns"]
CACHE = Path(DET["base"]["cache_dir"]); rng = np.random.default_rng(DET["seed"] + 71)


def fake_pipes(s_lo, s_hi, u, n, avoid=()):
    slots = [s for s in np.arange(s_lo, s_hi - 25, 28.0) if not any(a <= s + 20 and s <= b for a, b in avoid)]
    pick = rng.choice(len(slots), min(n, len(slots)), replace=False)
    return [dict(s0=float(slots[i]), s1=float(slots[i] + 20), u0=float(rng.uniform(u.min() - 0.2, u.max() + 0.2)), z=float(rng.uniform(0.3, 1.3)),
                 radius_m=float(rng.choice([0.0, 0.05, 0.1, 0.25])), amp=float(rng.choice([2.0, 3.0, 5.0, 8.0])), wander_m=0.15, wander_period_m=15.0) for i in pick]


def stacks(Z, s, u):
    X, cen = stack_along_s(Z, DET["grid"]["ds_m"], 2.0, 0.5); ug = np.arange(u.min() - 0.3, u.max() + 0.3 + 1e-9, 0.04)
    return X, s[cen], ug


def tracks_from(paths, sc, ug, zb, thr):
    return [dict(s_start=float(sc[t0]), s_end=float(sc[t1 - 1]), u_mean=float(ug[iu].mean()), depth_m=float(np.median(zb[iz] + 0.03)), score=float(sc_),
                 q=0.0 if sc_ > thr else 1.0, path_s=sc[t0:t1:4].tolist(), path_u=ug[iu][::4].tolist()) for t0, t1, iu, iz, sc_ in paths]


def recovery(tracks, segs):
    hit = np.zeros(len(segs), bool)
    for i, g in enumerate(segs):
        for t in tracks:
            if t["q"] > 0.05:
                continue
            ov = min(t["s_end"], g["s1"]) - max(t["s_start"], g["s0"])
            if ov >= 0.5 * (g["s1"] - g["s0"]) and abs(t["u_mean"] - g["u0"]) <= 0.3 and abs(t["depth_m"] - g["z"]) <= 0.2:
                hit[i] = True; break
    outside = sum(1 for t in tracks if t["q"] <= 0.05 and not any(min(t["s_end"], g["s1"]) - max(t["s_start"], g["s0"]) > 0 for g in segs))
    return hit, outside


def viterbi_with_null(llr_fn, X, u, ug, dt, n_null=8):
    L = llr_fn(X); paths = viterbi_uz_multi(L, k_paths=10, min_steps=20)
    nb = []
    for _ in range(n_null):
        Ln = llr_fn(X[:, rng.permutation(len(u)), :]); p = viterbi_uz_multi(Ln, k_paths=1, min_steps=20); nb.append(p[0][4] if p else 0.0)
    return paths, float(np.quantile(nb, 0.95))


def main():
    # ---- training: Corso Ricci 200 MHz grid (cached), fake pipes avoiding the probable real main (E26 zones)
    d = np.load(CACHE / f"v2grid_2_{cfg_hash(DET)}.npz"); Z = d["Z"].astype(np.float32); s = d["s_grid"]; u = d["u_ch"]; dt = float(d["dt_full_ns"])
    f = peak_frequency_mhz(Z, dt, (3.0, 15.0)); avoid = [(690, 790), (850, 950), (1080, 1215)]
    Xtr_all, ytr_all = [], []
    for rep in range(2):
        segs = fake_pipes(30, 1180, u, 22, avoid)
        X, sc, ug = stacks(inject_longitudinal(Z, u, s, dt, V, segs, f), s, u)
        F, zb = cell_features(X, u, ug, dt, V); pos, excl = labels_from_segments(segs, sc, ug, zb)
        zone = np.zeros(len(sc), bool)
        for a, b in avoid:
            zone |= (sc >= a) & (sc <= b)
        neg = ~excl & ~zone[:, None, None]
        ip = np.argwhere(pos); ineg = np.argwhere(neg); ineg = ineg[rng.choice(len(ineg), min(len(ineg), 20 * len(ip)), replace=False)]
        Xtr_all.append(np.r_[F[tuple(ip.T)], F[tuple(ineg.T)]]); ytr_all.append(np.r_[np.ones(len(ip)), np.zeros(len(ineg))])
        print(f"training rep {rep}: positives {len(ip)}, negatives {len(ineg)}", flush=True)
    Xtr = np.vstack(Xtr_all); ytr = np.concatenate(ytr_all); prior = float(ytr.mean())
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0).fit(Xtr, ytr)
    learned = lambda Xs: to_llr(clf.predict_proba(cell_features(Xs, uF, ugF, dtF, V)[0].reshape(-1, len(FEATURES)))[:, 1].reshape(Xs.shape[0], len(ugF), -1), prior)
    # ---- test street: Via Fontanassa (15 Oct Swath016)
    sw = Swath.load("/Users/manansarupria/ItalGas/Savona/savona 1410 matt_2025.10.15_09.07.28/Survey_2025-10-15_09-11-09", "Swath016")
    g = swath_geometry(sw, DET["base"]); S = g["s2"]; pp = preprocess_array(sw, 2, DET); LF = float(np.nanmax(np.nanmedian(S, 1)))
    sF = np.arange(0.0, LF, DET["grid"]["ds_m"]); ZF = grid_Z(pp["X"], S, sF); uF = np.nanmedian(g["u2"], 0); dtF = pp["dt_ns"]; del pp
    fF = peak_frequency_mhz(ZF, dtF, (3.0, 15.0))
    out = dict(features=FEATURES, n_train=int(len(ytr)), prior=prior)
    # (a) fake-pipe recovery on the other street: learned vs hand-built evidence
    segsF = fake_pipes(30, LF - 30, uF, 24)
    XF, scF, ugF = stacks(inject_longitudinal(ZF, uF, sF, dtF, V, segsF, fF), sF, uF)
    _, zb = cell_features(XF[:5], uF, ugF, dtF, V)
    hand = lambda Xs: (lambda E: np.clip((E - E.mean((0, 1))) / (E.std((0, 1)) + 1e-9) - 2.0, -2.0, 4.0))(handbuilt_evidence(Xs, uF, ugF, dtF, V)[0])
    for name, fn in (("learned", learned), ("hand-built", hand)):
        paths, thr = viterbi_with_null(fn, XF, uF, ugF, dtF); tr = tracks_from(paths, scF, ugF, zb, thr); hit, outside = recovery(tr, segsF)
        by = {f"strength {a:g}": f"{hit[[i for i, q in enumerate(segsF) if q['amp'] == a]].sum()}/{sum(q['amp'] == a for q in segsF)}" for a in (2.0, 3.0, 5.0, 8.0)}
        out[f"injection_{name}"] = dict(recovered=int(hit.sum()), of=len(segsF), false_significant_paths=int(outside), by_strength=by)
        print(f"fake pipes on Via Fontanassa, {name} evidence: recovered {hit.sum()}/{len(segsF)} {by}, significant paths outside fake pipes {outside}", flush=True)
    # (b) real data: municipal-record test (as E31)
    XF0, scF0, _ = stacks(ZF, sF, uF)
    paths, thr = viterbi_with_null(learned, XF0, uF, ugF, dtF); tr = tracks_from(paths, scF0, ugF, zb, thr)
    tg = json.load(open(CACHE / "along_20251015_090728_09-11-09_Swath016.json"))["targets"]
    hit = match(tr, tg, 0.05, 0.5, None, 0.3); offs = rng.uniform(20, 200, 1000) * rng.choice([-1, 1], 1000)
    ch = np.array([match(tr, shifted_targets(tg, o, 0.0, LF), 0.05, 0.5, None, 0.3).sum() for o in offs])
    out["records"] = dict(hits=int(hit.sum()), of=len(tg), chance=float(ch.mean()), p=float((ch >= hit.sum()).mean()), threshold=thr,
                          tracks=[{k: t[k] for k in ("s_start", "s_end", "u_mean", "depth_m", "score", "q")} for t in tr])
    print(f"records on Via Fontanassa (learned): {hit.sum()} of {len(tg)} hit vs {ch.mean():.2f} by chance (p = {(ch >= hit.sum()).mean():.3f})")
    for t in tr:
        print(f"   path s {t['s_start']:.0f}-{t['s_end']:.0f} u {t['u_mean']:+.2f} depth {t['depth_m']:.2f} score {t['score']:.1f} {'SIGNIFICANT' if t['q'] == 0 else ''}")
    json.dump(out, open(CACHE / "city" / "learned_viterbi.json", "w"))
    import pickle; pickle.dump(dict(clf=clf, prior=prior), open(CACHE / "city" / "learned_emission_model.pkl", "wb"))


if __name__ == "__main__":
    main()
