"""Independent checks of the v2 crossing detector (definitions fixed in DECISIONS.md,
3 Nov 2026, before scoring):
 T1  storm-drain records (Comune di Savona) vs detections, chance from shifted records;
 T2  600 vs 200 MHz agreement (separate antenna rows, 0.35 m apart), chance from shifted 200 MHz list.
Run: python eval_crossings_v2.py configs/dev_swath_v2.yaml xing_<hash>.json"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from build_v2 import load_cfg
from public_records import records_in_footprint

ROOT = Path(__file__).resolve().parents[1]


def drain_targets(cube: dict) -> list[dict]:
    """Storm-drain crossings in the footprint with |phi| <= 60 deg and recorded depth <= 150 cm
    or blank; records closer than 1 m in s0 are merged (same pipe split in two records)."""
    recs = records_in_footprint(ROOT / "data_public/gfi_all_condotte_acque_bianche.json",
                                cube["s_nodes"], cube["xn"], cube["yn"], float(cube["k"]))
    out = []
    for r in recs:
        dep = [abs(float(x)) for x in r["depth_cm"] if x not in ("", None)]
        if r["kind"] == "crossing" and abs(r["phi_deg"]) <= 60 and (not dep or min(dep) <= 150):
            r["depth_rec_m"] = min(dep) / 100 if dep else None
            out.append(r)
    out.sort(key=lambda r: r["s0"])
    merged = []
    for r in out:
        if merged and abs(r["s0"] - merged[-1]["s0"]) < 1.0:
            continue
        merged.append(r)
    return merged


def hits(targets_s: np.ndarray, det_s: np.ndarray, tol: float) -> np.ndarray:
    if len(det_s) == 0:
        return np.zeros(len(targets_s), bool)
    d = np.abs(targets_s[:, None] - det_s[None, :])
    return d.min(1) <= tol


def shifted(s: np.ndarray, off: float, lo: float, hi: float) -> np.ndarray:
    """Shift positions by off and wrap into [lo, hi) (keeps the spacing pattern)."""
    return lo + np.mod(s + off - lo, hi - lo)


def main(cfg_path: str, xing_json: str) -> dict:
    cfg = load_cfg(cfg_path)
    cache = Path(cfg["base"]["cache_dir"])
    cube = np.load(cache / cfg["cube_file"])
    res = json.load(open(cache / xing_json))
    rng = np.random.default_rng(cfg["seed"])
    qmax = cfg["crossing"]["q_max"]
    C = {a: res["arrays"][a]["candidates"] for a in res["arrays"]}
    sig = {a: [c for c in C[a] if c["q"] <= qmax] for a in C}
    lo, hi = 2.0, min(max(c["s0_m"] for c in C[a]) for a in C)
    out = {}
    # T1
    tg = drain_targets(cube)
    ts = np.array([t["s0"] for t in tg])
    det = np.array([c["s0_m"] for a in sig for c in sig[a]])
    h = hits(ts, det, 1.5)
    offs = rng.uniform(20, 200, 2000) * rng.choice([-1, 1], 2000)
    chance = np.array([hits(shifted(ts, o, lo, hi), det, 1.5).mean() for o in offs])
    out["T1"] = dict(n_targets=len(ts), hit_frac=float(h.mean()), chance_mean=float(chance.mean()),
                     p_perm=float((chance >= h.mean()).mean()),
                     per_target=[dict(s0=round(t["s0"], 1), phi=round(t["phi_deg"]), depth_rec_m=t["depth_rec_m"],
                                      diam=t["diam"], hit=bool(x),
                                      nearest_m=round(float(np.min(np.abs(det - t["s0"]))), 2) if len(det) else None)
                                 for t, x in zip(tg, h)])
    for a in sig:   # per frequency (exploratory, not pre-registered)
        da = np.array([c["s0_m"] for c in sig[a]])
        ha = hits(ts, da, 1.5); ca = np.array([hits(shifted(ts, o, lo, hi), da, 1.5).mean() for o in offs[:500]])
        out[f"T1_arr{a}"] = dict(hit_frac=float(ha.mean()), chance_mean=float(ca.mean()), p_perm=float((ca >= ha.mean()).mean()))
    # T2: shallow 600 MHz detections (depth <= 1.0 m) confirmed by a 200 MHz detection
    a6 = [c for c in sig["1"] if c["depth_m"] <= 1.0]
    s6 = np.array([c["s0_m"] for c in a6]); p6 = np.array([c["phi_deg"] for c in a6]); z6 = np.array([c["depth_m"] for c in a6])
    s2 = np.array([c["s0_m"] for c in sig["2"]]); p2 = np.array([c["phi_deg"] for c in sig["2"]]); z2 = np.array([c["depth_m"] for c in sig["2"]])

    def agree(s2x):
        ok = (np.abs(s6[:, None] - s2x[None]) <= 0.3) & (np.abs(p6[:, None] - p2[None]) <= 15) & (np.abs(z6[:, None] - z2[None]) <= 0.2)
        return ok.any(1)
    ag = agree(s2)
    ch2 = np.array([agree(shifted(s2, o, lo, hi)).mean() for o in offs[:500]])
    out["T2"] = dict(n_600_shallow=len(a6), agree_frac=float(ag.mean()), chance_mean=float(ch2.mean()),
                     p_perm=float((ch2 >= ag.mean()).mean()))
    # agreement as a function of the 600 MHz score decile (does confidence mean anything?)
    sc6 = np.array([c["score"] for c in a6])
    qs = np.quantile(sc6, [0, 0.5, 0.8, 0.9, 1.0]) if len(sc6) else []
    out["T2_by_score"] = [dict(score_range=[round(float(qs[i]), 2), round(float(qs[i + 1]), 2)],
                               agree_frac=float(ag[(sc6 >= qs[i]) & (sc6 <= qs[i + 1])].mean()))
                          for i in range(len(qs) - 1)]
    out["n_sig"] = {a: len(sig[a]) for a in sig}
    # T1b (EXPLORATORY, added after T1 was scored): place detections with the post-processed
    # trajectory instead of the real-time one, i.e. s -> s + along-track(PP - RT) at that s.
    from ids_reader import Swath
    from scipy.ndimage import median_filter
    from trajectory_qc import along_cross_disagreement
    sw = Swath.load(cfg["base"]["survey_dir"], cfg["base"]["swath"])
    dq = along_cross_disagreement(sw.path, sw.gga_pp, float(cube["k"]))
    along = median_filter(dq["along"], 51, mode="nearest")          # 5 s median, robust to PP spikes
    det_pp = det + np.interp(det, dq["s_rt"], along)
    h2 = hits(ts, det_pp, 1.5)
    chance2 = np.array([hits(shifted(ts, o, lo, hi), det_pp, 1.5).mean() for o in offs])
    out["T1b_exploratory"] = dict(hit_frac=float(h2.mean()), chance_mean=float(chance2.mean()),
                                  p_perm=float((chance2 >= h2.mean()).mean()), clock_offset_s=dq["clock_offset_s"],
                                  along_pp_minus_rt_quartiles=np.round(np.percentile(dq["along"], [25, 50, 75]), 2).tolist(),
                                  nearest_m=[round(float(np.min(np.abs(det_pp - x))), 2) for x in ts])
    return out


if __name__ == "__main__":
    r = main(sys.argv[1], sys.argv[2])
    print(json.dumps({k: v for k, v in r.items() if k != "T1"}, indent=1))
    t1 = dict(r["T1"]); per = t1.pop("per_target"); print("T1", t1)
    for p in per:
        print("  ", p)
