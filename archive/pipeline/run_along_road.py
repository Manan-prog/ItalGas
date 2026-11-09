"""L2: along-road detection on held-out swaths + municipal-record test. Everything is
processed in memory; only tracks, small plan maps and radargram excerpts at each record are
saved. Run: python run_along_road.py configs/along_road_heldout.yaml
Output: cache/along_<session>_<swath>.json/.npz per swath and cache/along_summary_<hash>.json."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from along_road import longitudinal_records, match, shifted_targets
from build_v2 import geometry_for, grid_Z, load_cfg, preprocess_array
from eval_longitudinal_v2 import agree
from ids_reader import Swath
from longitudinal import stack_along_s
from run_longitudinal_v2 import detect_longitudinal

ROOT = Path(__file__).resolve().parents[1]


def excerpts(Z, u, s, dt, targets, win_m=2.0, half_len_m=15.0):
    """Per target: 2 m coherent across-street section at the target middle (n_ch, n_t) and the
    along-street B-scan of the channel nearest the target (n_s window, n_t), float16."""
    out = {}
    for i, g in enumerate(targets):
        sm = 0.5 * (g["s0"] + g["s1"]); w = (s > sm - win_m / 2) & (s < sm + win_m / 2)
        out[f"x{i}"] = Z[:, w].mean(1).astype(np.float16)
        c = int(np.argmin(np.abs(u - g["u0"]))); w2 = (s > sm - half_len_m) & (s < sm + half_len_m)
        out[f"b{i}"] = Z[c, w2].astype(np.float16); out[f"bs{i}"] = np.array([s[w2][0], s[w2][-1], u[c]])
    return out


def run_swath(det_cfg: dict, raw_root: str, sess: str, surv: str, swath: str, records: list[Path], rng) -> dict:
    cfg = copy.deepcopy(det_cfg)
    cfg["base"]["survey_dir"] = str(Path(raw_root) / sess / surv); cfg["base"]["swath"] = swath; cfg["cube_file"] = None
    sw = Swath.load(cfg["base"]["survey_dir"], swath)
    geo = geometry_for(cfg, sw)
    s_max = float(min(np.nanmax(geo["s1"]), np.nanmax(geo["s2"])))
    s = np.arange(0.0, s_max, cfg["grid"]["ds_m"])
    res = dict(session=sess, survey=surv, swath=swath, length_m=s_max, arrays={}, corr={a: float(geo.get(f"corr{a}", np.nan)) for a in (1, 2)})
    u2 = np.nanmedian(geo["u2"], 0)
    tg = longitudinal_records(records, geo["s_nodes"], geo["xn"], geo["yn"], float(geo["k"]), (float(u2.min()), float(u2.max())))
    res["targets"] = tg
    npz = {}
    for a in (1, 2):
        pp = preprocess_array(sw, a, cfg)
        Z = grid_Z(pp["X"], geo[f"s{a}"], s); u = np.nanmedian(geo[f"u{a}"], 0); del pp["X"]
        tracks, info = detect_longitudinal(Z, u, s, pp["dt_ns"], pp["snr"], cfg, rng)
        for k_ in ("plan", "plan_t", "centres", "ug"):
            npz[f"{k_}{a}"] = info.pop(k_)
        for k_, v_ in excerpts(Z, u, s, pp["dt_ns"], tg).items():
            npz[f"{k_}_{a}"] = v_
        npz[f"u_ch{a}"] = u; npz[f"dt{a}"] = pp["dt_ns"]
        res["arrays"][a] = dict(**info, tracks=tracks, burst_frac=float(pp["burst"].mean()))
        del Z
    return res, npz


def main(cfg_path: str) -> Path:
    rc = yaml.safe_load(open(cfg_path))
    det = load_cfg(str(ROOT / "pipeline" / rc["detector_config"]))
    cache = Path(det["base"]["cache_dir"]); rng = np.random.default_rng(det["seed"])
    records = [ROOT / r for r in rc["records"]]; q_max = det["longitudinal"]["q_max"]
    summary = dict(swaths=[]); t0 = time.time()
    all_hits, all_chance = [], np.zeros(rc["n_chance"])
    n_targets = 0
    for sess, surv, swath in rc["swaths"]:
        tag = f"{sess[-19:]}_{surv[-8:]}_{swath}".replace(" ", "_").replace(".", "")
        fj = cache / f"along_{tag}.json"
        if fj.exists():
            res = json.load(open(fj))
        else:
            res, npz = run_swath(det, rc["raw_root"], sess, surv, swath, records, rng)
            json.dump(res, open(fj, "w")); np.savez(cache / f"along_{tag}.npz", **npz)
        tg = res["targets"]; trk = res["arrays"]["1"]["tracks"] + res["arrays"]["2"]["tracks"] if "1" in res["arrays"] else \
            res["arrays"][1]["tracks"] + res["arrays"][2]["tracks"]
        h = match(trk, tg, q_max, 0.5, None, 0.3)
        offs = rng.uniform(20, 200, rc["n_chance"]) * rng.choice([-1, 1], rc["n_chance"])
        ch = np.array([match(trk, shifted_targets(tg, o, 0.0, res["length_m"]), q_max, 0.5, None, 0.3).sum() for o in offs]) if tg else np.zeros(rc["n_chance"])
        all_hits += h.tolist(); all_chance += ch; n_targets += len(tg)
        arrs = res["arrays"]; A1 = arrs.get("1", arrs.get(1)); A2 = arrs.get("2", arrs.get(2))
        a10 = [t for t in A1["tracks"] if t["length_m"] >= 10]; a20 = [t for t in A2["tracks"] if t["length_m"] >= 10]
        t3 = float(agree(a10, a20, 0.0, 0.0, res["length_m"]).mean()) if a10 and a20 else None
        row = dict(tag=tag, length_m=round(res["length_m"]), n_targets=len(tg), hits=int(h.sum()), chance_mean=float(ch.mean()),
                   n_sig600=A1["n_sig"], n_sig200=A2["n_sig"], t3_agree=t3, t_search_ns={1: A1["t_search_ns"], 2: A2["t_search_ns"]})
        summary["swaths"].append(row)
        print(row, f"{time.time()-t0:.0f}s", flush=True)
    hits = np.array(all_hits)
    summary["pooled"] = dict(n_targets=n_targets, hit_frac=float(hits.mean()) if n_targets else None,
                             chance_frac=float(all_chance.mean() / max(1, n_targets)),
                             p_perm=float((all_chance >= hits.sum()).mean()))
    h_ = hashlib.md5(open(cfg_path, "rb").read()).hexdigest()[:8]
    out = cache / f"along_summary_{h_}.json"; json.dump(summary, open(out, "w"))
    print("POOLED", summary["pooled"]); print("saved", out)
    return out


if __name__ == "__main__":
    main(sys.argv[1])
