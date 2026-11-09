"""v2.3 on the 12 held-out swaths: keep only along-road tracks whose cross-section is pipe-shaped
(track_shape) and redo the municipal-record test (same targets, same chance procedure as E15).
Run: python rescore_heldout_v23.py configs/along_road_heldout.yaml -> cache/along_v23_summary.json"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from along_road import match, shifted_targets, track_shape
from build_v2 import geometry_for, grid_Z, load_cfg, preprocess_array
from ids_reader import Swath

ROOT = Path(__file__).resolve().parents[1]


def main(cfg_path: str) -> None:
    rc = yaml.safe_load(open(cfg_path)); det = load_cfg(str(ROOT / "pipeline" / rc["detector_config"]))
    cache = Path(det["base"]["cache_dir"]); v = det["migration"]["velocity_m_ns"]; q_max = det["longitudinal"]["q_max"]
    rng = np.random.default_rng(det["seed"] + 23)
    rows = []; H22 = H23 = 0; C22 = np.zeros(rc["n_chance"]); C23 = np.zeros(rc["n_chance"]); N = 0
    for sess, surv, swath in rc["swaths"]:
        tag = f"{sess[-19:]}_{surv[-8:]}_{swath}".replace(" ", "_").replace(".", "")
        res = json.load(open(cache / f"along_{tag}.json")); tg = res["targets"]
        sig = [t for t in res["arrays"]["2"]["tracks"] if t["q"] <= q_max]
        cfg = copy.deepcopy(det); cfg["base"]["survey_dir"] = str(Path(rc["raw_root"]) / sess / surv); cfg["base"]["swath"] = swath; cfg["cube_file"] = None
        if sig:
            sw = Swath.load(cfg["base"]["survey_dir"], swath); geo = geometry_for(cfg, sw)
            s = np.arange(0.0, res["length_m"], det["grid"]["ds_m"])
            pp = preprocess_array(sw, 2, cfg); Z = grid_Z(pp["X"], geo["s2"], s); u = np.nanmedian(geo["u2"], 0)
            for t in sig:
                t.update(track_shape(Z, u, s, pp["dt_ns"], v, t))
            del Z, pp
        shaped = [t for t in sig if t.get("pipe_like")]
        h22 = match(sig, tg, q_max, 0.5, None, 0.3); h23 = match(shaped, tg, q_max, 0.5, None, 0.3)
        offs = rng.uniform(20, 200, rc["n_chance"]) * rng.choice([-1, 1], rc["n_chance"])
        c22 = np.array([match(sig, shifted_targets(tg, o, 0.0, res["length_m"]), q_max, 0.5, None, 0.3).sum() for o in offs]) if tg else np.zeros(len(offs))
        c23 = np.array([match(shaped, shifted_targets(tg, o, 0.0, res["length_m"]), q_max, 0.5, None, 0.3).sum() for o in offs]) if tg else np.zeros(len(offs))
        H22 += h22.sum(); H23 += h23.sum(); C22 += c22; C23 += c23; N += len(tg)
        rows.append(dict(tag=tag, n_targets=len(tg), n_sig=len(sig), n_pipe_shaped=len(shaped), hits_v22=int(h22.sum()), hits_v23=int(h23.sum()),
                         chance_v22=float(c22.mean()), chance_v23=float(c23.mean()),
                         shaped_tracks=[{k: t[k] for k in ("s_start", "s_end", "u_mean", "depth_m", "q", "sections")} for t in shaped]))
        print(rows[-1]["tag"], {k: rows[-1][k] for k in ("n_targets", "n_sig", "n_pipe_shaped", "hits_v22", "hits_v23", "chance_v22", "chance_v23")}, flush=True)
    summ = dict(n_targets=N, hits_v22=int(H22), chance_v22=float(C22.mean()), p_v22=float((C22 >= H22).mean()),
                hits_v23=int(H23), chance_v23=float(C23.mean()), p_v23=float((C23 >= H23).mean()),
                n_sig=sum(r["n_sig"] for r in rows), n_pipe_shaped=sum(r["n_pipe_shaped"] for r in rows))
    print("SUMMARY", summ)
    json.dump(dict(summary=summ, swaths=rows), open(cache / "along_v23_summary.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1])
