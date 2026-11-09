"""E43 V2 + V4 on the 12 subset drives: ground-frame along-street tracks vs the instrument frame and the mirror control.
V2: tracks found only in the ground frame (no instrument track overlapping >= 50 percent within 0.3 m in the map), fraction on
stretches where the van moved >= 0.3 m sideways, and the same count for the mirror (-y_van) run; sign test over drives.
V4: municipal storm/sewer records running along the strip (E15 rule) hit by significant tracks of each frame, vs records
shifted 20-200 m (1000 draws). Run: python frame_analysis.py configs/dither.yaml -> cache/city/frame_analysis.json"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import binomtest

from along_road import longitudinal_records
from build_cube import swath_geometry
from build_v2 import load_cfg
from dither import shift_at, sideways, uc_at, y_at
from ids_reader import Swath
from run_injection_raw import records_score

ROOT = Path(__file__).resolve().parents[1]


def sig(tr): return [t for t in tr if t["q"] <= 0.05 and t["length_m"] >= 10]


def covered(t: dict, others: list[dict], k: float, d_max: float = 0.3, min_ov: float = 0.5) -> bool:
    """True if some track in others overlaps t by >= min_ov of t's length along s and lies within d_max m (map, median)."""
    for o in others:
        ov = min(t["s_end"], o["s_end"]) - max(t["s_start"], o["s_start"])
        if ov < min_ov * t["length_m"]:
            continue
        m = (np.array(t["path_s"]) >= o["s_start"]) & (np.array(t["path_s"]) <= o["s_end"])
        if not m.any():
            continue
        tx, ty = np.array(t["x"])[m], np.array(t["y"])[m]; ox, oy = np.array(o["x"]), np.array(o["y"])
        d = np.median(np.min(np.hypot(tx[:, None] - ox[None, :], ty[:, None] - oy[None, :]), 1)) * k
        if d <= d_max:
            return True
    return False


def main(cfg_path: str) -> None:
    cc = yaml.safe_load(open(cfg_path)); det = load_cfg(str(ROOT / "pipeline" / cc["detector_config"])); cache = Path(det["base"]["cache_dir"]) / "city"
    files = sorted(glob.glob(str(Path(cc["raw_root"]) / "*" / "Survey_*" / "Swath*.path"))); B = cc["along_travel_bias_m"]
    recs = [ROOT / "data_public" / "gfi_all_condotte_acque_bianche.json", ROOT / "data_public" / "gfi_all_condotte_acque_nere.json"]
    per = {}; V4 = {fr: dict(hits=0, targets=0, chance=0.0) for fr in ("instrument", "ground", "mirror")}; rng = np.random.default_rng(det["seed"] + 101)
    for i in cc["drives"]:
        T = {}
        for fr, tag in (("instrument", "_c"), ("ground", "_g"), ("mirror", "_gm")):
            f = cache / f"along{tag}_{i}.json"; T[fr] = sig(json.load(open(f))["tracks"]) if f.exists() else None
        if any(v is None for v in T.values()):
            per[i] = dict(missing=[fr for fr, v in T.items() if v is None]); continue
        fp = files[i]; sw = Swath.load(str(Path(fp).parent), Path(fp).stem); g = swath_geometry(sw, det["base"]); k = float(g["k"]); sv = sideways(g, B, cc["L_ref_m"])
        g_only = [t for t in T["ground"] if not covered(t, T["instrument"], k)]
        m_only = [t for t in T["mirror"] if not covered(t, T["instrument"], k)]
        per[i] = dict(n_inst=len(T["instrument"]), n_ground=len(T["ground"]), n_mirror=len(T["mirror"]), ground_only=len(g_only), mirror_only=len(m_only),
                      ground_only_moving=sum(t.get("y_range_m", 0) >= 0.3 for t in g_only), mirror_only_moving=sum(t.get("y_range_m", 0) >= 0.3 for t in m_only),
                      ground_only_len_m=round(sum(t["length_m"] for t in g_only), 1), km=round(float(g["s_nodes"][-1]) / 1e3, 2))
        # V4: records along the strip, matched in the drive frame (ground tracks converted: u_frame = u_ground - y_van)
        u2 = np.nanmedian(g["u2"], 0); L = float(g["s_nodes"][-1])
        tg = longitudinal_records(recs, g["s_nodes"], g["xn"], g["yn"], k, (float(np.nanmin(u2)), float(np.nanmax(u2))))
        for fr in ("instrument", "ground", "mirror"):
            tr = []
            for t in T[fr]:
                t2 = dict(t)
                if fr != "instrument":
                    ps = np.array(t["path_s"]); yv = (y_at(sv, ps) if fr == "ground" else -y_at(sv, ps)) - uc_at(sv, ps)
                    t2["u_mean"] = float(np.nanmean(np.array(t["path_u"]) - yv)) if np.isfinite(yv).any() else t["u_mean"]
                tr.append(t2)
            r = records_score(tr, tg, 0.0, L, rng) if tg else dict(n_targets=0, hits=0, chance_mean=0.0)
            V4[fr]["hits"] += r.get("hits", 0); V4[fr]["targets"] += r["n_targets"]; V4[fr]["chance"] += r.get("chance_mean", 0.0)
        print(i, per[i], flush=True)
    ok = [p for p in per.values() if "missing" not in p]
    wins = sum(p["ground_only"] > p["mirror_only"] for p in ok); ties = sum(p["ground_only"] == p["mirror_only"] for p in ok)
    summ = dict(n_drives=len(ok), ground_only=sum(p["ground_only"] for p in ok), mirror_only=sum(p["mirror_only"] for p in ok),
                ground_only_moving=sum(p["ground_only_moving"] for p in ok), mirror_only_moving=sum(p["mirror_only_moving"] for p in ok),
                drives_ground_gt_mirror=wins, drives_tied=ties,
                sign_p=float(binomtest(wins, len(ok) - ties, 0.5, alternative="greater").pvalue) if len(ok) - ties > 0 else None, V4=V4)
    json.dump(dict(per_drive=per, summary=summ), open(cache / "frame_analysis.json", "w")); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
