"""Real-data tests of the end-point detector on the subset drives (DECISIONS 2026-11-04, V2-V4).
Significant end points (q <= 0.05) not coinciding with a significant full crossing of the same array are pooled over
both rows; 'both-row' ends (600 and 200 within 0.5 m along, 0.25 m across, same side) are reported separately.
(V2) on a municipal main recorded along the strip (E15 targets) within 0.3 m sideways vs ends shifted 20-200 m along
the drive (1000 draws); (V3) the same end point seen by two drives in independent lanes (>= 0.5 m) within 0.3 m in the
map vs the other drive mirrored about its strip centre; (V4) main segments = >= 3 end points within +-0.2 m sideways
over 50 m along a drive vs end positions permuted among the drive's ends (200 draws; pre-registered 1000, reduced for run time).
Run: python ends_analysis.py configs/subset_merge.yaml -> cache/city/ends_analysis.json"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

from along_road import longitudinal_records
from build_v2 import load_cfg
from citywide_merge import strip_centre
from run_detection import su_to_map

ROOT = Path(__file__).resolve().parents[1]; MIN_SEP = 0.5


def sig_ends(E: list[dict], X: list[dict]) -> list[dict]:
    return [e for e in E if e["q"] <= 0.05 and not any(c["q"] <= 0.05 and abs(c["s0_m"] - e["s0_m"]) <= 0.5 and abs(c["phi_deg"] - e["phi_deg"]) <= 15
                                                           and abs(c["depth_m"] - e["depth_m"]) <= 0.15 for c in X)]


def segments(s: np.ndarray, u: np.ndarray, win: float = 50.0, du: float = 0.2, n_min: int = 3) -> list[tuple]:
    """Greedy main segments: windows of win m along s holding >= n_min ends within +-du of their median u."""
    o = np.argsort(s); s, u = s[o], u[o]; used = np.zeros(len(s), bool); out = []
    for i in range(len(s)):
        if used[i]:
            continue
        w = np.nonzero((s >= s[i]) & (s < s[i] + win) & ~used)[0]
        if len(w) < n_min:
            continue
        best = None
        for c in u[w]:                                          # densest +-du group in the window
            g = w[np.abs(u[w] - c) <= du]
            if best is None or len(g) > len(best):
                best = g
        if len(best) >= n_min:
            used[best] = True; out.append((float(s[best].min()), float(s[best].max()), float(np.median(u[best])), len(best)))
    return out


def main(cfg_path: str) -> None:
    cc = yaml.safe_load(open(cfg_path)); det = load_cfg(str(ROOT / "pipeline" / cc["detector_config"])); B = cc["along_travel_bias_m"]
    C = Path(det["base"]["cache_dir"]) / "city"; rng = np.random.default_rng(det["seed"] + 141)
    rec_files = [ROOT / "data_public" / "gfi_all_condotte_acque_bianche.json", ROOT / "data_public" / "gfi_all_condotte_acque_nere.json"]
    D = {}
    for i in cc["drives"]:
        fe, fs = C / f"ends_{i}.json", C / f"scan_{i}.json"
        if not fe.exists() or json.load(open(fe)).get("status") != "ok":
            continue
        E = json.load(open(fe)); S = json.load(open(fs)); fr = dict(np.load(C / f"scan_{i}_frame.npz"))
        ends = {a: sig_ends(E["ends"][a], S["per_array"][a]) for a in ("1", "2")}
        both = [e for e in ends["2"] if any(abs(f["s_end_m"] - e["s_end_m"]) <= 0.5 and abs(f["u_end_m"] - e["u_end_m"]) <= 0.25 and f["side"] == e["side"] for f in ends["1"])]
        pooled = ends["1"] + ends["2"]
        ok = np.isfinite(fr["lo"]); tg = longitudinal_records(rec_files, fr["s_nodes"], fr["xn"], fr["yn"], float(fr["k"]), (float(np.nanmin(fr["lo"])), float(np.nanmax(fr["hi"]))))
        D[i] = dict(L=S["length_m"], fr=fr, pooled=pooled, both=both, n600=len(ends["1"]), n200=len(ends["2"]), targets=tg)
    out = dict(drives={i: dict(km=d["L"] / 1e3, n600=d["n600"], n200=d["n200"], n_both=len(d["both"]), n_targets=len(d["targets"])) for i, d in D.items()})
    for cls in ("pooled", "both"):
        # V2
        def hits(shift=None):
            n = 0
            for i, d in D.items():
                for e in d[cls]:
                    s = e["s_end_m"] if shift is None else (e["s_end_m"] + shift[i]) % d["L"]
                    n += any(t["s0"] <= s <= t["s1"] and abs(e["u_end_m"] - t["u0"]) <= 0.3 for t in d["targets"])
            return n
        n_e = sum(len(d[cls]) for d in D.values()); real = hits()
        null = np.array([hits({i: rng.choice([-1, 1]) * rng.uniform(20, 200) for i in D}) for _ in range(1000)])
        v2 = dict(n_ends=n_e, n_on_record=real, chance=float(null.mean()), p=float((np.sum(null >= real) + 1) / 1001))
        # V3
        def pts(mirror=False):
            rows = []
            for i, d in D.items():
                fr = d["fr"]
                for e in d[cls]:
                    s = np.array([e["s_end_m"] + B]); c = strip_centre(fr, s - B); u = 2 * c - e["u_end_m"] if mirror else np.array([e["u_end_m"]])
                    x, y = su_to_map(s, u, fr["s_nodes"], fr["xn"], fr["yn"], float(fr["k"])); cx, cy = su_to_map(s, c, fr["s_nodes"], fr["xn"], fr["yn"], float(fr["k"]))
                    rows.append((i, x[0], y[0], e["depth_m"], cx[0], cy[0]))
            return np.array(rows) if rows else np.zeros((0, 6))
        k = float(np.median([float(d["fr"]["k"]) for d in D.values()])) if D else 1.0
        def agree(A, Bm):
            if not len(A) or not len(Bm):
                return 0
            tr = cKDTree(Bm[:, 1:3]); n = 0
            for p in A:
                for j in tr.query_ball_point(p[1:3], 0.3 / k):
                    q = Bm[j]
                    if q[0] != p[0] and abs(q[3] - p[3]) <= 0.15 and np.hypot(q[4] - p[4], q[5] - p[5]) * k >= MIN_SEP:
                        n += 1; break
            return n
        P, Pm = pts(), pts(True)
        v3 = dict(n_ends=len(P), repeated=agree(P, P), chance=agree(P, Pm))
        # V4
        segs, nulls = {}, []
        for i, d in D.items():
            s = np.array([e["s_end_m"] for e in d[cls]]); u = np.array([e["u_end_m"] for e in d[cls]]); segs[i] = segments(s, u) if len(s) else []
        for _ in range(200):                                    # 200 draws: the greedy segment search is slow in Python
            nulls.append(sum(len(segments(np.array([e["s_end_m"] for e in d[cls]]), rng.permutation([e["u_end_m"] for e in d[cls]]))) if d[cls] else 0 for d in D.values()))
        n_seg = sum(len(v) for v in segs.values())
        v4 = dict(n_segments=n_seg, chance=float(np.mean(nulls)), p=float((np.sum(np.array(nulls) >= n_seg) + 1) / (len(nulls) + 1)),
                  segments={i: v for i, v in segs.items() if v})
        out[cls] = dict(V2=v2, V3=v3, V4=v4); print(cls, json.dumps(dict(V2=v2, V3=v3, V4={k_: v4[k_] for k_ in ("n_segments", "chance", "p")})), flush=True)
    json.dump(out, open(C / "ends_analysis.json", "w"), default=float)


if __name__ == "__main__":
    main(sys.argv[1])
