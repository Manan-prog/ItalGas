"""Whole-survey checks and products from citywide_scan.py output (rules fixed in DECISIONS.md 2026-11-03 before results).
(C1) both-row agreement vs shifted 600 MHz list; (C2) repeat-drive crossing agreement (unregistered) vs drives shifted
along their own tracks; (C3) repeat-drive along-street agreement vs the other drive mirrored about its strip centre;
(C4) municipal storm/sewer record crossings hit by confident crossings vs records shifted along the drive.
Run: python citywide_merge.py configs/citywide_scan.yaml -> cache/city/citywide_merged.json"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from paths import load_yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from build_v2 import load_cfg
from cube import to_su
from geometry import lonlat_to_merc
from run_detection import su_to_map

ROOT = Path(__file__).resolve().parents[1]
Q = 0.05                                    # q threshold of a detection (frozen detector)
TOL_X = dict(perp=0.5, phi=15.0, along=6.0)  # m, deg, m (C2)
TOL_T = dict(du=0.4, dz=0.2, min_len=10.0)   # m (C3)
MIN_SEP = 0.5                                # m, independent-lane rule (DECISIONS 2026-11-04)
N_DRAWS = 50


def ratio(chance: float, real: float) -> float:
    """Estimated false fraction chance / real; 1.0 when nothing real was found (a rule cannot pass on no data)."""
    return float(chance / real) if real > 0 else 1.0


def load_all(cache: Path, along_tag: str | None = None, drives: list[int] | None = None) -> tuple[dict, dict]:
    """Crossing scans (scan_<i>.json + frame) of every drive; with along_tag, each drive's along-street tracks are
    taken from the separate along pass (along<tag>_<i>.json; empty, and flagged, where that pass has no result)."""
    S, F = {}, {}
    for f in glob.glob(str(cache / "scan_*.json")):
        r = json.load(open(f))
        if r.get("status") == "ok" and (drives is None or r["index"] in drives):
            S[r["index"]] = r; F[r["index"]] = dict(np.load(f.replace(".json", "_frame.npz")))
    if along_tag is not None:
        for i, r in S.items():
            fa = cache / f"along{along_tag}_{i}.json"; a = json.load(open(fa)) if fa.exists() else {}
            r["tracks"] = a.get("tracks", []) if a.get("status") == "ok" else []; r["along_ok"] = a.get("status") == "ok"
    return S, F


def strip_centre(fr: dict, s: np.ndarray) -> np.ndarray:
    lo, hi = fr["lo"], fr["hi"]; ok = np.isfinite(lo) & np.isfinite(hi)
    return np.interp(s, fr["s"][ok], 0.5 * (lo[ok] + hi[ok]))


# ------------------------------------------------------------------ C1
def c1_both_rows(S: dict, rng) -> dict:
    def matched(a6, a2):
        if not len(a6) or not len(a2):
            return 0
        d = np.abs(a6[:, None, :] - a2[None, :, :]); return int(((d[..., 0] <= 0.3) & (d[..., 1] <= 15) & (d[..., 2] <= 0.2)).any(1).sum())
    real = n6 = 0; null = np.zeros(N_DRAWS); per_day = {}
    for i, r in S.items():
        a6 = np.array([[c["s0_m"], c["phi_deg"], c["depth_m"]] for c in r["per_array"]["1"] if c["q"] <= Q]).reshape(-1, 3)
        a2 = np.array([[c["s0_m"], c["phi_deg"], c["depth_m"]] for c in r["per_array"]["2"] if c["q"] <= Q]).reshape(-1, 3)
        m = matched(a6, a2); real += m; n6 += len(a6); L = r["length_m"]; day = r["session"][-19:-9]
        pd = per_day.setdefault(day, [0, 0, 0.0]); pd[0] += m; pd[1] += len(a6)
        for d in range(N_DRAWS):
            sh = a6.copy(); sh[:, 0] = (sh[:, 0] + rng.choice([-1, 1]) * rng.uniform(5, 50)) % L
            mm = matched(sh, a2); null[d] += mm; pd[2] += mm / N_DRAWS
    return dict(n600=n6, matched=real, rate=real / max(n6, 1), chance_rate=float(null.mean() / max(n6, 1)),
                per_day={k: dict(n600=v[1], rate=v[0] / max(v[1], 1), chance=v[2] / max(v[1], 1)) for k, v in sorted(per_day.items())})


# ------------------------------------------------------------------ C2
def crossing_lines(S, F, B, shift=None):
    """Arrays over all drives: drive id, midpoint (merc), unit direction, k, and the index into r['crossings']."""
    rows = []
    for i, r in S.items():
        fr = F[i]; L = r["length_m"]
        for j, c in enumerate(r["crossings"]):
            s0 = c["s0_m"] if shift is None else (c["s0_m"] + shift[i]) % L
            jj = int(np.clip(round(s0 / 0.5), 0, len(fr["s"]) - 1)); u0, u1 = fr["lo"][jj], fr["hi"][jj]
            if not np.isfinite(u0):
                u0, u1 = -0.75, 0.75
            uu = np.array([u0, u1]); x, y = su_to_map(s0 + B + uu * np.tan(np.radians(c["phi_deg"])), uu, fr["s_nodes"], fr["xn"], fr["yn"], float(fr["k"]))
            rows.append((i, j, 0.5 * (x[0] + x[1]), 0.5 * (y[0] + y[1]), x[1] - x[0], y[1] - y[0], float(fr["k"])))
    a = np.array(rows, float); n = np.hypot(a[:, 4], a[:, 5]) + 1e-12
    return dict(drive=a[:, 0].astype(int), idx=a[:, 1].astype(int), mx=a[:, 2], my=a[:, 3], dx=a[:, 4] / n, dy=a[:, 5] / n, k=a[:, 6])


def crossing_groups(X) -> tuple[np.ndarray, np.ndarray]:
    """Link agreeing crossings of different drives; returns (group label, number of distinct drives in the group) per crossing."""
    k = float(np.median(X["k"])); P = np.c_[X["mx"], X["my"]]; tree = cKDTree(P)
    pr = tree.query_pairs(TOL_X["along"] / k, output_type="ndarray")
    a, b = pr[:, 0], pr[:, 1]; keep = X["drive"][a] != X["drive"][b]; a, b = a[keep], b[keep]
    vx, vy = P[b, 0] - P[a, 0], P[b, 1] - P[a, 1]
    perp = np.abs(vx * X["dy"][a] - vy * X["dx"][a]) * k
    cosang = np.abs(X["dx"][a] * X["dx"][b] + X["dy"][a] * X["dy"][b])
    ok = (perp <= TOL_X["perp"]) & (cosang >= np.cos(np.radians(TOL_X["phi"])))
    n = len(P); G = coo_matrix((np.ones(ok.sum()), (a[ok], b[ok])), shape=(n, n))
    _, lab = connected_components(G, directed=False)
    nd = np.zeros(n, int)
    for g in np.unique(lab[np.isin(lab, lab[a[ok]])]):
        m = lab == g; nd[m] = len(np.unique(X["drive"][m]))
    nd[nd == 0] = 1
    return lab, nd


def count_groups(lab, nd):
    u, first = np.unique(lab, return_index=True); n = nd[first]; return {f">={m}": int((n >= m).sum()) for m in (2, 3, 4)}


def c2_repeat_crossings(S, F, B, rng) -> tuple[dict, dict]:
    X = crossing_lines(S, F, B); lab, nd = crossing_groups(X); real = count_groups(lab, nd)
    null = {key: [] for key in real}
    for _ in range(N_DRAWS):
        sh = {i: rng.choice([-1, 1]) * rng.uniform(20, 200) for i in S}
        c = count_groups(*crossing_groups(crossing_lines(S, F, B, sh)))
        for key in null:
            null[key].append(c[key])
    out = {key: dict(real=real[key], chance=float(np.mean(null[key])), chance_over_real=ratio(np.mean(null[key]), real[key])) for key in real}
    return out, dict(X=X, lab=lab, nd=nd)


# ------------------------------------------------------------------ C3
def track_points(S, F, B, mirror=False):
    """1 m resampled points of every significant track (q <= Q, >= min_len): drive, track id, x, y, depth, and the map
    position of the drive's own strip centre at the same s (cx, cy; for the independent-lane rule)."""
    rows = []
    for i, r in S.items():
        fr = F[i]
        for j, t in enumerate(r["tracks"]):
            if t["q"] > Q or t["length_m"] < TOL_T["min_len"]:
                continue
            ps, pu = np.array(t["path_s"]), np.array(t["path_u"]); o = np.argsort(ps); ps, pu = ps[o], pu[o]
            ss = np.arange(ps[0], ps[-1] + 1e-9, 1.0)
            cx, cy = su_to_map(ss + B, strip_centre(fr, ss), fr["s_nodes"], fr["xn"], fr["yn"], float(fr["k"]))
            if t.get("frame") == "ground":               # E43: path_u is a ground offset; use the stored map path (from R)
                x = np.interp(ss, ps, np.array(t["x"])[o]); y = np.interp(ss, ps, np.array(t["y"])[o])
                if mirror:
                    x, y = 2 * cx - x, 2 * cy - y
            else:
                uu = np.interp(ss, ps, pu)
                if mirror:
                    uu = 2 * strip_centre(fr, ss) - uu
                x, y = su_to_map(ss + B, uu, fr["s_nodes"], fr["xn"], fr["yn"], float(fr["k"]))
            rows.append(np.c_[np.full(len(ss), i), np.full(len(ss), j), x, y, np.full(len(ss), t["depth_m"]), cx, cy])
    return np.concatenate(rows) if rows else np.zeros((0, 7))


def track_agreement(A, Bp, k):
    """For tracks in A: set of (drive_b, track_b) in Bp (other drives whose strip centre is >= MIN_SEP away at that place,
    i.e. an independent lane) sharing >= min_len 1 m points within du and dz."""
    tree = cKDTree(Bp[:, 2:4]); nb = tree.query_ball_point(A[:, 2:4], TOL_T["du"] / k)
    cnt = {}
    for p, lst in enumerate(nb):
        seen = set()
        for q in lst:
            if Bp[q, 0] != A[p, 0] and abs(Bp[q, 4] - A[p, 4]) <= TOL_T["dz"] and np.hypot(Bp[q, 5] - A[p, 5], Bp[q, 6] - A[p, 6]) * k >= MIN_SEP:
                seen.add((int(Bp[q, 0]), int(Bp[q, 1])))
        for key in seen:
            kk = (int(A[p, 0]), int(A[p, 1])) + key; cnt[kk] = cnt.get(kk, 0) + 1
    out = {}
    for (da, ta, db, tb), n in cnt.items():
        if n >= TOL_T["min_len"]:
            out.setdefault((da, ta), set()).add((db, tb))
    return out


def covered_by_other(T, S, F, B, k):
    """Tracks with >= min_len points inside the strip of another drive in an independent lane (strip centres >= MIN_SEP
    apart; coverage points every 1 m x 0.1 m, each carrying its drive's strip-centre position)."""
    cov = []
    for i, r in S.items():
        fr = F[i]; ok = np.isfinite(fr["lo"]); s = fr["s"][ok][::2]; lo = fr["lo"][ok][::2]; hi = fr["hi"][ok][::2]
        f = np.linspace(0, 1, 12); U = lo[:, None] + f * (hi - lo)[:, None]; Sg = np.repeat(s[:, None], 12, 1)
        x, y = su_to_map(Sg.ravel() + B, U.ravel(), fr["s_nodes"], fr["xn"], fr["yn"], float(fr["k"]))
        cx, cy = su_to_map(Sg.ravel() + B, (0.5 * (lo + hi))[:, None].repeat(12, 1).ravel(), fr["s_nodes"], fr["xn"], fr["yn"], float(fr["k"]))
        cov.append(np.c_[np.full(x.size, i), x, y, cx, cy])
    C = np.concatenate(cov); tree = cKDTree(C[:, 1:3]); nb = tree.query_ball_point(T[:, 2:4], 0.1 / k)
    out = {}
    for p, lst in enumerate(nb):
        if any(C[q, 0] != T[p, 0] and np.hypot(C[q, 3] - T[p, 5], C[q, 4] - T[p, 6]) * k >= MIN_SEP for q in lst):
            key = (int(T[p, 0]), int(T[p, 1])); out[key] = out.get(key, 0) + 1
    return {key for key, n in out.items() if n >= TOL_T["min_len"]}


def c3_repeat_tracks(S, F, B) -> tuple[dict, dict]:
    T = track_points(S, F, B); Tm = track_points(S, F, B, mirror=True); k = float(np.median([F[i]["k"] for i in F]))
    if not len(T):
        return dict(n_tracks=0), {}
    ag_r = track_agreement(T, T, k); ag_n = track_agreement(T, Tm, k); dbl = covered_by_other(T, S, F, B, k)
    keys = {(int(a), int(b)) for a, b in T[:, :2]}
    f_r = len([q for q in ag_r if q in dbl]) / max(len(dbl), 1); f_n = len([q for q in ag_n if q in dbl]) / max(len(dbl), 1)
    pairs_r = sum(len(v) for v in ag_r.values()) / 2; pairs_n = sum(len(v) for v in ag_n.values()) / 2   # unordered pairs
    return dict(n_tracks=len(keys), n_double_covered=len(dbl), confirmed_real=len([q for q in ag_r if q in dbl]),
                confirmed_chance=len([q for q in ag_n if q in dbl]), f_real=f_r, f_chance=f_n, pairs_real=pairs_r, pairs_chance=pairs_n,
                chance_over_real=ratio(pairs_n, pairs_r), w_single=max(0.0, (f_r - f_n) / max(1 - f_n, 1e-9))), dict(agree=ag_r)


# ------------------------------------------------------------------ C4
def record_crossings(S, F, B):
    """Municipal storm/sewer records crossing each drive's strip at >= 30 deg, depth <= 1.5 m or blank: (drive, s true)."""
    recs = []
    for fn in ("gfi_all_condotte_acque_bianche.json", "gfi_all_condotte_acque_nere.json"):
        for f in json.load(open(ROOT / "data_public" / fn))["features"]:
            if f["geometry"]["type"] != "LineString":
                continue
            pr = f["properties"]                                           # depths stored as text, negative centimetres
            z = [abs(float(v)) / 100 for v in (pr.get("prof_iniz"), pr.get("prof_fin")) if v not in ("", None) and float(v) != 0]
            if z and min(z) > 1.5:
                continue
            c = np.array(f["geometry"]["coordinates"], float).reshape(-1, 2); recs.append(np.c_[lonlat_to_merc(c[:, 0], c[:, 1])])
    out = []
    for i, r in S.items():
        fr = F[i]; L = r["length_m"]
        for c in recs:
            if np.hypot(c[:, 0, None] - fr["xn"][None, ::50], c[:, 1, None] - fr["yn"][None, ::50]).min() * float(fr["k"]) > 60:
                continue
            ss, uu = to_su(c[:, 0], c[:, 1], fr["s_nodes"], fr["xn"], fr["yn"], float(fr["k"]))
            du = uu - strip_centre(fr, ss)
            for a in range(len(c) - 1):
                if np.sign(du[a]) != np.sign(du[a + 1]) and abs(du[a]) < 15 and abs(du[a + 1]) < 15:
                    t = du[a] / (du[a] - du[a + 1]); s_r = ss[a] + t * (ss[a + 1] - ss[a])
                    ang = np.degrees(np.arctan2(abs(uu[a + 1] - uu[a]), abs(ss[a + 1] - ss[a]) + 1e-9))
                    if ang >= 30 and 5 < s_r < L - 5:
                        out.append((i, float(s_r)))
    return out


# drives used to set the detector or the along-travel bias (dev drive; Via Nizza drives of E16/E25): excluded in the held-out C4 line
DEV = {("Survey_2025-10-14_11-37-03", "Swath001")} | {("Survey_2025-10-13_10-01-51", f"Swath0{n}") for n in (37, 38, 40, 41, 46, 47)}


def c4_records(S, F, B, conf, rng, rc=None) -> dict:
    rc = record_crossings(S, F, B) if rc is None else rc
    def hits(sh=None):
        n = 0
        for i, s_r in rc:
            s = s_r if sh is None else (s_r + sh[i]) % S[i]["length_m"]
            n += any(abs(c["s0_m"] + B - s) <= 1.5 for c in conf[i])
        return n
    real = hits(); null = [hits({i: rng.choice([-1, 1]) * rng.uniform(20, 200) for i in S}) for _ in range(200)]
    return dict(n_record_crossings=len(rc), hit=real, rate=real / max(len(rc), 1), chance_rate=float(np.mean(null)) / max(len(rc), 1),
                p=float((np.sum(np.array(null) >= real) + 1) / (len(null) + 1)))


def main(cfg_path: str) -> None:
    cc = load_yaml(cfg_path); det = load_cfg(str(ROOT / "pipeline" / cc["detector_config"]))
    cache = Path(det["base"]["cache_dir"]) / "city"; B = cc["along_travel_bias_m"]; rng = np.random.default_rng(det["seed"] + 71)
    S, F = load_all(cache, cc.get("along_tag"), cc.get("drives")); print(f"drives scanned ok: {len(S)}, {sum(r['length_m'] for r in S.values()) / 1e3:.1f} km", flush=True)
    res = dict(n_drives=len(S), km=sum(r["length_m"] for r in S.values()) / 1e3)
    res["C1"] = c1_both_rows(S, rng); print("C1", {k: v for k, v in res["C1"].items() if k != "per_day"}, flush=True)
    res["C2"], g2 = c2_repeat_crossings(S, F, B, rng); print("C2", res["C2"], flush=True)
    res["C3"], g3 = c3_repeat_tracks(S, F, B); print("C3", res["C3"], flush=True)
    rep_ok = res["C2"][">=3"]["chance_over_real"] <= 0.2                  # pre-registered: repeat rule usable only if <= 20 % chance
    X = g2["X"]; conf = {i: [] for i in S}; label = {}
    for n in range(len(X["drive"])):
        i, j = int(X["drive"][n]), int(X["idx"][n]); c = S[i]["crossings"][j]
        is_conf = c["cls"] == "A" or (rep_ok and g2["nd"][n] >= 3)
        label[(i, j)] = dict(group=int(g2["lab"][n]), n_drives=int(g2["nd"][n]), confident=bool(is_conf))
        if is_conf:
            conf[i].append(c)
    rc = record_crossings(S, F, B); res["C4"] = c4_records(S, F, B, conf, rng, rc); print("C4", res["C4"], flush=True)
    held = [(i, s_) for i, s_ in rc if (S[i]["survey"], S[i]["swath"]) not in DEV]
    res["C4_heldout"] = c4_records(S, F, B, conf, rng, held); print("C4 held-out (excl. development drives)", res["C4_heldout"], flush=True)
    res["repeat_crossing_rule_used"] = bool(rep_ok)
    res["along_confirmed_rule_used"] = bool(res["C3"].get("chance_over_real", 1.0) <= 0.2)
    res["crossing_labels"] = [dict(drive=i, idx=j, **v) for (i, j), v in label.items()]
    res["track_agreement"] = [dict(drive=a, track=b, with_=[list(x) for x in sorted(v)]) for (a, b), v in g3.get("agree", {}).items()]
    out = cache / f"citywide_merged{cc.get('merge_tag', '')}.json"; json.dump(res, open(out, "w")); print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1])
