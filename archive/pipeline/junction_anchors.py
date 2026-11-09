"""Anchored along-road tracking (user's idea, 3 Nov 2026; test fixed in DECISIONS.md before running).
(1) Pins: at each place where two drives cross, the drive passing ACROSS a street sees that street's
    main as a crossing. Keep crossings (both radar rows, q <= 0.05) that run along the other drive's
    heading (within 20 deg), 0.4-1.5 m deep; correct the crossing drive's along-travel bias (+1.19 m)
    and project into the along-street drive's own frame -> pin (s, u, depth).
(2) Follow: on along-street drives with two pins 20-500 m apart, both inside the strip, track the pipe
    from pin 1 through 200 MHz point-vs-line evidence near the pinned depth; success if it reaches
    pin 2 within 0.6 m sideways; chance from random starting offsets.
Run: python junction_anchors.py configs/citywide.yaml -> cache/city/anchors.json, cache/city/anchored_tracks.json"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from build_cube import swath_geometry
from build_v2 import grid_Z, load_cfg, preprocess_array
from cube import to_su
from finalize_v2 import merge_crossings
from ids_reader import Swath
from junction_checkpoints import junction_places
from longitudinal import lateral_migrate, point_vs_line_contrast, stack_along_s
from network_dp import viterbi_anchored
from run_crossings_v2 import detect_crossings, migrate_grid
from run_detection import su_to_map
from run_multipass import longest_run

BIAS = 1.19          # m, along-travel correction (E25)
TOL = 0.6            # m, success tolerance and pin sigma (ASSUMED, DECISIONS.md)


def load(files, i):
    f = files[i]; sw = Swath.load(str(Path(f).parent), Path(f).stem); return sw, swath_geometry(sw, DET["base"])


def crossings_window(sw, g, s0, s1, rng):
    L = float(g["s_nodes"][-1]); s0, s1 = max(0.0, s0), min(L, s1)
    if s1 - s0 < 100.0:                                  # near a drive end: extend to the other side (chance test needs >= 2 x 30 m)
        s0, s1 = (max(0.0, s1 - 100.0), s1) if s0 <= 0.0 else (s0, min(L, s0 + 100.0))
    if s1 - s0 < 70.0:
        return None
    s_grid = np.arange(s0, s1, DET["grid"]["ds_m"]); C = {}
    for arr in (1, 2):
        S = g[f"s{arr}"]; sc = np.nanmedian(S, 1); sl = longest_run((sc > s_grid[0] - 15) & (sc < s_grid[-1] + 15))
        if sl is None or sl.stop - sl.start < 200:
            return None
        pp = preprocess_array(sw, arr, DET, scans=sl); G = migrate_grid(pp["X"], S[sl], s_grid, pp["dx_m"], pp["dt_ns"], DET)
        C[arr], _ = detect_crossings(G, np.nanmedian(g[f"u{arr}"][sl], 0), s_grid, pp["dt_ns"] * DET["grid"]["t_down"], pp["snr"], pp["dt_ns"], DET, rng)
    return merge_crossings(C[1], C[2], 0.05)


def heading_at(g, x, y):
    i = int(np.argmin(np.hypot(g["xn"] - x, g["yn"] - y))); i = int(np.clip(i, 1, len(g["xn"]) - 2))
    return float(np.arctan2(g["yn"][i + 1] - g["yn"][i - 1], g["xn"][i + 1] - g["xn"][i - 1]))


def make_pins(files, rng) -> list[dict]:
    """Pins for along-street drives that pass >= 2 junction places (or CC['only_along_drive']). Crossings of each
    (junction place, crossing drive) window are computed once and cached (cache/city/jx_<place>_<drive>.json)."""
    z = np.load(CC["swath_tracks"], allow_pickle=True); tracks = [z[f"s{i}"] for i in range(len(z["days"]))]
    J = junction_places(tracks); cache_d = {}
    get = lambda i: cache_d[i] if i in cache_d else cache_d.setdefault(i, load(files, i))
    places = {}
    for n, pl in enumerate(J):
        for p in pl["pairs"]:
            for cross, along in ((p["drive_a"], p["drive_b"]), (p["drive_b"], p["drive_a"])):
                places.setdefault(along, set()).add(n)
    want = {CC["only_along_drive"]} if CC.get("only_along_drive") is not None else {d for d, v in places.items() if len(v) >= 2}
    jdir = Path(DET["base"]["cache_dir"]) / "city"; pins = []
    shard = CC.get("shard"); n_sh = CC.get("n_shards", 1)
    for n, pl in enumerate(J):
        if shard is not None and n % n_sh != shard:
            continue                                     # parallel workers split the junction places
        for p in pl["pairs"]:
            for cross, along in ((p["drive_a"], p["drive_b"]), (p["drive_b"], p["drive_a"])):
                if along not in want:
                    continue
                try:
                    get(cross); get(along)
                except Exception as e:                       # drive too short for geometry: skip this pair
                    print(f"place {n}: drive {cross} or {along}: geometry failed ({e})", flush=True); continue
                fj = jdir / f"jx_{n}_{cross}.json"
                if fj.exists():
                    X = json.load(open(fj))
                else:
                    swc, gc = get(cross)
                    i0 = int(np.argmin(np.hypot(gc["xn"] - p["x"], gc["yn"] - p["y"]))); sj = float(gc["s_nodes"][i0])
                    try:
                        X = crossings_window(swc, gc, sj - 50, sj + 50, rng)   # +-50 m: the chance test shifts data by >= 30 m
                    except Exception as e:
                        print(f"place {n}: drive {cross}: window failed ({e})", flush=True); X = None
                    X = X if X is not None else []
                    json.dump(X, open(fj, "w"))
                gc = get(cross)[1]; ga = get(along)[1]; k = float(gc["k"]); hb = heading_at(ga, p["x"], p["y"]); kept = []
                for o in X:
                    s0 = o["s0_m"] + BIAS; tphi = np.tan(np.radians(o["phi_deg"]))
                    mx, my = su_to_map(np.array([s0 - 0.6 * tphi, s0, s0 + 0.9 * tphi]), np.array([-0.6, 0.0, 0.9]), gc["s_nodes"], gc["xn"], gc["yn"], k)
                    th = np.arctan2(my[2] - my[0], mx[2] - mx[0]); d = np.degrees(np.abs(np.angle(np.exp(1j * (th - hb))))); d = min(d, 180 - d)
                    if d <= 20 and 0.4 <= o["depth_m"] <= 1.5:
                        sb, ub = to_su(np.array([mx[1]]), np.array([my[1]]), ga["s_nodes"], ga["xn"], ga["yn"], float(ga["k"]))
                        if abs(float(ub[0])) <= 8.0:                          # on this street (not deep in the side street)
                            kept.append(dict(junction=n, cross_drive=int(cross), along_drive=int(along), s_along=float(sb[0]) - BIAS, u_along=float(ub[0]),
                                             depth_m=o["depth_m"], P=o["P"], cls=o["cls"], angle_to_street=float(d), x=float(mx[1]), y=float(my[1])))
                for q in kept:
                    q["n_alternatives"] = len(kept)
                pins += kept
                print(f"place {n}: drive {cross} across drive {along}: crossings {len(X)}, pins on the street {len(kept)}", flush=True)
    return pins


def consensus(pins: list[dict]) -> list[dict]:
    """Merge pins of the same along drive and junction place from different crossing drives that agree
    (|du| <= 1.0 m, |dz| <= 0.25 m): support = number of crossing drives."""
    out = []; by = {}
    for p in pins:
        by.setdefault((p["along_drive"], p["junction"]), []).append(p)
    for key, P in by.items():
        used = [False] * len(P)
        for i, a in enumerate(sorted(P, key=lambda q: -q["P"])):
            if used[i]:
                continue
            grp = [a] + [b for j, b in enumerate(P) if not used[j] and b is not a and b["cross_drive"] != a["cross_drive"]
                         and abs(b["u_along"] - a["u_along"]) <= 1.0 and abs(b["depth_m"] - a["depth_m"]) <= 0.25]
            for j, b in enumerate(P):
                if b in grp:
                    used[j] = True
            out.append(dict(along_drive=key[0], junction=key[1], s_along=float(np.mean([g["s_along"] for g in grp])),
                            u_along=float(np.mean([g["u_along"] for g in grp])), depth_m=float(np.mean([g["depth_m"] for g in grp])),
                            support=len({g["cross_drive"] for g in grp}), P=float(max(g["P"] for g in grp)),
                            x=float(np.mean([g["x"] for g in grp])), y=float(np.mean([g["y"] for g in grp]))))
    return out


def evidence(sw, g, s0, s1, z_pin, u_pad=0.3):
    """200 MHz along-road evidence (point-vs-line contrast, 2 m stack) on [s0, s1] of drive g's own frame,
    max over depth within z_pin +- 0.2 m. Returns (E (n_win, n_u), s centres, u grid, strip centre)."""
    S = g["s2"]; sc = np.nanmedian(S, 1); sl = longest_run((sc > s0 - 5) & (sc < s1 + 5))
    pp = preprocess_array(sw, 2, DET, scans=sl); s_grid = np.arange(s0, s1, DET["grid"]["ds_m"])
    Z = grid_Z(pp["X"], S[sl], s_grid); u = np.nanmedian(g["u2"][sl], 0); dt = pp["dt_ns"]; v = DET["migration"]["velocity_m_ns"]
    X, cen = stack_along_s(Z, DET["grid"]["ds_m"], 2.0, 0.5); ug = np.arange(u.min() - u_pad, u.max() + u_pad + 1e-9, 0.04)
    sc_ = point_vs_line_contrast(lateral_migrate(X, u, ug, dt, v))
    t0, t1 = int(2 * max(0.1, z_pin - 0.2) / v / dt), int(2 * (z_pin + 0.2) / v / dt) + 1
    return sc_[:, :, t0:t1].max(2), s_grid[cen], ug, float(np.median(u))


def follow(files, pins, rng) -> list[dict]:
    out = []; by = {}
    for p in pins:
        by.setdefault(p["along_drive"], []).append(p)
    for drv, P in by.items():
        P = sorted(P, key=lambda q: q["s_along"]); sw, g = load(files, drv)
        cen_u = float(np.median(np.nanmedian(g["u2"], 0)))
        for a, b in zip(P[:-1], P[1:]):
            L = b["s_along"] - a["s_along"]
            inside = all(abs(q["u_along"] - cen_u) <= 0.75 + TOL for q in (a, b))
            row = dict(drive=drv, s0=a["s_along"], s1=b["s_along"], u0=a["u_along"], u1=b["u_along"], length_m=L, inside=inside,
                       same_junction=a["junction"] == b["junction"])
            if not (20 <= L <= 500) or not inside or a["junction"] == b["junction"]:
                row["status"] = "not testable"; out.append(row); continue
            z = 0.5 * (a["depth_m"] + b["depth_m"])
            E, sc, ug, _ = evidence(sw, g, a["s_along"], b["s_along"], z)
            med = np.median(E); mad = 1.4826 * np.median(np.abs(E - med)) + 1e-9
            llr = np.clip((E - med) / mad - 2.0, -2.0, 4.0); du = float(ug[1] - ug[0])
            ui = lambda uu: (uu - ug[0]) / du
            path = viterbi_anchored(llr, ui(a["u_along"]), TOL / du)
            u_end = float(ug[path[-1]]); hit = abs(u_end - b["u_along"]) <= TOL
            starts = rng.uniform(ug[0], ug[-1], 200)
            ch = np.mean([abs(float(ug[viterbi_anchored(llr, ui(s_), TOL / du)[-1]]) - b["u_along"]) <= TOL for s_ in starts])
            both = viterbi_anchored(llr, ui(a["u_along"]), TOL / du, end=(ui(b["u_along"]), TOL / du))
            mx, my = su_to_map(sc[both] * 0 + sc, ug[both], g["s_nodes"], g["xn"], g["yn"], float(g["k"]))
            row.update(status="tested", depth_m=z, u_end=u_end, hit=bool(hit), chance=float(ch), mean_llr_on_path=float(llr[np.arange(len(path)), path].mean()),
                       path_x=mx[::4].tolist(), path_y=my[::4].tolist())
            out.append(row); print({k_: row[k_] for k_ in ("drive", "s0", "s1", "u0", "u1", "u_end", "hit", "chance")}, flush=True)
    return out


if __name__ == "__main__":
    CC = yaml.safe_load(open(sys.argv[1])); DET = load_cfg(str(Path(__file__).resolve().parents[1] / "pipeline" / CC["detector_config"]))
    cache = Path(DET["base"]["cache_dir"]) / "city"; rng = np.random.default_rng(DET["seed"] + 51)
    files = sorted(glob.glob(str(Path(CC["raw_root"]) / "*" / "Survey_*" / "Swath*.path")))
    t0 = time.time()
    sub = f"_drive{CC['only_along_drive']}" if CC.get("only_along_drive") is not None else ""
    fp = cache / f"anchors{sub}.json"
    if CC.get("shard") is not None:                      # worker: only fill the window cache
        make_pins(files, rng); print("shard done", flush=True); sys.exit(0)
    pins = json.load(open(fp)) if fp.exists() else []
    if not pins:                                         # recompute when nothing (or an empty list from an earlier bug) was saved
        pins = make_pins(files, rng)
    json.dump(pins, open(fp, "w")); print(f"pins: {len(pins)} ({time.time()-t0:.0f}s)", flush=True)
    cons = consensus(pins); json.dump(cons, open(cache / f"anchors_consensus{sub}.json", "w"))
    print(f"consensus pins: {len(cons)}; supported by >= 2 crossing drives: {sum(c['support'] >= 2 for c in cons)}", flush=True)
    best = {}
    for c in cons:                                             # one pin per (drive, junction): best supported, then highest P
        key = (c["along_drive"], c["junction"])
        if key not in best or (c["support"], c["P"]) > (best[key]["support"], best[key]["P"]):
            best[key] = c
    rows = follow(files, list(best.values()), rng)
    tested = [r for r in rows if r["status"] == "tested"]
    k = sum(r["hit"] for r in tested); exp = sum(r["chance"] for r in tested)
    # exact p-value of >= k hits under independent per-segment chance rates (Poisson-binomial by dynamic programming)
    pr = np.zeros(len(tested) + 1); pr[0] = 1.0
    for r in tested:
        pr[1:] = pr[1:] * (1 - r["chance"]) + pr[:-1] * r["chance"]; pr[0] *= (1 - r["chance"])
    summ = dict(n_pins=len(pins), n_segments=len(rows), n_tested=len(tested), hits=int(k), expected_by_chance=float(exp), p=float(pr[k:].sum()) if tested else None)
    print("SUMMARY", summ)
    json.dump(dict(summary=summ, segments=rows), open(cache / f"anchored_tracks{sub}.json", "w"))
