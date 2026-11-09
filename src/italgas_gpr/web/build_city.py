"""Interactive web map (one local HTML file) of the WHOLE survey, from the uniform scan of every drive
(crossing/scan.py) and its checks (crossing/survey_merge.py; rules in DECISIONS.md 2026-11-03). Layers:
 (1) scanned areas = each drive's radar strip (both arrays' channel span), with the drive's positioning quality;
 (2) chance of a pipe (glow) under every drive, 0.15 m cells: crossings (P from the column-shift null, spread along the
     drive by its position sigma) and along-street tracks (P from the measured repeat-drive confirmation rates, C3);
     highest over overlapping drives; shown at three resolutions (0.15 / 1.2 / 4.8 m) depending on zoom;
 (3) pipelines: crossing lines (confident = both radar rows, or >= 3 drives if C2 allows; probable = P >= 0.9) and
     along-street lines (confirmed by another drive / single-drive candidates).
 Positions: real-time track, every drive shifted +1.19 m along travel (ASSUMED, E25). Sigma = max(|RT - PP|, 0.3 m);
 unknown (PP missing or failed) -> 1.0 m ASSUMED in the glow.
 Usage: python -m italgas_gpr.web.build_city savona_pipes_map.html [merge config, default
 configs/crossing_scan.yaml; e.g. configs/crossing_merge_subset.yaml for the three sample regions]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from italgas_gpr.web.common import CACHE, junction_layers, ll, png_uri, png_uri_rg, records_near, survey_tracks
from italgas_gpr.geometry.swath import swath_geometry  # noqa: E402
from italgas_gpr.io.ids_reader import Swath  # noqa: E402
from italgas_gpr.crossing.survey_merge import load_all  # noqa: E402
from italgas_gpr.figures.map_layers import prob_on_cells  # noqa: E402
from italgas_gpr.geometry.frames import su_to_map  # noqa: E402
from italgas_gpr.along.evidence import search_columns, search_samples  # noqa: E402
from italgas_gpr.io.paths import config_path, load_yaml, swath_files  # noqa: E402

_RL = load_yaml(config_path("along_lines.yaml"))                            # evaluated band, one source
T_MIN_NS, U_MIN_M, U_MAX_M = float(_RL["t_min_ns"]), float(_RL["u_min_m"]), float(_RL["u_max_m"])

CC = load_yaml(config_path(sys.argv[2]) if (__name__ == "__main__" and len(sys.argv) > 2) else config_path("crossing_scan.yaml"))
BIAS = CC["along_travel_bias_m"]; SIG_FLOOR = CC["sigma_floor_m"]
SIG_UNKNOWN = 1.0    # m, ASSUMED along-street sigma where it cannot be measured (lane level, DECISIONS 2026-11-03)
PIX = 0.15           # m, finest raster cell (true metres)
LEVELS = ((1, 17), (8, 15), (32, 0))   # (pooling factor, min zoom): 0.15 m from zoom 17, 1.2 m from 15, 4.8 m below
TILE = 768           # px per raster tile at every level
P_LINE = 0.9         # probable crossing line threshold (P)
P_MIN = 0.02         # glow drawn from this P


def sig_of(v, floor=SIG_FLOOR):
    return None if v is None else round(max(float(v), floor), 2)


def drive_label(r):
    d = r["session"][-19:-9].replace(".", "-"); return f"{d} {r['survey'][-8:-3].replace('-', ':')} {r['swath']}"


def strip_poly(fr, step=2.0):
    """Footprint polygon(s) of one drive (split where coverage is missing), s shifted by BIAS."""
    s, lo, hi = fr["s"], fr["lo"].astype(float), fr["hi"].astype(float); ok = np.isfinite(lo) & np.isfinite(hi) & (hi - lo < 3.0)
    out = []; d = np.diff(np.r_[0, ok.astype(int), 0]); st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    for a, b in zip(st, en):
        if s[b - 1] - s[a] < 4:
            continue
        sd = np.r_[np.arange(s[a], s[b - 1], step), s[b - 1]]; l_ = np.interp(sd, s[a:b], lo[a:b]); h_ = np.interp(sd, s[a:b], hi[a:b])
        x, y = su_to_map(np.r_[sd, sd[::-1]] + BIAS, np.r_[h_, l_[::-1]], fr["s_nodes"], fr["xn"], fr["yn"], float(fr["k"]))
        c = ll(x, y); out.append(c + [c[0]])
    return out


def p_tiles(x0: float, y1: float, px: float, cache: Path = CACHE) -> tuple[dict, int]:
    """Red-green tiles of P (max over depth, 12 cm widening across the strip for display) for every drive that has
    prob_raster_<i>_*.npz files, at the same three levels as the glow. Returns (tiles by min zoom, n_drives)."""
    import glob as _glob
    from scipy.ndimage import maximum_filter
    files = swath_files()
    drives = sorted({int(Path(f).stem.split("_")[2]) for f in _glob.glob(str(cache / "prob_raster_*_*.npz"))})
    det = load_yaml(config_path("detector.yaml"))
    from italgas_gpr.processing.preprocess import load_cfg
    det = load_cfg(config_path("detector.yaml"))
    IX, IY, PV = [], [], []
    for i in drives:
        f = files[i]; sw = Swath.load(str(Path(f).parent), Path(f).stem); g = swath_geometry(sw, det["base"])
        for pf in sorted(_glob.glob(str(cache / f"prob_raster_{i}_*.npz"))):
            d = np.load(pf); it = search_samples(d, T_MIN_NS); iu = search_columns(d, U_MIN_M, U_MAX_M)     # evaluated band: raster_lines.yaml
            if not len(it) or not len(iu):
                continue
            plan = maximum_filter(d["P"].astype(np.float32)[:, iu][:, :, it].max(2), size=(1, 3)); cs, ug = d["centres_s"], d["ug"][iu]; keep = (cs >= d["c0"]) & (cs < d["c1"])
            for off in np.arange(-0.2, 0.21, 0.1):
                S, U = np.meshgrid(cs[keep] + off + BIAS, ug, indexing="ij")
                mx, my = su_to_map(S.ravel(), U.ravel(), g["s_nodes"], g["xn"], g["yn"], float(g["k"]))
                IX.append(((mx - x0) / px).astype(np.int32)); IY.append(((y1 - my) / px).astype(np.int32)); PV.append(plan[keep].ravel())
    if not IX:
        return {}, 0
    IX, IY, PV = np.concatenate(IX), np.concatenate(IY), np.concatenate(PV); tiles = {}
    ox, oy = int(IX.min()), int(IY.min()); IX -= ox; IY -= oy; x0 += ox * px; y1 -= oy * px     # own origin: rasters extend beyond the page's strips
    for f_, zmin in LEVELS:
        ix, iy = IX // f_, IY // f_; tx, ty = ix // TILE, iy // TILE; key = tx.astype(np.int64) * 100000 + ty
        o = np.argsort(key, kind="stable"); ks, idx = np.unique(key[o], return_index=True); bounds_ = np.r_[idx, len(o)]; out = []
        for n, kk in enumerate(ks):
            sel = o[bounds_[n]: bounds_[n + 1]]; TX, TY = int(kk // 100000), int(kk % 100000)
            img = np.full((TILE, TILE), np.nan, np.float32); np.fmax.at(img, (iy[sel] - TY * TILE, ix[sel] - TX * TILE), PV[sel])
            ys, xs = np.nonzero(np.isfinite(img)); r0, r1, q0, q1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            pf = px * f_; (lo0, la0), (lo1, la1) = ll([x0 + (TX * TILE + q0) * pf, x0 + (TX * TILE + q1) * pf], [y1 - (TY * TILE + r1) * pf, y1 - (TY * TILE + r0) * pf])
            out.append(dict(png=png_uri_rg(img[r0:r1, q0:q1]), bounds=[[la0, lo0], [la1, lo1]]))
        tiles[str(zmin)] = out; print(f"P raster level {PIX * f_:.2f} m: {len(out)} tiles, {sum(len(t['png']) for t in out) / 1e6:.1f} MB", flush=True)
    return tiles, len(drives)


def lane_confirmed(stats: dict) -> list[dict]:
    """Along-street stretches repeated by a drive in another lane after registration (E45 Via Nizza, E46 other regions):
    runs >= 10 m at P >= 0.5 from cache/mosaic_test.json (frame mp_frame.npz) and cache/mosaic_region_<r>.json (frame
    mosaic_frame_<r>.npz), only from regions whose test is positive (f_real > f_mirror and p_shift <= 0.05)."""
    feats = []; srcs = [("Via Nizza (E45)", CACHE / "mosaic_test.json", CACHE / "mp_frame.npz")]
    for f in sorted((CACHE / "city").glob("lane_*.json")):          # E47 components (whole survey); Via Nizza (ref 40) comes from E45 above
        if f.stem == "lane_summary" or f.stem == "lane_40":
            continue
        srcs.append((f"component ref {f.stem.replace('lane_', '')} (E47)", f, CACHE / f"mosaic_frame_c{f.stem.replace('lane_', '')}.npz"))
    for label, jf, ff in srcs:
        if not jf.exists() or not ff.exists():
            continue
        J = json.load(open(jf)); T = J.get("thresholds", {}).get("0.5")
        if not T or T["f_real"] is None or T["f_mirror"] is None or not (T["f_real"] > T["f_mirror"] and T["p_shift"] is not None and T["p_shift"] <= 0.05):
            continue
        fr = dict(np.load(ff)); k = float(fr["k"]); drives = J.get("passes") or {d: d for d in J.get("drives", [])}
        for r in T["runs"]:
            ss = np.arange(r["s0"], r["s1"] + 1e-9, 1.0); x, y = su_to_map(ss, np.full(len(ss), r["u_median"]), fr["s_nodes"], fr["xn"], fr["yn"], k)
            names = ", ".join(str(drives.get(p, p)) if isinstance(drives, dict) else str(p) for p in r["passes"])
            feats.append(dict(type="Feature", geometry=dict(type="LineString", coordinates=ll(x, y)),
                              properties=dict(kind="along-lanes", drive=label, P=round(T["f_real"], 2), depth_m=round(r["depth_median"], 2), length_m=round(r["length_m"], 1),
                                              support=f"{len(r['passes'])} drives in different lanes ({names}); region repeat fraction {T['f_real']:.2f} vs {T['f_mirror']:.2f} on pipe-free data, p {T['p_shift']:.3f}",
                                              q=None, sigma=None, rtk=None, pp_sd=None)))
            stats["along_lanes"] = stats.get("along_lanes", 0) + 1
    return feats


def main(out_path: str) -> None:
    S, F = load_all(CACHE / "city", CC.get("along_tag"), CC.get("drives")); M = json.load(open(CACHE / "city" / f"citywide_merged{CC.get('merge_tag', '')}.json"))
    lab = {(c["drive"], c["idx"]): c for c in M["crossing_labels"]}
    agree = {(a["drive"], a["track"]): a["with_"] for a in M["track_agreement"]}
    c3 = M["C3"]; P_conf_track = max(0.0, 1.0 - c3.get("chance_over_real", 1.0)); w_single = c3.get("w_single", 0.0)
    k_ref = float(np.median([F[i]["k"] for i in F])); px = PIX / k_ref
    allx = np.concatenate([F[i]["xn"] for i in F]); ally = np.concatenate([F[i]["yn"] for i in F]); x0, y1 = allx.min() - 50 / k_ref, ally.max() + 50 / k_ref
    strips, cross, along = [], [], []; IX, IY, PV = [], [], []
    stats = dict(conf=0, prob=0, along_conf=0, along_cand=0)
    for i in sorted(S):
        r, fr = S[i], F[i]; L = r["length_m"]; name = drive_label(r)
        # ---- detections of this drive (crossing positions are TRUE along-track positions in the drive frame: s0 + BIAS)
        X = []; A = []
        for j, c in enumerate(r["crossings"]):
            lb = lab.get((i, j), dict(confident=c["cls"] == "A", n_drives=1))
            X.append(dict(s0=c["s0_m"] + BIAS, phi=c["phi_deg"], P=c["P"], sig=sig_of(c["d_along_m"]) or SIG_UNKNOWN))
            if not (lb["confident"] or c["P"] >= P_LINE):
                continue
            kind = "confident" if lb["confident"] else "probable"; stats["conf" if lb["confident"] else "prob"] += 1
            seen = "both radar rows (600 + 200 MHz)" if c["cls"] == "A" else f"one radar row ({c['freq']} MHz)"
            if lb["n_drives"] >= 2:
                seen += f"; {lb['n_drives']} drives agree within 0.5 m"
            cross.append(dict(type="Feature", geometry=dict(type="LineString", coordinates=ll(c["x"], c["y"])),
                              properties=dict(kind=kind, drive=name, P=c["P"], depth_m=c["depth_m"], angle_deg=round(c["phi_deg"]), support=seen,
                                              sigma=None if c["d_along_m"] is None else [sig_of(c["d_along_m"]), sig_of(c["d_cross_m"])],
                                              rtk=c["rt_q"], pp_sd=c["pp_sd_m"])))
        for j, t in enumerate(r["tracks"]):
            if t["q"] > 0.05 or t["length_m"] < 10:
                continue
            w = agree.get((i, j)); conf = bool(w) and M["along_confirmed_rule_used"]
            P = P_conf_track if w else w_single
            A.append(dict(s0=min(t["path_s"]) + BIAS, s1=max(t["path_s"]) + BIAS, ps=np.array(t["path_s"]) + BIAS, pu=np.array(t["path_u"]), P=P,
                          sig=max(t["d_cross_m"] or SIG_UNKNOWN, 0.15)))
            kind = "along-conf" if conf else ("along-rep" if w else "along-cand"); stats["along_conf" if conf else "along_cand"] += 1
            sup = "1 drive (200 MHz)" if not w else f"{1 + len({d for d, _ in w})} drives (this one and {', '.join(drive_label(S[d]) for d in sorted({d for d, _ in w}))})"
            along.append(dict(type="Feature", geometry=dict(type="LineString", coordinates=ll(t["x"], t["y"])),
                              properties=dict(kind=kind, drive=name, P=round(P, 2), depth_m=t["depth_m"], length_m=t["length_m"], support=sup, q=t["q"],
                                              sigma=None if t["d_cross_m"] is None else [sig_of(t["d_along_m"]), sig_of(t["d_cross_m"])],
                                              rtk=t["rt_q"], pp_sd=t["pp_sd_m"])))
        # ---- glow samples on the strip (0.1 m along x 16 across), kept where P >= P_MIN
        s_, lo, hi = fr["s"].astype(float), fr["lo"].astype(float), fr["hi"].astype(float); ok = np.isfinite(lo) & np.isfinite(hi) & (hi - lo < 3.0)
        for c0 in np.arange(0.0, L, 100.0):
            near = [o for o in X if c0 - 6 <= o["s0"] - BIAS <= c0 + 106]; nearA = [a for a in A if a["s1"] - BIAS >= c0 - 1 and a["s0"] - BIAS <= c0 + 101]
            if not near and not nearA:
                continue
            ss = np.arange(c0, min(c0 + 100.0, L), 0.1); m = np.interp(ss, s_[ok], ok[ok].astype(float), left=0, right=0) > 0
            okk = np.interp(ss, s_, ok.astype(float)) > 0.5
            ss = ss[okk]
            if not len(ss):
                continue
            l_, h_ = np.interp(ss, s_[ok], lo[ok]), np.interp(ss, s_[ok], hi[ok]); f = np.linspace(0, 1, 16)
            St = np.repeat(ss[:, None], 16, 1) + BIAS; U = l_[:, None] + f[None, :] * (h_ - l_)[:, None]
            P = prob_on_cells(St, U, near, nearA, mass=True); keep = P >= P_MIN
            if keep.any():
                mx, my = su_to_map(St[keep], U[keep], fr["s_nodes"], fr["xn"], fr["yn"], float(fr["k"]))
                IX.append(((mx - x0) / px).astype(np.int32)); IY.append(((y1 - my) / px).astype(np.int32)); PV.append(P[keep].astype(np.float32))
        # ---- strip with drive summary
        rq = fr["rt_q"][ok]; dA = fr["d_along_m"][ok]
        props = dict(label=name, length_km=round(L / 1e3, 2), n_cross=len(r["crossings"]), n_A=sum(c["cls"] == "A" for c in r["crossings"]),
                     n_tracks=sum(t["q"] <= 0.05 and t["length_m"] >= 10 for t in r["tracks"]),
                     rtk_fixed=None if not np.isfinite(rq).any() else round(float(np.mean(rq[np.isfinite(rq)] == 4)), 2),
                     pp_ok=round(float(np.mean(np.isfinite(dA))), 2), d_along_med=None if not np.isfinite(dA).any() else round(float(np.nanmedian(dA)), 2))
        for poly in strip_poly(fr):
            strips.append(dict(type="Feature", geometry=dict(type="Polygon", coordinates=[poly]), properties=props))
    IX, IY, PV = np.concatenate(IX), np.concatenate(IY), np.concatenate(PV)
    print(f"drives {len(S)}, strips {len(strips)}, crossing lines {stats['conf']} confident + {stats['prob']} probable, "
          f"along lines {stats['along_conf']} confirmed + {stats['along_cand']} other, glow samples {len(PV) / 1e6:.1f} M", flush=True)
    tiles = {}
    for f_, zmin in LEVELS:
        ix, iy = IX // f_, IY // f_; tx, ty = ix // TILE, iy // TILE; key = tx.astype(np.int64) * 100000 + ty
        o = np.argsort(key, kind="stable"); ks, idx = np.unique(key[o], return_index=True); bounds_ = np.r_[idx, len(o)]; out = []
        for n, kk in enumerate(ks):
            sel = o[bounds_[n]: bounds_[n + 1]]; TX, TY = int(kk // 100000), int(kk % 100000)
            img = np.zeros((TILE, TILE), np.float32); np.maximum.at(img, (iy[sel] - TY * TILE, ix[sel] - TX * TILE), PV[sel])
            ys, xs = np.nonzero(img >= P_MIN); r0, r1, q0, q1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1     # crop to the used part
            pf = px * f_; (lo0, la0), (lo1, la1) = ll([x0 + (TX * TILE + q0) * pf, x0 + (TX * TILE + q1) * pf], [y1 - (TY * TILE + r1) * pf, y1 - (TY * TILE + r0) * pf])
            out.append(dict(png=png_uri(img[r0:r1, q0:q1]), bounds=[[la0, lo0], [la1, lo1]]))
        tiles[str(zmin)] = out; print(f"raster level {PIX * f_:.2f} m: {len(out)} tiles, {sum(len(t['png']) for t in out) / 1e6:.1f} MB", flush=True)
    pins, pinned = junction_layers(BIAS)
    SP = np.array([p for s in strips for p in s["geometry"]["coordinates"][0][::5]])
    recs = records_near(SP)
    fr_vn = dict(np.load(CACHE / "mp_frame.npz")); w = (fr_vn["s_nodes"] >= 2700) & (fr_vn["s_nodes"] <= 3640)
    def bnd(x, y):
        c = np.array(ll(x, y)); return [[c[:, 1].min(), c[:, 0].min()], [c[:, 1].max(), c[:, 0].max()]]
    find = lambda sess, surv, sw: next((i for i, r in S.items() if r["session"].endswith(sess) and r["survey"].endswith(surv) and r["swath"] == sw), None)
    areas = [dict(key="all", name=CC.get("label", "Whole survey"), note=f"{len(S)} drives, {M['km']:.0f} km", bounds=bnd(allx, ally)),
             dict(key="nizza", name="Via Nizza", note="6 drives on 13 Oct", bounds=bnd(fr_vn["xn"][w], fr_vn["yn"][w]))]
    for key, name, note, spec in (("ricci", "Corso Ricci", "development drive, 14 Oct", ("2025.10.14_11.31.18", "11-37-03", "Swath001")),
                                  ("fontanassa", "Via Fontanassa", "15 Oct", ("2025.10.15_09.07.28", "09-11-09", "Swath016"))):
        i = find(*spec)
        if i is not None:
            areas.append(dict(key=key, name=name, note=note, bounds=bnd(F[i]["xn"], F[i]["yn"])))
    checks = dict(label=CC.get("label", "Whole survey"), km=round(M["km"], 1), n_drives=M["n_drives"], C1=M["C1"], C2=M["C2"], C3={k: v for k, v in M["C3"].items()}, C4=M["C4"], C4h=M["C4_heldout"],
                  repeat_rule=M["repeat_crossing_rule_used"], along_rule=M["along_confirmed_rule_used"], stats=stats)
    along += lane_confirmed(stats)                                   # E45/E46/E47: stretches seen from two lanes after registration
    ptiles, n_p = p_tiles(x0, y1, px); print(f"P raster: {n_p} drives", flush=True)
    out = dict(areas=areas, strips=strips, cross=cross, along=along, tiles=tiles, ptiles=ptiles, pins=pins, pinned=pinned, tracks=survey_tracks(), records=recs, checks=checks)
    html = open(Path(__file__).parent / "templates" / "template_city.html").read().replace("/*__DATA__*/null", json.dumps(out, separators=(",", ":")))
    open(out_path, "w").write(html); print("saved", out_path, f"{len(html) / 1e6:.1f} MB; records {len(recs)}, pins {len(pins)}")


if __name__ == "__main__":
    main(sys.argv[1])
