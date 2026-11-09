"""Web map of the Savona survey (one local HTML file), 4 Nov 2026 design: ONE evidence raster and ONE line layer derived
from it for pipes running along the street, plus the crossing-pipe lines where the crossing scan was run.
 - raster: per-cell evidence P (prob_raster.py; red 0 -> green 1), max over depths and over overlapping drives;
 - along-street lines: runs of raster cells by the rule in pipeline/configs/raster_lines.yaml (raster_lines.py), each
   with its mean P, chance share (same rule on block-shuffled windows), repeat-drive agreement (E47 registration);
 - strips: every drive's radar footprint, classed 'scanned once' / 'repeat drives' (survey_pairs.py rule);
 - crossing lines: unchanged (citywide_scan crossings), only for drives that were scanned for crossings.
Hover shows the confidence of a line; click gives the details. Supersedes build_savona_city_web.py (glow, linker lines).
Usage (from figures/web/): python build_savona_pipes_web.py savona_pipes_map.html [cache root, default ~/ItalGas/cache]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from web_common import CACHE, ROOT, junction_layers, ll, records_near, survey_tracks
from build_savona_city_web import BIAS, P_LINE, PIX, SIG_FLOOR, drive_label, p_tiles, sig_of, strip_poly  # noqa: E402
from citywide_merge import load_all  # noqa: E402

RTK_FIX = 4


def main(out_path: str, cache: Path = CACHE) -> None:
    city = cache / "city"; RL = json.load(open(city / "raster_lines.json")); cfg = RL["config"]
    LC = json.load(open(city / "lane_components.json")); PD = LC.get("per_drive", {}); rule = LC.get("rule", {})
    LS = json.load(open(city / "lane_summary.json")) if (city / "lane_summary.json").exists() else None
    merged = [json.load(open(f)) for f in sorted((CACHE / "city").glob("citywide_merged*.json"))]            # citywide_merge.py outputs
    M = max(merged, key=lambda m: m.get("n_drives", 0), default=None)                                      # the merge that covers most drives
    S, F = load_all(CACHE / "city") if any((CACHE / "city").glob("scan_*.json")) else ({}, {})
    lab = {(c["drive"], c["idx"]): c for c in M["crossing_labels"]} if M else {}
    # ---- drives with a frame (= with rasters): strips and summary
    frames = {int(f.stem.split("_")[1]): dict(np.load(f)) for f in city.glob("rframe_*.npz")}
    rl_by_drive = {}
    for ft in RL["features"]:
        rl_by_drive.setdefault(ft["properties"]["drive"], []).append(ft)
    k_ref = float(np.median([fr["k"] for fr in frames.values()])); px = PIX / k_ref
    allx = np.concatenate([fr["xn"] for fr in frames.values()]); ally = np.concatenate([fr["yn"] for fr in frames.values()]); x0, y1 = allx.min() - 50 / k_ref, ally.max() + 50 / k_ref
    strips, n_single, n_repeat, km_single, km_repeat, n_raw = [], 0, 0, 0.0, 0.0, 0
    for i, fr in sorted(frames.items()):
        meta = json.load(open(city / f"rline_{i}.json")); name = drive_label(meta); L = float(fr["length_m"]); n_raw += int(meta.get("n_raw_runs", 0))
        pdv = PD.get(str(i), {}); usable = pdv.get("usable_repeat_m", 0); cls = "repeat" if usable >= rule.get("min_run_m", 50) else "single"
        rq = fr["rt_q"]; dA = fr["d_along_m"]; okq = np.isfinite(rq)
        props = dict(label=name, drive=i, cls=cls, length_km=round(L / 1e3, 2), usable_repeat_m=int(usable), partners=[drive_label(json.load(open(city / f"rline_{j}.json"))) if (city / f"rline_{j}.json").exists() else str(j) for j in pdv.get("partners", [])],
                     n_lines=len(rl_by_drive.get(i, [])), lines_m=round(sum(f_["properties"]["length_m"] for f_ in rl_by_drive.get(i, [])), 1),
                     chance_lines=RL["per_drive"].get(str(i), {}).get("chance_lines"), n_cross=len(S[i]["crossings"]) if i in S else None,
                     rtk_fixed=None if not okq.any() else round(float(np.mean(rq[okq] == RTK_FIX)), 2), pp_ok=round(float(np.mean(np.isfinite(dA))), 2),
                     d_along_med=None if not np.isfinite(dA).any() else round(float(np.nanmedian(dA)), 2))
        if cls == "single":
            n_single += 1; km_single += L / 1e3
        else:
            n_repeat += 1; km_repeat += L / 1e3
        for poly in strip_poly(fr):
            strips.append(dict(type="Feature", geometry=dict(type="Polygon", coordinates=[poly]), properties=props))
    # ---- along-street lines from the raster
    along = []; n_in_drive = {}
    for ft in RL["features"]:
        p = ft["properties"]; r = p.get("repeat"); i = p["drive"]; n_line = n_in_drive.get(i, 0); n_in_drive[i] = n_line + 1   # n-th line of drive i = rline_<i>.json order
        if r and r["partners"]:
            reps = [j for j, v in r["partners"].items() if v["repeats"]]; cov = [j for j, v in r["partners"].items() if v["covered_rows"] > 0]
            names = lambda js: ", ".join(drive_label(json.load(open(city / f"rline_{j}.json"))) if (city / f"rline_{j}.json").exists() else str(j) for j in js)
            if reps:
                rep_txt = f"yes: {names(reps)} (" + "; ".join(f"{int(100 * r['partners'][j]['share'])}% of {r['partners'][j]['covered_rows'] * 0.5:.0f} m" for j in reps) + ")"; rep_cls = "repeated"
            else:
                rep_txt = f"no: {names(cov)} covered " + "; ".join(f"{r['partners'][j]['covered_rows'] * 0.5:.0f} m, agreed on {int(100 * r['partners'][j]['share'])}%" for j in cov); rep_cls = "not-repeated"
        else:
            rep_txt = "no repeat drive in another lane covers this stretch"; rep_cls = "uncovered"
        conf = p["P_mean"]
        xy = np.array(ft["geometry"]["coordinates"], float)                                   # raster_lines.json is in EPSG:3857 metres
        along.append(dict(type="Feature", geometry=dict(type="LineString", coordinates=ll(xy[:, 0], xy[:, 1])),
                          properties=dict(kind="along", drive=drive_label(p), P=round(conf, 2), P_min=p["P_min"], depth_m=p["depth_m"], depth_range=p["depth_range_m"], length_m=round(p["length_m"], 1),
                                          chance_share=p["chance_share"], rep=rep_cls, rep_txt=rep_txt, u=p["u_mean"], n_runs=p.get("n_runs", 1), runs=p.get("runs", []),
                                          sigma=None if p.get("d_cross_m") is None else [sig_of(p["d_along_m"]), sig_of(p["d_cross_m"])], rtk=p.get("rt_q"), pp_sd=p.get("pp_sd_m"),
                                          img=f"evidence/lines/{i}_{n_line}.webp")))
    # ---- crossing lines (unchanged rule) for drives that were scanned for crossings
    cross = []; n_conf = n_prob = 0
    for i, r in sorted(S.items()):
        name = drive_label(r)
        for j, c in enumerate(r["crossings"]):
            lb = lab.get((i, j), dict(confident=c["cls"] == "A", n_drives=1))
            if not (lb["confident"] or c["P"] >= P_LINE):
                continue
            kind = "confident" if lb["confident"] else "probable"; n_conf += kind == "confident"; n_prob += kind == "probable"
            seen = "both radar rows (600 + 200 MHz)" if c["cls"] == "A" else f"one radar row ({c['freq']} MHz)"
            if lb["n_drives"] >= 2:
                seen += f"; {lb['n_drives']} drives agree within 0.5 m"
            cross.append(dict(type="Feature", geometry=dict(type="LineString", coordinates=ll(c["x"], c["y"])),
                              properties=dict(kind=kind, drive=name, P=c["P"], depth_m=c["depth_m"], angle_deg=round(c["phi_deg"]), support=seen,
                                              sigma=None if c["d_along_m"] is None else [sig_of(c["d_along_m"]), sig_of(c["d_cross_m"])], rtk=c["rt_q"], pp_sd=c["pp_sd_m"],
                                              img=f"evidence/crossings/{i}_{j}.webp")))
    ptiles, n_p = p_tiles(x0, y1, px, cache); print(f"P raster: {n_p} drives", flush=True)
    pins, pinned = junction_layers(BIAS)
    SP = np.array([p for s in strips for p in s["geometry"]["coordinates"][0][::5]]); recs = records_near(SP)

    def bnd(x, y):
        c = np.array(ll(x, y)); return [[c[:, 1].min(), c[:, 0].min()], [c[:, 1].max(), c[:, 0].max()]]
    km = sum(float(fr["length_m"]) for fr in frames.values()) / 1e3
    areas = [dict(key="all", name="Whole survey", note=f"{len(frames)} drives, {km:.0f} km", bounds=bnd(allx, ally))]
    for key, name, note, drives, win in (("nizza", "Via Nizza", "6 drives on 13 Oct", [37], (2700, 3340)), ("ricci", "Corso Ricci", "development drive, 14 Oct", [73], None), ("fontanassa", "Via Fontanassa", "15 Oct", [106], None)):
        ds = [d for d in drives if d in frames]
        if ds:
            fr = frames[ds[0]]; w = slice(None) if win is None else (fr["s_nodes"] >= win[0]) & (fr["s_nodes"] <= win[1])
            areas.append(dict(key=key, name=name, note=note, bounds=bnd(fr["xn"][w], fr["yn"][w])))
    sm = RL["summary"]; e50 = None
    if (city / "rule_compare.json").exists():                                                  # E50: why these thresholds (held-out test set)
        E = json.load(open(city / "rule_compare.json")); key = f"c{cfg['p_min']}_r0.0_g{cfg['max_gap']}_u{cfg['gate_u_m']}"   # the ADOPTED rule's own entry
        ch, cu = E["test"].get(key, E["test"][E["chosen"]]), E["test"][E["current"]]
        rank = lambda S: sorted(S, key=lambda k: -S[k]["excess_min_m"]).index(key) + 1 if key in S else None
        e50 = dict(n_rules=len(E["selection"]), chosen=E["chosen"], rule=key, rank_selection=rank(E["selection"]), rank_test=rank(E["test"]), share_test=ch["share"],
                   null_test=max(ch["null_share"] or 0, ch["null_lat_share"] or 0), excess_m=ch["excess_min_m"], share_current=cu["share"], covered_m=ch["covered_m"], n_test_groups=len(E["test_refs"]))
    checks = dict(n_drives=len(frames), km=round(km, 1), n_single=n_single, n_repeat=n_repeat, km_single=round(km_single, 1), km_repeat=round(km_repeat, 1),
                  repeat_rule=rule, lines=dict(n=sm["n_lines"], km=round(sm["length_m"] / 1e3, 2), chance_n=round(sm["chance_lines"], 1), chance_m=round(sm["chance_length_m"]),
                                               n_covered=sm["n_covered"], m_covered=round(sm["length_covered_m"]), n_repeated=sm["n_repeated"], m_repeated=round(sm["length_repeated_m"]),
                                               rule=dict(p_min=cfg["p_min"], gate_u_m=cfg["gate_u_m"], gate_ns=cfg["gate_ns"], depth_step_m=round(cfg["gate_ns"] * cfg["velocity_m_ns"] / 2, 2), min_len_m=cfg["min_len_m"],
                                                         t_min_ns=cfg["t_min_ns"], depth_min_m=round(cfg["t_min_ns"] * cfg["velocity_m_ns"] / 2, 2), gap_m=cfg["max_gap"] * 0.5,
                                                         merge_tol_m=round(cfg["merge_tol_cells"] * 0.04, 2), merge_min_share=cfg["merge_min_share"],
                                                         u_min_m=cfg["u_min_m"], u_max_m=cfg["u_max_m"],
                                                         block_m=cfg["chance"]["block_windows"] * 0.5, n_draws=cfg["chance"]["n_draws"], repeat_min_share=cfg["repeat_min_share"], tol_m=round(cfg["tol_cells"] * 0.04, 2), min_sep_m=cfg["min_sep_m"]),
                                               n_raw_runs=n_raw, e50=e50),
                  lanes=None if not LS else dict(n_groups=LS["n_tested_components"], n_positive=LS["n_positive"], expected=round(LS["expected_positive_on_pipe_free"], 2), p=LS["p_binomial"],
                                                 f=None if LS["f_pooled"] is None else round(LS["f_pooled"], 3), f_null=None if LS["f_pooled_null_mean"] is None else round(LS["f_pooled_null_mean"], 3), p_pooled=LS["p_pooled"]),
                  cross=None if not M else dict(n_drives=len(S), n_conf=n_conf, n_prob=n_prob, C1=M["C1"], C4=M["C4"], C4h=M["C4_heldout"]))
    out = dict(areas=areas, strips=strips, cross=cross, along=along, ptiles=ptiles, pins=pins, pinned=pinned, tracks=survey_tracks(), records=recs, checks=checks)
    html = open(Path(__file__).with_name("template_pipes.html")).read().replace("/*__DATA__*/null", json.dumps(out, separators=(",", ":")))
    open(out_path, "w").write(html)
    print(f"saved {out_path} {len(html) / 1e6:.1f} MB; drives {len(frames)} ({n_single} single, {n_repeat} repeat), along lines {len(along)}, crossings {len(cross)}, records {len(recs)}")


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]) if len(sys.argv) > 2 else CACHE)
