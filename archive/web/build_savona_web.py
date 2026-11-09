"""Interactive web map (one local HTML file) of the three studied streets: Via Nizza (6 registered drives, 13 Oct),
Corso Ricci (14 Oct 11:31 Swath001) and Via Fontanassa (15 Oct 09:11 Swath016). Layers:
 (1) scanned areas = each drive's radar strip (200 MHz channel span); every drive of the survey as a thin line;
 (2) probability raster = P that a pipe passes under each 0.15 m cell: crossing pipes only (P = 1 - local fdr against the
     antenna-column-shift null), spread along the street by the position sigma; highest over overlapping drives; bright = high;
 (3) pipelines: crossings seen by >= 2 drives or both radar rows = confident; one row / one drive with P >= 0.9 = probable;
     along-street candidates (no calibrated probability, unconfirmed) in their own layer.
 Optional layers: junction pins (city-wide, E30) and municipal storm/sewer records (Comune di Savona SIT, CC BY 4.0).
 Single drives are shifted +1.19 m along travel (ASSUMED along-travel bias, E25); Via Nizza's registration fits its own.
 Position sigma = max(|real-time - post-processed track|, 0.3 m) (DECISIONS 2026-11-03), a lower bound.
 The whole-survey page (savona_pipes_map.html) is now built by build_savona_city_web.py; this script keeps the three-street version.
 Usage (from figures/web/): python build_savona_web.py savona_pipes_map_3streets.html"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from web_common import CACHE, DET, RAW, STOPS, junction_layers, ll, png_uri, records_near, survey_tracks
from build_cube import swath_geometry  # noqa: E402  (web_common puts pipeline/ and figures/ on the path)
from build_v2 import cfg_hash  # noqa: E402
from ids_reader import Swath  # noqa: E402
from map_layers import is_confident, merge_drives, multipass_drives, prob_on_cells  # noqa: E402
from run_detection import su_to_map  # noqa: E402

BIAS = 1.19          # m along travel, ASSUMED (E25), single drives only
PIX = 0.15           # m, raster cell (true metres)
SIG_FLOOR = 0.3      # m, DECISIONS 2026-11-03
PP_FAIL = 5.0        # m, |real-time - post-processed| above which the post-processed solution is treated as failed
SIG_UNKNOWN = 1.0    # m, ASSUMED along-street sigma where it cannot be measured (lane-level, DECISIONS 2026-11-02)
P_LINE = 0.9         # probable crossing line threshold (P)
TILE = 768           # px per raster tile


def sigma_at(disp, s):
    """(sigma_along, sigma_across) m at s from a trajectory-disagreement record dict(s, ds, du); None if absent or if the
    post-processed solution has failed there (|disagreement| > PP_FAIL m, not a usable error estimate)."""
    if not disp:
        return None
    a = np.interp(s, disp["s"], np.abs(disp["ds"])); c = np.interp(s, disp["s"], np.abs(disp["du"]))
    if a > PP_FAIL or c > PP_FAIL:
        return None
    return round(float(max(a, SIG_FLOOR)), 2), round(float(max(c, SIG_FLOOR)), 2)


def raster_sigma(disp, s):
    """Along-street blur for the glow: measured sigma, else SIG_UNKNOWN (ASSUMED lane-level, DECISIONS 2026-11-03)."""
    sg = sigma_at(disp, s); return sg[0] if sg else SIG_UNKNOWN


class Area:
    """One street: frame (s_nodes, xn, yn, k), s shift (m), strips, crossings, along-street candidates."""

    def __init__(self, key, name, note, frame, shift):
        self.key, self.name, self.note, self.f, self.shift = key, name, note, frame, shift
        self.k = float(frame["k"]); self.strips, self.samples, self.cross, self.along = [], [], [], []

    def to_map(self, s, u):
        return su_to_map(np.asarray(s, float) + self.shift, np.asarray(u, float), self.f["s_nodes"], self.f["xn"], self.f["yn"], self.k)

    def add_strip(self, label, s, lo, hi, X, sig):
        """Footprint polygon + raster samples (0.05 m) of one drive; X = crossings dict(s0, phi, P) in this frame."""
        sd = np.arange(s.min(), s.max(), 1.0); lo_d = np.interp(sd, s, lo); hi_d = np.interp(sd, s, hi)
        x, y = self.to_map(np.r_[sd, sd[::-1]], np.r_[hi_d, lo_d[::-1]]); self.strips.append(dict(label=label, poly=ll(x, y)))
        for c0 in np.arange(s.min(), s.max(), 100.0):                     # chunks keep memory small
            ss = np.arange(c0, min(c0 + 100.0, s.max()), 0.05); f = np.linspace(0, 1, 32)
            l_, h_ = np.interp(ss, s, lo), np.interp(ss, s, hi); S = np.repeat(ss[:, None], 32, 1); U = l_[:, None] + f[None, :] * (h_ - l_)[:, None]
            near = [dict(o, sig=sig) for o in X if c0 - 5 <= o["s0"] <= c0 + 105]
            P = prob_on_cells(S, U, near, mass=True) if near else np.zeros_like(S)
            mx, my = self.to_map(S.ravel(), U.ravel()); self.samples.append((mx, my, P.ravel()))

    def crossing_line(self, s0, phi, u0, u1, props):
        uu = np.linspace(u0, u1, 12); x, y = self.to_map(s0 + uu * np.tan(np.radians(phi)), uu)
        self.cross.append(dict(type="Feature", geometry=dict(type="LineString", coordinates=ll(x, y)), properties=dict(props, area=self.name)))

    def along_line(self, s, u, props):
        x, y = self.to_map(s, u)
        self.along.append(dict(type="Feature", geometry=dict(type="LineString", coordinates=ll(x, y)), properties=dict(props, area=self.name)))

    def raster_tiles(self):
        """Max-P image on a north-up Mercator grid, cut into tiles; returns [dict(png data URI, bounds [[lat, lon], [lat, lon]])]."""
        mx = np.concatenate([a for a, _, _ in self.samples]); my = np.concatenate([b for _, b, _ in self.samples]); P = np.concatenate([c for _, _, c in self.samples])
        px = PIX / self.k; x0, y1 = mx.min() - px, my.max() + px
        ix = ((mx - x0) / px).astype(int); iy = ((y1 - my) / px).astype(int); W, H = ix.max() + 2, iy.max() + 2
        img = np.zeros((H, W), np.float32); np.maximum.at(img, (iy, ix), P.astype(np.float32))
        tiles = []
        for ty in range(0, H, TILE):
            for tx in range(0, W, TILE):
                t = img[ty: ty + TILE, tx: tx + TILE]
                if t.max() < STOPS[0, 0]:
                    continue
                h, w = t.shape
                (lo0, la0), (lo1, la1) = ll([x0 + tx * px, x0 + (tx + w) * px], [y1 - (ty + h) * px, y1 - ty * px])
                tiles.append(dict(png=png_uri(t), bounds=[[la0, lo0], [la1, lo1]]))
        return tiles, int(W * H), float(img.max())


def crossing_props(kind, P, depth, phi, n, sig, extra=""):
    return dict(kind=kind, P=round(float(P), 3), depth_m=round(float(depth), 2), angle_deg=round(float(phi), 0), support=n,
                sigma=sig, note=extra)


# ---------------------------------------------------------------- Via Nizza: 6 registered drives (E16, E17, E24)
REG = json.load(open(CACHE / "mp_registration.json"))["registration"]; FR = dict(np.load(CACHE / "mp_frame.npz"))
vn = Area("nizza", "Via Nizza", "6 drives on 13 Oct (3 each way), registered to each other (E16); 940 m studied stretch.", FR, 0.0)
DRV = multipass_drives(REG); disp_ref = json.load(open(CACHE / "mp_Swath041.json"))["arrays"]["2"]["pp_disp"]
for n, d in DRV.items():
    w = (d["s"] >= 2700) & (d["s"] <= 3640)
    vn.add_strip(f"Via Nizza, 13 Oct drive {n[-3:]} ({'with' if REG[n]['direction'] > 0 else 'against'} the reference direction)",
                 d["s"][w], d["lo"][w], d["hi"][w], [dict(s0=o["S0"], phi=o["phi"], P=o["P"]) for o in d["X"]], 0.3)
for g in merge_drives(DRV, 0.5, window=(2700.0, 3640.0)):           # E33: >= 3 drives within 0.5 m, chance 3.3 of 41
    conf = is_confident(g)
    if not (conf or g["P"] >= P_LINE):
        continue
    us = [v for n in g["members"] for v in (np.interp(g["S0"], DRV[n]["s"], DRV[n]["lo"], left=np.nan, right=np.nan),
                                            np.interp(g["S0"], DRV[n]["s"], DRV[n]["hi"], left=np.nan, right=np.nan)) if np.isfinite(v)]
    if not us:
        continue
    seen = f"{g['n_drives']} drive(s)" + (", both radar rows" if g["rows"] == 2 else ", one radar row")
    vn.crossing_line(g["S0"], g["phi"], min(us), max(us), crossing_props("confident" if conf else "probable", g["P"], g["depth"], g["phi"], seen,
                     sigma_at(disp_ref, g["S0"]), "Drawn across the lanes of the drives that saw it; registration residual 0.23 m RMS along the street (E16)."))
VP = json.load(open(CACHE / "vianizza_pipelines.json"))
groups = []
for f in VP["along_road"]:
    uc = float(np.mean([q for q in f["uc"] if q is not None])); g = next((G for G in groups if abs(G["u"] - uc) < 0.5 and f["s0"] <= G["s1"] + 2 and f["s1"] >= G["s0"] - 2), None)
    if g is None:
        groups.append(dict(u=uc, s0=f["s0"], s1=f["s1"], drives={f["a"], f["b"]}, z=list(f["z0"])))
    else:
        g["s0"] = min(g["s0"], f["s0"]); g["s1"] = max(g["s1"], f["s1"]); g["drives"] |= {f["a"], f["b"]}; g["z"] += f["z0"]
for g in groups:
    best = len(g["drives"]) >= 3
    vn.along_line([g["s0"], g["s1"]], [g["u"], g["u"]], dict(kind="along-best" if best else "along-weak", depth_m=round(float(np.median(g["z"])), 2),
                  length_m=round(g["s1"] - g["s0"], 1), support=f"{len(g['drives'])} drives ({', '.join(sorted(x[-3:] for x in g['drives']))})",
                  sigma=sigma_at(disp_ref, 0.5 * (g["s0"] + g["s1"])),
                  note=("The same pipe-shaped arch seen by 3 drives in neighbouring lanes, apex positions within 0.2 m (E17, E24; 4 of 54 repeated features vs 1 percent by chance). "
                        "Nearest municipal record: 600 mm sewer 1-2 m away recorded 2.6 m deep, so unrecorded utility or wrong record.") if best else
                       "Two drives in nearly the same lane agree; they are not independent views from different lanes (weak)."))

# ---------------------------------------------------------------- Corso Ricci: development drive (E09-E13, E26)
dev = Swath.load(str(RAW / "savona 14.10.2025 matt_2025.10.14_11.31.18/Survey_2025-10-14_11-37-03"), "Swath001"); gd = swath_geometry(dev, DET["base"])
cr = Area("ricci", "Corso Ricci", "1 drive on 14 Oct 11:31 (development drive), 1.2 km; positions shifted +1.19 m along travel (ASSUMED, E25).",
          {k: gd[k] for k in ("s_nodes", "xn", "yn", "k")}, BIAS)
h = cfg_hash(DET); u1 = np.load(CACHE / f"v2grid_1_{h}.npz")["u_ch"]; g2 = np.load(CACHE / f"v2grid_2_{h}.npz"); u2 = g2["u_ch"]; sg = g2["s_grid"]
FV = json.load(open(CACHE / "final_v2_xing_a65d608384_long_df5d12447d.json")); AD = FV["along_disagreement"]
disp_cr = dict(s=AD["s"], ds=AD["along"], du=AD["cross"]); lo_c, hi_c = min(u1.min(), u2.min()) - 0.04, max(u1.max(), u2.max()) + 0.04
cr.add_strip("Corso Ricci, 14 Oct 11:31 drive 001", sg, np.full(len(sg), lo_c), np.full(len(sg), hi_c), [], 0.3)
cr.samples = []                                                         # crossings carry their own sigma: resample with it
for c0 in np.arange(sg.min(), sg.max(), 100.0):
    ss = np.arange(c0, min(c0 + 100.0, sg.max()), 0.05); S, U = np.meshgrid(ss, np.linspace(lo_c, hi_c, 32), indexing="ij")
    near = [dict(s0=o["s0_m"], phi=o["phi_deg"], P=o["P"], sig=o["sigma_along_m"]) for o in FV["crossings"] if c0 - 5 <= o["s0_m"] <= c0 + 105]
    P = prob_on_cells(S, U, near, mass=True) if near else np.zeros_like(S); mx, my = cr.to_map(S.ravel(), U.ravel()); cr.samples.append((mx, my, P.ravel()))
for o in FV["crossings"]:
    if o["P"] >= P_LINE:
        kind = "confident" if o["cls"] == "A" else "probable"
        cr.crossing_line(o["s0_m"], o["phi_deg"], lo_c, hi_c, crossing_props(kind, o["P"], o["depth_m"], o["phi_deg"],
                         "both radar rows" if o["cls"] == "A" else f"one radar row ({o['freq']} MHz)", sigma_at(disp_cr, o["s0_m"])))
for t in FV["tracks"]:
    if t["freq"] == "200" and t["q"] <= 0.05:
        cr.along_line(t["path_s"], t["path_u"], dict(kind="along-weak", depth_m=t["depth_m"], length_m=round(t["length_m"], 1), support="1 drive, 200 MHz row",
                      sigma=sigma_at(disp_cr, 0.5 * (t["s_start"] + t["s_end"])),
                      note="Long reflector about 0.85 m deep, significant against the channel-shuffle test, but wide and flat across the whole strip: "
                           "a large pipe (main or collector) or a layer, undetermined (E12). At 700-775 m it lies on the Regione Liguria sewer-collector record."))

# ---------------------------------------------------------------- Via Fontanassa: one drive (E29, E31)
FZ = json.load(open(CACHE / "mp_fz_Swath016.json")); FFR = dict(np.load(CACHE / "mp_fz_frame.npz"))
fz = Area("fontanassa", "Via Fontanassa", "1 drive on 15 Oct 09:11, 1.07 km; positions shifted +1.19 m along travel (ASSUMED, E25).", FFR, BIAS)
DF = multipass_drives({"Swath016": dict(A=0.0, B=0.0)}, tag="fz_")["Swath016"]; disp_fz = (FZ["arrays"].get("2") or {}).get("pp_disp")
fz.add_strip("Via Fontanassa, 15 Oct 09:11 drive 016", DF["s"], DF["lo"], DF["hi"], [], 0.3); fz.samples = []
for c0 in np.arange(DF["s"].min(), DF["s"].max(), 100.0):
    ss = np.arange(c0, min(c0 + 100.0, DF["s"].max()), 0.05); f = np.linspace(0, 1, 32)
    l_, h_ = np.interp(ss, DF["s"], DF["lo"]), np.interp(ss, DF["s"], DF["hi"]); S = np.repeat(ss[:, None], 32, 1); U = l_[:, None] + f[None, :] * (h_ - l_)[:, None]
    near = [dict(s0=o["S0"], phi=o["phi"], P=o["P"], sig=raster_sigma(disp_fz, o["S0"])) for o in DF["X"] if c0 - 5 <= o["S0"] <= c0 + 105]
    P = prob_on_cells(S, U, near, mass=True) if near else np.zeros_like(S); mx, my = fz.to_map(S.ravel(), U.ravel()); fz.samples.append((mx, my, P.ravel()))
for o in DF["X"]:
    if o["P"] >= P_LINE:
        lo_, hi_ = np.interp(o["S0"], DF["s"], DF["lo"]), np.interp(o["S0"], DF["s"], DF["hi"])
        fz.crossing_line(o["S0"], o["phi"], lo_, hi_, crossing_props("confident" if o["rows"] == 2 else "probable", o["P"], o["depth"], o["phi"],
                         "both radar rows" if o["rows"] == 2 else "one radar row", sigma_at(disp_fz, o["S0"])))
AL = json.load(open(CACHE / "along_20251015_090728_09-11-09_Swath016.json"))
t = max((t for t in AL["arrays"]["2"]["tracks"] if t["s_end"] > 366 and t["s_start"] < 383 and abs(t["u_mean"] + 0.44) < 0.1 and t["depth_m"] < 0.4), key=lambda q: q["length_m"])
fz.along_line(t["path_s"], t["path_u"], dict(kind="along-weak", depth_m=t["depth_m"], length_m=round(t["length_m"], 1), support="1 drive, 200 MHz row",
              sigma=sigma_at(disp_fz, 0.5 * (t["s_start"] + t["s_end"])),
              note="Lies 0.03 m from the recorded 500 mm concrete storm pipe, at its inferred crown depth, running manhole to manhole (E29); "
                   "only 15 m inside the strip, too short to pass the chance test on its own (q 0.57)."))
VT = json.load(open(CACHE / "city" / "viterbi_Swath016_09-11-09.json")); sig_tr = [t for t in VT["tracks"] if t["significant"]]
if sig_tr:
    t = max(sig_tr, key=lambda q: q["score"])
    fz.along_line(t["path_s"], t["path_u"], dict(kind="along-weak", depth_m=t["depth_m"], length_m=round(t["s_end"] - t["s_start"], 1),
                  support=f"1 drive; {len(sig_tr)} significant continuous paths here (depths {', '.join('%.2f' % q['depth_m'] for q in sig_tr)} m)",
                  sigma=sigma_at(disp_fz, 0.5 * (t["s_start"] + t["s_end"])),
                  note="Continuous echo linked by the depth-aware Viterbi tracker, significant against channel-shuffled data (E31); 0.16 m beside a recorded "
                       "PVC sewer but deeper than its record (about 0.5 m), so not confirmed as that pipe."))

# ---------------------------------------------------------------- city-wide context and reference layers
pins, pinned = junction_layers(BIAS); tracks = survey_tracks()
areas = [vn, cr, fz]
recs = records_near(np.array([p for a in areas for s in a.strips for p in s["poly"]]))

# ---------------------------------------------------------------- assemble
out = dict(areas=[], strips=[], cross=[], along=[], tiles=[], pins=pins, pinned=pinned, tracks=tracks, records=recs)
for a in areas:
    tiles, npx, pmax = a.raster_tiles(); out["tiles"] += tiles
    pts = np.array([p for s in a.strips for p in s["poly"]])
    out["strips"] += [dict(type="Feature", geometry=dict(type="Polygon", coordinates=[s["poly"] + [s["poly"][0]]]), properties=dict(label=s["label"], area=a.name)) for s in a.strips]
    out["cross"] += a.cross; out["along"] += a.along
    nc = sum(f["properties"]["kind"] == "confident" for f in a.cross); npb = sum(f["properties"]["kind"] == "probable" for f in a.cross)
    out["areas"].append(dict(key=a.key, name=a.name, note=a.note, bounds=[[pts[:, 1].min(), pts[:, 0].min()], [pts[:, 1].max(), pts[:, 0].max()]],
                             n_conf=int(nc), n_prob=int(npb), n_along=len(a.along), n_drives=len(a.strips)))
    print(f"{a.name}: {len(a.strips)} strips, {nc} confident + {npb} probable crossing lines, {len(a.along)} along-street candidates, "
          f"{len(tiles)} raster tiles ({npx / 1e6:.1f} Mpx, max P {pmax:.2f})")
print(f"pins {len(pins)}, pinned segments kept {len(pinned)}, survey tracks {len(tracks)}, municipal records {len(recs)}")
html = open(Path(__file__).with_name("template.html")).read().replace("/*__DATA__*/null", json.dumps(out, separators=(",", ":")))
open(sys.argv[1], "w").write(html); print("saved", sys.argv[1], f"{len(html) / 1e6:.2f} MB")
