"""Shared pieces of the Savona web maps (build_savona_web.py: three streets; build_savona_city_web.py: whole survey):
lon/lat rounding, the glow colour ramp and PNG encoding, and the reference layers (survey tracks, junction pins,
pinned segments, municipal storm/sewer records)."""
from __future__ import annotations

import base64
import glob
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "figures")]
from build_cube import swath_geometry  # noqa: E402
from build_v2 import load_cfg  # noqa: E402
from geometry import merc_scale, merc_to_lonlat  # noqa: E402
from ids_reader import Swath  # noqa: E402
from paths import CACHE, RAW, load_yaml  # noqa: E402
from run_detection import su_to_map  # noqa: E402

DET = load_cfg(str(ROOT / "pipeline/configs/dev_swath_v2_2.yaml"))
# glow ramp: (P, r, g, b, alpha); dark orange to pale yellow, alpha rising with P (sequential, one hue family)
STOPS = np.array([[0.02, 90, 26, 10, 0.40], [0.30, 184, 64, 28, 0.70], [0.60, 235, 104, 52, 0.88], [0.85, 255, 179, 107, 0.96], [1.0, 255, 244, 204, 1.0]])


def ll(x, y):
    """Web Mercator (m) -> [[lon, lat], ...] rounded to 1e-7 deg."""
    lon, lat = merc_to_lonlat(np.asarray(x, float), np.asarray(y, float)); return np.c_[np.round(lon, 7), np.round(lat, 7)].tolist()


def png_uri(P: np.ndarray) -> str:
    """P image (H, W) in [0, 1] -> data-URI PNG with the glow ramp (transparent below the first stop), 128-colour palette
    (about 4.5 x smaller than RGBA; colour error <= 14/255 measured on test tiles, display only)."""
    from PIL import Image
    rgba = np.zeros(P.shape + (4,), np.float32)
    for c in range(4):
        rgba[..., c] = np.interp(P, STOPS[:, 0], STOPS[:, c + 1]) * (1.0 if c < 3 else 255.0)
    rgba[..., 3] *= P >= STOPS[0, 0]
    im = Image.fromarray(np.clip(rgba + 0.5, 0, 255).astype(np.uint8), "RGBA").quantize(colors=128, method=Image.Quantize.FASTOCTREE)
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True); return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


RG_STOPS = np.array([[0.0, 192, 57, 43, 0.85], [0.5, 239, 233, 220, 0.85], [1.0, 26, 152, 80, 0.85]])   # red -> neutral -> green (user request)


def png_uri_rg(P: np.ndarray) -> str:
    """P image (H, W) in [0, 1] -> data-URI PNG, red (0) to green (1), transparent where nan."""
    from PIL import Image
    Q = np.nan_to_num(P, nan=0.0); rgba = np.zeros(P.shape + (4,), np.float32)
    for c in range(4):
        rgba[..., c] = np.interp(Q, RG_STOPS[:, 0], RG_STOPS[:, c + 1]) * (1.0 if c < 3 else 255.0)
    rgba[..., 3] *= np.isfinite(P)
    im = Image.fromarray(np.clip(rgba + 0.5, 0, 255).astype(np.uint8), "RGBA").quantize(colors=128, method=Image.Quantize.FASTOCTREE)
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True); return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def survey_tracks(step_m: float = 4.0) -> list[dict]:
    """Every drive of the survey as a thin line (GPS track, decimated to step_m)."""
    T = np.load(load_yaml(ROOT / "pipeline/configs/citywide.yaml")["swath_tracks"], allow_pickle=True); out = []
    for i in range(len(T["days"])):
        tr = T[f"s{i}"]
        if len(tr) < 3:
            continue
        dd = np.r_[0, np.cumsum(np.hypot(*np.diff(tr, axis=0).T))] * float(merc_scale(tr[:1, 1])[0])
        keep = np.r_[np.unique(np.searchsorted(dd, np.arange(0, dd[-1], step_m))), len(tr) - 1]
        out.append(dict(type="Feature", geometry=dict(type="LineString", coordinates=ll(tr[keep, 0], tr[keep, 1])), properties=dict(day=str(T["days"][i]))))
    return out


def junction_layers(bias: float) -> tuple[list[dict], list[dict]]:
    """Junction pins (E30) and pinned segments drawn along the drive path (straightness >= 0.8)."""
    cc = json.load(open(CACHE / "city" / "anchors_consensus.json"))
    pins = [dict(type="Feature", geometry=dict(type="Point", coordinates=ll([c["x"]], [c["y"]])[0]), properties=dict(depth_m=c["depth_m"], support=c["support"])) for c in cc]
    files = sorted(glob.glob(str(RAW / "*" / "Survey_*" / "Swath*.path"))); pinned = []
    for sgm in json.load(open(CACHE / "city" / "pinned_segments.json")):
        f = files[sgm["drive"]]; g = swath_geometry(Swath.load(str(Path(f).parent), Path(f).stem), DET["base"])
        ss = np.linspace(sgm["s0"], sgm["s1"], 200) + bias; uu = np.linspace(sgm["u0"], sgm["u1"], 200)
        x, y = su_to_map(ss, uu, g["s_nodes"], g["xn"], g["yn"], float(g["k"])); kk = float(g["k"])
        if np.hypot(x[-1] - x[0], y[-1] - y[0]) * kk / (sgm["s1"] - sgm["s0"]) >= 0.8:      # skip pairings on strongly curving roads (E30)
            pinned.append(dict(type="Feature", geometry=dict(type="LineString", coordinates=ll(x[::5], y[::5])),
                               properties=dict(length_m=round(sgm["s1"] - sgm["s0"]), depth=f"{sgm['z0']:.2f}-{sgm['z1']:.2f}", u=f"{sgm['u0']:+.1f} to {sgm['u1']:+.1f}")))
    return pins, pinned


def records_near(lonlat: np.ndarray, radius_m: float = 30.0) -> list[dict]:
    """Municipal storm/sewer record lines (Comune di Savona SIT, CC BY 4.0) with any vertex or midpoint within radius_m of the given points."""
    from scipy.spatial import cKDTree
    kx = float(np.cos(np.radians(lonlat[:, 1].mean())))
    tree = cKDTree(np.c_[lonlat[:, 0] * 111320 * kx, lonlat[:, 1] * 110574])        # approx. metres, adequate for a 30 m keep radius
    recs = []
    for fn, kind in (("gfi_all_condotte_acque_bianche.json", "storm"), ("gfi_all_condotte_acque_nere.json", "sewer")):
        for f in json.load(open(ROOT / "data_public" / fn))["features"]:
            if f["geometry"]["type"] != "LineString":
                continue
            c = np.array(f["geometry"]["coordinates"], float).reshape(-1, 2); m = np.r_[c, 0.5 * (c[1:] + c[:-1])]
            if tree.query(np.c_[m[:, 0] * 111320 * kx, m[:, 1] * 110574], distance_upper_bound=radius_m)[0].min() < radius_m:
                pr = f["properties"]; recs.append(dict(type="Feature", geometry=dict(type="LineString", coordinates=np.round(c, 7).tolist()),
                                                       properties=dict(kind=kind, diam=pr.get("diametro", ""), mat=pr.get("materiale", ""),
                                                                       depth=f"{pr.get('prof_iniz', '')}/{pr.get('prof_fin', '')}", street=pr.get("via", ""))))
    return recs
