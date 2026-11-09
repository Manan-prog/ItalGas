"""Public utility records (Comune di Savona SIT, CC BY 4.0, data_public/) expressed in
swath coordinates (s along track, u across, m). Used only as an independent check:
record positional accuracy is not stated by the publisher."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cube import to_su
from geometry import lonlat_to_merc


def records_in_footprint(geojson: Path, s_nodes, xn, yn, k: float, u_lim=(-0.6, 0.9),
                         step_m: float = 0.1) -> list[dict]:
    """Every record line that enters the strip u_lim. For each: s0 = s where the line
    crosses u = 0 (or the mean s inside the strip), phi = deviation from perpendicular
    to the track (deg, same convention as crossings.py: s = s0 + u tan phi), length
    inside the strip, and the record attributes (diameter, material, depth in cm)."""
    feats = json.load(open(geojson))["features"]
    out = []
    for f in feats:
        g = f["geometry"]
        lines = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
        for ln in lines:
            ln = np.asarray(ln, float)
            X, Y = lonlat_to_merc(ln[:, 0], ln[:, 1])
            seg = np.hypot(np.diff(X), np.diff(Y)) * k
            n = np.maximum(1, np.ceil(seg / step_m).astype(int))
            xs = np.concatenate([np.linspace(X[i], X[i + 1], n[i], endpoint=False) for i in range(len(seg))] + [X[-1:]])
            ys = np.concatenate([np.linspace(Y[i], Y[i + 1], n[i], endpoint=False) for i in range(len(seg))] + [Y[-1:]])
            s, u = to_su(xs, ys, s_nodes, xn, yn, k)
            ins = (u >= u_lim[0]) & (u <= u_lim[1]) & (s > s_nodes[1]) & (s < s_nodes[-2])
            if ins.sum() < 2:
                continue
            si, ui = s[ins], u[ins]
            if np.ptp(ui) > 0.3:                                     # crosses the strip
                A = np.c_[np.ones_like(ui), ui]
                s0, tphi = np.linalg.lstsq(A, si, rcond=None)[0]
                phi = float(np.degrees(np.arctan(tphi)))
                kind = "crossing"
            else:
                s0, phi, kind = float(si.mean()), 90.0, "longitudinal"
            p = f["properties"]
            out.append(dict(id=f.get("id"), kind=kind, s0=float(s0), phi_deg=phi, u_mean=float(ui.mean()),
                            s_min=float(si.min()), s_max=float(si.max()), len_in_m=float(ins.sum() * step_m),
                            diam=p.get("diametro"), material=p.get("materiale"),
                            depth_cm=[p.get("prof_iniz"), p.get("prof_fin")], street=p.get("via")))
    return out
