"""Hosted variant of the Savona pipe map for a claude.ai artifact (6 Nov 2026).
The artifact frame loads images only from files published with the page, so the satellite tiles of the local page cannot be
used there. This script builds, from the LOCAL page (web/build.py output) and without touching its builder:
 1. base/base.json: streets, buildings, water, coast and rail of the survey area as compact vectors from OpenStreetMap
    (ODbL, attribution on the map). Buildings are kept within `bld_buffer_m` of the drives (that is where one zooms in).
 2. evidence/sheet_<drive>_<kind>_<k>.webp + index.json: the radar evidence images as sprite sheets (fetched on first click).
 3. index.html: the local page with the satellite layers replaced by the vector base (drawn on its own canvas, colours from
    the theme tokens), the Leaflet stylesheet inlined (the frame allows external stylesheets from Google Fonts only), the
    evidence images loaded from the packs, the export button hidden (downloads are inert in the frame), and the document
    skeleton removed (the publisher adds its own).
Usage: python -I -m italgas_gpr.web.host <osm overpass json> <out dir> [local page html]   (query: configs/osm_query.txt)
Every replacement below must match exactly once, so a change of the template that breaks this script is noticed."""
from __future__ import annotations

import base64
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from italgas_gpr.io.paths import EVIDENCE, FIGURES, config_path, load_yaml  # noqa: E402

BLD_BUFFER_M = 250.0                                   # buildings kept within this distance of any drive track
ROAD_CLASS = {"motorway": 3, "trunk": 3, "primary": 3, "secondary": 2, "tertiary": 2, "residential": 1, "unclassified": 1,
              "living_street": 1, "pedestrian": 1, "service": 0, "track": 0, "footway": 0, "path": 0, "cycleway": 0, "steps": 0}


def tracks_lonlat() -> np.ndarray:
    """Every drive track as lon/lat points (n, 2), thinned to about 5 m, from the swath track file of the city config."""
    T = np.load(load_yaml(config_path("survey.yaml"))["swath_tracks"], allow_pickle=True)
    pts = np.concatenate([T[f"s{i}"][::100] for i in range(len(T["days"])) if len(T[f"s{i}"]) > 2])   # 0.05 m x 100 = 5 m
    lon = pts[:, 0] / 6378137.0 * 180 / math.pi; lat = (2 * np.arctan(np.exp(pts[:, 1] / 6378137.0)) - math.pi / 2) * 180 / math.pi
    return np.c_[lon, lat]


def near_tracks(polys: list, tr: np.ndarray, buffer_m: float) -> list:
    """Keep the polygons whose first vertex is within buffer_m of a track point (grid hash on 0.005 deg cells)."""
    k = 0.005; cells = defaultdict(list)
    for j, (lo, la) in enumerate(tr):
        cells[(int(lo // k), int(la // k))].append(j)
    cl = math.cos(math.radians(float(tr[:, 1].mean()))); out = []
    for p in polys:
        lo, la = p[0]; c = (int(lo // k), int(la // k)); idx = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                idx += cells.get((c[0] + dx, c[1] + dy), [])
        if idx:
            q = tr[idx]; d = np.hypot((q[:, 0] - lo) * cl, q[:, 1] - la) * 111320.0
            if d.min() <= buffer_m:
                out.append(p)
    return out


def osm_to_base(osm_path: Path, out: Path) -> dict:
    """Compact vector base from an Overpass 'out geom' JSON (downloaded file: parsed as data only)."""
    d = json.load(open(osm_path)); r6 = lambda g: [[round(p["lon"], 6), round(p["lat"], 6)] for p in g]
    roads, bld, water, coast, rail, rivers = [], [], [], [], [], []
    for e in d["elements"]:
        if e.get("type") != "way" or "geometry" not in e:
            continue
        t = e.get("tags", {}); g = r6(e["geometry"])
        if "building" in t:
            bld.append(g)
        elif "highway" in t:
            roads.append([ROAD_CLASS.get(t["highway"], 0), g])
        elif t.get("natural") == "coastline":
            coast.append(g)
        elif t.get("natural") == "water":
            water.append(g)
        elif t.get("waterway"):
            rivers.append(g)
        elif t.get("railway"):
            rail.append(g)
    tr = tracks_lonlat(); n_bld_all = len(bld); bld = near_tracks(bld, tr, BLD_BUFFER_M)
    base = dict(roads=roads, buildings=bld, water=water, coast=coast, rail=rail, rivers=rivers,
                source="OpenStreetMap contributors, ODbL 1.0, via Overpass API", downloaded=d.get("osm3s", {}).get("timestamp_osm_base"))
    (out / "base").mkdir(parents=True, exist_ok=True); p = out / "base" / "base.json"
    json.dump(base, open(p, "w"), separators=(",", ":"))
    info = dict(roads=len(roads), buildings=len(bld), buildings_all=n_bld_all, water=len(water), coast=len(coast), rail=len(rail), rivers=len(rivers), mb=round(p.stat().st_size / 1e6, 2))
    print("base.json", info, flush=True); return info


def pack_evidence(src: Path, out: Path, per_sheet: int = 120, cols: int = 6) -> dict:
    """Sprite sheets: the evidence images of one drive and kind (lines / crossings), up to per_sheet per WebP sheet, in a grid of
    cols columns, plus evidence/index.json = {"lines/12_3.webp": [sheet file, x, y, w, h]}. Real image files, as the public-sharing
    review of the artifact requires (6 Nov 2026: JSON packs of base64 images could not be reviewed). Returns {sheet: bytes}."""
    from PIL import Image
    by = defaultdict(list)
    for sub in ("lines", "crossings"):
        for f in sorted((src / sub).glob("*.webp"), key=lambda f: (int(f.stem.split("_")[0]), int(f.stem.split("_")[1]))):
            by[(int(f.stem.split("_")[0]), sub)].append(f)
    (out / "evidence").mkdir(parents=True, exist_ok=True); index = {}; sizes = {}
    for (dr, sub), files in sorted(by.items()):
        for k in range(0, len(files), per_sheet):
            chunk = files[k:k + per_sheet]; ims = [Image.open(f).convert("RGB") for f in chunk]
            cw = max(im.width for im in ims); ch = max(im.height for im in ims); rows = (len(ims) + cols - 1) // cols
            sheet = Image.new("RGB", (cw * cols, ch * rows), (0, 0, 0)); name = f"sheet_{dr}_{sub}_{k // per_sheet}.webp"
            for n, (f, im) in enumerate(zip(chunk, ims)):
                x, y = (n % cols) * cw, (n // cols) * ch; sheet.paste(im, (x, y)); index[f"{sub}/{f.name}"] = [name, x, y, im.width, im.height]
            pth = out / "evidence" / name; sheet.save(pth, "WEBP", quality=88, method=4); sizes[name] = pth.stat().st_size
    json.dump(index, open(out / "evidence" / "index.json", "w"), separators=(",", ":"))
    print(f"evidence sheets: {len(sizes)} files, {len(index)} images, {sum(sizes.values())/1e6:.1f} MB, largest {max(sizes.values())/1e6:.1f} MB", flush=True)
    return sizes


def raster_sheets(ptiles: dict, out: Path, sheet_w: int = 4096) -> dict:
    """The evidence-raster tiles (base64 PNG strings inside the page's DATA.ptiles) as real PNG sprite sheets, shelf-packed:
    raster/sheet_<level>_<k>.png. Returns the new ptiles index {level: [{bounds, s, x, y, w, h}]}. Real image files, as the
    public-sharing review requires (6 Nov 2026)."""
    import io
    from PIL import Image
    (out / "raster").mkdir(parents=True, exist_ok=True); new = {}; n_files = 0; n_bytes = 0
    for z, tiles in ptiles.items():
        ims = [Image.open(io.BytesIO(base64.b64decode(t["png"].split(",", 1)[1]))).convert("RGBA") for t in tiles]
        order = sorted(range(len(ims)), key=lambda i: -ims[i].height); entries = [None] * len(ims)
        k = 0; x = y = row_h = 0; sheet = None; placed = []
        def flush():
            nonlocal sheet, k, x, y, row_h, placed, n_files, n_bytes
            if sheet is None: return
            img = Image.new("RGBA", (sheet_w, y + row_h), (0, 0, 0, 0))
            for (i, px, py) in placed: img.paste(ims[i], (px, py))
            pth = out / "raster" / f"sheet_{z}_{k}.png"; img.save(pth, "PNG", optimize=True); n_files += 1; n_bytes += pth.stat().st_size
            for (i, px, py) in placed: entries[i] = dict(bounds=tiles[i]["bounds"], s=pth.name, x=px, y=py, w=ims[i].width, h=ims[i].height)
            k += 1; x = y = row_h = 0; placed = []; sheet = None
        for i in order:
            im = ims[i]
            if x + im.width > sheet_w: x = 0; y += row_h; row_h = 0
            if y + im.height > sheet_w: flush()
            if sheet is None: sheet = True
            placed.append((i, x, y)); x += im.width; row_h = max(row_h, im.height)
        flush(); new[z] = entries
    print(f"raster sheets: {n_files} files, {n_bytes/1e6:.1f} MB, {sum(len(v) for v in new.values())} tiles", flush=True)
    return new


def once(html: str, old: str, new: str) -> str:
    n = html.count(old); assert n == 1, f"expected exactly one match, found {n}: {old[:90]!r}"
    return html.replace(old, new)


BASE_JS = r'''
// ---- Vector basemap (hosted variant): streets, buildings, water, coast, rail from OpenStreetMap, drawn on a canvas below the data.
map.createPane("base").style.zIndex = 300; const BASE_R = L.canvas({pane:"base"});
const baseGrp = L.layerGroup(); let BASE = null; let bldGrp = null; let cur = {setOpacity:v=>{ const c = BASE_R._container; if(c) c.style.opacity = v; }};
const tok = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function drawBase(){
  baseGrp.clearLayers(); if(bldGrp){ map.removeLayer(bldGrp); bldGrp = null; } if(!BASE) return;
  const ll = g => g.map(p=>[p[1], p[0]]); const o = {renderer:BASE_R, interactive:false};
  BASE.water.forEach(g=> L.polygon(ll(g), Object.assign({color:tok("--water"), weight:0.8, fillColor:tok("--water"), fillOpacity:0.5}, o)).addTo(baseGrp));
  BASE.rivers.forEach(g=> L.polyline(ll(g), Object.assign({color:tok("--water"), weight:1.2, opacity:0.8}, o)).addTo(baseGrp));
  BASE.coast.forEach(g=> L.polyline(ll(g), Object.assign({color:tok("--water"), weight:1.6, opacity:0.9}, o)).addTo(baseGrp));
  BASE.rail.forEach(g=> L.polyline(ll(g), Object.assign({color:tok("--rail"), weight:1.4, opacity:0.8, dashArray:"6 4"}, o)).addTo(baseGrp));
  const W = [0.6, 1.2, 2.2, 3.2];
  BASE.roads.forEach(([c,g])=> L.polyline(ll(g), Object.assign({color:c>=2?tok("--road2"):tok("--road"), weight:W[c], opacity:c?0.95:0.7}, o)).addTo(baseGrp));
  bldGrp = L.layerGroup(BASE.buildings.map(g=> L.polygon(ll(g), Object.assign({color:tok("--bld-edge"), weight:0.5, fillColor:tok("--bld"), fillOpacity:0.9}, o))));
  showBld();
}
function showBld(){ if(!bldGrp) return; const on = map.getZoom() >= 16 && map.hasLayer(baseGrp); if(on && !map.hasLayer(bldGrp)) bldGrp.addTo(map); if(!on && map.hasLayer(bldGrp)) map.removeLayer(bldGrp); }
map.on("zoomend", showBld); baseGrp.addTo(map);
fetch("base/base.json").then(r=>r.json()).then(d=>{ BASE = d; drawBase(); map.attributionControl.addAttribution("&copy; OpenStreetMap contributors (ODbL)"); }).catch(e=> console.warn("basemap not loaded", e));
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawBase);
new MutationObserver(drawBase).observe(document.documentElement, {attributes:true, attributeFilter:["data-theme"]});
'''

PACK_JS = r'''
// ---- Radar evidence images (hosted variant): WebP sprite sheets per drive and kind; index.json says where each image sits.
let EV_INDEX = null; const EV_SHEETS = new Map();
const evIndex = fetch("evidence/index.json").then(r=> r.ok ? r.json() : {}).catch(()=>({})).then(j=> EV_INDEX = j);
function sheetImage(path){ if(!EV_SHEETS.has(path)) EV_SHEETS.set(path, new Promise((res, rej)=>{ const im = new Image(); im.onload = ()=> res(im); im.onerror = rej; im.src = path; })); return EV_SHEETS.get(path); }
async function tileURL(t){ const im = await sheetImage("raster/" + t.s); const c = document.createElement("canvas"); c.width = t.w; c.height = t.h; c.getContext("2d").drawImage(im, t.x, t.y, t.w, t.h, 0, 0, t.w, t.h); return c.toDataURL("image/png"); }
async function evidenceURL(rel){ await evIndex; const e = EV_INDEX[rel.replace(/^evidence\//, "")]; if(!e) return null;
  const im = await sheetImage("evidence/" + e[0]); const c = document.createElement("canvas"); c.width = e[3]; c.height = e[4];
  c.getContext("2d").drawImage(im, e[1], e[2], e[3], e[4], 0, 0, e[3], e[4]); return c.toDataURL("image/png"); }
function hydrate(root){ root.querySelectorAll("img[data-src]").forEach(async el=>{ try{ const u = await evidenceURL(el.dataset.src); if(u) el.src = u; else el.closest("figure").style.display = "none"; }catch(err){ el.closest("figure").style.display = "none"; } }); }
'''


def hosted_html(local: Path, leaflet_css: Path, out: Path) -> Path:
    h = local.read_text()
    css = "\n".join(ln for ln in leaflet_css.read_text().splitlines() if "url(" not in ln)   # layers-control and marker icons, VML: unused here
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S); css = re.sub(r"\s+", " ", css).replace("; }", "}").replace(" { ", "{").replace("; ", ";").strip()
    assert "url(" not in css, "leaflet css still references a file"
    h = once(h, '<!doctype html>\n<html lang="en" data-theme="dark">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n', "")
    h = once(h, '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">\n', f"<style>{css}</style>\n")
    h = once(h, "</head>\n<body>\n", "")
    h = once(h, "</body>\n</html>\n", "")
    # theme tokens of the vector base (light / dark), and the map ground takes the token
    h = once(h, "--pipe:#1baf7a;--cross:#2a78d6;--high:#1baf7a;--mid:#8a93a0;--low:#eb6834;--shadow:0 6px 24px rgba(0,0,0,.14);--r:14px}",
             "--pipe:#1baf7a;--cross:#2a78d6;--high:#1baf7a;--mid:#8a93a0;--low:#eb6834;--shadow:0 6px 24px rgba(0,0,0,.14);--r:14px;\n"
             "  --mapbg:#eeece6;--road:#ffffff;--road2:#f6e9c8;--bld:#dcd8cf;--bld-edge:#c9c4b9;--water:#a9c8e6;--rail:#8d8d8d}\n"
             "@media (prefers-color-scheme:dark){:root:not([data-theme=\"light\"]){--mapbg:#15171a;--road:#3a3d43;--road2:#4a463a;--bld:#23262b;--bld-edge:#2e3238;--water:#243a52;--rail:#5a5e66}}\n"
             ":root[data-theme=\"dark\"]{--mapbg:#15171a;--road:#3a3d43;--road2:#4a463a;--bld:#23262b;--bld-edge:#2e3238;--water:#243a52;--rail:#5a5e66}")
    h = once(h, "#map{position:absolute;inset:0;background:#0d0f12}", "#map{position:absolute;inset:0;background:var(--mapbg)}")
    # basemap controls and layers
    h = once(h, '<select id="bm"><option value="sat">Satellite (Esri)</option><option value="osm">Streets (OpenStreetMap)</option><option value="none">None</option></select>',
             '<select id="bm"><option value="vec">Streets and buildings (OpenStreetMap)</option><option value="none">None</option></select>')
    old_base = re.search(r"const base = \{\n.*?\n\};\nlet cur = base\.sat\.addTo\(map\); cur\.setOpacity\(0\.8\);\n", h, re.S); assert old_base, "basemap block not found"
    h = h.replace(old_base.group(0), BASE_JS, 1)
    h = once(h, 'document.getElementById("bm").addEventListener("change",e=>{ map.removeLayer(cur); if(e.target.value!=="none"){ cur = base[e.target.value].addTo(map); cur.setOpacity(+document.getElementById("bop").value);} });',
             'document.getElementById("bm").addEventListener("change",e=>{ if(e.target.value==="none") map.removeLayer(baseGrp); else baseGrp.addTo(map); showBld(); });')
    # evidence images from packs
    h = once(h, 'const radar = (src, cap) => src ? `<figure class="radar"><img src="${src}" alt="radar evidence" loading="lazy" onerror="this.parentNode.style.display=\'none\'">',
             'const radar = (src, cap) => src ? `<figure class="radar"><img data-src="${src}" alt="radar evidence">')
    h = once(h, 'function showDetail(html){ document.getElementById("dbody").innerHTML = html; document.getElementById("detail").classList.remove("hide"); }',
             'function showDetail(html){ const b = document.getElementById("dbody"); b.innerHTML = html; hydrate(b); document.getElementById("detail").classList.remove("hide"); }' + PACK_JS)
    # evidence-raster tiles: base64 PNGs in DATA.ptiles -> PNG sprite sheets in raster/, index in DATA.ptiles
    m = re.search(r"const DATA = (\{.*?\});\n", h, re.S); assert m, "DATA block not found"
    D = json.loads(m.group(1)); D["ptiles"] = raster_sheets(D["ptiles"], out)
    h = h.replace(m.group(0), "const DATA = " + json.dumps(D, separators=(",", ":")) + ";\n", 1)
    h = once(h, 'for(const [k,l] of active){ if(!want.has(k)){ grp.removeLayer(l); active.delete(k);} }',
             'for(const [k,l] of active){ if(!want.has(k)){ if(l) grp.removeLayer(l); active.delete(k);} }')
    h = once(h, 'want.forEach(k=>{ if(!active.has(k)){ const [z,i] = k.split(":").map(Number); const t = tiles[z][i];\n'
             '      const l = L.imageOverlay(t.png, t.bounds, {pane:"raster", opacity:opacity, interactive:false}); grp.addLayer(l); active.set(k,l);} });',
             'want.forEach(k=>{ if(!active.has(k)){ const [z,i] = k.split(":").map(Number); const t = tiles[z][i]; active.set(k, null);\n'
             '      tileURL(t).then(u=>{ if(!active.has(k) || active.get(k)) return; const l = L.imageOverlay(u, t.bounds, {pane:"raster", opacity:+(document.getElementById("rop")||{value:opacity}).value, interactive:false}); grp.addLayer(l); active.set(k,l); }).catch(()=>{}); } });')
    assert "data:image" not in h, "page still embeds images"
    # export: downloads started by the page are inert in the frame
    h = once(h, '<div class="sec"><h2>Export</h2><button id="dl">Download lines and crossings (GeoJSON, WGS84)</button></div>', "")
    dl = re.search(r'// ---- Export: .*?\ndocument\.getElementById\("dl"\)\.onclick = \(\)=>\{\n.*?\n\};\n', h, re.S); assert dl, "export block not found"
    h = h.replace(dl.group(0), "", 1); assert '"dl"' not in h
    assert "arcgisonline" not in h and "tile.openstreetmap.org" not in h and 'src="evidence/' not in h
    p = out / "index.html"; p.write_text(h); print(f"index.html {p.stat().st_size/1e6:.2f} MB", flush=True); return p


def main(osm_path: str, out_dir: str, local_html: str | None = None) -> None:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    osm_to_base(Path(osm_path), out)
    pack_evidence(EVIDENCE, out)
    hosted_html(Path(local_html) if local_html else FIGURES / "web" / "savona_pipes_map.html", out / "leaflet.css", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
