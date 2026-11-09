"""Evidence thumbnails for the web page (6 Nov 2026): one WebP image per along-street line (cache/city/rline_<i>.json) and
per crossing (cache/city/scan_<i>.json), rendered from the raw traces with the v2.2 preprocessing.
Line image: A = across-street section at the line's midpoint (coherent 2 m stack of the 19 channels of the 200 MHz row,
Kirchhoff-migrated across the channels, envelope; u on x, depth on y; red cross and ring = the line's position and depth);
B = signed along-street B-scan of the channel nearest the line's lateral position, dashed = the line's depth,
bar = its extent. Crossing image: signed B-scan of the centre channel of each row (200 MHz, 600 MHz) +-4 m around the
crossing; red triangle, ring and dashed line = the detection. Depth at v = 0.10 m/ns (DERIVED, E03).
Output: $ITALGAS_DATA/cache/web_evidence/{lines,crossings}/<drive>_<n>.webp (outside Drive; link figures/web/evidence to it).
Run: python -m italgas_gpr.web.evidence_images [n_workers] [lines|crossings|all]   Test: python -m italgas_gpr.web.evidence_images test <drive>"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from italgas_gpr.geometry.swath import swath_geometry  # noqa: E402
from italgas_gpr.processing.preprocess import load_cfg, preprocess_array  # noqa: E402
from italgas_gpr.crossing.score import to_s_grid  # noqa: E402
from italgas_gpr.io.ids_reader import Swath  # noqa: E402
from italgas_gpr.along.focus import lateral_migrate  # noqa: E402
from italgas_gpr.lanes.multipass import longest_run  # noqa: E402

from italgas_gpr.io.paths import CACHE, CITY, EVIDENCE as OUT, config_path  # noqa: E402
DET = load_cfg(config_path("detector.yaml")); V = DET["migration"]["velocity_m_ns"]; DS = DET["grid"]["ds_m"]
Z_MAX = 1.6; CHUNK = 600.0; PAD = 60.0; HALF_X = 4.0; LINE_PAD = 6.0; LINE_MAX = 80.0
U_GRID = np.arange(-0.9, 0.9 + 1e-9, 0.04)
ORANGE = (235, 104, 52); BLUE = (42, 120, 214); RED = (222, 30, 30); WHITE = (252, 252, 251); INK = (31, 31, 31); MUTED = (107, 107, 107)   # markers: red with a white halo (user, 6 Nov)
SCALE = 1                                                     # 1 px per CSS px at the page's 330 px panel; WebP q 75 (size budget: < 60 MB for all)
W_A, W_B, H = 150 * SCALE, 300 * SCALE, 150 * SCALE
FONT = ImageFont.load_default(size=11 * SCALE)


def grey(a: np.ndarray, signed: bool, pct: float = 99.0) -> np.ndarray:
    """(n_x, n_z) -> uint8 (n_z, n_x) image: signed data symmetric grey, envelope white = 0 .. black = p99.5."""
    a = a.T
    if signed:
        lim = np.nanpercentile(np.abs(a), pct) + 1e-9; g = 0.5 + 0.5 * np.clip(a / lim, -1, 1)
    else:
        lim = np.nanpercentile(a, 99.5) + 1e-9; g = 1.0 - np.clip(a / lim, 0, 1)
    return (np.nan_to_num(g, nan=1.0) * 255).astype(np.uint8)


def panel(img8: np.ndarray, w: int, h: int) -> Image.Image:
    return Image.fromarray(img8, "L").resize((w, h), Image.BILINEAR).convert("RGB")


def frame(panels: list[tuple[Image.Image, str]], gap: int = 6 * SCALE) -> Image.Image:
    wt = sum(p.width for p, _ in panels) + gap * (len(panels) - 1); out = Image.new("RGB", (wt, H + 14 * SCALE), WHITE); x = 0; d = ImageDraw.Draw(out)
    for p, cap in panels:
        out.paste(p, (x, 14 * SCALE)); d.text((x + 2 * SCALE, 1 * SCALE), cap, fill=MUTED, font=FONT); x += p.width + gap
    return out


def depth_rows(dt: float, n_t: int) -> int:
    return int(min(n_t, round(2 * Z_MAX / V / dt)))


def line_image(Z: np.ndarray, s_grid: np.ndarray, X: np.ndarray, S: np.ndarray, u_ch: np.ndarray, dt: float, ln: dict) -> Image.Image:
    """Z (n_ch, n_s, n_t) gridded signed traces; X (n_scans, n_ch, n_t) raw-order traces with scan positions S (n_scans, n_ch)."""
    nz = depth_rows(dt, Z.shape[2]); s_mid = 0.5 * (ln["s0"] + ln["s1"]); c = int(np.argmin(np.abs(u_ch - ln["u_mean"])))
    # A: across-street section at s_mid: coherent mean over +-1 m per channel, migrated across the channels
    st = np.stack([np.nanmean(X[np.abs(S[:, k] - s_mid) <= 1.0, k, :], 0) for k in range(X.shape[1])])[None]     # (1, n_ch, n_t)
    sec = lateral_migrate(np.nan_to_num(st), u_ch, U_GRID, dt, V)[0][:, :nz]                                       # (n_u, nz) envelope
    A = panel(grey(sec, False), W_A, H); d = ImageDraw.Draw(A)
    ux = (ln["u_mean"] - U_GRID[0]) / (U_GRID[-1] - U_GRID[0]) * W_A; zy = ln["depth_m"] / Z_MAX * H; r = 9 * SCALE
    for col, w in ((WHITE, 6 * SCALE), (RED, 3 * SCALE)):                                    # white halo, then red
        d.line([(ux - r, zy), (ux + r, zy)], fill=col, width=w); d.line([(ux, zy - r), (ux, zy + r)], fill=col, width=w)
    d.ellipse([ux - 14 * SCALE, zy - 14 * SCALE, ux + 14 * SCALE, zy + 14 * SCALE], outline=RED, width=2 * SCALE)
    A = A.transpose(Image.FLIP_LEFT_RIGHT)                       # u positive = left of travel, shown on the left
    # B: along-street signed B-scan of the nearest channel over the line's extent (+- LINE_PAD, capped)
    half = min(LINE_MAX, ln["s1"] - ln["s0"] + 2 * LINE_PAD) / 2; w = (s_grid >= s_mid - half) & (s_grid <= s_mid + half)
    B = panel(grey(Z[c][w][:, :nz], True), W_B, H); d = ImageDraw.Draw(B); s_w = s_grid[w]
    sx = lambda s: (s - s_w[0]) / max(s_w[-1] - s_w[0], 1e-6) * W_B
    for x0 in np.arange(0, W_B, 10 * SCALE):
        d.line([(x0, zy), (x0 + 6 * SCALE, zy)], fill=WHITE, width=4 * SCALE); d.line([(x0, zy), (x0 + 6 * SCALE, zy)], fill=RED, width=2 * SCALE)
    d.line([(sx(ln["s0"]), H - 4 * SCALE), (sx(ln["s1"]), H - 4 * SCALE)], fill=WHITE, width=7 * SCALE); d.line([(sx(ln["s0"]), H - 4 * SCALE), (sx(ln["s1"]), H - 4 * SCALE)], fill=RED, width=4 * SCALE)
    return frame([(A, "A  across, 1.8 m"), (B, f"B  along, {2 * half:.0f} m")])


def crossing_image(parts: list[tuple[np.ndarray, np.ndarray, float, str]], c: dict) -> Image.Image:
    """parts: [(Zc (n_s, n_t) signed centre-channel traces on s_grid, s_grid, dt, label)] for the rows available."""
    pans = []
    for Zc, s_grid, dt, lab in parts:
        nz = depth_rows(dt, Zc.shape[1]); w = (s_grid >= c["s0_m"] - HALF_X) & (s_grid <= c["s0_m"] + HALF_X)
        if w.sum() < 10:
            continue
        P = panel(grey(Zc[w][:, :nz], True), W_A + 60 * SCALE, H); d = ImageDraw.Draw(P); s_w = s_grid[w]
        x = (c["s0_m"] - s_w[0]) / max(s_w[-1] - s_w[0], 1e-6) * P.width; zy = c["depth_m"] / Z_MAX * H
        d.polygon([(x - 9 * SCALE, 0), (x + 9 * SCALE, 0), (x, 12 * SCALE)], fill=WHITE); d.polygon([(x - 7 * SCALE, 0), (x + 7 * SCALE, 0), (x, 9 * SCALE)], fill=RED)
        for x0 in np.arange(0, P.width, 10 * SCALE):
            d.line([(x0, zy), (x0 + 6 * SCALE, zy)], fill=WHITE, width=4 * SCALE); d.line([(x0, zy), (x0 + 6 * SCALE, zy)], fill=RED, width=2 * SCALE)
        d.ellipse([x - 14 * SCALE, zy - 14 * SCALE, x + 14 * SCALE, zy + 14 * SCALE], outline=WHITE, width=4 * SCALE); d.ellipse([x - 14 * SCALE, zy - 14 * SCALE, x + 14 * SCALE, zy + 14 * SCALE], outline=RED, width=2 * SCALE)
        pans.append((P, f"{lab}, {2 * HALF_X:.0f} m"))
    return frame(pans) if pans else None


def chunks(L: float):
    for c0 in np.arange(0.0, L, CHUNK):
        yield c0, min(L, c0 + CHUNK)


def drive_lines(i: int) -> str:
    os.environ.setdefault("OMP_NUM_THREADS", "1"); t0 = time.time(); od = OUT / "lines"; od.mkdir(parents=True, exist_ok=True)
    m = json.load(open(CITY / f"rline_{i}.json")); lines = m["lines"]
    if not lines or all((od / f"{i}_{n}.webp").exists() for n in range(len(lines))):
        return f"{i}: {len(lines)} lines, nothing to do"
    f = Path(m["file"]); sw = Swath.load(str(f.parent), f.stem); g = swath_geometry(sw, DET["base"]); L = float(g["s_nodes"][-1])
    sc = np.nanmedian(g["s2"], 1); u_ch = np.nanmedian(g["u2"], 0); n_done = 0
    for c0, c1 in chunks(L):
        todo = [(n, ln) for n, ln in enumerate(lines) if c0 <= 0.5 * (ln["s0"] + ln["s1"]) < c1 and not (od / f"{i}_{n}.webp").exists()]
        if not todo:
            continue
        sl = longest_run((sc > c0 - PAD) & (sc < c1 + PAD))
        if sl is None or sl.stop - sl.start < 200:
            continue
        pp = preprocess_array(sw, 2, DET, scans=sl); X = pp["X"]; S = g["s2"][sl]; dt = pp["dt_ns"]
        s_grid = np.arange(max(0.0, c0 - PAD), min(L, c1 + PAD), DS)
        Z = np.stack([to_s_grid(X[:, k], S[:, k], s_grid) for k in range(X.shape[1])])
        for n, ln in todo:
            img = line_image(Z, s_grid, X, S, u_ch, dt, ln); img.save(od / f"{i}_{n}.webp", format="WEBP", quality=75, method=4); n_done += 1
        del X, Z, pp
    return f"{i}: {n_done} line images in {time.time() - t0:.0f} s"


def drive_crossings(i: int) -> str:
    os.environ.setdefault("OMP_NUM_THREADS", "1"); t0 = time.time(); od = OUT / "crossings"; od.mkdir(parents=True, exist_ok=True)
    r = json.load(open(CITY / f"scan_{i}.json")); cr = r["crossings"]
    if not cr or all((od / f"{i}_{j}.webp").exists() for j in range(len(cr))):
        return f"{i}: {len(cr)} crossings, nothing to do"
    f = Path(r["file"]); sw = Swath.load(str(f.parent), f.stem); g = swath_geometry(sw, DET["base"]); L = float(g["s_nodes"][-1])
    sc = {a: np.nanmedian(g[f"s{a}"], 1) for a in (1, 2) if f"s{a}" in g}; n_done = 0
    for c0, c1 in chunks(L):
        todo = [(j, c) for j, c in enumerate(cr) if c0 <= c["s0_m"] < c1 and not (od / f"{i}_{j}.webp").exists()]
        if not todo:
            continue
        parts = []; s_grid = np.arange(max(0.0, c0 - PAD), min(L, c1 + PAD), DS)
        for a, lab in ((2, "200 MHz"), (1, "600 MHz")):
            if a not in sc:
                continue
            sl = longest_run((sc[a] > c0 - PAD) & (sc[a] < c1 + PAD))
            if sl is None or sl.stop - sl.start < 200:
                continue
            pp = preprocess_array(sw, a, DET, scans=sl); u_ch = np.nanmedian(g[f"u{a}"][sl], 0); c = int(np.argmin(np.abs(u_ch)))
            parts.append((to_s_grid(pp["X"][:, c], g[f"s{a}"][sl][:, c], s_grid), s_grid, pp["dt_ns"], lab)); del pp
        for j, c in todo:
            img = crossing_image(parts, c)
            if img is not None:
                img.save(od / f"{i}_{j}.webp", format="WEBP", quality=75, method=4); n_done += 1
        del parts
    return f"{i}: {n_done} crossing images in {time.time() - t0:.0f} s"


def main(n_workers: int, what: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    jobs = []
    if what in ("lines", "all"):
        jobs += [(drive_lines, int(Path(f).stem.split("_")[1])) for f in sorted(CITY.glob("rline_*.json"))]
    if what in ("crossings", "all"):
        jobs += [(drive_crossings, int(Path(f).stem.split("_")[1])) for f in sorted(CITY.glob("scan_*.json"))]
    with ProcessPoolExecutor(n_workers) as ex:
        futs = [ex.submit(fn, i) for fn, i in jobs]
        for fu in futs:
            try:
                print(fu.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print("FAILED:", repr(e), flush=True)
    n = len(list((OUT / "lines").glob("*.webp"))) if (OUT / "lines").exists() else 0; m = len(list((OUT / "crossings").glob("*.webp"))) if (OUT / "crossings").exists() else 0
    mb = sum(p.stat().st_size for p in OUT.rglob("*.webp")) / 1e6
    print(f"done: {n} line images, {m} crossing images, {mb:.1f} MB, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    if sys.argv[1:2] == ["test"]:
        i = int(sys.argv[2]); print(drive_lines(i) if (CITY / f"rline_{i}.json").exists() else "no rline"); print(drive_crossings(i) if (CITY / f"scan_{i}.json").exists() else "no scan")
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 4, sys.argv[2] if len(sys.argv) > 2 else "all")
