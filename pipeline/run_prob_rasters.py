"""E47: P rasters (prob_raster.py) for every drive of every lane component, one process per 600 m piece.
Run: python run_prob_rasters.py [n_workers] -> cache/prob_raster_<drive>_<s0>.npz (existing pieces are skipped)."""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from paths import CACHE  # noqa: E402

HERE = Path(__file__).resolve().parent


def drive_length_m(i: int) -> float:
    """Length of drive i from its real-time track (cache/swaths.npz, EPSG:3857 x cos(lat))."""
    t = np.load(CACHE / "swaths.npz")[f"s{i}"]; k = float(np.cos(np.radians(44.3)))
    return float(np.sum(np.hypot(np.diff(t[:, 0]), np.diff(t[:, 1])) * k))


def run_piece(job: tuple[int, float]) -> str:
    i, c0 = job; out = CACHE / f"prob_raster_{i}_{int(c0)}.npz"
    if out.exists():
        return f"{i} {int(c0)} exists"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log = open(CACHE / f"prob_raster_{i}_{int(c0)}.log", "w"); t = time.time()
    r = subprocess.run([sys.executable, str(HERE / "prob_raster.py"), str(HERE / "configs" / "prob_raster.yaml"), str(i), str(c0)], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
    return f"{i} {int(c0)} rc {r.returncode} {time.time() - t:.0f}s"


def main(n_workers: int = 7, which: str = "lanes") -> None:
    """which: 'lanes' = drives of the repeat-drive components only; 'all' = every drive of 100 m or more."""
    if which == "all":
        sw = np.load(CACHE / "swaths.npz"); drives = sorted(int(k[1:]) for k in sw.files if k.startswith("s") and len(sw[k]) > 1 and drive_length_m(int(k[1:])) >= 100)
    else:
        comps = json.load(open(CACHE / "city" / "lane_components.json"))["components"]
        drives = sorted({d for c in comps for d in c["drives"]})
    jobs = []
    for i in drives:
        L = drive_length_m(i)
        jobs += [(i, float(c0)) for c0 in np.arange(0.0, L, 600.0) if min(L, c0 + 600) - c0 > 60]
    todo = [j for j in jobs if not (CACHE / f"prob_raster_{j[0]}_{int(j[1])}.npz").exists()]
    print(f"{len(drives)} drives, {len(jobs)} pieces, {len(todo)} to compute, {n_workers} workers", flush=True); t0 = time.time()
    with ProcessPoolExecutor(n_workers) as ex:
        for n, msg in enumerate(ex.map(run_piece, todo)):
            print(f"[{n + 1}/{len(todo)}] {msg} (elapsed {time.time() - t0:.0f}s)", flush=True)
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    # default workers: each piece holds about 2 GB (19 channels x 600 m x 512 samples, 9 migrations); 7 workers thrashed a 16 GB machine
    main(int(sys.argv[1]) if len(sys.argv) > 1 else max(1, min(5, (os.cpu_count() or 2) - 1)), sys.argv[2] if len(sys.argv) > 2 else "lanes")
