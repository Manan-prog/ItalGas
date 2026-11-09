"""GPS track (EPSG:3857 x, y of the real-time .path) of every drive, indexed like every cached product (sorted list of
raw_root/*/Survey_*/Swath*.path, see paths.swath_files), with its survey day.
Run: python -m italgas_gpr.geometry.tracks configs/survey.yaml  -> the file named by swath_tracks in that config"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

from italgas_gpr.io.ids_reader import read_path
from italgas_gpr.io.paths import config_path, load_yaml, swath_files


def main(cfg_path: str) -> None:
    cc = load_yaml(config_path(cfg_path)); files = swath_files(cc["raw_root"])
    P = [read_path(f) for f in files]; days = [re.search(r"_(2025\.10\.\d\d)_", Path(f).parents[1].name).group(1) for f in files]
    np.savez(cc["swath_tracks"], **{f"s{i}": np.c_[p["x"], p["y"]] for i, p in enumerate(P)}, days=np.array(days))
    print(len(files), "drives ->", cc["swath_tracks"])


if __name__ == "__main__":
    main(sys.argv[1])
