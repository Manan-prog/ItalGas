"""GPS track (EPSG:3857 x, y of the real-time .path) of every drive, indexed like the city scripts (sorted glob of
raw_root/*/Survey_*/Swath*.path), with its survey day. Replaces the scratchpad copy made by figures/geom.py.
Run: python build_swath_tracks.py configs/citywide.yaml  -> the file named by swath_tracks in that config"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
from paths import load_yaml

from ids_reader import read_path

cc = load_yaml(sys.argv[1]); files = sorted(glob.glob(str(Path(cc["raw_root"]) / "*" / "Survey_*" / "Swath*.path")))
P = [read_path(f) for f in files]; days = [re.search(r"_(2025\.10\.\d\d)_", Path(f).parents[1].name).group(1) for f in files]
np.savez(cc["swath_tracks"], **{f"s{i}": np.c_[p["x"], p["y"]] for i, p in enumerate(P)}, days=np.array(days))
print(len(files), "drives ->", cc["swath_tracks"])
