"""Single source of every location (9 Nov 2026). Everything large lives outside the repository:

    $ITALGAS_DATA/                 default ~/ItalGas  (override with the environment variable ITALGAS_DATA)
        Savona/<session folder>/Survey_*/Swath*.{data,index,path,nmea}   raw IDS Stream UP data, unzipped from S3
        cache/                                                           every derived product (rasters, lines, scans, page data)
        cache/city/                                                      per-drive results of the survey-wide steps
        cache/web_evidence/                                              radar images shown in the web page popups

Inside the repository: configs/ (YAML, referenced by file name through config_path), figures/ (result PNGs and the web
page), data_public/ (municipal open data). Configs refer to the data folders as ${ITALGAS_DATA}/Savona and
${ITALGAS_DATA}/cache; load_yaml() expands the variable (and ~) in every string value."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import yaml

DATA = Path(os.environ.get("ITALGAS_DATA", str(Path.home() / "ItalGas"))).expanduser()
os.environ.setdefault("ITALGAS_DATA", str(DATA))          # so ${ITALGAS_DATA} expands inside configs
RAW = DATA / "Savona"                                     # raw survey data (read-only, never modified)
CACHE = DATA / "cache"                                    # derived products
CITY = CACHE / "city"
EVIDENCE = CACHE / "web_evidence"

REPO = Path(__file__).resolve().parents[3]                # src/italgas_gpr/io/paths.py -> repository root (editable install)
CONFIGS = REPO / "configs"                                # YAML configs of the working chain
FIGURES = REPO / "figures"                                # result figures (PNG) and the web page
DATA_PUBLIC = REPO / "data_public"                        # Comune di Savona open data (CC BY 4.0)

# config names used by the logs (EXPERIMENTS.md, DECISIONS.md, archive/) before the 9 Nov 2026 restructure -> current names
CONFIG_RENAMES = {"dev_swath.yaml": "detector_base.yaml", "dev_swath_v2_2.yaml": "detector.yaml", "prob_raster.yaml": "along_raster.yaml",
                  "dither.yaml": "along_raster.yaml", "raster_lines.yaml": "along_lines.yaml", "citywide.yaml": "survey.yaml",
                  "citywide_scan.yaml": "crossing_scan.yaml", "citywide_along.yaml": "along_tracks.yaml",
                  "subset_merge.yaml": "crossing_merge_subset.yaml", "multipass_aurelia.yaml": "lane_registration.yaml"}


def expand(obj):
    """Recursively expand environment variables and ~ in every string of a parsed YAML structure."""
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj)) if ("$" in obj or obj.startswith("~")) else obj
    if isinstance(obj, dict):
        return {k: expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand(v) for v in obj]
    return obj


def load_yaml(path) -> dict:
    """yaml.safe_load with path expansion (the one way configs are read in this project)."""
    with open(path) as fh:
        return expand(yaml.safe_load(fh))


def config_path(name) -> Path:
    """Resolve a config reference: an existing or absolute path is returned as is; otherwise the file name (with or without a
    leading 'configs/'; old names of the logs accepted) is looked up in CONFIGS."""
    p = Path(os.path.expanduser(os.path.expandvars(str(name))))
    if p.is_absolute() or p.exists():
        return p
    base = p.name if p.parent.name in ("configs", "") else str(p)
    return CONFIGS / CONFIG_RENAMES.get(base, base)


def swath_files(raw_root=RAW) -> list[str]:
    """Every swath of the survey, sorted; the position in this list is the drive index used in all cached file names."""
    return sorted(glob.glob(str(Path(raw_root) / "*" / "Survey_*" / "Swath*.path")))
