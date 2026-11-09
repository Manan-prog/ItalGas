"""Single source of the data locations (9 Nov 2026). Everything large lives outside the repository:

    $ITALGAS_DATA/                 default ~/ItalGas  (override with the environment variable ITALGAS_DATA)
        Savona/<session folder>/Survey_*/Swath*.{data,index,path,nmea}   raw IDS Stream UP data, unzipped from S3
        cache/                                                           every derived product (rasters, lines, scans, page data)
        cache/city/                                                      per-drive results of the survey-wide steps
        cache/web_evidence/                                              radar images shown in the web page popups

Configs refer to these folders as ${ITALGAS_DATA}/Savona and ${ITALGAS_DATA}/cache; load_yaml() expands the variable
(and ~) in every string value, so a config never carries a user-specific path."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

DATA = Path(os.environ.get("ITALGAS_DATA", str(Path.home() / "ItalGas"))).expanduser()
os.environ.setdefault("ITALGAS_DATA", str(DATA))          # so ${ITALGAS_DATA} expands inside configs
RAW = DATA / "Savona"                                     # raw survey data (read-only, never modified)
CACHE = DATA / "cache"                                    # derived products
CITY = CACHE / "city"
EVIDENCE = CACHE / "web_evidence"


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


def swath_files(raw_root=RAW) -> list[str]:
    """Every swath of the survey, sorted; the position in this list is the drive index used in all cached products."""
    import glob
    return sorted(glob.glob(str(Path(raw_root) / "*" / "Survey_*" / "Swath*.path")))
