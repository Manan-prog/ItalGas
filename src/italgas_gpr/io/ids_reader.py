"""Reader for IDS GeoRadar Stream UP survey folders (reverse-engineered formats).

One source of truth for the binary layouts. All functions are pure and return
NumPy arrays. Canonical array order is (n_scans, n_channels, n_samples).

Layouts (verified on the Savona 2025 data, see DECISIONS.md):
  .data   36-byte header, then int16 little-endian scans of [n_channels x n_samples]
  .index  (60 + 4*n_channels)-byte header, then 20-byte records
          (int32 scan, int32 t_us, int32 flag, float64 odometer_m)
  .path   112-byte header (contains 'EPSG:3857'), then 36-byte records
          (float64 x, float64 y, float64 z, int32 seq, int32 t_ms time-of-day, int32 flag)
  .nmea / _Ext.nmea   $GPGGA at 10 Hz, real-time RTK / post-processed
  TrjData/*_QF.csv    tab-separated 1 Hz post-processed quality (StdDev m, AmbStatus, Q)
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

INDEX_DTYPE = np.dtype([("scan", "<i4"), ("t_us", "<i4"), ("flag", "<i4"), ("odo_m", "<f8")])
INDEX_DTYPE_UNWRAPPED = np.dtype([("scan", "<i4"), ("t_us", "<i8"), ("flag", "<i4"), ("odo_m", "<f8")])
PATH_DTYPE = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("seq", "<i4"),
                       ("t_ms", "<i4"), ("flag", "<i4")])
DATA_HEADER_BYTES = 36
PATH_HEADER_BYTES = 112
INDEX_HEADER_BASE_BYTES = 60


@dataclass(frozen=True)
class Channel:
    """One transmit-receive pair. Positions are in the vehicle frame (m):
    x forward, y positive to the LEFT of travel, z up."""
    id: int                    # 1-based IDS channel id
    array: int                 # 1 = 600 MHz (Array01), 2 = 200 MHz (Array02)
    slot: int                  # 0-based position inside its .data file
    freq_mhz: float            # nominal centre frequency (vendor label)
    range_ns: float            # time window (measured: from Survey XML)
    n_samples: int
    tx_xy: tuple[float, float]  # TX element position, vehicle frame (m)
    rx_xy: tuple[float, float]  # RX element position, vehicle frame (m)

    @property
    def mid_xy(self) -> tuple[float, float]:
        """Midpoint between TX and RX: the nominal footprint centre (m)."""
        return ((self.tx_xy[0] + self.rx_xy[0]) / 2, (self.tx_xy[1] + self.rx_xy[1]) / 2)

    @property
    def dt_ns(self) -> float:
        """Sampling interval (ns) = range / (n_samples - 1)."""
        return self.range_ns / (self.n_samples - 1)


@dataclass(frozen=True)
class SurveyConfig:
    epsg: str
    gps_xyz: tuple[float, float, float]      # GPS antenna in vehicle frame (m)
    channels: tuple[Channel, ...]
    encoder_step_m: float                     # XStep * CalibrationFactor (nominal)
    swaths: tuple[str, ...]

    def channels_of(self, array: int) -> tuple[Channel, ...]:
        return tuple(c for c in self.channels if c.array == array)


def _num(v: str) -> float:
    """Survey XML numbers: some sessions write decimals with a comma ('1,94'); fields are ';'-separated."""
    return float(v.replace(",", "."))


def parse_survey(survey_dir: str | Path) -> SurveyConfig:
    """Parse the 'Survey' XML file of a Survey_* folder.

    Element positions are built from Antenna xyz (vehicle frame) plus the TX/RX
    'xy' offsets of the AntennaType, both read from the file (measured by vendor)."""
    root = ET.parse(Path(survey_dir) / "Survey").getroot()
    epsg = root.find("Workspace/Projection").get("Name")
    radar = root.find("RadarConfiguration/Radar")
    gps = tuple(_num(v) for v in radar.find("GPSs/GPS").get("xyz").split(";"))
    enc = radar.find("Encoders/Encoder")
    step = _num(enc.get("XStep")) * _num(enc.get("CalibrationFactor"))
    types: dict[str, dict] = {}
    for at in root.find("RadarConfiguration/AntennaTypes"):
        tx = {int(e.get("Id")): tuple(_num(v) for v in e.get("xy").split(";")) for e in at.find("TXGroup")}
        rx = {int(e.get("Id")): tuple(_num(v) for v in e.get("xy").split(";")) for e in at.find("RXGroup")}
        types[at.get("Type")] = {"tx": tx, "rx": rx, "freq": _num(at.find("TXGroup/TX").get("Frequency"))}
    ants = {}
    for a in radar.find("Antennas"):
        xyz = tuple(_num(v) for v in a.get("xyz").split(";"))
        ants[int(a.get("Id"))] = (a.get("Type"), xyz)
    chans = []
    slot_counter = {1: 0, 2: 0}
    for c in radar.find("Channels"):
        cid = int(c.get("Id"))
        ta, te = (int(v) for v in c.get("TX").split(";"))
        ra, re_ = (int(v) for v in c.get("RX").split(";"))
        ttype, txyz = ants[ta]
        rtype, rxyz = ants[ra]
        tx_off = types[ttype]["tx"][te]
        rx_off = types[rtype]["rx"][re_]
        array = 1 if ttype == "AM600" else 2
        chans.append(Channel(
            id=cid, array=array, slot=slot_counter[array], freq_mhz=types[ttype]["freq"],
            range_ns=_num(c.get("Range_ns")), n_samples=int(c.get("Samples")),
            tx_xy=(txyz[0] + tx_off[0], txyz[1] + tx_off[1]),
            rx_xy=(rxyz[0] + rx_off[0], rxyz[1] + rx_off[1])))
        slot_counter[array] += 1
    swaths = tuple(s.text for s in root.find("Swaths"))
    return SurveyConfig(epsg=epsg, gps_xyz=gps, channels=tuple(chans), encoder_step_m=step, swaths=swaths)


def read_index(path: str | Path, n_channels: int) -> np.ndarray:
    """Return structured array (scan, t_us, flag, odo_m), one record per scan.
    t_us is stored as a signed 32-bit microsecond counter that wraps every 2^32 us (71.6 min; MEASURED on
    13 Oct Survey 10-01-51 Swath050: +2147.48 s -> -2147.48 s mid-swath); it is returned unwrapped as int64."""
    size = os.path.getsize(path)
    header = INDEX_HEADER_BASE_BYTES + 4 * n_channels
    n = (size - header) // INDEX_DTYPE.itemsize
    raw = np.fromfile(path, dtype=INDEX_DTYPE, count=n, offset=header)
    out = np.empty(n, dtype=INDEX_DTYPE_UNWRAPPED)
    for f in INDEX_DTYPE.names:
        out[f] = raw[f]
    t = raw["t_us"].astype(np.int64)
    out["t_us"] = t + (np.cumsum(np.r_[0, np.diff(t) < -(1 << 31)]) << 32)
    return out


def open_data(path: str | Path, n_channels: int, n_samples: int = 512) -> np.memmap:
    """Memory-map a .data file as int16 (n_scans, n_channels, n_samples). Read-only."""
    size = os.path.getsize(path)
    n_scans = (size - DATA_HEADER_BYTES) // (n_channels * n_samples * 2)
    return np.memmap(path, dtype="<i2", mode="r", offset=DATA_HEADER_BYTES,
                     shape=(n_scans, n_channels, n_samples))


def read_path(path: str | Path) -> np.ndarray:
    """Return structured array (x, y, z, seq, t_ms, flag); x,y in EPSG:3857 metres
    (unscaled Web Mercator), t_ms = UTC time of day in ms, 10 Hz."""
    size = os.path.getsize(path)
    n = (size - PATH_HEADER_BYTES) // PATH_DTYPE.itemsize
    return np.fromfile(path, dtype=PATH_DTYPE, count=n, offset=PATH_HEADER_BYTES)


_GGA = re.compile(r"^\$G[PN]GGA,(\d{2})(\d{2})(\d{2}(?:\.\d+)?),(\d{2})(\d+\.\d+),([NS]),(\d{3})(\d+\.\d+),([EW]),(\d),(\d+),([\d.]*),([-\d.]*),M")


def read_gga(path: str | Path) -> np.ndarray:
    """Parse $GPGGA lines. Returns structured array with
    t_s (UTC seconds of day), lat, lon (deg), q (fix quality), nsat, hdop, h (m)."""
    rows = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = _GGA.match(line)
            if not m:
                continue
            hh, mm, ss, latd, latm, ns, lond, lonm, ew, q, nsat, hdop, h = m.groups()
            lat = int(latd) + float(latm) / 60
            lon = int(lond) + float(lonm) / 60
            rows.append((int(hh) * 3600 + int(mm) * 60 + float(ss),
                         -lat if ns == "S" else lat, -lon if ew == "W" else lon,
                         int(q), int(nsat), float(hdop or "nan"), float(h or "nan")))
    return np.array(rows, dtype=[("t_s", "<f8"), ("lat", "<f8"), ("lon", "<f8"), ("q", "<i4"),
                                 ("nsat", "<i4"), ("hdop", "<f8"), ("h", "<f8")])


def read_trj_quality(csv_path: str | Path) -> np.ndarray:
    """Parse the Inertial Explorer *_QF.csv (1 Hz). Returns (t_s GPS seconds of week,
    lat, lon, sd_m, amb (0 float, 1 fixed), q)."""
    rows = []
    with open(csv_path) as fh:
        next(fh)
        for line in fh:
            f = line.split("\t")
            if len(f) < 15:
                continue
            rows.append((float(f[1]), float(f[2]), float(f[3]), float(f[8]),
                         0 if f[9].strip().lower().startswith("float") else 1, int(f[-1])))
    return np.array(rows, dtype=[("t_sow", "<f8"), ("lat", "<f8"), ("lon", "<f8"), ("sd_m", "<f8"),
                                 ("fixed", "<i4"), ("q", "<i4")])


@dataclass
class Swath:
    """All files of one swath. Data arrays are memmaps (lazy)."""
    survey_dir: Path
    name: str
    config: SurveyConfig
    index: dict[int, np.ndarray] = field(default_factory=dict)   # array -> index records
    data: dict[int, np.memmap] = field(default_factory=dict)     # array -> (n_scans, n_ch, 512)
    path: np.ndarray | None = None
    gga_rt: np.ndarray | None = None
    gga_pp: np.ndarray | None = None
    trj_quality: np.ndarray | None = None

    @classmethod
    def load(cls, survey_dir: str | Path, name: str | None = None) -> "Swath":
        survey_dir = Path(survey_dir)
        cfg = parse_survey(survey_dir)
        name = name or cfg.swaths[0]
        sw = cls(survey_dir=survey_dir, name=name, config=cfg)
        for arr in (1, 2):
            nch = len(cfg.channels_of(arr))
            if nch == 0:
                continue
            base = survey_dir / f"{name}_Array{arr:02d}"
            sw.index[arr] = read_index(base.with_suffix(".index"), nch)
            sw.data[arr] = open_data(base.with_suffix(".data"), nch)
            assert sw.data[arr].shape[0] == len(sw.index[arr]), (name, arr)
        sw.path = read_path(survey_dir / f"{name}.path")
        rt = survey_dir / f"{name}.nmea"
        pp = survey_dir / f"{name}_Ext.nmea"
        sw.gga_rt = read_gga(rt) if rt.exists() else None
        sw.gga_pp = read_gga(pp) if pp.exists() else None
        trj = sorted((survey_dir.parent).glob("TrjData_*/*_QF.csv"))
        if trj:
            # pick the TrjData folder whose GGA csv covers this swath's start time
            sw.trj_quality = np.concatenate([read_trj_quality(t) for t in trj])
        return sw
