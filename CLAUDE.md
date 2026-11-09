# ItalGas Savona: pipe localisation from array GPR

Goal: recover the 2D position (lat/lon) and along-street continuity of buried gas pipes
from the IDS Stream UP radargrams. Depth is secondary. Position accuracy and continuity
are the success criteria; every modelling choice is judged against them.

## 1. Response format (every reply, no exceptions)
1. One or two sentences stating the conclusion or result.
2. Bullet points, concise, carrying the substance of the answer.
3. Elaboration only if it adds information the bullets cannot carry.
No preamble, no restating the question, no closing recap. Numbers come from files or
code that was run in the reply, never from memory; if a number is from memory, say so.

## 2. Scientific rigour
- **No unverified physical assumptions.** Wave velocity, time zero, antenna offsets,
  trace spacing, sampling interval and GPS lever arm are read from the data or
  measured, never guessed. If a value must be assumed, label it ASSUMED in code and
  prose and record it in `DECISIONS.md` with the reason and the planned verification.
- **Three classes of quantity, always distinguished:** measured (from the instrument
  or GNSS), derived (computed with a stated formula), assumed. Mixing them silently
  is an error.
- **Units on every quantity.** Metres, nanoseconds, samples, degrees. Web Mercator
  distances must be scaled by cos(latitude) before being called metres.
- **Uncertainty travels with the estimate.** A pipe position without a position error
  bound is not a result. Positioning quality of the trajectory (RTK fix state,
  post-processed standard deviation) is propagated into every detection.
- **Baseline before complexity.** A classical method (migration, hyperbola fitting,
  cross-channel coherence) is run and scored first. A learned model must beat the
  baseline on the same held-out streets or it is not adopted.
- **Held-out evaluation only.** Train/validation/test splits are by street and by day,
  never by trace, to prevent leakage through spatial autocorrelation.
- **Ground truth is the arbiter.** Until ItalGas GIS pipe maps are obtained, all
  accuracy claims are provisional and are labelled as such.
- **Negative results are reported** with the same care as positive ones.
- **Literature is traced.** Any technique taken from the web or papers is logged in
  `REFERENCES.md` with author, year, venue, URL, and the one sentence of what we use
  from it. Prefer peer-reviewed sources; note when a source is a vendor document.
  Cross-disciplinary analogies (seismic, medical ultrasound, SAR, radio astronomy)
  are welcome but their assumptions are checked against GPR physics before use.
- **Reproducibility.** Every experiment has a config file, a fixed random seed, the
  git commit hash, and an entry in `EXPERIMENTS.md` (date, config, data subset,
  metric, one-line conclusion). A result that cannot be regenerated from the log
  does not exist.

## 3. Project facts (verified 2 Nov 2026, not derivable from the code)
- Instrument: IDS GeoRadar Stream UP, vehicle towed. 10 antennas in two rows:
  5 x AM600 (600 MHz, front row, x = +0.347 m) and 5 x AM200 (200 MHz, rear row,
  x = 0). Lateral positions y = +0.671, +0.335, 0, -0.335, -0.671 m. 29 channels
  (10 at 600 MHz, 19 at 200 MHz: each AM200 holds TX, RX, TX, RX 8 cm apart, giving
  3 pairs per box and 4 between neighbouring boxes). Channel footprints span
  y = -0.751 to +0.751 m (1.50 m) in both rows: every 8 to 9 cm at 200 MHz, every
  16 to 18 cm at 600 MHz. 512 samples per
  trace; 60 ns window at 600 MHz, 100 ns at 200 MHz. Scan spacing (measured, dev
  swath index files, 5 Nov 2026): 4.54 cm for the 600 MHz row, 7.57 cm for the
  200 MHz row.
- Vehicle frame: x forward, y positive to the LEFT of travel (verified against the
  IDS KML channel lines). GPS antenna at (0.695, -0.175, 1.968) m in that frame.
- Survey: Savona, 13 to 17 Oct 2025, 10 sessions, 155 swaths, about 160 km of track.
- Raw data: `$ITALGAS_DATA/Savona/` (default `~/ItalGas/Savona/`; local, 90 GB, never synced; `src/italgas_gpr/io/paths.py`). Zips remain in S3
  bucket `smart-auger-italgas-ingest-286665918804` and in this Drive folder as
  online-only placeholders. AWS profile `italgas-manan-cli`.
- File formats (reverse engineered; reader of record `src/italgas_gpr/io/ids_reader.py`, exploration scripts in `archive/figures/`):
  `.data` = 36-byte header then int16 little-endian scans of [channels x 512];
  `.index` = header of 60 + 4 x n_channels bytes (100 for the 600 MHz array, 136 for the 200 MHz array) then 20-byte records (scan, time us, flag, odometer m);
  `.path` = 112-byte header then 36-byte records (x, y, z EPSG:3857, seq, time ms,
  flag); `.nmea` = real-time RTK GGA at 10 Hz; `_Ext.nmea` = post-processed track.
- Positioning caveat: post-processed trajectory (Inertial Explorer, tightly coupled
  GNSS+IMU) is below 10 cm standard deviation for only about a quarter of epochs and
  above 1 m for 40 to 90 percent depending on the day. This is the dominant error
  source for 2D pipe position and must be mapped before detection work.
- Ground truth: none yet. Request ItalGas GIS network layer and surface feature
  positions (valves, manholes, meters) for the surveyed streets.
- Related work: `../Fathom/` holds the racetrack pipeline (Sensors & Software
  .dt1 format) and Horseshoe survey (ImpulseRadar). Different formats, same B-scan
  abstraction downstream. Read-only reference.

## 4. Code standards
- Python 3.11, NumPy/SciPy first. Vectorised operations; no Python loops over traces
  or samples. Use `np.memmap` for `.data` files; never load a whole swath unless the
  step needs it.
- Pure functions with type hints and docstrings stating units and array shapes
  (`(n_scans, n_channels, n_samples)` is the canonical order).
- One source of truth per concept: a single reader module for the IDS format, a
  single geometry module for vehicle-frame to map-frame transforms. No copy-paste.
- Profile before optimising; optimise only what the profile shows. Record the
  before/after timing in the commit message.
- Intermediate products (georeferenced traces, feature maps) are cached locally as
  `.npz` or Zarr with a hash of the inputs in the filename. Caches live outside Drive.
- Configuration in YAML, never hard-coded constants in function bodies.
- Tests: the format reader, the geometry transform and every metric have unit tests
  with known answers (for geometry, the IDS KML lines are the reference).
- Notebooks are for looking, not for logic. Anything reused moves to a module.
- Code is the installable package `src/italgas_gpr/` (subpackages io, geometry, processing, along, crossing, lanes, web,
  figures; `pip install -e .`; one CLI `italgas-gpr`), configs in `configs/`, tests in `tests/`. Virtual environments,
  caches, raw data and model checkpoints live locally and are git-ignored.
- Run scripts with `python -I` when they read untrusted downloaded files.

## 5. Figures
- Follow the dataviz method: pick the form for the data's job, assign colour by role,
  validate the palette, thin marks, recessive axes, legend for two or more series,
  direct labels where they help. Light surface `#fcfcfb`; categorical slots
  blue `#2a78d6`, orange `#eb6834`, aqua `#1baf7a`; never a rainbow for magnitude.
- Every map figure has a basemap (Google satellite at 75 percent opacity for
  overviews, zoom 21 for detail panels), a scale bar, and corner coordinates.
- Every figure answers one question stated in its title. Panels are lettered and
  the caption line under the title says what each panel shows.
- Figures are produced by a module of `src/italgas_gpr/figures/` that writes the PNG into `figures/`, at 150 dpi
  or more, and are opened and inspected before being delivered.
- Innovation is welcome (small multiples along the street, uncertainty ribbons on
  the pipe trace, radargram-on-map perspective views) but never at the cost of
  readability or honesty about the data.

## 6. Working practices
- **Deepening check, before any substantive task** (a new method, analysis, model,
  figure series or research question; not for small edits or fixes). Answer these
  four in writing, then restate the task as the deeper version and do that:
  1. What would a top professional in this field do here that is not being asked for?
  2. What is the rawest material they would start from, instead of what has already
     been digested (raw traces and GNSS logs over exported images; papers over blog
     summaries; the KML and config files over my own earlier conclusions)?
  3. Which specific methods and tools would they use?
  4. What, other than the author or the reviewer, would show the result is right
     (ground truth, an independent dataset, a physical constraint, a reproduction
     by a different method)?
  The aim is a deeper task, not a bigger one. If the deeper version changes the
  scope, say so in one line before starting.
- Plan in writing before coding anything longer than a screen. State what will be
  measured to decide if it worked.
- Develop on one swath (14 Oct 11:31 session, 1.2 km, fully extracted) until the
  method is correct, then scale to a day, then to all 155 swaths.
- Definition of done for a task: code runs end to end from the config, the metric is
  logged in `EXPERIMENTS.md`, the figure is inspected, and the next step is written.
- Decisions with lasting consequences (coordinate system, velocity model, metric
  definition, split strategy) are recorded in `DECISIONS.md` with date and reason.
- Never modify or delete raw data. Never write large files into this Drive folder.
- Stop and ask before: deleting anything, spending money (cloud compute, APIs),
  changing the evaluation metric, or using a data source whose licence is unclear.

## 7. Terminology (use these words, nothing else)
- **trace**: one recorded time series at one position for one channel.
- **scan**: all 29 channels at one firing position.
- **channel**: one transmit-receive pair (29 per scan).
- **column**: one of the five lateral antenna positions.
- **swath**: one continuous drive, one `.data` file pair.
- **session**: one DataLogger run, one `Survey_*` folder.
- **B-scan**: traces of one channel along one swath, shown as an image.
- **track**: the GPS antenna trajectory.
