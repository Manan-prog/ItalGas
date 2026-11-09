# archive/

Code of experiments that did not become part of the working chain (moved here 9 Nov 2026, unchanged, unmaintained).
Every experiment is logged in `../EXPERIMENTS.md` with its config and metric, and the decisions it led to are in
`../DECISIONS.md`; this folder keeps those entries reproducible.

- `pipeline/`  experiment modules (injection tests, learned emission, Viterbi tracking, polarisation, end-point detector,
  junction pins, apex agreement, multi-pass registration, town-plan overlay, the first detector `run_detection`/`detect`/`detect2`, ...).
  They import the working modules by their OLD flat names (`from build_v2 import ...`); to run one, put a folder of shims or
  the old layout on `sys.path`, or rewrite its imports with the table below. Paths inside are the original machine's.
- `configs/`   their YAML configs (old names; `italgas_gpr.io.paths.config_path` also accepts the old names of the kept configs).
- `figures/`   their figure scripts (the PNGs are experiment output and are not in git; the INTERNAL town-plan overlay never is).
- `web/`       the first two web pages (three-street page, city page template v1).
- `tests/`     unit tests of archived modules.

## Module names: logs and archive (before 9 Nov 2026) -> package

| old flat module | now |
|---|---|
| `ids_reader.py` | `italgas_gpr/io/ids_reader.py` |
| `paths.py` | `italgas_gpr/io/paths.py` |
| `geometry.py`, `run_detection.su_to_map` | `italgas_gpr/geometry/frames.py` |
| `cube.py` | `italgas_gpr/geometry/cube.py` |
| `build_cube.py` | `italgas_gpr/geometry/swath.py` |
| `build_swath_tracks.py` | `italgas_gpr/geometry/tracks.py` |
| `dither.py` | `italgas_gpr/geometry/dither.py` |
| `trajectory_qc.py` | `italgas_gpr/geometry/trajectory_qc.py` |
| `processing.py` | `italgas_gpr/processing/filters.py` |
| `build_v2.py` | `italgas_gpr/processing/preprocess.py` |
| `longitudinal.py` | `italgas_gpr/along/focus.py` |
| `prob_raster.py` | `italgas_gpr/along/evidence.py` |
| `run_prob_rasters.py` | `italgas_gpr/along/rasters.py` |
| `raster_lines.py` | `italgas_gpr/along/lines.py` |
| `rule_compare.py` | `italgas_gpr/along/rules.py` |
| `run_longitudinal_v2.py` | `italgas_gpr/along/tracks.py` |
| `terminations.py` | `italgas_gpr/along/endpoints.py` |
| `crossings.py` | `italgas_gpr/crossing/score.py` |
| `run_crossings_v2.py` | `italgas_gpr/crossing/detect.py` |
| `finalize_v2.py` | `italgas_gpr/crossing/merge.py` |
| `citywide_scan.py` | `italgas_gpr/crossing/scan.py` |
| `citywide_merge.py` | `italgas_gpr/crossing/survey_merge.py` |
| `survey_pairs.py` | `italgas_gpr/lanes/pairs.py` |
| `mosaic_test.py` | `italgas_gpr/lanes/agreement.py` |
| `mosaic_fast.py` | `italgas_gpr/lanes/register.py` |
| `run_multipass.py` | `italgas_gpr/lanes/multipass.py` |
| `evidence_images.py` | `italgas_gpr/web/evidence_images.py` |
| `figures/web/web_common.py`, `build_savona_city_web.py`, `build_savona_pipes_web.py`, `host_savona_pipes.py` | `italgas_gpr/web/common.py`, `build_city.py`, `build.py`, `host.py` |
| `figures/basemap_tools.py`, `map_layers.py` | `italgas_gpr/figures/basemap.py`, `map_layers.py` |
| `figures/investor_coverage_savona.py`, `team_street_pipes.py` | `italgas_gpr/figures/coverage.py`, `street_pipes.py` |
| `figures/plot_rule_compare.py`, `plot_near_surface_floor.py`, `plot_prob_raster.py` | `italgas_gpr/figures/rule_compare.py`, `near_surface_floor.py`, `prob_raster.py` |
| `figures/plot_green_vs_lines.py`, `plot_pipelines_map.py`, `plot_mosaic_test.py`, `plot_mosaic_raster.py` | `italgas_gpr/figures/raster_vs_lines.py`, `crossing_map.py`, `lane_agreement.py`, `lane_raster.py` |

| old config | now |
|---|---|
| `dev_swath.yaml` | `configs/detector_base.yaml` |
| `dev_swath_v2_2.yaml` | `configs/detector.yaml` |
| `dither.yaml` (also `prob_raster.yaml`) | `configs/along_raster.yaml` |
| `raster_lines.yaml` | `configs/along_lines.yaml` |
| `citywide.yaml` | `configs/survey.yaml` |
| `citywide_scan.yaml` | `configs/crossing_scan.yaml` |
| `citywide_along.yaml` | `configs/along_tracks.yaml` |
| `subset_merge.yaml` | `configs/crossing_merge_subset.yaml` |
| `multipass_aurelia.yaml` | `configs/lane_registration.yaml` |

Cache file names (`prob_raster_<drive>_<s0>.npz`, `rline_<drive>.json`, `scan_<drive>.json`, `lane_<ref>.npz`, `mosaic_test.json`, ...)
did not change.
