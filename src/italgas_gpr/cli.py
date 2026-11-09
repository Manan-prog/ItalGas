"""One command for the whole chain: `italgas-gpr <step> [args]` runs the module of that step with the given arguments,
exactly as `python -m italgas_gpr.<module> [args]` would. `italgas-gpr --help` lists the steps in running order."""
from __future__ import annotations

import runpy
import sys

STEPS = [  # (command, module, arguments, what it produces) in the order they are run
    ("tracks", "italgas_gpr.geometry.tracks", "configs/survey.yaml", "cache/swaths.npz: every drive's RTK track (drive index = sorted swath list)"),
    ("pairs", "italgas_gpr.lanes.pairs", "", "cache/city/lane_components.json: which drives repeat which street in another lane"),
    ("rasters", "italgas_gpr.along.rasters", "[n_workers] [lanes|all]", "cache/prob_raster_<drive>_<s0>.npz: along-street evidence raster per 600 m piece"),
    ("raster", "italgas_gpr.along.evidence", "configs/along_raster.yaml <drive> <piece start m>", "one raster piece (inspection)"),
    ("lanes", "italgas_gpr.lanes.register", "[n_workers] [component ...]", "cache/city/lane_<ref>.npz, lane_summary.json: lane registration and agreement (E47)"),
    ("lines", "italgas_gpr.along.lines", "[n_workers] [cache dir]", "cache/city/rline_<drive>.json, raster_lines.json: along-street lines by configs/along_lines.yaml"),
    ("rules", "italgas_gpr.along.rules", "[n_workers]", "cache/city/rule_compare.json: held-out comparison of 54 line rules (E50, optional)"),
    ("scan", "italgas_gpr.crossing.scan", "configs/crossing_scan.yaml --workers <n>", "cache/city/scan_<drive>.json: crossing pipes, both radar rows"),
    ("merge", "italgas_gpr.crossing.survey_merge", "configs/crossing_scan.yaml", "cache/city/citywide_merged.json: checks C1..C4 and repeat-drive labels"),
    ("images", "italgas_gpr.web.evidence_images", "[n_workers] [lines|crossings|all]", "cache/web_evidence/: radargram image per detection"),
    ("page", "italgas_gpr.web.build", "figures/web/savona_pipes_map.html [cache dir]", "the interactive map, one HTML file"),
    ("host", "italgas_gpr.web.host", "<osm overpass json> <out dir>", "hosted variant of the page (vector basemap, image sheets)"),
    ("fig-coverage", "italgas_gpr.figures.coverage", "[out.png]", "figures/investor_coverage_savona.png"),
    ("fig-street", "italgas_gpr.figures.street_pipes", "[out.png]", "figures/team_street_pipes.png"),
    ("fig-rules", "italgas_gpr.figures.rule_compare", "", "figures/rule_compare.png"),
    ("fig-floor", "italgas_gpr.figures.near_surface_floor", "", "figures/near_surface_floor.png"),
    ("fig-raster", "italgas_gpr.figures.prob_raster", "<drive> <s0> <stretch start m> <stretch length m>", "figures/savona_prob_raster_<drive>_<s0>.png"),
    ("fig-raster-lines", "italgas_gpr.figures.raster_vs_lines", "", "figures/savona_green_vs_lines_nizza.png"),
    ("fig-crossings", "italgas_gpr.figures.crossing_map", "out.png", "crossing map of a street group"),
    ("fig-lane-agreement", "italgas_gpr.figures.lane_agreement", "", "figures/savona_mosaic_test_nizza.png"),
    ("fig-lane-raster", "italgas_gpr.figures.lane_raster", "<out.png> <drive,...> <ref drive> <s0> <s1> [cell m]", "lane raster of a street"),
]


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip() + "\n\nSteps, in running order (details in README.md):\n")
        for cmd, mod, args, what in STEPS:
            print(f"  {cmd:19s} {args:52s} {what}")
        print("\nData live in $ITALGAS_DATA (default ~/ItalGas): Savona/ raw data, cache/ products.")
        return
    steps = {c: m for c, m, _, _ in STEPS}
    if argv[0] not in steps:
        sys.exit(f"unknown step {argv[0]!r}; run italgas-gpr --help")
    sys.argv = [steps[argv[0]]] + argv[1:]
    runpy.run_module(steps[argv[0]], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
