"""Build the method report PDF (new-user guide): python build_method_report.py -> ItalGas_Savona_pipe_detection_method.pdf
Text is for a new user; every number comes from the cached result files read here."""
import json
import sys
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[1]; FIG = ROOT / "figures"; sys.path.insert(0, str(ROOT / "pipeline"))
from paths import CACHE, CITY  # noqa: E402
OUT = Path(__file__).with_name("ItalGas_Savona_pipe_detection_method.pdf")
# numbers from files
RL = json.load(open(CITY / "raster_lines.json")); sm = RL["summary"]; cfg = RL["config"]
LS = json.load(open(CITY / "lane_summary.json")); RC = json.load(open(CITY / "rule_compare.json")); T = RC["test"][f"c{cfg['p_min']}_r0.0_g{cfg['max_gap']}_u{cfg['gate_u_m']}"]
frames = {int(f.stem.split("_")[1]): float(np.load(f)["length_m"]) for f in CITY.glob("rframe_*.npz")}
scans = [int(f.stem.split("_")[1]) for f in CITY.glob("scan_*.json") if json.load(open(f)).get("crossings")]
n_cross = sum(len(json.load(open(CITY / f"scan_{i}.json"))["crossings"]) for i in scans)
n_A = sum(sum(c["cls"] == "A" for c in json.load(open(CITY / f"scan_{i}.json"))["crossings"]) for i in scans)
km_all = sum(frames.values()) / 1e3; km_scan = sum(frames[i] for i in scans if i in frames) / 1e3
sw = np.load(CACHE / "swaths.npz", allow_pickle=True); n_swaths = len(sw.files)

ss = getSampleStyleSheet(); INK = colors.HexColor("#1f1f1f"); MUTED = colors.HexColor("#6b6b6b"); BLUE = colors.HexColor("#2a78d6")
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, leading=20, textColor=INK, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, leading=16, textColor=BLUE, spaceBefore=10, spaceAfter=4)
P = ParagraphStyle("p", parent=ss["Normal"], fontSize=10, leading=14, textColor=INK, spaceAfter=6)
CAP = ParagraphStyle("cap", parent=P, fontSize=8.5, leading=11, textColor=MUTED, spaceAfter=10)
BUL = ParagraphStyle("bul", parent=P, leftIndent=12, bulletIndent=2, spaceAfter=3)
TITLE = ParagraphStyle("t", parent=ss["Title"], fontSize=20, leading=24, alignment=TA_LEFT, textColor=INK, spaceAfter=4)
SUB = ParagraphStyle("s", parent=P, fontSize=10.5, textColor=MUTED, spaceAfter=14)
W = A4[0] - 40 * mm


def fig(path, caption, width=W):
    im = PILImage.open(path); h = width * im.height / im.width
    return [KeepTogether([Image(str(path), width=width, height=h), Paragraph(caption, CAP)])]


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


story = [Paragraph("Finding gas pipes with a radar van", TITLE),
         Paragraph("How the Savona pipe map is made, step by step. A guide for new users. SmartAuger, November 2026.", SUB)]
story += [Paragraph("1. What was measured", H1),
          Paragraph(f"In October 2025 a van towing an IDS Stream UP ground-penetrating radar drove {km_all:.0f} km of streets in Savona ({n_swaths} drives on four days). "
                    "The radar sends short pulses into the ground and records the echoes. Every buried object that differs from the soil around it, a pipe, a cable, a wall, "
                    "a change of soil, sends back an echo. The time the echo takes to return tells the depth; with a ground speed of 0.10 m per nanosecond, measured from the data, "
                    "the radar sees to about 1.5 m.", P)]
story += fig(FIG / "report/array_schematic.png", "Figure 1. The array under the van. The 200 MHz row has 19 channels every 8 cm and sees deepest; the 600 MHz row has 10 channels and sees finer detail near the surface. Each pass covers a 1.5 m strip.", W * 0.7)
story += fig(FIG / "investor_coverage_savona.png", "Figure 2. Everything that was driven. Green where the GPS had a precise fix, orange where it did not. Positions on the map are uncertain by 0.3 to 1 m, more in orange stretches.", W * 0.5)
story += [Paragraph("2. Why a pipe looks like an arch", H1),
          Paragraph("The radar does not see a pipe as a pipe. As the van approaches a pipe, the echo comes back from far away and arrives late; above the pipe it arrives earliest; "
                    "then it gets later again. Plotted along the street, the echo traces an arch. Everything in the method comes back to this arch.", P)]
story += fig(FIG / "report/arch_geometry.png", "Figure 3. A: a pipe crossing the street makes an arch along the drive. B: a pipe running along the street makes the same arch across the 19 side-by-side channels, but the array is only 1.5 m wide, so only the top of the arch is seen.")
story += [Paragraph("3. Cleaning the recordings", H1),
          Paragraph("Three steps are applied to every channel before anything is detected. The average trace of the whole drive is subtracted, which removes the instrument's own ringing and the flat surface echo. "
                    "A filter keeps the frequencies where the pipe echoes live. The amplitude at every depth is divided by the typical noise at that depth, so deep and shallow echoes can be compared.", P)]
story += fig(FIG / "report/processing_steps.png", "Figure 4. One crossing pipe on Via Nizza through the chain: as recorded, cleaned, and focused. Focusing (migration) adds the echoes along the arch so a pipe becomes a compact spot at its true position and depth.")
story += [Paragraph("4. Pipes crossing the street", H1),
          Paragraph("For a pipe that crosses the street, each channel sees the arch along the drive. The method focuses every channel, then adds the five antenna columns together along the line a pipe would make "
                    "(perpendicular to the street or up to 60 degrees off). A pipe gives a bright, narrow spot; a flat layer gives a wide band and is rejected by comparing the spot with its surroundings 0.6 to 1.5 m either side.", P)]
story += bullets(["Both rows must agree. A crossing is called confident only when the 200 MHz and the 600 MHz rows find it at the same place and depth.",
                  "A chance rate is measured on the same data with the channels shifted, so a detection is reported with the share of equally strong spots that noise alone produces.",
                  f"So far {len(scans)} drives ({km_scan:.0f} km) have been scanned: {n_cross} crossings, of which {n_A} are seen by both rows."])
story += fig(FIG / "team_street_pipes.png", "Figure 5. 135 m of Via Nizza: blue bars are crossing pipes found by both rows, with their depth; the three radargrams show the arch behind three of them.")
story += [Paragraph("5. Pipes running along the street", H1),
          Paragraph("A pipe under the van that runs along the street looks the same in every trace, so it hides in the background. The trick is to look across the street instead of along it, in five steps.", P)]
story += bullets(["Average along the street: traces are averaged over 2 m, stepping 0.5 m. A pipe along the street survives the average; noise and short objects shrink.",
                  "Focus across the street: the 19 averaged traces are added along the arch a pipe would make for every possible sideways position (every 4 cm) and depth. A pipe collapses to a spot; a flat layer stays a line.",
                  "Score spot against line: each cell's brightness minus the brightest continuation 12 to 32 cm to the side. A layer scores about zero.",
                  "Shuffle test: the order of the 19 channels is scrambled 8 times and the focusing repeated. A real pipe cannot survive this; layers and noise do. For every score, P = the share of such cells that the shuffled data does not produce. P is the evidence value shown as the raster on the map.",
                  f"Join into lines: cells with P at least {cfg['p_min']} are followed along the street within {cfg['gate_u_m']} m sideways, bridging up to {cfg['max_gap'] * 0.5:.0f} m of gaps; runs shorter than {cfg['min_len_m']:.0f} m are not drawn."])
story += fig(FIG / "report/shuffle_demo.png", "Figure 6. The along-street method on one real line. Left: the 19 averaged channels. Middle: focused with the true positions, the pipe is a spot (red ring). Right: the same data with the channel order shuffled, the spot is gone. This is the comparison behind every P value.")
story += [Paragraph(f"Result: {sm['n_lines']} lines, {sm['length_m'] / 1e3:.0f} km in total, on {len(frames)} drives.", P)]
story += [Paragraph("6. How we know it is not noise (without a gas map)", H1),
          Paragraph("No pipe map from ItalGas has been received yet, so no detection has been checked against a real pipe. Three internal checks stand in for it.", P)]
story += bullets(["Shuffled channels (section 5) and shifted channels (section 4) give the rate at which the method invents a pipe from nothing. Every mark on the map carries that rate.",
                  f"Repeat drives in a neighbouring lane: {LS['n_tested_components']} streets were driven twice in lanes 0.5 to 1.2 m apart. On {LS['n_positive']} of them the second drive finds the same evidence cells more often than chance "
                  f"({LS['f_pooled'] * 100:.0f}% of cells repeated against {LS['f_pooled_null_mean'] * 100:.0f}% when the drives are shifted against each other).",
                  f"Held-out streets: the rule for drawing lines was chosen on the streets of 13 and 14 October and judged on those of 15 to 17 October, which it had never seen. There, {T['share'] * 100:.0f}% of the drawn line length is found again by the other lane, "
                  f"against {max(T['null_share'], T['null_lat_share']) * 100:.0f}% for shifted lines.",
                  "Municipal sewer and storm records are drawn on the map for comparison; they are not gas pipes but they cross the same streets."])
story += [Paragraph("7. Reading the map", H1),
          Paragraph("The interactive map (savona_pipes_map.html) shows the drive footprints, the evidence raster (red = like pipe-free data, green = beyond it), green along-street lines and blue crossing pipes. "
                    "Hover a line or a bar for the summary; click it for the details and the radar picture behind it, with the detection marked in red. The Checks section lists every threshold and every chance rate used.", P)]
story += [KeepTogether([Paragraph("8. What to keep in mind", H1)] + bullets(["Everything is provisional until compared with the ItalGas network map. The method finds buried objects that behave like pipes, not gas pipes specifically.",
                  "Positions depend on the GPS track: 0.3 to 1 m uncertainty, more where the fix was lost. The uncertainty is given on every mark.",
                  "Along-street lines are the weak point. A line seen by one drive can be a reflection fixed to the vehicle or the trench the pipe lies in; only a repeat from another lane rules that out, and few streets have one.",
                  "The top 0.33 m and the outer 0.2 m of each side of the strip are not evaluated, because the instrument itself produces strong, fixed signals there.",
                  f"The crossing detector has run on {len(scans)} of {len(frames)} drives; the rest is machine time, not method work."]))]
story += [Spacer(1, 8), Paragraph("Code: pipeline/ in the project folder; every figure in this report has a script next to it under figures/. Experiments are logged in EXPERIMENTS.md, decisions in DECISIONS.md.", CAP)]
doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=18 * mm, title="Finding gas pipes with a radar van", author="SmartAuger")
doc.build(story); print(OUT)
