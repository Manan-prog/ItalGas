"""Figure: results of the improvement round (3 Nov 2026) in plain language.
A opposite-direction drives vs same-direction drives along the street (patch test) per street;
B where under the van a single drive finds a pipe running along the road (injection);
C along-road detections vs recorded pipes on 12 held-out drives; D crossing pipes continuing across the street.
Usage (from figures/): python plot_next_steps_results.py out.png"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURF = "#fcfcfb"; BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"; INK = "#1f1f1f"; MUTED = "#6b6b6b"; GREY = "#c9c9c9"
CACHE = Path("/Users/manansarupria/ItalGas/cache")
CL = [("", "Via Nizza, 13 Oct"), ("c2_", "street C2, 14 Oct"), ("c3_", "street C3, 15 Oct"), ("c4_", "street C4, 14 Oct"), ("c5_", "street C5, 17 Oct")]
fig = plt.figure(figsize=(17, 10.5), facecolor=SURF)
fig.suptitle("What the improvement round showed", x=0.02, y=0.985, ha="left", fontsize=15, color=INK, fontweight="bold")
gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.3, left=0.15, right=0.98, top=0.9, bottom=0.08)

# A: patch test
ax = fig.add_subplot(gs[0, 0]); y = 0; labels = []
for tag, name in CL:
    f = CACHE / f"mp_{tag}registration.json"
    if not f.exists():
        continue
    R = json.load(open(f)); ref = [n for n, g in R["registration"].items() if g.get("A_rel", 1) == 0.0 and g["B"] == 0.0]
    ref = ref[0] if ref else next(iter(R["registration"]))
    for p in R["pairs"]:
        if ref not in (p["a"], p["b"]):
            continue
        other = p["b"] if p["a"] == ref else p["a"]; sgn = 1 if p["a"] == ref else -1
        same = p["dir_a"] == p["dir_b"]
        val = None; strong = False
        if p["tie_used"]:
            val = sgn * p["A_tie"]; strong = True
        elif p["overlap"] and p["overlap"]["spread_A"] <= 0.5:
            val = sgn * p["overlap"]["dA"]; strong = p["overlap"]["r_best"] >= 0.5 and abs(p["overlap"]["dA"]) < 2.7
        elif p["tie_peak"] >= 3:
            val = sgn * p["A_tie"]; strong = False
        if val is None:
            continue
        col = BLUE if same else ORANGE
        ax.plot(val, y, "o", ms=9, color=col if strong else "white", mec=col, mew=1.8)
        labels.append((y, f"{name}: {other.split('_')[-1]}")); y += 1
    y += 0.6
ax.axvline(0, color=GREY, lw=1)
ax.set_yticks([l[0] for l in labels]); ax.set_yticklabels([l[1] for l in labels], fontsize=8)
ax.set_xlabel("shift along the street needed to line the drive up with the reference drive (m)", color=MUTED, fontsize=9)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, mec="white", ms=9, label="drive in the SAME direction"),
                   Line2D([], [], marker="o", ls="", color=ORANGE, mec="white", ms=9, label="drive in the OPPOSITE direction"),
                   Line2D([], [], marker="o", ls="", color="white", mec=MUTED, mew=1.8, ms=9, label="hollow = weak estimate")], fontsize=8.5, loc="lower right", frameon=False)
ax.set_title("A  Drives in opposite directions come out shifted along the street\n(a timing error would do this; cause still to be pinned down)", loc="left", fontsize=10.5, color=INK)
ax.invert_yaxis()

# B: injection by position
ax = fig.add_subplot(gs[0, 1])
J = json.load(open(CACHE / "injection_v23.json"))
cls = ["centre", "off-centre", "edge"]; x = np.arange(3)
for k, (key, col, lab) in enumerate((("recall_v22", GREY, "without the arch check"), ("recall_v23", AQUA, "with the arch check"))):
    vals = [np.mean(next(r for r in J["runs"] if r["amp"] == 5.0 and r["cls"] == c)[key]) for c in cls]
    ax.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=col, label=lab)
    for xi, vv in zip(x, vals):
        ax.text(xi + (k - 0.5) * 0.36, vv + 0.02, f"{vv * 12:.0f}/12", ha="center", fontsize=8.5, color=INK)
fp = {c: (next(r for r in J["runs"] if r["amp"] == 5.0 and r["cls"] == c)["false_tracks_v22"], next(r for r in J["runs"] if r["amp"] == 5.0 and r["cls"] == c)["false_tracks_v23"]) for c in cls}
NICE = {"centre": "under the middle", "off-centre": "0.45-0.7 m off centre", "edge": "at the strip edge"}
ax.set_xticks(x); ax.set_xticklabels([f"{NICE[c]}\n(false finds {fp[c][0]} -> {fp[c][1]})" for c in cls], fontsize=8.5)
ax.set_ylim(0, 1.15); ax.set_ylabel("share of fake pipes found", color=MUTED, fontsize=9); ax.legend(fontsize=8.5, frameon=False, loc="upper right")
ax.set_title("B  One drive finds a pipe running along the road only if it lies under the middle of the van\n(fake pipes of typical real strength, wandering 15 cm sideways)", loc="left", fontsize=10.5, color=INK)

# C: held-out records
ax = fig.add_subplot(gs[1, 0])
H = json.load(open(CACHE / "along_v23_summary.json"))["summary"]
bars = [("without the arch check", H["hits_v22"], H["chance_v22"]), ("with the arch check", H["hits_v23"], H["chance_v23"])]
for i, (lab, h, c) in enumerate(bars):
    ax.bar(i - 0.18, h, 0.34, color=AQUA); ax.bar(i + 0.18, c, 0.34, color=GREY)
    ax.text(i - 0.18, h + 0.2, f"{h}", ha="center", fontsize=9); ax.text(i + 0.18, c + 0.2, f"{c:.1f}", ha="center", fontsize=9)
ax.set_xticks([0, 1]); ax.set_xticklabels([b[0] for b in bars], fontsize=9)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=AQUA), plt.Rectangle((0, 0), 1, 1, color=GREY)], labels=["recorded pipes matched", "matches expected by pure chance"], fontsize=8.5, frameon=False)
ax.set_ylabel(f"of {H['n_targets']} recorded pipes", color=MUTED, fontsize=9)
ax.set_title("C  On 12 new drives, along-road finds still do not match recorded sewer and storm pipes\n(the arch check removes most false finds: 70 -> 6 tracks)", loc="left", fontsize=10.5, color=INK)

# D: crossing continuity
ax = fig.add_subplot(gs[1, 1]); rows = []
for tag, name in CL:
    f = CACHE / f"mp_{tag}network.json"
    if f.exists():
        c = json.load(open(f)).get("crossing_continuity")
        if c:
            rows.append((name, c["frac_continuing"], c["chance"], c["n_A"]))
for i, (name, fr, ch, n) in enumerate(rows):
    ax.barh(i - 0.18, fr, 0.34, color=ORANGE); ax.barh(i + 0.18, ch, 0.34, color=GREY)
    ax.text(fr + 0.01, i - 0.18, f"{fr * 100:.0f}%", va="center", fontsize=9); ax.text(ch + 0.01, i + 0.18, f"{ch * 100:.0f}% by chance", va="center", fontsize=8.5, color=MUTED)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9); ax.set_xlim(0, 1); ax.invert_yaxis()
ax.set_xlabel("share of crossing pipes seen in one direction's lanes that also appear in the other direction's lanes", color=MUTED, fontsize=9)
ax.set_title("D  Crossing pipes found in one lane continue across the street\n(the rest stop between the lanes or were missed there)", loc="left", fontsize=10.5, color=INK)
for a_ in fig.axes:
    a_.set_facecolor(SURF); a_.tick_params(colors=MUTED, labelsize=8.5)
    for sp in a_.spines.values():
        sp.set_color("#cfcfcf")
fig.savefig(sys.argv[1], dpi=150, facecolor=SURF); print("saved", sys.argv[1])
