"""Which line rule does a second lane reproduce best? (E50, along/rules.py -> cache/city/rule_compare.json)
A: selection groups (13-14 Oct): excess repeated length against total line length, one point per rule, the chosen and
the current rule labelled. B: the same on the held-out groups (15-17 Oct). Run: python -m italgas_gpr.figures.rule_compare"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
from italgas_gpr.io.paths import CITY, FIGURES, config_path, load_yaml  # noqa: E402

R = json.load(open(CITY / "rule_compare.json"))
_C = load_yaml(config_path("along_lines.yaml"))
ADOPTED = f"c{_C['p_min']}_r0.0_g{_C['max_gap']}_u{_C['gate_u_m']}"                 # the rule on the page (DECISIONS 2026-11-05)
BLUE, ORANGE, AQUA, SURF, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#fcfcfb", "#9a9a9a"


def main(out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5), facecolor=SURF, sharey=True)
    fig.suptitle("Which along-street line rule does a second lane reproduce best?  54 rules, excess repeated length vs lines drawn", x=0.01, ha="left", fontsize=12, weight="bold")
    fig.text(0.01, 0.915, "A: the six lane groups of 13–14 Oct (selection).   B: the twelve groups of 15–17 Oct (held out; the result).", fontsize=8.5, color="#444")
    fig.text(0.01, 0.885, "Excess = line rows the other lane repeats within ±0.08 m, minus the larger of two nulls: its lines shifted ≥ 20 m along the street, or 0.2–0.6 m across.", fontsize=8.5, color="#444")
    for a, key, title in ((ax[0], "selection", "A"), (ax[1], "test", "B")):
        S = R[key]
        for k, v in S.items():
            c = {0.5: BLUE, 0.7: AQUA, 0.9: ORANGE}[v["rule"]["p_cell"]]
            a.scatter(v["line_km"], v["excess_min_m"], s=22 + 30 * (v["rule"]["max_gap"] == 8), color=c, alpha=0.75, edgecolor="white", linewidth=0.6)
        for k, lab, dy in ((R["chosen"], "best on A", 10), (ADOPTED, "adopted (page)", 10), (R["current"], "rule of 4 Nov", -14)):
            v = S[k]; a.scatter(v["line_km"], v["excess_min_m"], s=120, facecolor="none", edgecolor="#222", linewidth=1.4)
            a.annotate(f"{lab}: {k}", (v["line_km"], v["excess_min_m"]), xytext=(-8 if v["line_km"] > 45 else 8, dy), textcoords="offset points", fontsize=8, color="#222", ha="right" if v["line_km"] > 45 else "left")
        a.set_title(title, loc="left", weight="bold"); a.set_xlabel("lines drawn in the groups (km)"); a.set_facecolor(SURF)
        a.grid(True, color="#e6e6e6", lw=0.6); [a.spines[s].set_visible(False) for s in ("top", "right")]
    ax[0].set_ylabel("excess repeated length (m)")
    for c, lab in ((BLUE, "cell threshold 0.5"), (AQUA, "0.7"), (ORANGE, "0.9")):
        ax[0].scatter([], [], color=c, label=lab)
    ax[0].scatter([], [], color=GREY, s=52, label="large marker: 4 m gaps allowed"); ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig(out, dpi=150, facecolor=SURF); print(f"saved {out}")


if __name__ == "__main__":
    main(FIGURES / "rule_compare.png")
