"""Figure (E52): does bridging gaps between along-street lines add length that a second lane reproduces?
A: length of joins per variant (gap limit, corridor support) on the choosing and judging street groups, next to the
lines they would bridge. B: of the join rows a drive in another lane covers, the share that lane reproduces (lines +
joins), with the shifted-null share, per variant and set; the lines' own share drawn as reference.
Usage (from figures/): python plot_join_compare.py out.png"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF = "#fcfcfb"; ORANGE = "#eb6834"; BLUE = "#2a78d6"; AQUA = "#1baf7a"; INK = "#1f1f1f"; MUTED = "#6b6b6b"
R = json.load(open(Path.home() / "ItalGas/cache/city/join_compare.json"))
keys = [k for k in R["selection"] if k != "lines"]; lab = [k.replace("gap", "gap ").replace("_s", " m, support ≥ ") for k in keys]
fig, ax = plt.subplots(1, 2, figsize=(15, 5.4), facecolor=SURF, gridspec_kw=dict(wspace=0.3))
fig.suptitle("Do joins between along-street lines add length that a second lane reproduces? On held-out streets, none of the covered join rows is repeated",
             x=0.01, ha="left", fontsize=13, color=INK, weight="bold", y=1.0)
fig.text(0.01, 0.935, "A: join length per variant (blue: choosing streets 13–14 Oct, orange: judging streets 15–17 Oct), with the lines of the adopted rule for scale.  "
         "B: share of covered join rows that a drive in another lane reproduces within 0.08 m (dots), the shifted-null share (ticks); dashed = the lines' own share.", fontsize=9.5, color=MUTED)
x = np.arange(len(keys)); w = 0.38
for k_, (name, col, off) in enumerate((("selection", BLUE, -w / 2), ("test", ORANGE, w / 2))):
    S = R[name]; a = ax[0]
    a.bar(x + off, [S[k]["join_km"] * 1000 for k in keys], w, color=col, label=f"{'choosing' if name == 'selection' else 'judging'} streets: lines {S['lines']['line_km']:.1f} km")
    b = ax[1]; sh = [S[k]["joins_vs_both"]["share"] for k in keys]; nl = [max(S[k]["joins_vs_both"]["null_share"] or 0, S[k]["joins_vs_both"]["null_lat_share"] or 0) if S[k]["joins_vs_both"]["share"] is not None else None for k in keys]
    cov = [S[k]["joins_vs_both"]["covered_m"] for k in keys]
    b.plot(x + off, [np.nan if s is None else s for s in sh], "o", color=col, ms=7, label=f"{'choosing' if name == 'selection' else 'judging'} streets")
    b.plot(x + off, [np.nan if s is None else s for s in nl], "_", color=col, ms=12, mew=2)
    for xi, c, s in zip(x + off, cov, sh):
        if c > 0:
            b.text(xi, s + 0.04, f"{c} m", ha="center", fontsize=7.5, color=col)
    b.axhline(S["lines"]["share"], ls="--", lw=1, color=col)
ax[0].set_ylabel("join length (m)", fontsize=9); ax[0].set_title("A  Length the joins would add", loc="left", fontsize=10, color=INK); ax[0].legend(fontsize=8, frameon=False)
ax[1].set_ylabel("share of covered join rows reproduced by the other lane", fontsize=9); ax[1].set_ylim(-0.02, 1.08); ax[1].set_title("B  Second-lane reproduction of the join rows (label: metres covered; no label = nothing covered)", loc="left", fontsize=10, color=INK)
ax[1].legend(fontsize=8, frameon=False, loc="upper right")
for a in ax:
    a.set_facecolor(SURF); a.set_xticks(x); a.set_xticklabels(lab, rotation=35, ha="right", fontsize=8)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.tick_params(labelsize=8.5, colors=MUTED)
fig.text(0.01, -0.3, f"Lines: adopted rule (P ≥ {R['line_rule']['p_min']}, {R['line_rule']['max_gap']} windows bridged, gate {R['line_rule']['gate_u_m']} m, ≥ {R['line_rule']['min_len_m']:.0f} m, plan-view merged). Join: two line ends within the linking gates, straight corridor; "
         f"support = share of gap windows with a P ≥ {R['line_rule']['p_min']} peak within ±{R['line_rule']['tol_cells'] * 0.04:.2f} m / ±2 ns of it. Null: the other lane's lines and joins shifted ≥ {R['shift_min_m']:.0f} m along or 0.2–0.6 m across ({R['n_shift']} draws). "
         f"{len(R['select_refs'])} choosing and {len(R['test_refs'])} judging lane groups. No ground truth: provisional.", fontsize=8, color=MUTED, wrap=True)
out = sys.argv[1] if len(sys.argv) > 1 else "join_compare.png"
fig.savefig(out, dpi=150, facecolor=SURF, bbox_inches="tight"); print(out)
