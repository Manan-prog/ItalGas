"""T3: 600 vs 200 MHz agreement of longitudinal tracks (independent antenna rows).
A 600 MHz track (>= min_len) agrees if a 200 MHz track overlaps it along s by >= 50
percent of the shorter, with |du| <= 0.2 m and |d depth| <= 0.15 m. Chance: 200 MHz
tracks shifted along s by random offsets (20 to 200 m, wrapped).
Run: python eval_longitudinal_v2.py long_<hash>.json"""
from __future__ import annotations

import json
import sys

import numpy as np


def agree(a, b, shift=0.0, lo=0.0, hi=1.0):
    out = []
    for t in a:
        ok = False
        for r in b:
            s0, s1 = r["s_start"] + shift, r["s_end"] + shift
            if s0 > hi:
                s0, s1 = s0 - (hi - lo), s1 - (hi - lo)
            if s1 < lo:
                s0, s1 = s0 + (hi - lo), s1 + (hi - lo)
            ov = min(t["s_end"], s1) - max(t["s_start"], s0)
            if ov >= 0.5 * min(t["length_m"], r["length_m"]) and abs(t["u_mean"] - r["u_mean"]) <= 0.2 \
                    and abs(t["depth_m"] - r["depth_m"]) <= 0.15:
                ok = True; break
        out.append(ok)
    return np.array(out)


def main(path: str, min_len: float = 10.0, seed: int = 0) -> dict:
    r = json.load(open(path)); rng = np.random.default_rng(seed)
    a = [t for t in r["arrays"]["1"]["tracks"] if t["length_m"] >= min_len]
    b = [t for t in r["arrays"]["2"]["tracks"] if t["length_m"] >= min_len]
    lo, hi = 0.0, max(t["s_end"] for t in a + b)
    ag = agree(a, b, 0.0, lo, hi)
    offs = rng.uniform(20, 200, 300) * rng.choice([-1, 1], 300)
    ch = np.array([agree(a, b, o, lo, hi).mean() for o in offs])
    return dict(n600=len(a), n200=len(b), agree_frac=float(ag.mean()), chance_mean=float(ch.mean()),
                p_perm=float((ch >= ag.mean()).mean()),
                agreeing=[{k: t[k] for k in ("s_start", "s_end", "u_mean", "depth_m", "q")} for t, x in zip(a, ag) if x])


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1]), indent=1))
