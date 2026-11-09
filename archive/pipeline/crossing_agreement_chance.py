"""Chance rate of multi-drive crossing agreement on Via Nizza (E33). Same merge rule as the maps (map_layers.merge_drives);
chance = every non-reference drive's crossings shifted along the street by a random 20-200 m (wrapped in the window), 200 draws.
Run: python crossing_agreement_chance.py -> cache/crossing_agreement_chance.json"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
from map_layers import CACHE, merge_drives, multipass_drives  # noqa: E402

W0, W1 = 2700.0, 3640.0


def shifted(DRV, rng, ref):
    out = {}
    for n, d in DRV.items():
        a = 0.0 if n == ref else rng.uniform(20, 200) * rng.choice([-1, 1])
        out[n] = dict(d, X=[dict(o, S0=W0 + (o["S0"] + a - W0) % (W1 - W0)) for o in d["X"] if W0 <= o["S0"] <= W1])
    return out


def main():
    REG = json.load(open(CACHE / "mp_registration.json"))["registration"]; DRV = multipass_drives(REG); ref = list(REG)[0]
    rng = np.random.default_rng(0); res = {}
    rules = {f"n>={k} within {tol} m": (tol, (lambda g, k=k: g["n_drives"] >= k)) for tol in (1.0, 0.5) for k in (2, 3)}
    for name, (tol, f) in rules.items():
        real = sum(map(f, merge_drives(DRV, tol, window=(W0, W1))))
        ch = np.array([sum(map(f, merge_drives(shifted(DRV, rng, ref), tol))) for _ in range(200)])
        res[name] = dict(real=int(real), chance_mean=float(ch.mean()), chance_q95=float(np.quantile(ch, 0.95)), est_false_frac=float(ch.mean() / max(real, 1)))
        print(f"{name}: real {real}, chance {ch.mean():.1f} (q95 {np.quantile(ch, 0.95):.0f}), estimated false fraction {ch.mean() / max(real, 1):.2f}", flush=True)
    both = sum(g["rows"] == 2 for g in merge_drives(DRV, 0.5, window=(W0, W1))); res["both_rows_groups"] = int(both)
    print(f"groups with both radar rows in a drive: {both} (a within-drive agreement, validated separately in E13; drive shifts cannot test it)")
    json.dump(res, open(CACHE / "crossing_agreement_chance.json", "w"), indent=1)


if __name__ == "__main__":
    main()
