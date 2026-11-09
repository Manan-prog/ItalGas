"""Shape test of repeated along-road features in the multi-pass mosaic vs random control positions
(same depths, same coverage). Run: python classify_repeated.py configs/multipass_aurelia.yaml
Output: cache/mp_shape_test.json"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

from along_road import pipe_shape_test
from build_v2 import load_cfg
from register_multipass import cross_section

ROOT = Path(__file__).resolve().parents[1]


def main(cfg_path: str) -> None:
    mc = yaml.safe_load(open(cfg_path)); det = load_cfg(str(ROOT / "pipeline" / mc["detector_config"]))
    cache = Path(det["base"]["cache_dir"]); v = det["migration"]["velocity_m_ns"]; rng = np.random.default_rng(det["seed"])
    reg = json.load(open(cache / "mp_registration.json"))["registration"]
    feats = json.load(open(cache / "mp_repeated_features.json"))
    D = {n: np.load(cache / f"mp_{n}_2.npz") for n in reg}
    s_grid = D[next(iter(D))]["s_grid"]
    u_grid = np.arange(mc["mosaic"]["u_range_m"][0], mc["mosaic"]["u_range_m"][1] + 1e-9, mc["mosaic"]["du_m"])
    SL = mc["mosaic"]["depth_slices_m"]
    out = dict(features=[], controls=[])
    for f in feats:
        sm = 0.5 * (f["s0"] + f["s1"]); X, z = cross_section(D, reg, s_grid, u_grid, sm, v)
        r = pipe_shape_test(X, u_grid, z, f["u"], float(np.mean(SL[f["zi"]])))
        out["features"].append(dict(f, s_mid=sm, **r))
    # controls: random (s, u) inside the covered mosaic, depth slices drawn like the features'
    zis = [f["zi"] for f in feats]
    tried = 0
    while len(out["controls"]) < 300 and tried < 3000:
        tried += 1
        sm = float(rng.uniform(s_grid[0] + 5, s_grid[-1] - 5)); X, z = cross_section(D, reg, s_grid, u_grid, sm, v)
        cov = np.nonzero(np.isfinite(X).all(0))[0]
        if len(cov) < 25:
            continue
        u0 = float(u_grid[rng.choice(cov[5:-5])]) if len(cov) > 12 else float(u_grid[cov[len(cov) // 2]])
        r = pipe_shape_test(X, u_grid, z, u0, float(np.mean(SL[int(rng.choice(zis))])))
        if r["verdict"] in ("pipe-like", "layer-like"):
            out["controls"].append(dict(s_mid=sm, u=u0, **{k: r[k] for k in ("verdict", "rms_hyperbola_m", "rms_plane_m")}))
    pf = np.mean([x["verdict"] == "pipe-like" for x in out["features"] if x["verdict"] in ("pipe-like", "layer-like")])
    pc = np.mean([x["verdict"] == "pipe-like" for x in out["controls"]])
    nF = sum(x["verdict"] in ("pipe-like", "layer-like") for x in out["features"]); nC = len(out["controls"])
    kF = sum(x["verdict"] == "pipe-like" for x in out["features"])
    from scipy.stats import binomtest
    p = binomtest(kF, nF, pc, alternative="greater").pvalue if nF else None
    out["summary"] = dict(n_features_tested=nF, pipe_like_features=kF, frac_features=float(pf), n_controls=nC, frac_controls=float(pc), p_binomial=p)
    print(out["summary"])
    for x in sorted(out["features"], key=lambda x: -x["len_m"])[:10]:
        print(round(x["s0"]), round(x["s1"]), round(x["u"], 2), SL[x["zi"]], x["a"][-3:], x["b"][-3:], x["verdict"],
              round(x.get("rms_hyperbola_m", np.nan), 3), round(x.get("rms_plane_m", np.nan), 3), x.get("uc") and round(x["uc"], 2), x.get("z0") and round(x["z0"], 2))
    json.dump(out, open(cache / "mp_shape_test.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1])
