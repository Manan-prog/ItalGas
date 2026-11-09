"""E47: which drives are usable repeats of which (tracks 0.5-1.2 m apart for >= 50 m continuous), components and references.
Tracks: cache/swaths.npz (real-time, EPSG:3857), resampled every 1 m, distances x cos(lat).
Run: python -m italgas_gpr.lanes.pairs -> cache/city/lane_components.json"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from italgas_gpr.io.paths import CACHE  # noqa: E402

LO, HI, MIN_RUN = 0.5, 1.2, 50


def main() -> None:
    d = np.load(CACHE / "swaths.npz"); k = float(np.cos(np.radians(44.3)))
    R = {}
    for x in d.files:
        t = d[x]
        if not x.startswith("s") or len(t) < 2:
            continue
        seg = np.hypot(np.diff(t[:, 0]), np.diff(t[:, 1])) * k; s = np.r_[0, np.cumsum(seg)]
        if s[-1] < 100:
            continue
        g = np.arange(0, s[-1], 1.0); R[int(x[1:])] = np.c_[np.interp(g, s, t[:, 0]), np.interp(g, s, t[:, 1])]
    trees = {i: cKDTree(v) for i, v in R.items()}
    pairs = {}
    for i, v in R.items():
        for j, tr in trees.items():
            if j == i:
                continue
            dist, _ = tr.query(v, distance_upper_bound=HI / k * 1.01); dist = dist * k
            ok = (dist >= LO) & (dist <= HI)
            # longest continuous run of usable offset (m)
            best = cur = 0
            for o in ok:
                cur = cur + 1 if o else 0; best = max(best, cur)
            if best >= MIN_RUN:
                pairs[(i, j)] = dict(usable_m=int(ok.sum()), longest_run_m=int(best))
    # symmetric pairs -> components (union-find)
    parent = {i: i for i in R}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for (i, j) in pairs:
        if (j, i) in pairs:
            parent[find(i)] = find(j)
    comps = {}
    for i in R:
        comps.setdefault(find(i), []).append(i)
    comps = [sorted(c) for c in comps.values() if len(c) > 1]
    out = []
    for c in comps:
        ref = max(c, key=lambda i: len(R[i]))
        partners = [j for j in c if j != ref and ((ref, j) in pairs or (j, ref) in pairs)]
        indirect = [j for j in c if j != ref and j not in partners]
        out.append(dict(reference=ref, drives=[ref] + partners, indirect=indirect, length_km=round(len(R[ref]) / 1e3, 2),
                        usable_m={str(j): pairs.get((ref, j), pairs.get((j, ref)))["usable_m"] for j in partners}))
    tot = sum(len(v) for v in R.values()); n_usable = 0; per_drive = {}
    for i, v in R.items():
        dmin = np.full(len(v), np.inf); dany = np.full(len(v), np.inf)
        for j, tr in trees.items():
            if j != i:
                dd, _ = tr.query(v, distance_upper_bound=HI / k * 1.01); dmin = np.minimum(dmin, dd * k)
                da, _ = tr.query(v, distance_upper_bound=3.0 / k); dany = np.minimum(dany, da * k)
        u_ = int(((dmin >= LO) & (dmin <= HI)).sum()); n_usable += u_
        per_drive[str(i)] = dict(length_m=len(v), usable_repeat_m=u_, any_repeat_3m_m=int((dany <= 3.0).sum()),
                                 partners=sorted({b for (a, b) in pairs if a == i} | {a for (a, b) in pairs if b == i}))
    res = dict(n_drives=len(R), total_km=round(tot / 1e3, 1), usable_km=round(n_usable / 1e3, 1), n_pairs=len(pairs) // 2, components=sorted(out, key=lambda c: -c["length_km"]),
               per_drive=per_drive, rule=dict(usable_offset_m=[LO, HI], min_run_m=MIN_RUN, any_repeat_within_m=3.0))
    json.dump(res, open(CACHE / "city" / "lane_components.json", "w"), indent=1)
    print(f"drives {len(R)}, total {tot/1e3:.1f} km, usable-repeat length {n_usable/1e3:.1f} km, pairs {len(pairs)//2}, components {len(out)}, drives in components {sum(len(c['drives']) for c in out)} (+{sum(len(c['indirect']) for c in out)} indirect)")
    for c in res["components"][:12]:
        print(c["reference"], c["drives"], c["length_km"], "km", c["usable_m"])


if __name__ == "__main__":
    main()
