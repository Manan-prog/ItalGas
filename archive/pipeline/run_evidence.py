"""Clean + line-evidence for one cube file. Usage: python run_evidence.py cube.npz tag"""
import sys, time
import numpy as np
from detect import resample_cube, remove_layers, evidence_pipeline
cube_file, tag = sys.argv[1], sys.argv[2]
z = np.load(cube_file)
for arr in (1, 2):
    t0 = time.time()
    env = z[f"env{arr}"]; dt = float(z[f"dt{arr}"])
    f = max(1, int(round(2.0 / dt))); nt = env.shape[2] // f
    env = env[:, :, :nt * f].reshape(env.shape[0], env.shape[1], nt, f).mean(3)
    cube, sg, ug = resample_cube(env, z[f"s{arr}"], z[f"u{arr}"], 0.10, 0.08, (-0.58, 0.93))
    cl = remove_layers(cube, 0.10, 3.0)
    dtv = dt * f; jmax = int((45 if arr == 1 else 70) / dtv); j0 = int(4 / dtv); cl = cl[:, :, j0:jmax]
    res = evidence_pipeline(cl, 0.10, 0.08, line_len_m=2.0, sigmas=(1.0, 1.5), seed=0)
    np.savez(f"/Users/manansarupria/ItalGas/cache/prob_{tag}_arr{arr}.npz", evidence=res["evidence"], angle=res["angle"],
             clean=cl, sg=sg, ug=ug, dt=dtv, t_offset_ns=j0 * dtv)
    print(tag, arr, f"{time.time()-t0:.1f}s", flush=True)
