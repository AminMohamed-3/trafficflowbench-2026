"""Compare a hooked-build npz (gap_hook_build.py) with a stored one, key by key (worker R2 2026-10-21).
Every panel key of A must be in B; the fields compared are those both files hold (stored row-25/26 files hold d,t,s,v_old,q_old,
v_new,q_new; row-27's also chain/spec/cls/cs). Exact equality (np.array_equal). Prints one JSON line; exit 0 = EXACT.
Usage: python cmp_hook.py A.npz B.npz"""
import sys, json
import numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
ka = [str(k) for k in a["keys"]]; kb = set(str(k) for k in b["keys"])
fa = {f.split("|", 2)[2] for f in a.files if f != "keys"}; fb = {f.split("|", 2)[2] for f in b.files if f != "keys"}
rep = {"keys": ka, "fields": sorted(fa & fb), "missing_keys": [k for k in ka if k not in kb], "diff": {}}
for k in ka:
    if k not in kb:
        continue
    for f in sorted(fa & fb):
        x, y = a[f"{k}|{f}"], b[f"{k}|{f}"]
        if x.shape != y.shape or not np.array_equal(x, y):
            rep["diff"][f"{k}|{f}"] = "shape" if x.shape != y.shape else float(np.nanmax(np.abs(x.astype(float) - y.astype(float))))
rep["result"] = "EXACT" if not rep["diff"] and not rep["missing_keys"] and ka else "FAIL"
print(json.dumps(rep)); sys.exit(0 if rep["result"] == "EXACT" else 1)
