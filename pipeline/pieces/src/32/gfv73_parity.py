# R3 (rows 28-35) copy of exp/L_gapfv/gfv73_parity.py (sha256 4fa43c2262c4cbe66d5ab6b08206d3d279866c09f1fa19b1b72d54521d4b5050).
# Edits: paths from $TFB_ROOT; build folder $GFV_BUILD_DIR; build references $GFV_REF_VAL / $GFV_REF_PRIV (the FV-INEL March and FV2 April build parquets of this rebuild; default the stored ones); json to $GFV_PARITY_JSON. Logic unchanged.
"""L_gapfv c73p step 1b (copy of gfv_parity.py; folders and references differ): merge the per-panel hook files
(gfv73_build.py --panels P -> c73p/build/t1.csv.P.gaprows.npz) into c73p/build/t1.csv.gaprows.npz and check build parity at the
gap rows: the hook's old-weight arm (asserted in the build == the build's own clip output) must equal the shipped builds' gap rows
bit for bit: validation == artifacts/claude_ref/l_fvinel/mar/build/t1.csv.state.parquet (FV-INEL March build, written by
exp/L_fvinel/fvi_finish.sh with fvi_tune_mar.json; its non-gap validation rows ship in C73), private ==
artifacts/claude_ref/l_fvft/fv2/apr/build/t1.csv.state.parquet (FV2 April build, fv2_tune_apr.json; non-gap private rows ship in C73P).
Keys must equal the S2b hook's. Information (method check, as L_gapfv attempt 2): the S2b hook old arm vs the private rows of the
FV-INEL March build (its private arm = S2b arm48k + S7b April models) at the gap rows.
Usage (repo root): python exp/L_gapfv/gfv73_parity.py"""
import json, sys, os
from pathlib import Path
import numpy as np
import pandas as pd
CR = Path(os.environ["TFB_ROOT"]) / "artifacts/claude_ref"; BD = Path(os.environ["GFV_BUILD_DIR"])
SPLITS = ("validation", "private")
sys.path.insert(0, os.path.join(os.environ["TFB_ROOT"], "exp/C_gap2c47")); sys.path.insert(0, os.path.join(os.environ["TFB_ROOT"], "docs/reference_code")); sys.path.insert(0, os.path.join(os.environ["TFB_ROOT"], "src"))
S2B = np.load(CR / "l_gap2/build/t1.csv.gaprows.npz")
keys = [str(k) for k in S2B["keys"]]
FL = ("d", "t", "s", "v_old", "q_old", "v_new", "q_new", "chain_v", "chain_q", "spec_v", "spec_q", "cls", "cs")
cap = {}
for f in sorted(BD.glob("t1.csv.*.gaprows.npz")):
    z = np.load(f)
    for k in z["keys"]:
        k = str(k); assert k not in cap, (k, "twice")
        cap[k] = {fl: z[f"{k}|{fl}"] for fl in FL}
assert sorted(cap) == sorted(keys), ("hook keys != S2b hook keys", sorted(cap), keys)
np.savez_compressed(BD / "t1.csv.gaprows.npz", keys=np.array(sorted(cap)), **{f"{k}|{fl}": cap[k][fl] for k in sorted(cap) for fl in cap[k]})
s2 = {k: {fl: S2B[f"{k}|{fl}"] for fl in ("d", "t", "s", "v_old", "q_old")} for k in keys}
km = np.load(CR / "xfer/keymap.npz"); sid, st, ksplit, kpanel = km["sid"], km["state"], km["split"], km["panel"]
kd, kt, ks = km["d"].astype(np.int64), km["t"].astype(np.int64), km["s"].astype(np.int64)
spos = np.cumsum(st) - 1
import g2_build as B
assert tuple(B.SPLITS) == SPLITS
FVI = Path(os.environ.get("GFV_REF_VAL") or CR / "l_fvinel/mar/build/t1.csv.state.parquet")
REF = {"validation": FVI, "private": Path(os.environ.get("GFV_REF_PRIV") or CR / "l_fvft/fv2/apr/build/t1.csv.state.parquet")}


def rows_of(sp, k, z):
    si, pi = B.SPLITS.index(sp), B.PANELS.index(k.split("|")[1])
    sel = np.flatnonzero(st & (ksplit == si) & (kpanel == pi))
    kk = kd[sel] * 1000000 + kt[sel] * 1000 + ks[sel]; o = np.argsort(kk)
    qk = z["d"].astype(np.int64) * 1000000 + z["t"].astype(np.int64) * 1000 + z["s"].astype(np.int64)
    j = o[np.searchsorted(kk[o], qk)]; assert (kk[j] == qk).all()
    return spos[sel[j]]


def compare(path, sp, src):
    R = pd.read_parquet(path); assert np.array_equal(R.submission_id.to_numpy(np.int64), sid[st].astype(np.int64))
    rv, rq = R.speed_kmh.to_numpy(float), R.flow_vph.to_numpy(float); del R
    n = bad = 0
    for k in sorted(src):
        if k.split("|")[0] != sp:
            continue
        z = src[k]; rows = rows_of(sp, k, z)
        bad += int(((z["v_old"] != rv[rows]) | (z["q_old"] != rq[rows])).sum()); n += len(rows)
    return dict(gap_rows=n, mismatches=bad, ref=str(path).replace("\\", "/"))


res = {sp: compare(REF[sp], sp, cap) for sp in SPLITS}
print("GFV73_BUILD_PARITY", json.dumps(res), flush=True)
info = compare(FVI, "private", s2)
print("GFV73_METHOD_CHECK (info) S2b hook old arm vs FV-INEL March build private gap rows:", json.dumps(info),
      "EXACT" if info["mismatches"] == 0 else "DIFFERS", flush=True)
assert all(v["mismatches"] == 0 for v in res.values()) and sum(v["gap_rows"] for v in res.values()) == 128465, "FAIL"
res["method_check_info"] = dict(info, what="S2b hook old arm == FV-INEL March build private gap rows (S2b arm48k + S7b April models)")
Path(os.environ["GFV_PARITY_JSON"]).write_text(json.dumps(res, indent=1))
print("GFV73_BUILD_PARITY EXACT")
