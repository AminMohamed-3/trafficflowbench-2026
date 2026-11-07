"""Post step of the hooked build (rows 25 and 26; worker R2 2026-10-21). COPY of exp/L_b6/bundle18_official.py
(sha256 f9fddecefc6ed74a097f17fb8fa7f00eac7f62fbbe07099a9af4596c5ce88d97) = exp/L_b6/gapw_official.py (662ffa02) with folders as
arguments. Logic unchanged; parameterised: no output-folder assert, the new parquet name is an argument (row 25 t1_gapw, row 26
t1_gapwf), check 1 (rebuild == REF parquet, the determinism check against an earlier build) is skipped when REF is "none".
  check 1  BUILD/t1.csv.state.parquet = REF parquet bit for bit
  check 2  hook old-weight gap rows = build parquet gap rows bit for bit; each of the 128,465 official gap&target rows once
  write    OUTDIR/t1_old.state.parquet (gap rows <- old arm) and OUTDIR/NEWNAME.state.parquet (gap rows <- GAPW W3 arm)
Usage: python gap_post.py BUILD_PREFIX REF_PARQUET|none OUTDIR NEWNAME
  (BUILD_PREFIX = the OUT given to gap_hook_build.py; reads BUILD_PREFIX.state.parquet and BUILD_PREFIX.gaprows.npz)"""
import os, sys, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(os.environ.get("TFB_OLD_REPO") or Path(__file__).resolve().parents[5]).resolve()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "exp/C_gap2c47"))
CR = ROOT / "artifacts/claude_ref"
N_GAP2 = 128465


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main(B, REF, O, NEW):
    import g2_build as GB
    O = Path(O); O.mkdir(parents=True, exist_ok=True)
    P = pd.read_parquet(B + ".state.parquet")
    if REF != "none":
        S = pd.read_parquet(REF)
        same = all(np.array_equal(P[c].to_numpy(), S[c].to_numpy()) for c in ("submission_id", "speed_kmh", "flow_vph")) and list(P.columns) == list(S.columns)
        print("POST_CHECK1 rebuild == REF build parquet:", "EXACT" if same else "DIFFERENT", flush=True)
        if not same:
            sys.exit(2)
        del S
    else:
        print("POST_CHECK1 skipped (no REF)", flush=True)
    km = np.load(CR / "xfer/keymap.npz")
    sid, st, gapk, ksplit, kpanel = km["sid"], km["state"], km["gap"], km["split"], km["panel"]
    kd, kt, ks = km["d"].astype(np.int64), km["t"].astype(np.int64), km["s"].astype(np.int64)
    gap = gapk & st; assert int(gap.sum()) == N_GAP2
    assert np.array_equal(P.submission_id.to_numpy(np.int64), sid[st].astype(np.int64))
    spos = np.cumsum(st) - 1
    z = np.load(B + ".gaprows.npz")
    v = P.speed_kmh.to_numpy(float); q = P.flow_vph.to_numpy(float)
    arms = {a: (v.copy(), q.copy()) for a in ("old", "new")}
    hit = np.zeros(len(sid), int); per = {}
    for key in z["keys"]:
        sp, pn = str(key).split("|"); si, pi = GB.SPLITS.index(sp), GB.PANELS.index(pn)
        full = np.flatnonzero(gap & (ksplit == si) & (kpanel == pi))
        kk = kd[full] * 1000000 + kt[full] * 1000 + ks[full]; o = np.argsort(kk)
        qk = z[f"{key}|d"].astype(np.int64) * 1000000 + z[f"{key}|t"].astype(np.int64) * 1000 + z[f"{key}|s"].astype(np.int64)
        j = o[np.searchsorted(kk[o], qk)]; assert (kk[j] == qk).all() and len(np.unique(j)) == len(j) == len(full), key
        rows = full[j]; hit[rows] += 1; sp_ = spos[rows]
        for a in ("old", "new"):
            arms[a][0][sp_] = z[f"{key}|v_{a}"]; arms[a][1][sp_] = z[f"{key}|q_{a}"]
        assert np.array_equal(z[f"{key}|v_old"], v[sp_]) and np.array_equal(z[f"{key}|q_old"], q[sp_]), (key, "old arm != build parquet")
        per[str(key)] = dict(rows=len(rows), mean_abs_dv=float(np.abs(z[f"{key}|v_new"] - z[f"{key}|v_old"]).mean()),
                             mean_abs_dq=float(np.abs(z[f"{key}|q_new"] - z[f"{key}|q_old"]).mean()))
    assert (hit[gap] == 1).all() and not hit[~gap].any(), "gap row coverage"
    print("POST_CHECK2 old-weight gap rows == build parquet: EXACT; coverage", int(hit.sum()), "of", N_GAP2, flush=True)
    ng = ~gap[st]
    for a in ("old", "new"):
        assert np.array_equal(arms[a][0][ng], v[ng]) and np.array_equal(arms[a][1][ng], q[ng])
    assert np.array_equal(arms["old"][0], v) and np.array_equal(arms["old"][1], q)
    for a, name in (("old", "t1_old"), ("new", NEW)):
        out = pd.DataFrame({"submission_id": P.submission_id.to_numpy(), "speed_kmh": arms[a][0], "flow_vph": arms[a][1]})
        assert out.dtypes.tolist() == P.dtypes.tolist()
        out.to_parquet(O / f"{name}.state.parquet", index=False, compression="zstd")
    dv = np.abs(arms["new"][0] - v)[~ng]; dq = np.abs(arms["new"][1] - q)[~ng]
    s = dict(build=str(B), ref=str(REF), per_panel=per, mean_abs_dv=float(dv.mean()), mean_abs_dq=float(dq.mean()), max_dv=float(dv.max()),
             max_dq=float(dq.max()), changed=int(((arms["new"][0] != v) | (arms["new"][1] != q)).sum()),
             files={n: sha(O / f"{n}.state.parquet") for n in ("t1_old", NEW)}, build_parquet_sha256=sha(B + ".state.parquet"))
    (O / "post_step.json").write_text(json.dumps(s, indent=1))
    print("POST_STEP", json.dumps({k: s[k] for k in ("mean_abs_dv", "mean_abs_dq", "max_dv", "max_dq", "changed")}), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])
