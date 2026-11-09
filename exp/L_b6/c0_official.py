"""L_b6 A5 = C0, official side: the 128,465 gap2 rows rebuilt as the shipped gap2 step (C48, exp/C_gap2c47/asm2_build_gap2w.py
Stages.gap2_rows: clip((1 - w) row + w mean(3 U-Net members, smooth 1)), weights_mar.json m3 = onset v .5 q .2, ongoing v .7 q .35)
applied over the S2 state build (exp/L_b6/s2_build.py, S2 B6R members in the chain value of the gap rows) instead of the C47/C54 build.
  parity  the same step over the C54 control build (artifacts/claude_ref/l_micro/a2/t1.csv.state.parquet = 24k members; its gap rows
          equal the C47 build's) must give C59's 128,465 gap rows BIT FOR BIT; and the parent's gap rows must equal C59's (else HOLD, exit 4)
  build   PARENT (the S2 candidate, or the S2_ADOPTED parent) streamed line by line; only the gap rows get the new speed/flow, every other
          line is copied byte for byte. Out: artifacts/claude_ref/l_b6/c0/NAME.csv, NAME.delta.csv.gz (tools/probe.py make), NAME.info.json
Usage (repo root): python exp/L_b6/c0_official.py PARENT.csv STATE.parquet NAME [--parity-only]"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, hashlib, subprocess, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "docs/reference_code", "exp/C_gap2c47"):
    sys.path.insert(0, str(ROOT / p))
CR = ROOT / "artifacts/claude_ref"; D = Path(os.environ.get("C0_DIR", CR / "l_b6/c0"))   # C0_DIR: smoke tests only
C59 = CR / "l_tinyad/a1/on_c57/delta/c_tinyad_on_c57.csv"; C59SHA = "c9a635fd35917ec279b09d8c034792051896592676848850eb670d89fd72019f"
CTL = CR / "l_micro/a2/t1.csv.state.parquet"
N = 6980503; N_GAP2 = 128465
def rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def state_rows(path, st, sid):
    """speed/flow of the state rows (full-file order); asserts the row ids follow the keymap."""
    v = np.full(len(st), np.nan); q = np.full(len(st), np.nan); stl = st.tolist(); sl = sid.tolist()
    with open(path, "rb") as f:
        f.readline()
        for i, line in enumerate(f):
            a = line.split(b",", 4)
            if int(a[0]) != sl[i]:
                raise SystemExit(f"{path}: row {i} id {a[0]} differs from the keymap")
            if stl[i]:
                v[i] = float(a[2]); q[i] = float(a[3])
    assert i == len(st) - 1
    return v, q


def parquet_vq(path, sid_state):
    t = pq.read_table(path, columns=["submission_id", "speed_kmh", "flow_vph"])
    assert np.array_equal(np.asarray(t.column("submission_id").to_numpy(), np.int64), sid_state), f"{path}: row order differs from the keymap"
    return t.column("speed_kmh").to_numpy().astype(float), t.column("flow_vph").to_numpy().astype(float)


def main(parent, state_parquet, name, parity_only=False):
    t0 = time.time()
    import g2_build as B
    import asm2_build_gap2w as AB
    assert sha(C59) == C59SHA, "C59 sha"
    km = np.load(ROOT / "artifacts/claude_ref/xfer/keymap.npz")
    sid, st, gapk, ksplit, kpanel = km["sid"], km["state"], km["gap"], km["split"], km["panel"]
    kd, kt, ks = km["d"].astype(int), km["t"].astype(int), km["s"].astype(int)
    assert len(sid) == N; gap = gapk & st; assert int(gap.sum()) == N_GAP2
    spos = np.cumsum(st) - 1                                    # full row -> state-row index
    cv59, cq59 = state_rows(C59, st, sid)
    pv, pq_ = state_rows(parent, st, sid)
    # the parent must still carry C59's gap rows (only then is "replace the gap rows" the whole change)
    pg = bool(np.array_equal(pv[gap], cv59[gap]) and np.array_equal(pq_[gap], cq59[gap]))
    if not pg:
        print("C0_HOLD parent gap rows differ from C59 in", int(((pv != cv59) | (pq_ != cq59))[gap].sum()), "rows", flush=True); sys.exit(4)
    ctl_v, ctl_q = parquet_vq(CTL, sid[st])
    new_v, new_q = (ctl_v, ctl_q) if parity_only else parquet_vq(state_parquet, sid[st])
    S = AB.Stages(["gap", "clip", "gap2"], None)
    out = {"ctl": (cv59.copy(), cq59.copy()), "new": (pv.copy(), pq_.copy())}
    per = {}
    for si, sp in enumerate(B.SPLITS):
        for pi, pn in enumerate(B.PANELS):
            sel = np.flatnonzero(st & (ksplit == si) & (kpanel == pi))
            if not len(sel):
                continue
            cs = B.clipvec(pn)[ks[sel]]; g = gap[sel]
            for key, (bv, bq) in (("ctl", (ctl_v, ctl_q)), ("new", (new_v, new_q))):
                v0, q0 = bv[spos[sel]], bq[spos[sel]]
                ov, oq = S.gap2_rows(sp, pn, kd[sel], kt[sel], ks[sel], v0, q0, cs)
                assert np.array_equal(ov[~g], v0[~g]) and np.array_equal(oq[~g], q0[~g]), (sp, pn, key, "step changed a non-gap row")
                out[key][0][sel[g]] = ov[g]; out[key][1][sel[g]] = oq[g]
            dv = np.abs(out["new"][0][sel[g]] - cv59[sel[g]]); dq = np.abs(out["new"][1][sel[g]] - cq59[sel[g]])
            per[f"{sp}|{pn}"] = dict(gap_rows=int(g.sum()), changed=int(((dv > 0) | (dq > 0)).sum()), mean_abs_dv=float(dv.mean()) if len(dv) else 0.,
                                    mean_abs_dq=float(dq.mean()) if len(dq) else 0., max_dv=float(dv.max()) if len(dv) else 0., max_dq=float(dq.max()) if len(dq) else 0.)
    # PARITY: control build (24k members) through the step = C59 gap rows bit for bit
    bad = int(((out["ctl"][0] != cv59) | (out["ctl"][1] != cq59))[gap].sum())
    worst = float(max(np.abs(out["ctl"][0] - cv59)[gap].max(), np.abs(out["ctl"][1] - cq59)[gap].max()))
    print("C0_PARITY official", "EXACT" if bad == 0 else "FAIL", "| gap rows", N_GAP2, "| differ from C59", bad, "| largest", worst, flush=True)
    if bad:
        sys.exit(2)
    if parity_only:
        print("C0_OFFICIAL parity only, %.0f s" % (time.time() - t0), flush=True); return
    nv, nq = out["new"]
    assert np.isfinite(nv[gap]).all() and np.isfinite(nq[gap]).all() and (nv[gap] > 0).all() and (nq[gap] >= 0).all()
    chg = ((nv != pv) | (nq != pq_)) & st
    assert not (chg & ~gap).any()
    # ---- stream the parent; replace only the gap rows
    D.mkdir(parents=True, exist_ok=True); full = D / f"{name}.csv"; assert not full.exists(), full
    tmp = D / f"{name}.csv.tmp"; gl = gap.tolist(); nvl, nql = nv.tolist(), nq.tolist()
    with open(parent, "rb") as f, open(tmp, "wb") as o:
        o.write(f.readline())
        for i, line in enumerate(f):
            if gl[i]:
                a = line.split(b",", 4); eol = b"\r\n" if line.endswith(b"\r\n") else b"\n"
                rest = a[4]
                o.write(b",".join([a[0], a[1], repr(nvl[i]).encode(), repr(nql[i]).encode(), rest]))
                assert rest.endswith(eol)
            else:
                o.write(line)
    os.replace(tmp, full)
    # ---- verify the written file
    wv, wq = state_rows(full, st, sid)
    assert np.array_equal(wv[gap], nv[gap]) and np.array_equal(wq[gap], nq[gap]), "written gap rows differ"
    assert np.array_equal(wv[st & ~gap], pv[st & ~gap]) and np.array_equal(wq[st & ~gap], pq_[st & ~gap]), "non-gap state rows changed"
    with open(parent, "rb") as f, open(full, "rb") as g_:
        f.readline(); g_.readline()
        for i, (a, b) in enumerate(zip(f, g_)):
            if not gl[i] and a != b:
                raise SystemExit(f"line {i} changed outside the gap rows")
    dv = np.abs(nv - pv)[gap]; dq = np.abs(nq - pq_)[gap]
    by_split = {sp: dict(gap_rows=int((gap & (ksplit == si)).sum()), rows_changed=int((chg & (ksplit == si)).sum())) for si, sp in enumerate(B.SPLITS)}
    info = dict(probe=rel(full), sha256=sha(full), parent=rel(parent), parent_sha256=sha(parent), rows=N, gap2_rows=N_GAP2,
                rows_changed=int(chg.sum()), rows_changed_by_split=by_split,
                change_size=dict(mean_abs_dv=float(dv.mean()), q999_dv=float(np.quantile(dv, .999)), max_dv=float(dv.max()),
                                 mean_abs_dq=float(dq.mean()), q999_dq=float(np.quantile(dq, .999)), max_dq=float(dq.max())),
                per_panel=per, non_gap_rows_unchanged=True, queue_odme_rows_unchanged=True, state_parquet=rel(state_parquet),
                parity=dict(control=rel(CTL), step="exp/C_gap2c47/asm2_build_gap2w.py Stages.gap2_rows", rows_differ_from_c59=0),
                gap2_weights=rel(CR / "c_gap2c47/weights_mar.json"), gap2_weights_sha256=sha(CR / "c_gap2c47/weights_mar.json"),
                gap2_members=rel(CR / "c_gap2c47/pred_cpu"), seconds=round(time.time() - t0))
    (D / f"{name}.info.json").write_text(json.dumps(info, indent=1))
    print("C0_PROBE", json.dumps({k: info[k] for k in ("probe", "sha256", "parent", "rows_changed", "rows_changed_by_split", "change_size")}), flush=True)
    subprocess.run([sys.executable, str(ROOT / "tools/probe.py"), "make", str(parent), str(full), str(D / f"{name}.delta.csv.gz")], check=True)
    subprocess.run([sys.executable, str(ROOT / "tools/probe.py"), "diff", str(parent), str(full)], check=True)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0], a[1], a[2], "--parity-only" in sys.argv)
