"""gap2 step on the official gap rows + streamed write over the parent (rows 25 and 26; worker R2 2026-10-21).
Values: COPY of the build part of exp/L_b6/c0_official.py (sha256 27144d173cbc770a4f47a5cfe1370b893e9dd476519c6fa50f728770a5459aa5):
per split/panel, Stages(["gap", "clip", "gap2"]).gap2_rows (exp/C_gap2c47/asm2_build_gap2w.py, C48 gap2 step: clip((1 - w) row +
w mean(3 gap2 U-Net members, smooth 1)), weights_mar.json m3) applied to the 128,465 official gap&target rows of STATE.parquet;
every other line of PARENT copied byte for byte; changed lines = a[0],a[1],repr(v),repr(q),a[4] (as c0_official).
Row 26 option --nongap NEW.parquet [--ref REF.parquet]: also every non-gap state line takes NEW's values, written the same way
(= exp/L_b6/bundle18_stack.py, sha256 8d330c75254b2a0915fa6cff1f87c01fe22396ecbbf1d5b53b5d317ea821e99e, which rewrote every
non-gap state line of its FILE12 with repr(NEW)); with --ref, its check b: NEW gap rows == REF gap rows exactly.
Differences to the originals (no effect on values): c0_official's two parity checks are left out (control build
l_micro/a2 parquet through the step == C59 gap rows; parent gap rows == C59 gap rows): they tie the step to row 20/23 files and
are covered by this piece's parity runner (output sha). The member folder and m3 weights are arguments (defaults = the stored
artifacts/claude_ref/c_gap2c47/pred_cpu and weights_mar.json, i.e. row 19's served members and weights).
Usage: python gap2_step.py PARENT.csv STATE.parquet OUT.csv [--members DIR] [--gap2w FILE] [--nongap NEW.parquet] [--ref REF.parquet]"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, hashlib, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT = Path(os.environ.get("TFB_OLD_REPO") or Path(__file__).resolve().parents[5]).resolve()
G2SRC = Path(os.environ.get("GAP2W_SRC") or ROOT / "exp/C_gap2c47")
for p in (str(ROOT / "src"), str(ROOT / "docs/reference_code"), str(G2SRC)):
    sys.path.insert(0, p)
CR = ROOT / "artifacts/claude_ref"
N = 6980503; N_GAP2 = 128465


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def state_rows(path, st, sid):
    """speed/flow of the state rows (full-file order); asserts the row ids follow the keymap (c0_official.state_rows)."""
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


def main(parent, state_parquet, outp, members=None, gap2w=None, nongap=None, ref=None):
    t0 = time.time()
    import g2_build as B
    import asm2_build_gap2w as AB
    if members:
        AB.GAP2_PRED = Path(members).resolve()
    if gap2w:
        AB.GAP2_W = Path(gap2w).resolve()
    _in0 = AB.Stages._in

    def _in(s, p):   # files outside the development repo (clean machine): no ROOT-relative name, same file read
        p = Path(p)
        try:
            p.resolve().relative_to(AB.ROOT.resolve()) if p.is_absolute() else None
        except ValueError:
            s.inputs[str(p)] = sha(p); return p
        return _in0(s, p)
    AB.Stages._in = _in
    km = np.load(CR / "xfer/keymap.npz")
    sid, st, gapk, ksplit, kpanel = km["sid"], km["state"], km["gap"], km["split"], km["panel"]
    kd, kt, ks = km["d"].astype(int), km["t"].astype(int), km["s"].astype(int)
    assert len(sid) == N; gap = gapk & st; assert int(gap.sum()) == N_GAP2
    spos = np.cumsum(st) - 1
    pv, pq_ = state_rows(parent, st, sid)
    new_v, new_q = parquet_vq(state_parquet, sid[st])
    S = AB.Stages(["gap", "clip", "gap2"], None)
    nv, nq = pv.copy(), pq_.copy()
    for si, sp in enumerate(B.SPLITS):
        for pi, pn in enumerate(B.PANELS):
            sel = np.flatnonzero(st & (ksplit == si) & (kpanel == pi))
            if not len(sel):
                continue
            cs = B.clipvec(pn)[ks[sel]]; g = gap[sel]
            v0, q0 = new_v[spos[sel]], new_q[spos[sel]]
            ov, oq = S.gap2_rows(sp, pn, kd[sel], kt[sel], ks[sel], v0, q0, cs)
            assert np.array_equal(ov[~g], v0[~g]) and np.array_equal(oq[~g], q0[~g]), (sp, pn, "step changed a non-gap row")
            nv[sel[g]] = ov[g]; nq[sel[g]] = oq[g]
    assert np.isfinite(nv[gap]).all() and np.isfinite(nq[gap]).all() and (nv[gap] > 0).all() and (nq[gap] >= 0).all()
    ng = st & ~gap
    if nongap:
        xv, xq = parquet_vq(nongap, sid[st].astype(np.int64))
        if ref:
            rv, rq = parquet_vq(ref, sid[st].astype(np.int64)); gs = gap[st]
            assert np.array_equal(rv[gs], xv[gs]) and np.array_equal(rq[gs], xq[gs]), "b: NEW gap rows differ from REF"
            print("STEP check b: NEW gap rows == REF gap rows EXACT", flush=True); del rv, rq
        nv[ng] = xv[spos[ng]]; nq[ng] = xq[spos[ng]]; del xv, xq
    rw = gap | ng if nongap else gap
    outp = Path(outp); outp.parent.mkdir(parents=True, exist_ok=True); tmp = Path(str(outp) + ".tmp")
    rl = rw.tolist(); nvl, nql = nv.tolist(), nq.tolist()
    with open(parent, "rb") as f, open(tmp, "wb") as o:
        o.write(f.readline())
        for i, line in enumerate(f):
            if rl[i]:
                a = line.split(b",", 4); eol = b"\r\n" if line.endswith(b"\r\n") else b"\n"
                rest = a[4]
                o.write(b",".join([a[0], a[1], repr(nvl[i]).encode(), repr(nql[i]).encode(), rest]))
                assert rest.endswith(eol)
            else:
                o.write(line)
    os.replace(tmp, outp)
    wv, wq = state_rows(outp, st, sid)
    assert np.array_equal(wv[rw], nv[rw]) and np.array_equal(wq[rw], nq[rw]), "written rows differ"
    with open(parent, "rb") as f, open(outp, "rb") as g_:
        f.readline(); g_.readline()
        for i, (a, b) in enumerate(zip(f, g_)):
            if not rl[i] and a != b:
                raise SystemExit(f"line {i} changed outside the rewritten rows")
    chg = ((nv != pv) | (nq != pq_)) & st
    info = dict(out=str(outp), sha256=sha(outp), parent=str(parent), parent_sha256=sha(parent), state_parquet=str(state_parquet),
                nongap=str(nongap) if nongap else None, rows_changed=int(chg.sum()), gap_rows_changed=int((chg & gap).sum()),
                members=str(AB.GAP2_PRED), gap2_weights=str(AB.GAP2_W), seconds=round(time.time() - t0))
    Path(str(outp) + ".info.json").write_text(json.dumps(info, indent=1))
    print("GAP2_STEP_DONE", json.dumps(info), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    def opt(f):
        if f in a:
            i = a.index(f); v = a[i + 1]; del a[i:i + 2]; return v
    m, w, n, r = opt("--members"), opt("--gap2w"), opt("--nongap"), opt("--ref")
    main(a[0], a[1], a[2], m, w, n, r)
