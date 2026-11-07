# R3 (rows 28-35) parameterised copy of exp/L_isinel/isi_stack.py (sha256 5c81e279833002cd71bee38869e5032eb02f7affbdd0b076c10f324181fa4960; = fvi_stack.py / fv2_stack.py body; the writing
# loop is the one of mar_stack.py / apr_stack.py too). Edits: ROOT = $TFB_ROOT; no output folder check; PARITY.json may be 'none'
# (then no parity json is required: the piece script checks the build itself); optional 7th arg BREF = the file check b reads
# (apr_stack.py form: the other split is checked against C66, not BASE); tags R3_. The written bytes are unchanged.
"""L_isinel copy of exp/L_fvinel/fvi_stack.py (outdir check l_isinel, line tags R3_; parity takes one parquet per split: parity VALPARQ PRIVPARQ OUT.json VAL_REF PRIV_REF). fvi_stack.py text: L_fvinel copy of exp/L_fvft/fv2_stack.py (outdir check l_fvinel, line tags R3_; parity text: the BASE arm). fv2_stack.py text: L_fvft FV2 (mar_stack.py / apr_stack.py form, split given): candidate = BASE with ONLY the non-gap state rows of SPLIT from a build.
  parity PARITY.parquet OUT.json VAL_REF PRIV_REF
      build parity: the non-gap state rows of the parity build == the reference file's non-gap state rows EXACTLY, per split
      (REF = a submission csv or a .state.parquet). March: both refs = C70 (build with the it54000 FV arm, mar_tune.json).
      April: VAL_REF = c_fv2_mar_on_c70.csv, PRIV_REF = c_fvft_both_on_c69.csv (build with the April it54000 FV arm).
  stack BASE.csv CAND.parquet PARITY.json SPLIT OUTDIR NAME
      needs PARITY.json ok and its reference for the OTHER split == BASE (sha); checks:
        b  CAND non-gap state rows of the other split == BASE EXACTLY (determinism: that arm is the parity build's arm)
        c  only SPLIT non-gap state lines change vs BASE, written values == CAND, every other line byte for byte
      Writes OUTDIR/NAME.csv + NAME.info.json."""
import os, sys, json, hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT = Path(os.environ["TFB_ROOT"])
N = 6980503


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def keys():
    km = np.load(ROOT / "artifacts/claude_ref/xfer/keymap.npz")
    sid, st, gapk, spl = km["sid"], km["state"], km["gap"], km["split"]
    gap = gapk & st
    return sid, st, gap, spl == 0


def vq(path, sid_state):
    t = pq.read_table(path, columns=["submission_id", "speed_kmh", "flow_vph"])
    assert np.array_equal(np.asarray(t.column("submission_id").to_numpy(), np.int64), sid_state), path
    return t.column("speed_kmh").to_numpy().astype(float), t.column("flow_vph").to_numpy().astype(float)


def csv_vq(path, rows, sid):
    """speed, flow of the csv lines where rows (bool over the N lines) is True, in line order (ids checked)."""
    rl = rows.tolist(); v, q = [], []; sl = sid.tolist()
    with open(path, "rb") as f:
        f.readline()
        for i, line in enumerate(f):
            if rl[i]:
                a = line.split(b",", 4)
                assert int(a[0]) == sl[i], f"{path}: line {i} id differs from the keymap"
                v.append(float(a[2])); q.append(float(a[3]))
    assert i == N - 1
    return np.array(v), np.array(q)


def ref_vq(path, rows, sid, st):
    """values of a reference (csv or state parquet) on rows (bool over the N lines, a subset of the state rows)."""
    if str(path).endswith(".parquet"):
        v, q = vq(path, sid[st].astype(np.int64)); k = (np.cumsum(st) - 1)[rows]
        return v[k], q[k]
    return csv_vq(path, rows, sid)


def parity(parv, parp, outp, val_ref, priv_ref):
    sid, st, gap, val = keys(); ng = st & ~gap; spos = np.cumsum(st) - 1
    res = dict(rule="non-gap state rows of the split's base build parquet == the reference's non-gap state rows exactly, per split", parity={"validation": str(parv), "private": str(parp)},
               parity_sha256={"validation": sha(parv), "private": sha(parp)})
    for nm, sel, ref, par in (("validation", ng & val, val_ref, parv), ("private", ng & ~val, priv_ref, parp)):
        pv, pq_ = vq(par, sid[st].astype(np.int64))
        x, y = ref_vq(ref, sel, sid, st); k = spos[sel]; a, b = pv[k], pq_[k]
        res[nm] = dict(ref=str(ref).replace("\\", "/"), ref_sha256=sha(ref), rows=int(sel.sum()), exact=bool(np.array_equal(a, x) and np.array_equal(b, y)),
                       n_diff=int(((a != x) | (b != y)).sum()), max_dv=float(np.abs(a - x).max()), max_dq=float(np.abs(b - y).max()))
        print("R3_BUILD_PARITY", nm, json.dumps(res[nm]), flush=True)
    res["ok"] = res["validation"]["exact"] and res["private"]["exact"]
    Path(outp).write_text(json.dumps(res, indent=1))
    print("R3_BUILD_PARITY", "PASS" if res["ok"] else "FAIL", flush=True)
    sys.exit(0 if res["ok"] else 1)


def stack(base, cand, parj, split, outdir, name, bref=None):
    outdir = Path(outdir); assert split in ("validation", "private")
    other = "private" if split == "validation" else "validation"
    bsha = sha(base); pj = None
    if parj != "none":
        pj = json.loads(Path(parj).read_text())
        assert pj["ok"], "build parity not PASS"
        assert pj[other]["ref_sha256"] == bsha, f"the parity reference for the {other} rows is not BASE"
    sid, st, gap, val = keys(); ng = st & ~gap
    sv = val if split == "validation" else ~val
    take = ng & sv; spos = np.cumsum(st) - 1
    nv, nq = vq(cand, sid[st].astype(np.int64))
    ot = ng & ~sv
    bv, bq = csv_vq(bref or base, ot, sid); j = spos[ot]
    assert np.array_equal(nv[j], bv) and np.array_equal(nq[j], bq), f"b: CAND {other} non-gap state rows differ from {bref or 'BASE'}"
    print(f"R3_STACK check b: CAND {other} non-gap state rows == BASE EXACT ({int(ot.sum())} rows)", flush=True)
    del bv, bq
    tl = take.tolist(); sl = sid.tolist(); spl_ = spos.tolist(); nvl, nql = nv.tolist(), nq.tolist()
    full = outdir / f"{name}.csv"; tmp = outdir / f"{name}.csv.tmp"; assert not full.exists()
    changed = 0; old_v, old_q = [], []
    with open(base, "rb") as f, open(tmp, "wb") as o:
        o.write(f.readline())
        for i, line in enumerate(f):
            if tl[i]:
                a = line.split(b",", 4)
                assert int(a[0]) == sl[i]
                k = spl_[i]; ov, oq = float(a[2]), float(a[3]); old_v.append(ov); old_q.append(oq)
                if nvl[k] != ov or nql[k] != oq:
                    changed += 1
                o.write(b",".join([a[0], a[1], repr(nvl[k]).encode(), repr(nql[k]).encode(), a[4]]))
            else:
                o.write(line)
    assert i == N - 1
    os.replace(tmp, full)
    with open(base, "rb") as f, open(full, "rb") as g:
        assert f.readline() == g.readline()
        for i, (x, y) in enumerate(zip(f, g)):
            if not tl[i] and x != y:
                raise SystemExit(f"c: line {i} changed outside the {split} non-gap state rows")
            if tl[i]:
                b = y.split(b",", 4); k = spl_[i]
                assert float(b[2]) == nvl[k] and float(b[3]) == nql[k]
    assert i == N - 1
    print(f"R3_STACK check c: only {split} non-gap state lines changed vs BASE; written values == CAND", flush=True)
    k = spos[take]
    dv = np.abs(nv[k] - np.array(old_v)); dq = np.abs(nq[k] - np.array(old_q))
    info = dict(file=str(full.resolve()).replace("\\", "/"), bref=str(bref or base).replace("\\", "/"), sha256=sha(full), split=split, parent=str(Path(base)).replace("\\", "/"),
                parent_sha256=bsha, cand=str(cand).replace("\\", "/"), cand_sha256=sha(cand), parity=pj, rows_taken=int(take.sum()), rows_changed=changed,
                mean_abs_dv=float(dv.mean()), mean_abs_dq=float(dq.mean()), max_abs_dv=float(dv.max()), max_abs_dq=float(dq.max()),
                checks=["parity: build path with the BASE arm == BASE non-gap state rows EXACT (both splits)",
                        f"b CAND {other} non-gap state rows == BASE EXACT (determinism)", f"c only {split} non-gap state lines changed vs BASE"])
    (outdir / f"{name}.info.json").write_text(json.dumps(info, indent=1))
    print("R3_STACK_DONE", full, info["sha256"], json.dumps({k_: info[k_] for k_ in ("rows_taken", "rows_changed", "mean_abs_dv", "mean_abs_dq", "max_abs_dv", "max_abs_dq")}), flush=True)


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    dict(parity=parity, stack=stack)[cmd](*a)
