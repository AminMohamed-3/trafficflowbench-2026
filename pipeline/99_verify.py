"""99_verify.py: compare a rebuilt submission file with the submitted one (worker R1, 2026-10-21).

  python pipeline/99_verify.py CANDIDATE.csv --ref FULL.csv [--ref-sha HEX] [--state-atol 1e-8] [--json OUT.json]
  python pipeline/99_verify.py CANDIDATE.csv --ref-parent PARENT.csv --ref-delta DELTA.csv.gz --ref-sha HEX ...

The reference may be stored as a parent file + a delta (tools/probe.py layout: submission_id + changed columns): it is
rebuilt in memory and, unless --no-ref-sha-check, its bytes (written as tools/probe.py expand writes them: round-trip
floats, the parent's line ending) are hashed and must equal --ref-sha, so the comparison is against the submitted file.
Reports:
  1. sha256 of the candidate as written, and with its line endings normalised to CRLF and to LF; BYTE_EQUAL if one of
     them equals the reference sha;
  2. per task (state, queue, odme) and value column: rows that differ, max and mean |difference|, rows over 1e-6 / .1 / 1;
  3. PASS rule (the build's own --check rule, docs/reference_code/asm2_final.py): every queue and ODME value exact, state
     speed_kmh / flow_vph within --state-atol (default 1e-8), identical ids, row order, tasks and missing-value pattern.
Last line: VERIFY_PASS ... or VERIFY_FAIL ... . Exit code 0 on PASS.
"""
import argparse, hashlib, io, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

VALUE_COLS = ["speed_kmh", "flow_vph", "queue_pred", "path_flow"]


class HashWriter(io.TextIOBase):
    def __init__(self): self.h = hashlib.sha256()
    def write(self, s): self.h.update(s.encode("utf-8")); return len(s)


def file_shas(path):
    native, crlf, lf = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    tail = b""
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            native.update(chunk)
            buf = tail + chunk
            tail = b"\r" if buf.endswith(b"\r") else b""   # a CR split from its LF is kept for the next chunk
            if tail:
                buf = buf[:-1]
            norm = buf.replace(b"\r\n", b"\n")
            lf.update(norm); crlf.update(norm.replace(b"\n", b"\r\n"))
        if tail:
            lf.update(tail); crlf.update(tail)
    return dict(native=native.hexdigest(), crlf=crlf.hexdigest(), lf=lf.hexdigest())


def line_ending(path):
    with open(path, "rb") as f:
        return "\r\n" if b"\r\n" in f.read(4096) else "\n"


def read_full(path):
    df = pd.read_csv(path, float_precision="round_trip", dtype={"task": "category"})
    assert list(df.columns[:2]) == ["submission_id", "task"], f"{path}: unexpected columns {list(df.columns)}"
    return df


def reference(a):
    if a.ref:
        return read_full(a.ref), dict(kind="full", path=str(a.ref))
    p = read_full(a.ref_parent)
    d = pd.read_csv(a.ref_delta, float_precision="round_trip")
    idx = pd.Index(p.submission_id).get_indexer(d.submission_id)
    assert (idx >= 0).all(), "delta has ids missing from the parent"
    for c in d.columns[1:]:
        col = p[c].to_numpy().copy(); col[idx] = d[c].to_numpy(); p[c] = col
    info = dict(kind="parent+delta", parent=str(a.ref_parent), delta=str(a.ref_delta), delta_rows=len(d))
    if a.ref_sha and not a.no_ref_sha_check:
        hw = HashWriter()
        p.to_csv(hw, index=False, lineterminator=line_ending(a.ref_parent))
        got = hw.h.hexdigest()
        info["rebuilt_sha256"] = got
        assert got.startswith(a.ref_sha), f"rebuilt reference sha {got} != {a.ref_sha}: wrong parent or delta"
    return p, info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate", type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ref", type=Path); g.add_argument("--ref-parent", type=Path)
    ap.add_argument("--ref-delta", type=Path); ap.add_argument("--ref-sha")
    ap.add_argument("--no-ref-sha-check", action="store_true")
    ap.add_argument("--state-atol", type=float, default=1e-8)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    assert not a.ref_parent or a.ref_delta, "--ref-parent needs --ref-delta"
    t0 = time.time()
    out = dict(candidate=str(a.candidate), candidate_sha256=file_shas(a.candidate), reference_sha256=a.ref_sha)
    out["byte_equal"] = bool(a.ref_sha) and any(v.startswith(a.ref_sha) for v in out["candidate_sha256"].values())
    ref, out["reference"] = reference(a)
    if a.ref and a.ref_sha:
        out["reference"]["sha256_ok"] = any(v.startswith(a.ref_sha) for v in file_shas(a.ref).values())
    cand = read_full(a.candidate)
    problems = []
    if len(cand) != len(ref) or not (cand.submission_id.to_numpy() == ref.submission_id.to_numpy()).all():
        problems.append("ids or row order differ")
        cand = cand.set_index("submission_id").reindex(ref.submission_id).reset_index()
    task = ref.task.astype(str).to_numpy()
    if not (cand.task.astype(str).to_numpy() == task).all():
        problems.append("task column differs")
    rows = {}
    for t in sorted(set(task)):
        m = task == t
        for c in [c for c in VALUE_COLS if c in ref.columns]:
            x = cand[c].to_numpy(float)[m]; y = ref[c].to_numpy(float)[m]
            nan_x, nan_y = np.isnan(x), np.isnan(y)
            both = ~nan_x & ~nan_y
            d = np.abs(x[both] - y[both])
            diff = int((d != 0).sum()) + int((nan_x != nan_y).sum())
            r = dict(rows=int(m.sum()), differ=diff, nan_pattern_differs=int((nan_x != nan_y).sum()),
                     max_abs=float(d.max()) if d.size else 0.0, mean_abs=float(d.mean()) if d.size else 0.0,
                     over_1e_6=int((d > 1e-6).sum()), over_0_1=int((d > .1).sum()), over_1=int((d > 1).sum()))
            rows[f"{t}.{c}"] = r
            if r["nan_pattern_differs"]:
                problems.append(f"{t}.{c}: missing-value pattern differs in {r['nan_pattern_differs']} rows")
            tol = a.state_atol if (t == "state" and c in ("speed_kmh", "flow_vph")) else 0.0
            if r["max_abs"] > tol:
                problems.append(f"{t}.{c}: max |d| {r['max_abs']:.3g} > {tol:g} ({r['differ']} rows differ)")
    out.update(columns=rows, state_atol=a.state_atol, problems=problems, seconds=round(time.time() - t0, 1))
    for k, r in rows.items():
        print(f"{k:22s} rows {r['rows']:>9d}  differ {r['differ']:>9d}  max|d| {r['max_abs']:.3g}  mean|d| {r['mean_abs']:.3g}"
              f"  >1e-6 {r['over_1e_6']}  >.1 {r['over_0_1']}  >1 {r['over_1']}")
    print("candidate sha256 native/crlf/lf: " + " ".join(v[:16] for v in out["candidate_sha256"].values())
          + f"; reference {a.ref_sha[:16] if a.ref_sha else '-'}; byte equal: {out['byte_equal']}")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))
    changed = {k: r["differ"] for k, r in rows.items() if r["differ"]}
    verdict = "VERIFY_PASS" if not problems else "VERIFY_FAIL"
    print(f"{verdict} byte_equal={out['byte_equal']} changed_rows={json.dumps(changed)} "
          + (f"state max|d| speed {rows.get('state.speed_kmh', {}).get('max_abs', 0):.3g} flow {rows.get('state.flow_vph', {}).get('max_abs', 0):.3g}" if "state.speed_kmh" in rows else "")
          + ("" if not problems else " problems: " + "; ".join(problems[:5])))
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
