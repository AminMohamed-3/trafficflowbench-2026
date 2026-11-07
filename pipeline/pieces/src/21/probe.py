"""Shared probe tool (C, 2026-10-06, retro fix): one way to store, rebuild and check submission files.

Probes live on disk as DELTAS (submission_id + changed columns, changed rows only) on a named parent file. The full
400 MB CSV is built only when it is uploaded or needed, and its sha256 is checked against the value recorded when the
delta was made. Replaces the per-helper probe builders and delta appliers.

  python tools/probe.py make   PARENT.csv FULL.csv OUT.delta.csv.gz   # delta + OUT.json (parent sha, full sha, counts)
  python tools/probe.py expand PARENT.csv DELTA OUT.csv [--sha HEX]   # rebuild the full file; checks sha (or the json's)
  python tools/probe.py diff   A.csv B.csv                             # changed rows/columns by task, max abs change
  python tools/probe.py sha    FILE

Byte-exact rebuilds: files are read with float_precision='round_trip' and written with the parent's line ending, so a
file written by pandas rebuilds to the same bytes. If a sha does not match, the values still match (diff reports 0).
"""
import argparse, gzip, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

VALUE_COLS = ["speed_kmh", "flow_vph", "queue_pred", "path_flow"]
N_ROWS = 6_980_503


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def line_ending(path):
    with open(path, "rb") as f:
        head = f.read(4096)
    return "\r\n" if b"\r\n" in head else "\n"


def read_full(path):
    df = pd.read_csv(path, float_precision="round_trip", dtype={"task": "category"})
    assert list(df.columns[:2]) == ["submission_id", "task"], f"{path}: unexpected columns {list(df.columns)}"
    return df


def changed_mask(a, b, cols):
    out = {}
    for c in cols:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        out[c] = ~((x == y) | (pd.isna(x) & pd.isna(y)))
    return out


def cmd_make(a):
    p, f = read_full(a.parent), read_full(a.full)
    assert len(p) == len(f) and (p.submission_id.to_numpy() == f.submission_id.to_numpy()).all(), "row order differs"
    assert (p.task.astype(str).to_numpy() == f.task.astype(str).to_numpy()).all(), "task column differs"
    cols = [c for c in VALUE_COLS if c in p.columns]
    m = changed_mask(p, f, cols)
    cc = [c for c in cols if m[c].any()]
    rows = np.zeros(len(p), bool)
    for c in cc:
        rows |= m[c]
    d = f.loc[rows, ["submission_id"] + cc]
    d.to_csv(a.out, index=False)
    meta = dict(parent=Path(a.parent).name, parent_sha256=sha256(a.parent), full=Path(a.full).name,
                full_sha256=sha256(a.full), delta=Path(a.out).name, delta_sha256=sha256(a.out), rows_changed=int(rows.sum()),
                columns=cc, rows_by_task=f.loc[rows, "task"].astype(str).value_counts().to_dict())
    js = Path(str(a.out).replace(".delta.csv.gz", "").replace(".csv.gz", "").replace(".csv", "") + ".json")
    js.write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def cmd_expand(a):
    p = read_full(a.parent)
    d = pd.read_csv(a.delta, float_precision="round_trip")
    idx = pd.Index(p.submission_id).get_indexer(d.submission_id)
    assert (idx >= 0).all(), "delta has ids missing from the parent"
    for c in d.columns[1:]:
        assert c in p.columns, c
        col = p[c].to_numpy().copy()
        col[idx] = d[c].to_numpy()
        p[c] = col
    assert len(p) == N_ROWS, len(p)
    p.to_csv(a.out, index=False, lineterminator=line_ending(a.parent))
    h = sha256(a.out)
    want = a.sha
    if not want:
        js = Path(str(a.delta).replace(".delta.csv.gz", "").replace(".csv.gz", "").replace(".csv", "") + ".json")
        if js.exists():
            want = json.loads(js.read_text()).get("full_sha256")
    status = "no sha to check" if not want else ("MATCH" if h.startswith(want) else "MISMATCH")
    print(f"wrote {a.out} sha256 {h} ({status}); {len(d)} rows from delta, columns {list(d.columns[1:])}")
    if status == "MISMATCH":
        sys.exit(2)


def cmd_diff(a):
    x, y = read_full(a.a), read_full(a.b)
    assert (x.submission_id.to_numpy() == y.submission_id.to_numpy()).all(), "row order differs"
    cols = [c for c in VALUE_COLS if c in x.columns]
    m = changed_mask(x, y, cols)
    task = x.task.astype(str)
    for c in cols:
        if m[c].any():
            dv = np.abs(x[c].to_numpy()[m[c]] - y[c].to_numpy()[m[c]])
            print(f"{c}: {int(m[c].sum())} rows changed {task[m[c]].value_counts().to_dict()}, max abs {np.nanmax(dv):.6g}")
        else:
            print(f"{c}: 0 rows changed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="op", required=True)
    s = sp.add_parser("make"); s.add_argument("parent"); s.add_argument("full"); s.add_argument("out")
    s = sp.add_parser("expand"); s.add_argument("parent"); s.add_argument("delta"); s.add_argument("out"); s.add_argument("--sha")
    s = sp.add_parser("diff"); s.add_argument("a"); s.add_argument("b")
    s = sp.add_parser("sha"); s.add_argument("file")
    a = ap.parse_args()
    if a.op == "sha":
        print(sha256(a.file))
    else:
        {"make": cmd_make, "expand": cmd_expand, "diff": cmd_diff}[a.op](a)


if __name__ == "__main__":
    main()
