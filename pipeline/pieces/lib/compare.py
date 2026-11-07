"""Parity check of a rebuilt submission file against the stored one (worker R2, 2026-10-21).

  python compare.py BUILT.csv EXPECTED_SHA256 [STORED.csv]

PASS-SHA  : sha256(BUILT) == EXPECTED_SHA256.
PASS-VALUE: sha differs but every value equals the stored file (row-wise max diff 0; needs STORED.csv, may be .csv.gz).
FAIL      : otherwise; prints changed rows and max abs diff per task and column.
Prints one JSON line; exit 0 on PASS-*, 1 on FAIL.
"""
import hashlib, json, sys
import numpy as np
import pandas as pd

COLS = ["speed_kmh", "flow_vph", "queue_pred", "path_flow"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    built, exp = sys.argv[1], sys.argv[2]
    stored = sys.argv[3] if len(sys.argv) > 3 else None
    s = sha256(built)
    out = {"built": built, "sha256": s, "expected": exp}
    if s == exp:
        out["result"] = "PASS-SHA"
        print(json.dumps(out)); return 0
    if stored is None:
        out["result"] = "FAIL"; out["why"] = "sha differs, no stored file given"
        print(json.dumps(out)); return 1
    a = pd.read_csv(built, float_precision="round_trip")
    b = pd.read_csv(stored, float_precision="round_trip")
    if len(a) != len(b) or not (a.submission_id.to_numpy() == b.submission_id.to_numpy()).all():
        out["result"] = "FAIL"; out["why"] = "row order or count differs"
        print(json.dumps(out)); return 1
    rep, bad = {}, 0
    for c in COLS:
        if c not in a.columns:
            continue
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        d = np.abs(x - y); d[np.isnan(x) & np.isnan(y)] = 0; d[np.isnan(d)] = np.inf
        for t in sorted(a.task.astype(str).unique()):
            m = (a.task.astype(str) == t).to_numpy()
            n = int((d[m] > 0).sum())
            if n:
                rep[f"{t}.{c}"] = {"rows": n, "max": float(d[m].max())}
                bad += n
    out["diff"] = rep
    out["result"] = "PASS-VALUE" if bad == 0 else "FAIL"
    print(json.dumps(out)); return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
