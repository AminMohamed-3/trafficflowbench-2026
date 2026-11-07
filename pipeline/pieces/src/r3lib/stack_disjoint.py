# R3 (rows 28-35) copy of tools/rebase/stack_disjoint.py (sha256 cdd671fb698907cb0a1ace7ca2eeb395b9a419dd4c9570186714a372e563ced4). Edit: ROOT = $TFB_ROOT (tools/probe.py of the old-layout root). Logic unchanged.
"""Stack a piece on a new parent when the two pieces change DISJOINT rows of the same old parent (C 2026-10-17).
Example: S2b changes the non-gap state rows of C64, GAPW changes the gap rows of C64; S2b on C65 = C65 + S2b's delta.
  python tools/rebase/stack_disjoint.py NEW_PARENT.csv PIECE.delta.csv.gz OTHER.delta.csv.gz OUT.csv
PIECE and OTHER must be deltas on the same old parent (their .json 'parent_sha256' must match); asserts the row ids
are disjoint, applies PIECE on NEW_PARENT (probe.py expand rules), then writes OUT.delta.csv.gz + OUT.json on NEW_PARENT
via probe.py make. Last line STACK_DONE OUT sha or a traceback."""
import os, sys, json, subprocess, importlib.util
from pathlib import Path
import pandas as pd
ROOT = Path(os.environ["TFB_ROOT"])
spec = importlib.util.spec_from_file_location("probe", ROOT / "tools/probe.py"); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)


def js(delta):
    return json.loads(Path(str(delta).replace(".delta.csv.gz", "") + ".json").read_text())


def main():
    new_parent, piece, other, out = map(Path, sys.argv[1:5])
    jp, jo = js(piece), js(other)
    assert jp["parent_sha256"] == jo["parent_sha256"], "pieces are not on the same old parent"
    assert P.sha256(new_parent) == jo["full_sha256"], "NEW_PARENT is not OTHER's full file"
    ip = pd.read_csv(piece, usecols=["submission_id"]).submission_id
    io = pd.read_csv(other, usecols=["submission_id"]).submission_id
    both = len(set(ip) & set(io))
    print(f"piece rows {len(ip)}, other rows {len(io)}, overlap {both}", flush=True)
    assert both == 0, "rows overlap: rebuild, do not stack"
    p = P.read_full(new_parent)
    d = pd.read_csv(piece, float_precision="round_trip")
    idx = pd.Index(p.submission_id).get_indexer(d.submission_id)
    assert (idx >= 0).all(), "delta has ids missing from the new parent"
    for c in d.columns[1:]:
        col = p[c].to_numpy().copy(); col[idx] = d[c].to_numpy(); p[c] = col
    assert len(p) == P.N_ROWS, len(p)
    p.to_csv(out, index=False, lineterminator=P.line_ending(new_parent)); del p, d
    subprocess.run([sys.executable, "-B", str(ROOT / "tools/probe.py"), "make", str(new_parent), str(out),
                    str(out).replace(".csv", ".delta.csv.gz")], check=True)
    print("STACK_DONE", out, P.sha256(out))


if __name__ == "__main__":
    main()
