"""Weights resolver for the piece scripts of rows 19 to 27 (worker R2, 2026-10-21).

Every saved weight or fitted file a piece reads is listed in final_solution/weights/rows/<NN>.csv, one file per
piece (columns: repo_path, sha256, bytes, old_repo_path, piece); the union is merged into weights/MANIFEST.csv. A piece asks for a file by its repo_path
(path under final_solution/weights/). Resolution order:
  1. $FS/weights/<repo_path> (after pipeline/01_fetch_weights.sh, or on a clean machine);
  2. $TFB_OLD_REPO/<old_repo_path> (laptop mode: the development repo, no copy needed).
The sha256 is always checked; a mismatch stops the run.

CLI:
  python weights.py path <repo_path>          print the resolved absolute path (sha checked)
  python weights.py check [<piece>]           check every row (of one piece)
"""
import csv, hashlib, os, sys
from pathlib import Path

HERE = Path(__file__).resolve()
FS = Path(os.environ.get("FS", HERE.parents[3]))            # final_solution/
OLD = Path(os.environ.get("TFB_OLD_REPO", FS.parent))         # development repo root (laptop)
MAN = FS / "weights" / "rows"
_SHA_CACHE = {}


def sha256(path):
    p = str(path)
    if p in _SHA_CACHE:
        return _SHA_CACHE[p]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    _SHA_CACHE[p] = h.hexdigest()
    return _SHA_CACHE[p]


def rows(piece=None):
    out = []
    for fn in sorted(MAN.glob("*.csv")):
        with open(fn, newline="") as f:
            out += list(csv.DictReader(f))
    return [r for r in out if piece is None or r["piece"] == piece]


def resolve(repo_path, check=True):
    rs = [r for r in rows() if r["repo_path"] == repo_path]
    if not rs:
        raise SystemExit(f"weights: {repo_path} not in {MAN}")
    r = rs[0]
    for cand in (FS / "weights" / r["repo_path"], OLD / r["old_repo_path"]):
        if cand.is_file():
            if check and sha256(cand) != r["sha256"]:
                raise SystemExit(f"weights: sha mismatch {cand}: {sha256(cand)} != {r['sha256']}")
            return cand
    raise SystemExit(f"weights: {repo_path} missing (looked in {FS/'weights'} and {OLD})")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "path":
        print(resolve(a[1]))
    elif a and a[0] == "check":
        rr = rows(a[1] if len(a) > 1 else None)
        for r in rr:
            resolve(r["repo_path"])
        print(f"weights: {len(rr)} files OK")
    else:
        print(__doc__)
        sys.exit(2)
