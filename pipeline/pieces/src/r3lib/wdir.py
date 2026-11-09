"""R3 (rows 28-35): resolve a saved-weights folder by its OLD path (relative to artifacts/claude_ref) and check every file in it.
  python wdir.py OLDDIR        e.g. l_fvft/mar/w/it54000_w32
Rows: final_solution/weights/rows/*.csv (repo_path, sha256, bytes, old_repo_path, piece). The folder is
$FS/weights/<repo dir> when fetched there (clean machine), else $TFB_OLD_REPO/artifacts/claude_ref/OLDDIR (laptop). Every listed
file must exist with its sha256 and no listed file may be missing. Prints the folder."""
import csv, hashlib, os, sys
from pathlib import Path
FS = Path(os.environ.get("FS") or Path(__file__).resolve().parents[4])
OLD = Path(os.environ.get("TFB_OLD_REPO") or FS.parent)
old = "artifacts/claude_ref/" + sys.argv[1].strip("/") + "/"
rows = [r for f in sorted((FS / "weights/rows").glob("*.csv")) for r in csv.DictReader(open(f, newline=""))
        if r["old_repo_path"].startswith(old)]
assert rows, f"no weights rows for {old}"
rdirs = {r["repo_path"][: len(r["repo_path"]) - len(r["old_repo_path"]) + len(old)] for r in rows}
assert len(rdirs) == 1, rdirs
rdir = FS / "weights" / rdirs.pop()
_n = lambda p: os.path.normcase(os.path.abspath(p))
if _n(FS).startswith(_n(OLD) + os.sep):   # laptop: repo copy first
    base = rdir if rdir.is_dir() else OLD / old
else:   # clean machine: the devroot copy (pipeline/15_make_devroot.py), pieces record inputs relative to TFB_ROOT
    base = OLD / old if (OLD / old).is_dir() else rdir
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()
for r in rows:
    p = base / r["old_repo_path"][len(old):]
    assert p.is_file(), f"missing {p}"
    assert sha(p) == r["sha256"], f"sha mismatch {p}"
print(str(base).replace("\\", "/"))
