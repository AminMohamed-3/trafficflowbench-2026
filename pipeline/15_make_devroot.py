"""Lay out the development-repo folder ("devroot") that the piece scripts of rows 19-35 read, on a machine that has only this
repository, the weights release, the devroot extras and the competition data. Run after pipeline/10_base_c47.sh.

  python pipeline/15_make_devroot.py DEVROOT --extra EXTRA_DIR [--copy]

Why: the pieces run copied and original development code that opens its inputs at their development paths under one root
(TFB_ROOT = TFB_OLD_REPO; on the author's laptop the development repository itself). This script rebuilds that root from
what is published, at the same relative paths:
  code     docs/reference_code, exp, src, tools, scripts, kaggle_kernels of this repository (copied)
  weights  every row of weights/MANIFEST.csv: the restored file (weights/<repo_path> for pieces, <repo_path> for the base)
           placed at its development path (old_repo_path; the member path for rows packed in a tar ::)
  extras   EXTRA_DIR = the unzipped release asset devroot_extra.zip (70 files, 730 MB: stored member predictions and rows,
           fitted gap tables, serve receipts, two LightGBM files; sha256 checked against EXTRA_DIR/DEVROOT_EXTRA.tsv;
           made by tools/release/make_devroot_extra.py from the audited runs and checks/clean_room/fetch_list.txt)
  cache    artifacts/cache = the panel cache the base build wrote (byte-equal to the development cache)
  data     data/raw/kaggle_public = the competition release
  base     outputs/c47 (the base build) at final_solution/outputs/c47, its development path
Files are hard-linked (no extra disk) when DEVROOT is on the same volume, else copied (--copy forces copies).
Then run the chain with  TFB_OLD_REPO=DEVROOT CHAIN_WORK=DEVROOT/work  bash pipeline/20_pieces.sh
(the pieces of rows 28-35 require their work folder inside TFB_ROOT). Writes DEVROOT/DEVROOT.json (counts, sources).
"""
import csv, hashlib, json, os, shutil, sys, time
from pathlib import Path

FS = Path(__file__).resolve().parents[1]
CODE = ("docs/reference_code", "docs/research", "exp", "src", "tools", "scripts", "kaggle_kernels")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def place(src, dst, copy):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.stat().st_size == src.stat().st_size:
            return "kept"
        dst.unlink()
    if not copy:
        try:
            os.link(src, dst); return "linked"
        except OSError:
            pass
    shutil.copyfile(src, dst); return "copied"


def tree(src, dst, copy, n, skip=("__pycache__",)):
    for p in src.rglob("*"):
        if p.is_file() and not any(s in p.parts for s in skip):
            n[place(p, dst / p.relative_to(src), copy)] += 1


def main():
    a = sys.argv[1:]
    dev = Path(a[0]).resolve(); copy = "--copy" in a
    extra = Path(a[a.index("--extra") + 1]).resolve() if "--extra" in a else None
    if extra is None or not (extra / "DEVROOT_EXTRA.tsv").exists():
        sys.exit("--extra EXTRA_DIR (the unzipped devroot_extra.zip, with DEVROOT_EXTRA.tsv) is required")
    if dev == FS.resolve() or dev.parent == FS.resolve() and dev.name != "devroot":
        sys.exit("DEVROOT must be a folder of its own: ./devroot inside the repository, or any folder outside it")
    t0 = time.time(); n = {"kept": 0, "linked": 0, "copied": 0}; miss = []
    for c in CODE:   # code: plain copies (small, and edits to the repo must not reach a running chain through links)
        if (FS / c).is_dir():
            tree(FS / c, dev / c, True, n)
    with open(FS / "weights/MANIFEST.csv", newline="", encoding="utf-8") as f:
        man = list(csv.DictReader(f))
    for r in man:
        src = (FS / "weights" / r["repo_path"]) if r["location"] == "weights" else (FS / r["repo_path"])
        dst = dev / r["old_repo_path"].split("::")[-1]
        if not src.is_file():
            miss.append(r["repo_path"]); continue
        n[place(src, dst, copy)] += 1
    with open(extra / "DEVROOT_EXTRA.tsv", newline="", encoding="utf-8") as f:
        ex = list(csv.DictReader(f, delimiter="\t"))
    bad = [r["path"] for r in ex if sha256(extra / r["path"]) != r["sha256"]]
    if bad:
        sys.exit(f"devroot extras: sha256 mismatch {bad}")
    for r in ex:
        n[place(extra / r["path"], dev / r["path"], copy)] += 1
    for sub in ("artifacts/cache", "data/raw/kaggle_public"):
        src = (FS / sub).resolve()
        if not src.is_dir():
            sys.exit(f"{FS / sub} missing (run pipeline/00_fetch_data.sh and pipeline/10_base_c47.sh first)")
        tree(src, dev / sub, copy, n, skip=())
    base = FS / "outputs/c47"   # the base build at its development path (row 20 needs it inside the root)
    if (base / "final_A.csv").is_file():
        tree(base.resolve(), dev / "final_solution/outputs/c47", copy, n, skip=())
    else:
        print("note: outputs/c47 missing (run pipeline/10_base_c47.sh first); the chain then needs CHAIN_BASE / CHAIN_BASE_RUN")
    if miss:
        sys.exit(f"{len(miss)} weights rows not restored (run pipeline/01_fetch_weights.sh first), e.g. {miss[:3]}")
    (dev / "work").mkdir(exist_ok=True)
    plan = dev / "PLAN.md"   # project-root marker of src/tfb/paths.py (TFB_ROOT must hold it; the devroot has no .git)
    if not plan.exists():
        plan.write_text("Development-path root made by pipeline/15_make_devroot.py (project-root marker for src/tfb/paths.py).\n", encoding="utf-8")
    rec = dict(made=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), repo=str(FS), weights_rows=len(man), extras=len(ex),
               files=n, seconds=round(time.time() - t0))
    (dev / "DEVROOT.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print("DEVROOT_OK", dev, json.dumps(n))


if __name__ == "__main__":
    main()
