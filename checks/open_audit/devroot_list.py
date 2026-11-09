"""Which development-machine files does the chain read that a clean machine would not have? (C, 2026-11-07)

  python checks/open_audit/devroot_list.py OUT_TSV DEV_ROOT AUDIT_DIR [AUDIT_DIR ...]

Reads AUDIT_DIR/open_audit.log (Python-level opens) and AUDIT_DIR/atime_reads.tsv (NTFS last-access superset, paths
relative to DEV_ROOT) of each audited run, and classifies every file:
  weights   listed in weights/MANIFEST.csv (old_repo_path or repo_path): the release restores it
  code      a source file whose byte copy is in this repository (same relative path under docs/reference_code, src/tfb,
            scripts, tools, exp, pipeline; a .pyc counts by its .py source)
  code_diff a source file present in this repository under the same path with different bytes (line endings ignored)
  data      the competition release (data/raw/kaggle_public)
  build     written by the build itself (final_solution/{artifacts/cache,outputs,work}, __pycache__ of build code, .venv*)
  meta      repository metadata and documents the atime pass touched but no build step needs (.git, .claude, *.md, logs)
  NEEDED    opened by Python and in no class above: a clean machine lacks it (the devroot package must carry it)
  NEEDED_native?  seen only by atime, with a file type native readers load: check, then carry if a step reads it
  atime_noise     seen only by atime, other types (indexers, earlier tools)
Writes OUT_TSV (class, path, bytes, source) and prints a summary by class and top folders of NEEDED.
"""
import csv, os, sys, collections
from pathlib import Path

FS = Path(__file__).resolve().parents[2]
CODE_ROOTS = ("docs/reference_code/", "src/tfb/", "scripts/", "tools/", "exp/", "pipeline/", "kaggle_kernels/")
TFB_ALIASES = ("work/tfb-wt/",)
NATIVE = (".parquet", ".txt", ".lgb", ".model", ".bin", ".feather", ".arrow", ".pt", ".pth", ".npy", ".npz", ".ubj", ".cbm")          # other checkouts whose src/tfb the dev venv imported


def norm(p):
    return p.replace("\\", "/")


def main():
    out, dev, audits = Path(sys.argv[1]), Path(sys.argv[2]), [Path(a) for a in sys.argv[3:]]
    devl = norm(str(dev)).lower().rstrip("/") + "/"
    parent = norm(str(dev.parent)).lower().rstrip("/") + "/"
    man = {}
    with open(FS / "weights/MANIFEST.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in (r["old_repo_path"].split("::")[0], r["repo_path"]):
                man[k.lower()] = r["repo_path"]
    seen = {}
    for a in audits:
        lg = a / "open_audit.log"
        if lg.exists():
            for ln in open(lg, encoding="utf-8"):
                p = norm(ln.split("\t")[2]).lower()
                if p.startswith(devl):
                    seen.setdefault(p[len(devl):], f"open:{a.name}")
                elif p.startswith(parent):
                    seen.setdefault("../" + p[len(parent):], f"open:{a.name}")
        at = a / "atime_reads.tsv"
        if at.exists():
            with open(at, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    seen.setdefault(norm(r["path"]).lower(), f"atime:{a.name}")
    rows = []
    for rel, src in sorted(seen.items()):
        real = (dev / rel) if not rel.startswith("../") else (dev.parent / rel[3:])
        try:
            nb = real.stat().st_size
        except OSError:
            nb = -1
        cls = classify(rel, real, man)
        if cls == "NEEDED" and src.startswith("atime:"):
            # atime is a superset (anything touched near the run); keep it as NEEDED only for files native readers load
            cls = "NEEDED_native?" if rel.endswith(NATIVE) else "atime_noise"
        rows.append((cls, rel, nb, src))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["class", "path", "bytes", "source"]); w.writerows(rows)
    by = collections.Counter(); bb = collections.Counter(); top = collections.Counter(); topb = collections.Counter()
    for cls, rel, nb, _ in rows:
        by[cls] += 1; bb[cls] += max(nb, 0)
        if cls in ("NEEDED", "NEEDED_native?", "code_diff"):
            k = cls + " " + "/".join(rel.split("/")[:3]); top[k] += 1; topb[k] += max(nb, 0)
    for c in by:
        print(f"{c:10s} {by[c]:6d} files {bb[c] / 1e9:7.2f} GB")
    for k, n in top.most_common(40):
        print(f"  {n:5d} {topb[k] / 1e6:9.1f} MB  {k}")


def eol(b):
    return b.replace(bytes([13, 10]), bytes([10]))


def classify(rel, real, man):
    r = rel.lower()
    if r in man:
        return "weights"
    if r.startswith("data/raw/kaggle_public/"):
        return "data"
    if r.startswith(("final_solution/artifacts/cache/", "final_solution/outputs/", "final_solution/work/", ".venv", "../tfb-wt/") ) and not r.endswith((".py", ".pyc")):
        return "build"
    if r.startswith((".git/", ".claude/")) or r.endswith((".md", ".log", ".txt")) and not r.startswith(("artifacts/", "exp/")):
        return "meta"
    src = r
    if src.endswith(".pyc"):
        d, name = src.rsplit("/__pycache__/", 1) if "/__pycache__/" in src else (None, None)
        if d is None:
            return "build"
        src = d + "/" + name.split(".")[0] + ".py"
    for al in TFB_ALIASES:   # ../tfb-wt/<wt>/src/tfb/x -> src/tfb/x
        a = "../" + al[len("work/"):]
        if src.startswith(a) and "/src/tfb/" in src:
            src = "src/tfb/" + src.split("/src/tfb/", 1)[1]
    if src.startswith(CODE_ROOTS) and src.endswith((".py", ".sh", ".json", ".yaml", ".yml", ".txt", ".csv")):
        mine = FS / src
        if mine.exists():
            real_src = real if not rel.endswith(".pyc") else None
            if real_src is None:
                return "code"     # bytecode: judged by its source, which is listed separately when read
            try:
                return "code" if eol(mine.read_bytes()) == eol(real_src.read_bytes()) else "code_diff"
            except OSError:
                return "code"
    if "/__pycache__/" in r:
        return "build"
    return "NEEDED"


if __name__ == "__main__":
    main()
