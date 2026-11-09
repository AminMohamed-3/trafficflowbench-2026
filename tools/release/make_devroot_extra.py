"""Collect the development-repo files the chain (rows 19-35) reads that neither the repository nor the weights release carries.

  python tools/release/make_devroot_extra.py OUT_DIR DEV_ROOT LIST_TSV [LIST_TSV ...]

LIST_TSV = outputs of checks/open_audit/devroot_list.py (one per audited run): every row of class NEEDED (opened by Python
during the run, in no other class); or a .txt list, one relative path per line (the files a fetch-mode discovery run of the
clean room copied from the development repo and read or tested, checks/open_audit/fetch_mode.py; __pycache__ entries are
skipped, Python rebuilds them). Each path is sorted:
  reference  a stored single-step result (artifacts/claude_ref/**/c_*.csv[.gz], *.delta.csv.gz, or any stored_full / stored_delta
             path of pipeline/pieces/src/r3lib/expected.tsv): only the per-step comparison
             reads it; a clean machine reports NOT_COMPARED for that step and compares the final file instead. Not collected.
  code       source or config under docs/, exp/, src/, tools/, scripts/: copied INTO this repository at its development path
             (real letter case), so pipeline/15_make_devroot.py lays it out with the rest of the code.
  extra      anything else (fitted tables, models): copied to OUT_DIR at its development path; OUT_DIR/DEVROOT_EXTRA.tsv
             lists path, bytes, sha256. Zip OUT_DIR as the release asset devroot_extra.zip.
Prints one line per class and the files it copied.
"""
import csv, hashlib, os, re, shutil, sys
from pathlib import Path

FS = Path(__file__).resolve().parents[2]
REF = re.compile(r"^artifacts/claude_ref/.*(/c_[^/]*\.csv(\.gz)?|\.delta\.csv\.gz)$")
with open(FS / "pipeline/pieces/src/r3lib/expected.tsv", newline="", encoding="utf-8") as _f:
    EXPECTED = {v.lower() for r in csv.DictReader(_f, delimiter="	") for v in (r["stored_full"], r["stored_delta"]) if v != "-"}
CODE = re.compile(r"^(docs|exp|src|tools|scripts)/.*\.(py|sh|json|yaml|yml)$")


def real_case(root, rel):
    """Resolve a lower-cased relative path to its on-disk letter case (Windows audit logs are lower case)."""
    cur = Path(root); parts = []
    for comp in rel.split("/"):
        names = {n.lower(): n for n in os.listdir(cur)}
        n = names.get(comp.lower())
        if n is None:
            raise FileNotFoundError(f"{rel}: no {comp} in {cur}")
        parts.append(n); cur = cur / n
    return "/".join(parts)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out, dev, lists = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    need = set()
    for t in lists:
        with open(t, newline="", encoding="utf-8") as f:
            if t.endswith(".txt"):
                need |= {l.strip().replace("\\", "/").lower() for l in f
                         if l.strip() and "__pycache__/" not in l and not l.strip().endswith(".gitkeep")}
            else:
                need |= {r["path"] for r in csv.DictReader(f, delimiter="\t") if r["class"] == "NEEDED"}
    got = {"reference": [], "code": [], "extra": []}
    rows = []
    for rel in sorted(need):
        if rel.startswith("../"):
            sys.exit(f"outside the development root, handle by hand: {rel}")
        rc = real_case(dev, rel)
        if REF.match(rel) or rel.lower() in EXPECTED:
            got["reference"].append(rc); continue
        if CODE.match(rel):
            dst = FS / rc
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(dev / rc, dst); got["code"].append(rc); continue
        dst = out / rc
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(dev / rc, dst)
        rows.append((rc, dst.stat().st_size, sha256(dst))); got["extra"].append(rc)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "DEVROOT_EXTRA.tsv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n"); w.writerow(["path", "bytes", "sha256"]); w.writerows(rows)
    for k, v in got.items():
        print(f"{k:9s} {len(v):4d}")
        for p in v:
            print("   ", p)
    print(f"extra bytes {sum(r[1] for r in rows) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
