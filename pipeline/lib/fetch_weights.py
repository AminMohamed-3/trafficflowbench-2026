"""Place every saved weight / fitted file of final_solution where the pipeline reads it, sha256-checked (worker R1, 2026-10-21).

  python pipeline/lib/fetch_weights.py merge                         rebuild weights/MANIFEST.csv from weights/base/*.csv + weights/rows/*.csv
  python pipeline/lib/fetch_weights.py fetch --from-repo OLD [--pieces base,reference,...]
  python pipeline/lib/fetch_weights.py fetch --from-dir DIR  [--pieces ...]   (unpacked public weight release, same relative paths)
  python pipeline/lib/fetch_weights.py check [--pieces ...]          sha256 of every placed file
  python pipeline/lib/fetch_weights.py snapshot OUT.json [--pieces ...]   size + mtime of the placed files (before/after a build)
  python pipeline/lib/fetch_weights.py compare OUT.json             the placed files did not change since the snapshot

Targets: pieces base, base_s08fit and reference -> final_solution/<repo_path> (the base build binds artifacts/... paths);
every other piece (rows of R2/R3) -> final_solution/weights/<repo_path>.
Sources (--from-repo): old_repo_path inside the development repo; "<archive>::<member>" = that member of a package archive.
Loose files are HARD-LINKED (no disk use; same volume), else copied; archive members are extracted (new files).
Safety: nothing is ever written below the source repo. A target that exists with the wrong bytes is unlinked first (never
written through, so a hard link can not carry a write back into the development repo).
"""
import argparse, csv, hashlib, json, os, shutil, sys, tarfile
from pathlib import Path

FS = Path(__file__).resolve().parents[2]
FIELDS = ["repo_path", "sha256", "bytes", "old_repo_path", "piece"]
ROOT_PIECES = {"base", "base_s08fit", "reference"}


def sha_stream(f):
    h = hashlib.sha256()
    for b in iter(lambda: f.read(1 << 22), b""):
        h.update(b)
    return h.hexdigest()


def sha(p):
    with open(p, "rb") as f:
        return sha_stream(f)


def load(pieces=None):
    out = []
    for d in ("weights/base", "weights/rows"):
        for fn in sorted((FS / d).glob("*.csv")):
            with open(fn, newline="", encoding="utf-8") as f:
                out += list(csv.DictReader(f))
    return [r for r in out if not pieces or r["piece"] in pieces]


def target(r):
    t = (FS / r["repo_path"]) if r["piece"] in ROOT_PIECES else (FS / "weights" / r["repo_path"])
    t = Path(os.path.abspath(t))
    assert str(t).startswith(str(FS)), f"target outside final_solution: {t}"
    return t


def ok(t, r, full=True):
    return t.is_file() and t.stat().st_size == int(r["bytes"]) and (not full or sha(t) == r["sha256"])


def place_file(src, t, r):
    t.parent.mkdir(parents=True, exist_ok=True)
    if t.exists() or t.is_symlink():
        t.unlink()
    try:
        os.link(src, t); how = "link"
    except OSError:
        shutil.copyfile(src, t); how = "copy"
    assert sha(t) == r["sha256"], f"sha mismatch after {how}: {t}"
    return how


def cmd_fetch(a):
    rows = load(a.pieces)
    stats = dict(present=0, link=0, copy=0, extract=0)
    by_arch = {}
    for r in rows:
        t = target(r)
        if ok(t, r, full=a.verify_present):
            stats["present"] += 1; continue
        src = r["old_repo_path"]
        if a.from_dir:
            stats[place_file(Path(a.from_dir) / r["repo_path"], t, r)] += 1
        elif "::" in src:
            arch, member = src.split("::", 1); by_arch.setdefault(arch, {})[member] = r
        else:
            stats[place_file(Path(a.from_repo) / src, t, r)] += 1
    for arch, members in by_arch.items():
        with tarfile.open(Path(a.from_repo) / arch, "r:gz") as tf:
            for m in tf:
                r = members.get(m.name)
                if r is None or not m.isfile():
                    continue
                t = target(r); t.parent.mkdir(parents=True, exist_ok=True)
                if t.exists() or t.is_symlink():
                    t.unlink()
                tmp = t.with_name(t.name + ".part")
                with tf.extractfile(m) as fi, open(tmp, "wb") as fo:
                    shutil.copyfileobj(fi, fo, 1 << 22)
                os.replace(tmp, t)
                assert sha(t) == r["sha256"], f"sha mismatch after extract: {t}"
                stats["extract"] += 1; members.pop(m.name)
        assert not members, f"{len(members)} members not found in {arch}"
    print(f"FETCH_OK {len(rows)} files: " + ", ".join(f"{k} {v}" for k, v in stats.items()), flush=True)


def cmd_check(a):
    bad = [r["repo_path"] for r in load(a.pieces) if not ok(target(r), r)]
    print(f"CHECK {'OK' if not bad else 'FAIL'} {len(load(a.pieces))} files, {len(bad)} bad" + (f": {bad[:5]}" if bad else ""))
    sys.exit(1 if bad else 0)


def cmd_merge(a):
    rows = load()
    with open(FS / "weights/MANIFEST.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS + ["location"], lineterminator="\n"); w.writeheader()
        for r in rows:
            w.writerow({**{k: r[k] for k in FIELDS}, "location": "root" if r["piece"] in ROOT_PIECES else "weights"})
    print(f"weights/MANIFEST.csv: {len(rows)} rows, {sum(int(r['bytes']) for r in rows)} bytes")


def cmd_snapshot(a):
    snap = {r["repo_path"]: [target(r).stat().st_size, target(r).stat().st_mtime_ns] for r in load(a.pieces)}
    Path(a.out).write_text(json.dumps(snap))
    print(f"snapshot {len(snap)} files -> {a.out}")


def cmd_compare(a):
    snap = json.loads(Path(a.out).read_text()); changed = []
    for rp, (size, mt) in snap.items():
        p = (FS / rp) if (FS / rp).exists() else (FS / "weights" / rp)
        st = p.stat()
        if st.st_size != size or st.st_mtime_ns != mt:
            changed.append(rp)
    print(f"INPUTS {'UNCHANGED' if not changed else 'CHANGED'} {len(snap)} files" + (f": {changed[:5]}" if changed else ""))
    sys.exit(1 if changed else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="op", required=True)
    for name in ("fetch", "check", "merge", "snapshot", "compare"):
        s = sp.add_parser(name)
        s.add_argument("--pieces", type=lambda x: set(x.split(",")) if x else None, default=None)
        if name == "fetch":
            g = s.add_mutually_exclusive_group(required=True)
            g.add_argument("--from-repo"); g.add_argument("--from-dir")
            s.add_argument("--verify-present", action="store_true", help="sha-check files already in place (default: size only)")
        if name in ("snapshot", "compare"):
            s.add_argument("out")
    a = ap.parse_args()
    {"fetch": cmd_fetch, "check": cmd_check, "merge": cmd_merge, "snapshot": cmd_snapshot, "compare": cmd_compare}[a.op](a)


if __name__ == "__main__":
    main()
