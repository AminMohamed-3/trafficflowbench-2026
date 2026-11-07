"""Pack the saved weights and fitted files (weights/MANIFEST.csv) into GitHub release parts, one part at a time.

  python tools/release/pack_weights.py plan [--max-gb 1.9]                 assign files to parts -> weights/RELEASE_PARTS.csv
  python tools/release/pack_weights.py build N OUT_DIR                     write part N (zip, member names = repo_path)
  python tools/release/pack_weights.py index OUT_DIR                       weights/RELEASE.tsv: part name, bytes, sha256

Sources: the placed files of a fetched tree (pipeline/01_fetch_weights.sh --from-repo ...), i.e. the target of each row
(final_solution/<repo_path> for root pieces, final_solution/weights/<repo_path> otherwise); every file is sha256-checked
before it is written. Parts hold whole files, are at most --max-gb, and are filled in MANIFEST order. Text files are deflated,
binaries stored. Build and upload one part at a time (gh release upload TAG part; then delete it) to keep disk use to one part.
"""
import csv, hashlib, sys, zipfile
from pathlib import Path

FS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FS / "pipeline" / "lib"))
import fetch_weights as fw  # noqa: E402

PARTS = FS / "weights" / "RELEASE_PARTS.csv"
INDEX = FS / "weights" / "RELEASE.tsv"
TEXT = {".csv", ".json", ".tsv", ".txt", ".md", ".py", ".sh", ".yaml", ".yml"}


def part_name(n):
    return f"tfb2026_weights_part{n:02d}.zip"


def cmd_plan(max_gb):
    cap, rows, n, used, out = int(max_gb * 1e9), fw.load(), 1, 0, []
    for r in rows:
        b = int(r["bytes"])
        assert b < cap, f"file larger than a part: {r['repo_path']}"
        if used + b > cap:
            n, used = n + 1, 0
        used += b
        out.append(dict(part=n, repo_path=r["repo_path"], piece=r["piece"], bytes=b, sha256=r["sha256"]))
    with open(PARTS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n"); w.writeheader(); w.writerows(out)
    print(f"{len(out)} files, {sum(o['bytes'] for o in out) / 1e9:.2f} GB in {n} parts -> {PARTS}")


def cmd_build(n, out_dir):
    rows = {r["repo_path"]: r for r in fw.load()}
    want = [r for r in csv.DictReader(open(PARTS, newline="", encoding="utf-8")) if int(r["part"]) == n]
    out = Path(out_dir) / part_name(n); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as z:
        for w in want:
            r = rows[w["repo_path"]]; t = fw.target(r)
            assert fw.ok(t, r), f"missing or sha mismatch, fetch first: {t}"
            ct = zipfile.ZIP_DEFLATED if t.suffix.lower() in TEXT else zipfile.ZIP_STORED
            z.write(t, arcname=r["repo_path"], compress_type=ct)   # fetch_weights.py --from-dir DIR reads DIR/<repo_path>
    tmp.replace(out)
    print(f"{out.name}: {len(want)} files, {out.stat().st_size / 1e9:.2f} GB")


def cmd_index(out_dir):
    with open(INDEX, "w", encoding="utf-8", newline="\n") as f:
        f.write("name\tbytes\tsha256\n")
        for p in sorted(Path(out_dir).glob("tfb2026_weights_part*.zip")):
            h = hashlib.sha256()
            with open(p, "rb") as fi:
                for c in iter(lambda: fi.read(1 << 22), b""):
                    h.update(c)
            f.write(f"{p.name}\t{p.stat().st_size}\t{h.hexdigest()}\n")
    print(INDEX.read_text(encoding="utf-8"))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["plan"]:
        cmd_plan(float(a[a.index("--max-gb") + 1]) if "--max-gb" in a else 1.9)
    elif a[:1] == ["build"]:
        cmd_build(int(a[1]), a[2])
    elif a[:1] == ["index"]:
        cmd_index(a[1])
    else:
        print(__doc__); sys.exit(2)
