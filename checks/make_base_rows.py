"""Write weights/base/00_base.csv and weights/base/00_reference.csv from the development repo (worker R1, 2026-10-21).

  python checks/make_base_rows.py --src-root OLD_REPO

00_base.csv = every fitted / weight file the C47 base build reads (docs/reference_code/asm2_run_all.sh, default flags):
  the 888 files of artifacts/claude_ref/asm2/serving_inputs_sha256.txt (the build's own input manifest, checked by
  asm2_inputs.py before any computation), the manifest itself and the package index serving_archives.json.
  Every file is hashed here; a sha that differs from the build manifest stops the script. A file that is not on the
  development disk as a loose file is looked up in the package archives of the index (index order, last one wins, as
  docs/reference_code/asm2_unpack.py unpacks) and recorded as old_repo_path = "<archive>::<member>".
00_base_s08fit.csv = the two S08 stack-input exports of the C47 file (127 leaves x 4800 rounds, laptop fit; the Linux
  refit of the build reproduced them exactly, docs/report/REPRO_README.md); used only by 10_base_c47.sh --s08-fitted.
00_reference.csv = the submitted C47 file, stored as C44 (full CSV) + the C47 delta (tools/probe.py layout) + its json.
Columns (shared with the piece rows of R2/R3 in weights/rows/): repo_path, sha256, bytes, old_repo_path, piece.
For the base files repo_path is relative to the final_solution ROOT (the build binds artifacts/... paths); piece rows
in weights/rows/ are relative to final_solution/weights/ (pipeline/pieces/lib/weights.py).
"""
import argparse, csv, hashlib, json, tarfile
from pathlib import Path

FINAL = Path(__file__).resolve().parents[1]
MANIFEST = "artifacts/claude_ref/asm2/serving_inputs_sha256.txt"
INDEX = "artifacts/claude_ref/asm2/package/serving_archives.json"
S08FIT = ["artifacts/claude_ref/c_budget/s08/official/l127_r4800/official_validation.parquet",
          "artifacts/claude_ref/c_budget/s08/official/l127_r4800/official_private.parquet"]
REFERENCE = [  # submitted C47 = C44 full CSV + delta (artifacts/claude_ref/probes/manifest.csv, C47 sha 1606b1c5...)
    ("reference/c_b6l12k_on_c43.csv", "artifacts/claude_ref/probes/c_b6l12k_on_c43.csv"),
    ("reference/c_b6l24k_on_c44.delta.csv.gz", "artifacts/claude_ref/probes/c_b6l24k_on_c44.delta.csv.gz"),
    ("reference/c_b6l24k_on_c44.json", "artifacts/claude_ref/probes/c_b6l24k_on_c44.json"),
]
FIELDS = ["repo_path", "sha256", "bytes", "old_repo_path", "piece"]


def sha_stream(f):
    h = hashlib.sha256()
    for b in iter(lambda: f.read(1 << 22), b""):
        h.update(b)
    return h.hexdigest()


def sha(p):
    with open(p, "rb") as f:
        return sha_stream(f)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    print(f"{path.relative_to(FINAL).as_posix()}: {len(rows)} rows, {sum(int(r['bytes']) for r in rows)} bytes", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", type=Path, required=True)
    root = ap.parse_args().src_root.resolve()
    want, rows, missing = {}, {}, set()
    for line in (root / MANIFEST).read_text().splitlines():
        if line.strip():
            digest, size, name = line.split(maxsplit=2)
            want[name] = (digest, int(size))
    for name, (digest, size) in want.items():
        p = root / name
        if not p.is_file():
            missing.add(name); continue
        assert sha(p) == digest and p.stat().st_size == size, f"old repo file differs from the build manifest: {name}"
        rows[name] = dict(repo_path=name, sha256=digest, bytes=size, old_repo_path=name, piece="base")
    found = {}
    if missing:
        for e in json.loads((root / INDEX).read_text()):
            with tarfile.open(root / e["path"], "r:gz") as tf:
                for m in tf:
                    if m.isfile() and m.name in missing:
                        got = sha_stream(tf.extractfile(m))
                        if got == want[m.name][0] and m.size == want[m.name][1]:
                            found[m.name] = e["path"]
            print(f"scanned {e['path']}: {len(found)} of {len(missing)} missing files located", flush=True)
    lost = sorted(missing - set(found))
    assert not lost, f"{len(lost)} manifest files neither loose nor in an archive, e.g. {lost[:3]}"
    for name, arch in found.items():
        rows[name] = dict(repo_path=name, sha256=want[name][0], bytes=want[name][1], old_repo_path=f"{arch}::{name}", piece="base")
    out = [rows[n] for n in want]   # manifest order
    for name in (MANIFEST, INDEX):
        p = root / name
        out.append(dict(repo_path=name, sha256=sha(p), bytes=p.stat().st_size, old_repo_path=name, piece="base"))
    write(FINAL / "weights/base/00_base.csv", out)
    write(FINAL / "weights/base/00_base_s08fit.csv",
          [dict(repo_path=n, sha256=sha(root / n), bytes=(root / n).stat().st_size, old_repo_path=n, piece="base_s08fit") for n in S08FIT])
    write(FINAL / "weights/base/00_reference.csv",
          [dict(repo_path=rp, sha256=sha(root / old), bytes=(root / old).stat().st_size, old_repo_path=old, piece="reference")
           for rp, old in REFERENCE])


if __name__ == "__main__":
    main()
