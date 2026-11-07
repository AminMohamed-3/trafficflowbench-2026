"""Unpack the shipped fitted inputs of the final build into the repository root and check them (KB2 2026-10-13,
exp/C_kbuild/KBUILD.md section 8, gaps W1 and W13).

The index is artifacts/claude_ref/asm2/package/serving_archives.json: one entry per archive (path relative to the
repository root, bytes, sha256, files). This step
  1. checks size and sha256 of every archive of the index (nothing is unpacked if one differs or is missing);
  2. unpacks them IN INDEX ORDER at the repository root (members must be regular files under artifacts/; a file that
     is already there with the same bytes is not written again, so a rerun is cheap). The order
     matters: serving_03.tar.gz is last and carries the current manifest artifacts/claude_ref/asm2/serving_inputs_sha256.txt;
     serving_00.tar.gz holds an older copy of it;
  3. checks that the manifest on disk is the one of the index (manifest_sha256 / manifest_lines of the last entry),
     so an older copy unpacked later (for example from asm2/adaptation/B_0512_clean3/adaptation_01.tar.gz, the C26
     record, which is NOT part of this index) cannot go unseen;
  4. runs the input check of asm2_inputs.py (every manifest file present, size and sha256 equal).
  python docs/reference_code/asm2_unpack.py [--check-only] [--from DIR]
--from DIR: the archives are read from DIR/<path of the index> (for example a mounted copy of the package) and the
index itself from DIR; default: this repository. --check-only: steps 1, 3 and 4, nothing is unpacked."""
import argparse, hashlib, json, subprocess, sys, tarfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
INDEX = "artifacts/claude_ref/asm2/package/serving_archives.json"; MAN = "artifacts/claude_ref/asm2/serving_inputs_sha256.txt"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check-only", action="store_true"); ap.add_argument("--from", dest="src", type=Path, default=ROOT)
    a = ap.parse_args(); entries = json.loads((a.src / INDEX).read_text())
    assert entries and entries[-1]["path"].endswith("serving_03.tar.gz") and "manifest_sha256" in entries[-1], "index: the last entry must be serving_03.tar.gz with manifest_sha256"
    for e in entries:
        p = a.src / e["path"]
        assert p.is_file(), f"archive missing: {e['path']}"
        assert p.stat().st_size == e["bytes"] and sha(p) == e["sha256"], f"archive differs from the index: {e['path']}"
    print("archives of the index verified:", len(entries), sum(e["bytes"] for e in entries), "bytes", flush=True)
    if not a.check_only:
        for e in entries:
            n = wrote = 0
            with tarfile.open(a.src / e["path"]) as t:
                for m in t:
                    assert m.isfile() and m.name.startswith("artifacts/") and ".." not in Path(m.name).parts, f"unexpected member {m.name} in {e['path']}"
                    out = ROOT / m.name; out.parent.mkdir(parents=True, exist_ok=True)
                    data = t.extractfile(m).read(); n += 1
                    if out.is_file() and out.stat().st_size == len(data) and sha(out) == hashlib.sha256(data).hexdigest():
                        continue   # already there with the same bytes (a rerun writes nothing)
                    out.write_bytes(data); wrote += 1
            assert n == e["files"], f"{e['path']}: {n} files, index says {e['files']}"
            print("unpacked", e["path"], n, "files,", wrote, "written,", n - wrote, "already equal", flush=True)
    last = entries[-1]; man = ROOT / MAN
    assert man.is_file(), f"manifest missing: {MAN}"
    lines = sum(1 for ln in man.read_text().splitlines() if ln.strip())
    assert sha(man) == last["manifest_sha256"] and lines == last["manifest_lines"], (
        f"{MAN} on disk ({lines} lines) is not the manifest of the index ({last['manifest_lines']} lines): an older copy was unpacked over it; unpack serving_03.tar.gz again")
    print("manifest verified:", lines, "lines, sha256", last["manifest_sha256"][:16], flush=True)
    subprocess.run([sys.executable, str(ROOT / "docs/reference_code/asm2_inputs.py")], check=True)
    print("ASM2_UNPACK_OK", flush=True)


if __name__ == "__main__": main()
