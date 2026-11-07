"""Verify the shipped fitted-input manifest before serving; no raw data is included."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "artifacts/claude_ref/asm2/serving_inputs_sha256.txt")
    ap.add_argument("--environment", action="store_true", help="fail before serving if Python or focused package pins differ")
    ap.add_argument("--environment-out", type=Path)
    a = ap.parse_args(); count = 0; total = 0
    if a.environment:
        assert sys.version_info[:2] == (3, 12), "organizer build requires Python 3.12.x"
        pins = ROOT / "docs/report/requirements-final.txt"
        expected = dict(line.strip().split("==", 1) for line in pins.read_text().splitlines()
                        if "==" in line and not line.lstrip().startswith("#"))
        assert expected, "focused pinned requirements missing"
        measured, mismatches = {}, {}
        # KB 2026-10-12: TFB_ASM2_CPU_ONLY=1 = a CPU build box (torch +cpu wheel; no CUDA runtime packages, no triton)
        cpu_only = os.environ.get("TFB_ASM2_CPU_ONLY") == "1"
        if cpu_only:
            expected = {n: (v.split("+")[0] + "+cpu" if n == "torch" else v) for n, v in expected.items()
                        if not (n.startswith("nvidia-") or n == "triton")}
        for name, version in expected.items():
            try:
                measured[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                measured[name] = None
            if measured[name] != version:
                mismatches[name] = dict(expected=version, installed=measured[name])
        assert not mismatches, "install docs/report/requirements-final.txt: " + json.dumps(mismatches, sort_keys=True)
        environment = dict(python=platform.python_version(), packages=measured, cpu_only=cpu_only,
                           requirements_sha256=hashlib.sha256(pins.read_bytes()).hexdigest())
        if a.environment_out:
            a.environment_out.write_text(json.dumps(environment, indent=2) + "\n")
        print("verified environment pins", len(measured), "python", environment["python"], flush=True)
    # KB2 2026-10-13 (KBUILD.md 7c W13): the manifest must be the one of the package index (an older copy sits in
    # serving_00.tar.gz and in the C26 adaptation archives; unpacked after serving_03.tar.gz it replaces the current one)
    index = ROOT / "artifacts/claude_ref/asm2/package/serving_archives.json"
    if index.is_file() and a.manifest.resolve() == (ROOT / "artifacts/claude_ref/asm2/serving_inputs_sha256.txt").resolve():
        want = json.loads(index.read_text())[-1].get("manifest_sha256")
        assert want is None or hashlib.sha256(a.manifest.read_bytes()).hexdigest() == want, \
            "serving_inputs_sha256.txt is not the manifest of serving_archives.json: unpack serving_03.tar.gz again (docs/reference_code/asm2_unpack.py)"
    for line in a.manifest.read_text().splitlines():
        if not line.strip(): continue
        digest, size, name = line.split(maxsplit=2)
        rel = Path(name)
        assert not rel.is_absolute() and ".." not in rel.parts and name.startswith("artifacts/"), name
        p = ROOT / rel  # VM worktrees intentionally share the main root's artifact symlink.
        assert p.is_file() and p.stat().st_size == int(size), f"missing/size mismatch: {name}"
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(2**20), b""): h.update(b)
        assert h.hexdigest() == digest, f"input sha mismatch: {name}"
        count += 1; total += int(size)
    print("verified fitted inputs", count, "bytes", total, flush=True)


if __name__ == "__main__": main()
