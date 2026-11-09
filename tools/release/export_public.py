"""Copy the public repository out of final_solution/ into a clean folder (no history), then sweep it.

  python tools/release/export_public.py OUT_DIR [--site-branch-dir DIR]

Copies every git-tracked file of final_solution/ except the internal notes listed in DROP, refuses to finish if the sweep
finds a credential-like string, and writes OUT_DIR/.export_receipt.json (file count, bytes, source commit).
With --site-branch-dir, also writes the solution site (report/site/index.html) as DIR/index.html + DIR/.nojekyll for the
gh-pages branch. Existing files in OUT_DIR are overwritten; files that are no longer exported are listed, not deleted.
"""
import json, re, shutil, subprocess, sys
from pathlib import Path

FS = Path(__file__).resolve().parents[2]
DROP = re.compile(r"^(report/html/|report/OPEN_ITEMS\.md$|report/figures/TODO_F6_F7\.md$|pipeline/pieces/BRIEF_R2\.md$|outputs/)")
SECRET = re.compile(rb"gho_[A-Za-z0-9]{20}|ghp_[A-Za-z0-9]{20}|KGAT[A-Za-z0-9]{8}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY")
LOCAL = re.compile(rb"(?i)(?:[a-z]:|/[a-z]|/mnt/[a-z])(?:\\\\|\\|/)+data(?:\\\\|\\|/)+projects(?:\\\\|\\|/)+money making"
                   rb"(?:\\\\|\\|/)+work(?:\\\\|\\|/)+ieee big data-traffic")


def main():
    out = Path(sys.argv[1]).resolve()
    site = Path(sys.argv[sys.argv.index("--site-branch-dir") + 1]).resolve() if "--site-branch-dir" in sys.argv else None
    files = subprocess.run(["git", "-c", "core.quotepath=off", "ls-files", "."], cwd=FS, capture_output=True, text=True,
                           encoding="utf-8", check=True).stdout.splitlines()
    keep = [f for f in files if not DROP.match(f) and (FS / f).is_file()]
    hashed = {r.split(",")[0] for r in (FS / "checks/code_files.csv").read_text(encoding="utf-8").splitlines()[1:]}
    bad, nbytes, masked = [], 0, []
    for f in keep:
        src, dst = FS / f, out / f
        b = src.read_bytes()
        if SECRET.search(b):
            bad.append(f)
        if LOCAL.search(b) and not f.endswith((".py", ".sh")) and f not in hashed:
            b = LOCAL.sub(b"<DEVROOT>", b); masked.append(f)   # logs and notes: hide the author's local folder
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b)
        nbytes += len(b)
    if bad:
        sys.exit(f"SECRET-LIKE STRING in: {bad}")
    old = [str(p.relative_to(out)).replace("\\", "/") for p in out.rglob("*") if p.is_file() and ".git" not in p.parts
           and p.name != ".export_receipt.json"]
    stale = sorted(set(old) - set(keep))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=FS, capture_output=True, text=True, check=True).stdout.strip()
    rec = dict(files=len(keep), bytes=nbytes, source_commit=head, dropped=[f for f in files if DROP.match(f)], stale=stale,
               local_path_masked=masked)
    (out / ".export_receipt.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    if site:
        site.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FS / "report/site/index.html", site / "index.html")
        (site / ".nojekyll").write_bytes(b"")
    print(f"EXPORT_OK {len(keep)} files, {nbytes / 1e6:.1f} MB, dropped {len(rec['dropped'])}, stale {len(stale)}")


if __name__ == "__main__":
    main()
