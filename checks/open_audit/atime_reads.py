"""List files under a development repo whose last-access time is at or after a cutoff (NTFS atime trace of a build run).

  python atime_reads.py REPO_ROOT CUTOFF_EPOCH OUT_DIR

Skips junctions and symlinks (the repo links data, caches and venvs elsewhere), .git, .claude, .venv*, __pycache__ and the
build's own work and output folders. Writes OUT_DIR/atime_reads.tsv (relpath, bytes, atime UTC) and a per-top-folder summary.
"""
import os, sys, time
from collections import Counter

root, cutoff, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
SKIP_DIRS = {".git", ".claude", "__pycache__", "node_modules"}
SKIP_REL = ("final_solution" + os.sep + "work", "final_solution" + os.sep + "outputs")
rows = []


def walk(d):
    try:
        it = list(os.scandir(d))
    except OSError:
        return
    for e in it:
        try:
            if e.is_symlink() or os.path.isjunction(e.path):
                continue
            rel = os.path.relpath(e.path, root)
            if e.is_dir(follow_symlinks=False):
                if e.name in SKIP_DIRS or e.name.startswith(".venv") or rel.startswith(SKIP_REL):
                    continue
                walk(e.path)
            else:
                st = e.stat(follow_symlinks=False)
                if st.st_atime >= cutoff:
                    rows.append((rel.replace(os.sep, "/"), st.st_size, st.st_atime))
        except OSError:
            pass


walk(root)
rows.sort()
with open(os.path.join(out, "atime_reads.tsv"), "w", encoding="utf-8", newline="\n") as f:
    f.write("path\tbytes\tatime_utc\n")
    for p, b, a in rows:
        f.write(f"{p}\t{b}\t{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(a))}\n")
top = Counter(); size = Counter()
for p, b, _ in rows:
    k = "/".join(p.split("/")[:3]); top[k] += 1; size[k] += b
with open(os.path.join(out, "atime_reads_summary.txt"), "w", encoding="utf-8", newline="\n") as f:
    for k, n in top.most_common():
        f.write(f"{n:6d} files {size[k] / 1e6:10.1f} MB  {k}\n")
print(f"atime_reads: {len(rows)} files, {sum(r[1] for r in rows) / 1e9:.2f} GB at or after {time.strftime('%FT%TZ', time.gmtime(cutoff))}")
