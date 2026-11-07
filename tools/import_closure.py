"""Static import closure of entry scripts, with git tracking status (C, 2026-10-06, retro fix).

  python tools/import_closure.py ENTRY.py [ENTRY2.py ...] [--sh docs/reference_code/asm2_run_all.sh] [--untracked]

Follows `import x` / `from x import y` (and importlib-free sys.path tricks are not followed: list such files as
entries) through docs/reference_code/, scripts/ and src/. Prints each local module with TRACKED/UNTRACKED and, with
--untracked, a `git add` line for the user or L (Claude makes no commits). --sh adds every *.py named in a shell script.
"""
import argparse, ast, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = [ROOT / "docs/reference_code", ROOT / "scripts", ROOT / "src", ROOT / "tools"]


def resolve(name):
    parts = name.split(".")
    for d in DIRS:
        for cand in (d.joinpath(*parts).with_suffix(".py"), d.joinpath(*parts) / "__init__.py"):
            if cand.exists():
                return cand
    return None


def imports(path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.append(n.module)
            out += [f"{n.module}.{a.name}" for a in n.names]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entries", nargs="*"); ap.add_argument("--sh", action="append", default=[])
    ap.add_argument("--untracked", action="store_true")
    a = ap.parse_args()
    todo = [Path(e).resolve() for e in a.entries]
    for sh in a.sh:
        text = Path(sh).read_text(encoding="utf-8")
        for m in sorted(set(re.findall(r"([\w/]+\.py)", text))):
            p = resolve(Path(m).stem) or (ROOT / m if (ROOT / m).exists() else None)
            if p:
                todo.append(p.resolve())
    seen = set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        for name in imports(p):
            q = resolve(name)
            if q and q.resolve() not in seen:
                todo.append(q.resolve())
    tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.splitlines())
    rel = sorted(p.relative_to(ROOT).as_posix() for p in seen)
    un = [r for r in rel if r not in tracked]
    for r in rel:
        print(("TRACKED   " if r in tracked else "UNTRACKED ") + r)
    print(f"\n{len(rel)} local modules, {len(un)} untracked")
    if a.untracked and un:
        print("\ngit add " + " ".join(f'"{u}"' for u in un))


if __name__ == "__main__":
    main()
