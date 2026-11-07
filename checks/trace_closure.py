"""Static code closure of the base build (C47) for final_solution/ (worker R1, 2026-10-21).

  python checks/trace_closure.py --src-root OLD_REPO [--copy] [--out checks/code_files.csv]

Entries: every *.py named in docs/reference_code/asm2_run_all.sh (default path and its optional flags) plus asm2_unpack.py.
Follows, conservatively (imports inside functions and on every branch count, as exp/L_stinel/s8i_make.py did):
  import x / from x import y (relative imports resolved), importlib.import_module("x"), __import__("x"),
  importlib.util.spec_from_file_location(..., ROOT / "tools" / "diskguard.py") style path chains,
and every string constant or ROOT / "a" / "b" chain that names an existing non-artifact, non-data repo file
(json/yaml/txt/py/sh/md/csv read at run time: protocol.json, requirements-final.txt, configs).
Module search dirs = the sys.path entries the build code adds: src/ (package tfb), docs/reference_code/, scripts/, tools/.
A name found in several dirs is taken from all of them (the copy keeps every candidate, so shadowing stays as in the repo).
With --copy the files are copied byte for byte to the same relative paths under final_solution/ (the build binds
paths: ROOT = parents[2] of docs/reference_code/*.py, sys.path ROOT/src and ROOT/docs/reference_code, receipt source hashes).
"""
import argparse, ast, csv, hashlib, re, shutil, subprocess, sys
from pathlib import Path

FINAL = Path(__file__).resolve().parents[1]
SCRIPT_DIRS = ["docs/reference_code", "scripts", "tools"]
SKIP_PREFIX = ("artifacts/", "data/", ".git/", ".venv", "final_solution/",
               "submissions/")   # prior submission CSVs are never build inputs (asm2_final runs with --queue-base none)
SKIP_FILES = {"LEDGER.md": "project work ledger, written only by experiment runners (tools/ledger.py), not by the build",
              "PLAN.md": "root marker for tfb.paths.project_root (TFB_ROOT check); final_solution ships its own stub"}
EXTRA = ["docs/reference_code/asm2_run_all.sh", "docs/reference_code/asm2_unpack.py", "docs/reference_code/asm2_step.py",
         "docs/report/requirements-final.txt"]


def consts(tree):
    """String constants and joined ROOT / "a" / "b" chains."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and 2 < len(n.value) < 200:
            out.add(n.value)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
            parts, cur = [], n
            while isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Div):
                if isinstance(cur.right, ast.Constant) and isinstance(cur.right.value, str):
                    parts.insert(0, cur.right.value)
                else:
                    parts = []
                cur = cur.left
            if isinstance(cur, ast.Constant) and isinstance(cur.value, str):
                parts.insert(0, cur.value)
            if len(parts) >= 2:
                out.add("/".join(parts))
        if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", "")) == "join":
            ps = [a.value for a in n.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            if len(ps) >= 2:
                out.add("/".join(ps))
    return out


def imports(root, rel):
    tree = ast.parse((root / rel).read_text(encoding="utf-8", errors="replace"))
    pkg = rel[4:-3].replace("/", ".") if rel.startswith("src/") else None
    if pkg and pkg.endswith(".__init__"):
        pkg = pkg[:-9] + ".x"
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            base = n.module or ""
            if n.level and pkg:
                parts = pkg.split(".")[:-n.level]; base = ".".join(parts + ([base] if base else []))
            names.add(base); names |= {f"{base}.{a.name}" for a in n.names}
        elif isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) in ("import_module", "__import__") \
                and n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
            names.add(n.args[0].value)
    return names, consts(tree)


def resolve(root, name):
    parts = [p for p in name.split(".") if p]
    if not parts:
        return []
    fs = []
    if parts[0] == "tfb":
        fs += [f"src/{'/'.join(parts[:k])}/__init__.py" for k in range(1, len(parts) + 1)]
        fs += [f"src/{'/'.join(parts)}.py"]
    for d in SCRIPT_DIRS:
        fs += [f"{d}/{'/'.join(parts)}.py", f"{d}/{'/'.join(parts)}/__init__.py"]
        if len(parts) > 1:   # from module import name: the module itself
            fs += [f"{d}/{'/'.join(parts[:-1])}.py"]
    return [f for f in dict.fromkeys(fs) if (root / f).is_file()]


def literal_files(root, s):
    s = s.strip().lstrip("./").replace("\\", "/")
    if not s or s.startswith(SKIP_PREFIX) or s in SKIP_FILES or "*" in s or "{" in s or "\n" in s or " " in s:
        return []
    if not re.search(r"\.(py|json|ya?ml|txt|sh|csv|md|toml)$", s):
        return []
    p = root / s
    return [s] if p.is_file() else []


def closure(root):
    sh = (root / "docs/reference_code/asm2_run_all.sh").read_text(encoding="utf-8")
    todo = []
    for m in sorted(set(re.findall(r"([\w/$]+\.py)", sh))):
        stem = m.split("/")[-1]
        cands = [m] if (root / m).is_file() else [f"docs/reference_code/{stem}", f"scripts/{stem}"]
        todo += [c for c in cands if (root / c).is_file()]
    todo += [e for e in EXTRA if e.endswith(".py")]
    todo += [m for m in sorted(set(re.findall(r"((?:docs|exp|configs)/[\w/]+\.(?:json|ya?ml))", sh))) if (root / m).is_file()]
    seen, why = set(), {}
    for t in todo:
        why.setdefault(t, "entry (asm2_run_all.sh)")
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        if not f.endswith(".py"):
            continue
        names, lits = imports(root, f)
        for m in names:
            for g in resolve(root, m):
                if g not in seen:
                    why.setdefault(g, f"import {m} in {f}"); todo.append(g)
        for s in lits:
            for g in literal_files(root, s):
                if g not in seen:
                    why.setdefault(g, f"path literal in {f}"); todo.append(g)
    for e in EXTRA:
        seen.add(e); why.setdefault(e, "entry / read by the build")
    # exp/<dir> named anywhere in the code (also inside f-strings and parents[k] / "exp/..." forms): its tracked
    # json/yaml records directly in that dir (protocols, references, configs read at run time), < 1 MB each
    tracked = set(subprocess.run(["git", "ls-files", "exp"], cwd=root, capture_output=True, text=True).stdout.splitlines())
    dirs = set()
    for f in [s for s in seen if s.endswith(".py")]:
        dirs |= set(re.findall(r"exp/([A-Za-z0-9_]+)", (root / f).read_text(encoding="utf-8", errors="replace")))
    for d in sorted(dirs):
        for t in sorted(tracked):
            if t.startswith(f"exp/{d}/") and t.count("/") == 2 and re.search(r"\.(json|ya?ml)$", t) \
                    and ".lane_branch." not in t and (root / t).stat().st_size < 2**20 and t not in seen:
                seen.add(t); why[t] = f"record in exp/{d} (dir named in the code)"
    return sorted(seen), why


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=FINAL / "checks/code_files.csv")
    ap.add_argument("--copy", action="store_true")
    a = ap.parse_args()
    root = a.src_root.resolve()
    files, why = closure(root)
    rows = []
    for f in files:
        src = root / f
        rows.append(dict(path=f, sha256=sha(src), bytes=src.stat().st_size, reason=why[f]))
        if a.copy:
            dst = FINAL / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            assert sha(dst) == rows[-1]["sha256"], f
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "sha256", "bytes", "reason"], lineterminator="\n"); w.writeheader(); w.writerows(rows)
    by = {}
    for r in rows:
        k = r["path"].split("/")[0] if not r["path"].startswith("src/tfb/") else "/".join(r["path"].split("/")[:3])
        by[k] = by.get(k, 0) + 1
    print(f"{len(rows)} files, {sum(r['bytes'] for r in rows)} bytes; by dir: {by}")


if __name__ == "__main__":
    main()
