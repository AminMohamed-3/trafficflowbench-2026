"""Import smoke test of the final_solution code closure (worker R1, 2026-10-21).

  python checks/import_smoke.py [--log outputs/logs/import_smoke_audit.tsv]

For every *.py entry named in docs/reference_code/asm2_run_all.sh: a fresh process, cwd = final_solution, TFB_ROOT =
final_solution, PYTHONPATH = final_solution/src + checks/open_audit, runs the script's top level with
runpy.run_path(run_name="__smoke__") (all entries guard main(), so nothing is computed: sys.path set-up and every
top-level import run). One more process imports every src/tfb module of checks/code_files.csv. Any exception = FAIL.
The open audit (checks/open_audit/sitecustomize.py) logs every Python-level open outside final_solution and the
interpreter; the release link is allowed. PASS = no failure and an empty audit log.
"""
import argparse, csv, os, re, subprocess, sys
from pathlib import Path

FS = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=FS / "outputs/logs/import_smoke_audit.tsv")
    a = ap.parse_args()
    a.log.parent.mkdir(parents=True, exist_ok=True)
    if a.log.exists():
        a.log.unlink()
    env = dict(os.environ, TFB_ROOT=str(FS), TFB_OPEN_AUDIT_LOG=str(a.log), CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join([str(FS / "src"), str(FS / "checks/open_audit")]),
               TFB_OPEN_AUDIT_ALLOW=os.path.realpath(FS / "data/raw/kaggle_public"))
    sh = (FS / "docs/reference_code/asm2_run_all.sh").read_text(encoding="utf-8")
    entries = []
    for m in sorted(set(re.findall(r"([A-Za-z0-9_]+\.py)", sh))):
        for d in ("docs/reference_code", "scripts"):
            if (FS / d / m).is_file():
                entries.append(f"{d}/{m}"); break
        else:
            print(f"MISSING entry {m}"); entries.append(None)
    with open(FS / "checks/code_files.csv", newline="", encoding="utf-8") as f:
        mods = [r["path"][4:-3].replace("/", ".").removesuffix(".__init__") for r in csv.DictReader(f)
                if r["path"].startswith("src/tfb/") and r["path"].endswith(".py")]
    fails = [e for e in entries if e is None]
    for e in [e for e in entries if e]:
        code = (f"import os, runpy, sys; sys.argv = [{e!r}]; sys.path[0] = os.path.abspath(os.path.dirname({e!r})); "   # as asm2_step.py
                f"runpy.run_path({e!r}, run_name='__smoke__')")
        r = subprocess.run([sys.executable, "-c", code], cwd=FS, env=env, capture_output=True, text=True)
        print(f"{'ok  ' if r.returncode == 0 else 'FAIL'} {e}" + ("" if r.returncode == 0 else "\n    " + r.stderr.strip().splitlines()[-1][:300]), flush=True)
        if r.returncode:
            fails.append(e)
    code = "import importlib, sys\nbad = []\nfor m in sys.argv[1:]:\n    try: importlib.import_module(m)\n    except Exception as x: bad.append(f'{m}: {type(x).__name__}: {x}'[:300])\nprint('\\n'.join(bad)); sys.exit(1 if bad else 0)"
    r = subprocess.run([sys.executable, "-c", code, *mods], cwd=FS, env=env, capture_output=True, text=True)
    print(f"{'ok  ' if r.returncode == 0 else 'FAIL'} import of {len(mods)} src/tfb modules" + ("" if r.returncode == 0 else "\n    " + r.stdout.strip().replace(chr(10), chr(10) + "    ")), flush=True)
    if r.returncode:
        fails.append("src/tfb modules")
    audit = a.log.read_text(encoding="utf-8").splitlines() if a.log.exists() else []
    # Windows development laptop only: exp/0354_d_f4_density_score/inputs.json names the sibling worktree
    # C:/.../work/tfb-wt/lane-s as S_source_root; docs/reference_code/asm2_phys_lwr_diag.py uses it when it exists
    # (laptop) and falls back to final_solution/src on any other machine (Linux, clean clone): reported, not counted.
    wt = [l for l in audit if "tfb-wt" in l.split("\t")[2]]
    audit = [l for l in audit if l not in wt]
    if wt:
        print(f"NOTE {len(wt)} opens under the sibling worktree tfb-wt (laptop-only fallback of asm2_phys_lwr_diag.py)")
    for line in audit[:20]:
        print("AUDIT " + line)
    print(f"IMPORT_SMOKE {'PASS' if not fails and not audit else 'FAIL'}: {len(entries)} entries, {len(mods)} modules, "
          f"{len(fails)} failures, {len(audit)} opens outside final_solution")
    sys.exit(1 if fails or audit else 0)


if __name__ == "__main__":
    main()
