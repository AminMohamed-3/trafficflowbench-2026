"""R3 (rows 28-35): render a tune template. The templates in src/r3lib/tunes are the original tune files with
  - each b6 arm folder replaced by @S:<KEY>@ (KEY = serve key; given as KEY=FOLDER arguments; S2B = the S2b 48k served arm of row 26:
    $R3_S2B_SERVE if set, else the stored artifacts/claude_ref/l_b6/s2b/arm48k), written as a path relative to artifacts/claude_ref
    (asm2_build reads CR / arm dir; an absolute Windows path would break the ':' split of the arm string);
  - each S7c tree file of row 35 replaced by @W:<old path under artifacts/claude_ref>@ (resolved like wdir.py: fetched copy under
    $FS/weights or the old path; sha checked by the piece script's manifest check), written absolute.
  python tune.py TEMPLATE.json OUT.json [KEY=FOLDER ...]"""
import json, os, re, sys
from pathlib import Path
ROOT = Path(os.environ["TFB_ROOT"]); CR = ROOT / "artifacts/claude_ref"
FS = Path(os.environ.get("FS") or Path(__file__).resolve().parents[4])
tpl, out, *kv = sys.argv[1:]
S = dict(x.split("=", 1) for x in kv)
S.setdefault("S2B", os.environ.get("R3_S2B_SERVE") or str(CR / "l_b6/s2b/arm48k"))
t = Path(tpl).read_text()
def srv(m):
    k = m.group(1); assert k in S, f"no folder for serve key {k}"
    return os.path.relpath(S[k], CR).replace("\\", "/")
def wts(m):
    import subprocess
    d, f = m.group(1).rsplit("/", 1)
    base = subprocess.run([sys.executable, str(Path(__file__).with_name("wdir.py")), d], capture_output=True, text=True, check=True).stdout.strip()
    return base + "/" + f
t = re.sub(r"@S:([A-Z0-9_]+)@", srv, t)
t = re.sub(r"@W:([^@]+)@", wts, t)
assert "@" not in t, "unrendered placeholder"
json.loads(t)
Path(out).write_text(t)
print("tune", out)
