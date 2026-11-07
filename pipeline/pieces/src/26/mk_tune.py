"""Write a copy of a tune json with the registered files substituted (worker R2 2026-10-21; values unchanged, only paths):
  --arm-dir DIR   b6 arm folder (relative to artifacts/claude_ref, the form asm2_build.parse_arm expects)
  --b6w FILE      b6 blend weights of both splits
  --s3 SPLIT=Q,V  stage-3 model pair of tune "s3_split"[SPLIT] (row 26 S7b models)
Paths are written relative to the development repo root when inside it (asm2_build joins ROOT / path), else absolute.
Usage: python mk_tune.py IN.json OUT.json [--arm-dir DIR] [--b6w FILE] [--s3 validation=Q,V] [--s3 private=Q,V]"""
import os, sys, json
from pathlib import Path
ROOT = Path(os.environ.get("TFB_OLD_REPO") or Path(__file__).resolve().parents[5]).resolve()
a = sys.argv[1:]; src, dst = a[0], a[1]; a = a[2:]
T = json.loads(Path(src).read_text())


def rp(x):
    x = Path(x).resolve()
    try:
        return x.relative_to(ROOT).as_posix()
    except ValueError:
        return x.as_posix()


while a:
    k, v = a[0], a[1]; a = a[2:]
    if k == "--arm-dir":
        d = os.path.relpath(Path(v).resolve(), ROOT / "artifacts/claude_ref").replace("\\", "/"); assert ":" not in d
        for x in T["b6"].values():
            s = x["arm"].split(":"); s[0] = d; x["arm"] = ":".join(s)
    elif k == "--b6w":
        for x in T["b6"].values():
            x["weights"] = rp(v)
    elif k == "--s3":
        sp, pair = v.split("=", 1); q, vv = pair.split(",")
        assert sp in T["s3_split"]; T["s3_split"][sp] = [[rp(q), rp(vv)]]
    else:
        raise SystemExit(f"unknown {k}")
Path(dst).write_text(json.dumps(T, indent=1))
print("TUNE", dst, json.dumps({k: T[k] for k in ("b6", "s3_split") if k in T}), flush=True)
