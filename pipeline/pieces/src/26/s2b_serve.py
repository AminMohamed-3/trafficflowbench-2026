"""Row 26 step 1: CPU official-view serve of the 60 S2b B6R members (worker R2 2026-10-21). COPY of exp/L_b6/s2ab_serve.py mode b
(sha256 990fe0648770b97bd5233eda1a842adc8236e93d2b1f5d1121cd843c33755e68): s2b_b6r.B.serve(out, root, seeds=(0, 1, 2)) =
exp/L_b6/s2_b6r.py serve (UNetS2 cin 16, s2_inputs, official view, CPU), unchanged. Parameterised: the weights root (the folder that
holds it48000_w32/ with sd_*.pt + run_*.json; default the registered weights of row 26) and --panels (cheap parity: one panel).
Model code imported from the development repo ($TFB_OLD_REPO/exp/L_b6: s2b_b6r.py, s2_b6r.py; env L_B6_SRC overrides).
Usage: python s2b_serve.py OUT_DIR [--root DIR] [--panels P1,P2]"""
import os, sys, json
from pathlib import Path
ROOT = Path(os.environ.get("TFB_OLD_REPO") or Path(__file__).resolve().parents[5]).resolve()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
a = sys.argv[1:]
def opt(f):
    if f in a:
        i = a.index(f); v = a[i + 1]; del a[i:i + 2]; return v
root, panels = opt("--root"), opt("--panels")
out = a[0]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(os.environ.get("L_B6_SRC") or ROOT / "exp/L_b6")))
os.chdir(ROOT)
import s2b_b6r as BB
r = os.path.relpath(Path(root).resolve(), ROOT).replace("\\", "/") if root else "artifacts/claude_ref/l_b6/s2b"
BB.B.serve(out, r, panels=panels.split(",") if panels else None, seeds=(0, 1, 2))
rec = json.loads((Path(out) / "receipt.json").read_text())
n = 60 if not panels else 6 * len(panels.split(","))
assert len(rec["files"]) == n, (len(rec["files"]), n)
print("S2X_SERVE_OK", len(rec["files"]), flush=True)
