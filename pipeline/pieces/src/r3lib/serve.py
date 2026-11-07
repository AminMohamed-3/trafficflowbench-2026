"""Rows 28-35 (worker R3, 2026-10-21): CPU official-view serve of 30 B6R members (FV-FT, FV2, FV-INEL, FV-INEL W1.0).

Body = exp/L_fvft/fv2_serve.py serve() / exp/L_fvinel/fvi_serve.py serve(): mar_serve.members (vendor/mar_serve.py, verbatim copy
of exp/L_fvft/mar_serve.py = the s2_b6r.serve body) + F._save, with the weights folder, iterations, period and output folder as
arguments instead of fixed artifacts/ paths. Needs TFB_ROOT (old-layout root holding exp/L_b6, src, data) in the environment.

  python serve.py WEIGHTS_DIR ITERS PERIOD OUTDIR [STORED_RECEIPT.json]

STORED_RECEIPT (the original serve receipt.json): every member's speed and flow array must have the stored sha256 (serve parity,
array level; the npz container bytes are not compared). Last line SERVE_OK <n> <outdir> [parity EXACT] or an exception.
Run on CPU (CUDA_VISIBLE_DEVICES=-1), as the original serves."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, sys, time, hashlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
assert os.environ.get("TFB_ROOT"), "set TFB_ROOT"
sys.path.insert(0, str(HERE / "vendor"))
import mar_serve as MS
B = MS.B; F = MS.F


def main(wdir, iters, period, outdir, stored=None):
    import torch as _t
    wdir = Path(wdir); out = Path(outdir); out.mkdir(parents=True, exist_ok=True); iters = int(iters)
    ref = json.loads(Path(stored).read_text())["files"] if stored else None
    rec = dict(iterations=iters, width=32, arch="UNetS2", seeds=[0, 1, 2], period=period, root=str(wdir).replace("\\", "/"), device="cpu",
               torch=_t.__version__, code="final_solution/pipeline/pieces/src/r3lib/serve.py (mar_serve.members = s2_b6r.serve body)", files={})
    t0 = time.time(); bad = []
    for pn, s, name, iv, iq, dates, w, h, run in MS.members(wdir, iters, list(B.PANELS), period=period):
        f = out / name; F._save(f, speed=iv, flow=iq, dates=np.array(dates))
        sv = hashlib.sha256(np.ascontiguousarray(iv).tobytes()).hexdigest(); sq = hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest()
        rec["files"][name] = dict(weights=str(w).replace("\\", "/"), weights_sha256=h, speed_sha256=sv, flow_sha256=sq, shape=list(iv.shape))
        if ref is not None:
            r = ref.get(name, {})
            ok = r.get("speed_sha256") == sv and r.get("flow_sha256") == sq and r.get("weights_sha256") == h
            if not ok:
                bad.append(name)
        print("serve", period, pn, s, round(time.time() - t0), "s", flush=True)
    assert len(rec["files"]) == 30, len(rec["files"])
    if ref is not None:
        rec["parity"] = dict(stored_receipt=str(stored).replace("\\", "/"), exact=not bad, differing=bad)
    (out / "receipt.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    if ref is not None and bad:
        raise SystemExit(f"SERVE_PARITY FAIL {len(bad)} members differ from {stored}: {bad[:5]}")
    print("SERVE_OK", len(rec["files"]), out, "parity EXACT (30 members, speed+flow sha == stored receipt)" if ref is not None else "", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
