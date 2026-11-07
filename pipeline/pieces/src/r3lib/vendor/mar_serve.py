"""L_fvft attempt 2: CPU official-view serve of B6R members, line for line the s2_b6r.serve body (M.layer + s2_inputs, predict,
F._save, receipt), for ONE split and a given weights folder.
  serve OUTDIR [PANELS]     FV members (artifacts/claude_ref/l_fvft/mar/w/it54000_w32, iters 54000) -> OUTDIR/imp_valown_*.npz + receipt.json
  parity OUT.json [PANELS]  S2b weights (artifacts/claude_ref/l_b6/s2b/it48000_w32, iters 48000) through the same code, compared with the
                            shipped S2b served files (artifacts/claude_ref/l_b6/s2b/arm48k) array for array: must be EXACT (nothing written
                            but OUT.json)
Run on CPU (CUDA_VISIBLE_DEVICES=-1), as the S2b serve."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, sys, time, hashlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fvft_step as FS
B = FS.B; F = FS.F; ROOT = FS.ROOT
PERIOD = os.environ.get("FVFT_PERIOD", "validation")
FVW = ROOT / "artifacts/claude_ref/l_fvft/mar/w/it54000_w32"
S2BW = ROOT / "artifacts/claude_ref/l_b6/s2b/it48000_w32"
S2B_SERVED = ROOT / "artifacts/claude_ref/l_b6/s2b/arm48k"


def members(wdir, iters, panels, seeds=(0, 1, 2), period=PERIOD):
    """Yields (panel, seed, file name, iv, iq, dates, weights path, sha, run) exactly as s2_b6r.serve computes them."""
    torch = F._flags(); M = F.K.b6()
    assert M.DEV == "cpu", "serve on CPU (as the S2b serve)"
    for pn in panels:
        sp, fl, vis, stations, dates = M.layer(B.MODE[period], pn); dates = [str(x) for x in dates]
        Xn, vf, cap, lanes = B.s2_inputs(pn, period, sp, fl, vis, stations)
        for s in seeds:
            key = f"{B.TAG[period]}_{pn}_s{s}"; w, r = wdir / f"sd_{key}.pt", wdir / f"run_{key}.json"
            run = json.loads(r.read_text()); h = B.fsha(w)
            assert h == run["sd_sha256"], f"weights do not match their receipt: {w}"
            z = torch.load(w, map_location="cpu", weights_only=True)
            assert (z["iters"], z["width"], z["seed"], z["period"], z["panel"], z["arch"]) == (iters, 32, s, period, pn, "UNetS2"), w
            model = B.make_model(torch, M).to(M.DEV); model.load_state_dict(z["state_dict"]); model.eval()
            iv, iq = B.predict(torch, M, model, Xn, vf, cap); del model
            assert iv.shape == sp.shape and np.isfinite(iv).all() and np.isfinite(iq).all()
            yield pn, s, f"imp_{B.OTAG[period]}_{pn}_s{s}.npz", iv, iq, dates, w, h, run
        B._CONST.clear()


def serve(outdir, panels=None):
    import torch as _t
    panels = panels.split(",") if panels else list(B.PANELS)
    out = ROOT / outdir; out.mkdir(parents=True, exist_ok=True); rp = out / "receipt.json"
    rec = json.loads(rp.read_text()) if rp.exists() else dict(iterations=54000, width=32, arch="UNetS2", seeds=[0, 1, 2], period=PERIOD,
                                                               root=str(FVW.relative_to(ROOT)).replace("\\", "/"), device="cpu", torch=_t.__version__,
                                                               inputs="exp/L_b6/s2_b6r.py s2_inputs (official view, OWN-fitted constants)",
                                                               code="exp/L_fvft/mar_serve.py (s2_b6r.serve body)", files={})
    t0 = time.time()
    for pn, s, name, iv, iq, dates, w, h, run in members(FVW, 54000, panels):
        f = out / name; F._save(f, speed=iv, flow=iq, dates=np.array(dates))
        rec["files"][name] = dict(weights=str(w.relative_to(ROOT)).replace("\\", "/"), weights_sha256=h, trained_on=run.get("device"),
                                  speed_sha256=hashlib.sha256(np.ascontiguousarray(iv).tobytes()).hexdigest(),
                                  flow_sha256=hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest(), shape=list(iv.shape),
                                  npz_sha256=B.fsha(f))
        print("fv serve", PERIOD, pn, s, round(time.time() - t0), "s", flush=True)
        rp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("FVFT_SERVE_OK", len(rec["files"]), "member files in", out, flush=True)


def parity(outp, panels=None):
    panels = panels.split(",") if panels else list(B.PANELS)
    res = dict(rule="S2b weights through mar_serve.members == arm48k arrays exactly", rows=[], ok=True); t0 = time.time()
    shipped = json.loads((S2B_SERVED / "receipt.json").read_text())["files"]
    for pn, s, name, iv, iq, dates, w, h, run in members(S2BW, 48000, panels):
        z = np.load(S2B_SERVED / name)
        ok = bool(np.array_equal(iv, z["speed"]) and np.array_equal(iq, z["flow"]) and list(z["dates"]) == dates
                  and shipped[name]["weights_sha256"] == h)
        res["rows"].append(dict(file=name, exact=ok, max_dv=float(np.abs(iv - z["speed"]).max()), max_dq=float(np.abs(iq - z["flow"]).max())))
        res["ok"] &= ok
        print("parity", name, "EXACT" if ok else "DIFF", round(time.time() - t0), "s", flush=True)
    Path(outp).write_text(json.dumps(res, indent=1))
    print("FVFT_SERVE_PARITY", "PASS" if res["ok"] else "FAIL", len(res["rows"]), "members", flush=True)
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    dict(serve=serve, parity=parity)[cmd](*a)
