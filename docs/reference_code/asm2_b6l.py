"""asm2 build step for the long-trained B6 imputer members (ADOPTED row 16; exp/C_b6long/B6LONG.md sections 9.4 and 14).

The Task 1 chain reads, per split, panel and seed, one official-view member file imp_<valown|privown>_<panel>_s<seed>.npz
(the U-Net trained on the split's OWN layer, applied to the official view). This step makes these 60 files in the run
directory from the SHIPPED trained weights, on CPU, with no training:
  serve --it 12000 [--panel-it PANEL:ITERS,...] [--roots DIR,DIR] --out RUNDIR [--refit]
      weights  <root>/it<ITERS>_w32/sd_<own|ownapr>_<panel>_s<seed>.pt   (first root that has the file; default root
               artifacts/claude_ref/b6l), with the training receipt run_<own|ownapr>_<panel>_s<seed>.json next to it:
               the weight file's sha256 must equal the receipt's sd_sha256, and iterations / width / seed / panel / split
               must match (so a 24000-iteration set is a drop-in: --it 24000, or per panel with --panel-it).
      output   RUNDIR/imp_<valown|privown>_<panel>_s<seed>.npz (speed, flow, dates; same arrays as
               c_b6long_full_step.py official) and RUNDIR/receipt.json (weights used, their sha256, array sha256).
      --refit  first retrain every member on the GPU (c_b6long_full_step.py trainf, deterministic kernels, the OWN mask
               rebuilt from the cache with the hash check) into RUNDIR/refit, then serve from those weights.
  tune --shipped TUNE.json [--base USER.json] --out OUT.json
      the tune file the Task 1 chain gets: the user's knobs plus the shipped b6 block. Prints USER_B6 and writes nothing
      when the user's file has its own b6 block (then that block is used as it is and no member is served).
Run from the repo root. CPU inference of a GPU-trained net is not bit-equal to the GPU's own inference (float32 noise);
exp/C_b6long/B6LONG.md section 14 gives the measured size."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
PANELS = ["D12_I405_N", "D12_I405_S", "D12_I5_N", "D12_I5_S", "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S"]
SPLITS = {"validation": ("own", "valown", "val"), "private": ("ownapr", "privown", "priv")}   # training tag, official tag, layer mode
WIDTH = 32; SEEDS = (0, 1, 2); DEFAULT_ROOT = "artifacts/claude_ref/b6l"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")


def find(roots, it, tag, panel, seed):
    cfg = f"it{it}_w{WIDTH}"; name = f"sd_{tag}_{panel}_s{seed}.pt"
    for r in roots:
        d = ROOT / r / cfg
        if (d / name).is_file():
            return d / name, d / f"run_{tag}_{panel}_s{seed}.json"
    raise SystemExit(f"B6 weights missing: {cfg}/{name} under {roots} (ship them, pass --b6-root, or use --refit-b6)")


def serve(a):
    import torch
    import c_b6long_full_step as F
    torch = F._flags(); M = F.K.b6()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True); roots = [r.strip().rstrip("/") for r in a.roots.split(",") if r.strip()]
    pit = dict(x.split(":") for x in a.panel_it.split(",")) if a.panel_it else {}
    assert set(pit) <= set(PANELS), f"unknown panel in --panel-it: {sorted(set(pit) - set(PANELS))}"
    its = {pn: int(pit.get(pn, a.it)) for pn in PANELS}
    if a.refit:   # GPU retrain, deterministic kernels (B6LONG_DET default 1); weights + receipts into RUNDIR/refit
        assert M.DEV == "cuda", "--refit-b6 needs one CUDA GPU"
        assert F.FW == 0, "B6LONG_FRONT_W must be unset for the build"
        for period in SPLITS:
            for pn in PANELS:
                for s in SEEDS:
                    F.trainf(period, pn, s, its[pn], WIDTH, str(out / "refit"), "0")
        roots = [rel(out / "refit")]
    rec = dict(iterations=its, width=WIDTH, seeds=list(SEEDS), roots=roots, refit=bool(a.refit), device=M.DEV, torch=torch.__version__, files={}); t0 = time.time()
    for period, (tag, otag, mode) in SPLITS.items():
        for pn in PANELS:
            sp, fl, vis, stations, dates = M.layer(mode, pn); dates = [str(x) for x in dates]
            for s in SEEDS:
                w, r = find(roots, its[pn], tag, pn, s)
                assert r.is_file(), f"training receipt missing next to the weights: {r}"
                run = json.loads(r.read_text()); h = sha(w)
                assert h == run["sd_sha256"], f"weight file does not match its training receipt: {w}"
                assert (run["iters"], run["width"], run["seed"], run["period"], run["panel"]) == (its[pn], WIDTH, s, period, pn), f"receipt of another run: {r}"
                z = torch.load(w, map_location="cpu", weights_only=True)
                assert (z["iters"], z["width"], z["seed"], z["period"], z["panel"]) == (its[pn], WIDTH, s, period, pn), f"weights of another run: {w}"
                model = M.UNet(cin=z["cin"], w=z["width"]).to(M.DEV); model.load_state_dict(z["state_dict"]); model.eval()
                iv, iq = F._predict(M, model, sp, fl, vis, stations); del model
                assert iv.shape == sp.shape and np.isfinite(iv).all() and np.isfinite(iq).all()
                f = out / f"imp_{otag}_{pn}_s{s}.npz"; F._save(f, speed=iv, flow=iq, dates=np.array(dates))
                rec["files"][f.name] = dict(weights=rel(w), weights_sha256=h, iterations=its[pn], trained_on=run.get("device"), train_loss=run.get("loss"),
                                            speed_sha256=hashlib.sha256(np.ascontiguousarray(iv).tobytes()).hexdigest(),
                                            flow_sha256=hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest(), shape=list(iv.shape))
            print("b6l", period, pn, f"it{its[pn]}", "3 members", round(time.time() - t0), "s", flush=True)
    (out / "receipt.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("B6L_SERVE_OK", len(rec["files"]), "member files in", a.out, "device", M.DEV, flush=True)


def tune(a):
    ship = json.loads((ROOT / a.shipped).read_text()); base = json.loads((ROOT / a.base).read_text()) if a.base else {}
    assert set(ship) == {"b6"}, "the shipped B6 tune file must carry the b6 block only"
    if base.get("b6"):
        print("USER_B6", flush=True); return
    (ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / a.out).write_text(json.dumps(dict(base, b6=ship["b6"]), indent=1) + "\n", encoding="utf-8", newline="\n")
    print("B6L_TUNE_OK", a.out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("serve"); p.add_argument("--it", type=int, default=12000); p.add_argument("--panel-it", default="")
    p.add_argument("--roots", default=DEFAULT_ROOT); p.add_argument("--out", required=True); p.add_argument("--refit", action="store_true")
    p = sub.add_parser("tune"); p.add_argument("--shipped", required=True); p.add_argument("--base", default=""); p.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(serve=serve, tune=tune)[a.cmd](a)
