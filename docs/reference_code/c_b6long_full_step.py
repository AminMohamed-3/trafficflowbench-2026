"""K6 helper 2026-10-11 (ledger 0801, exp/C_b6long/B6LONG.md section 6): B6 long FULL run, steps for Kaggle and the laptop.
Builds on c_b6long_kg_step.py (OWN mask rebuild from the raw release, layer digests, b6full_imp stub); the training code is
b6full_imp.train_predict UNCHANGED. One training per panel / split / seed on the OWN layer serves both the I-S-OWN read
and the rows the build would ship.
  gpuinfo
  cache P1,P2,..                      validation + private caches
  ref all | P1,P2                     LAPTOP ONLY: digests of every panel's layer through the ORIGINAL code path
                                      (flow_own + own_instrument.load_item) -> exp/C_b6long/kg_ref_all.json
  layer P1,P2,.. [OUT.json]           rebuild masks + layers, compare with kg_ref_all.json; exit 1 on any mismatch
  trainf PERIOD PANEL SEED ITERS WIDTH OUTROOT OFFICIAL(0|1)
        -> OUTROOT/it<ITERS>_w<WIDTH>/: imp_<own|ownapr>_<PANEL>_s<SEED>.npz (OWN view = the I-S-OWN read),
           sd_<own|ownapr>_<PANEL>_s<SEED>.pt (weights), with OFFICIAL=1 also imp_<valown|privown>_<PANEL>_s<SEED>.npz (the
           same weights on the official view: arm format of asm2_build.parse_arm), run_<own|ownapr>_<PANEL>_s<SEED>.json
           written LAST (= the run is done; restart skips it)
  official CFGDIR PERIOD PANEL SEED   official-view prediction from saved weights (sets trained with OFFICIAL=0)
  repeat PANEL ITERS                  LAPTOP GPU: the same short run twice in this process configuration; prints sha + it/s
B6LONG_FRONT_W=<w> (> 0; B6LONG.md section 13.3): trainf uses a front-weighted loss (weight 1 + w on training-hidden cells
whose visible speed steps by 15 km/h or more into or out of the cell in time) and writes to it<ITERS>_w<WIDTH>_fw<w>/.
Unset or 0: b6full_imp.train_predict UNCHANGED and the usual directory.
Deterministic kernels (the TFB_ASM2_DET=1 rule of asm2_b6imp.py) are on unless B6LONG_DET=0 (cudnn.deterministic False,
benchmark False; used on the laptop only if the same-GPU repeat test passes, see exp/C_b6long/PROGRESS.md)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_b6long_kg_step as K

ROOT = K.ROOT; PANELS = K.PANELS; TAG = K.TAG; sha = K.sha
REF_ALL = ROOT / "exp/C_b6long/kg_ref_all.json"
OTAG = {"validation": "valown", "private": "privown"}
MODE = {"validation": "val", "private": "priv"}
DET = os.environ.get("B6LONG_DET", "1") != "0"
FW = float(os.environ.get("B6LONG_FRONT_W", "0") or 0)


def cache(panels):
    for pn in panels.split(","):
        K.cache(pn)


def ref_all(panels):
    import b6full_imp as M
    import state5_imp_apr as SA
    panels = PANELS if panels == "all" else panels.split(",")
    mr = json.loads((ROOT / "exp/0222_s07_vm_continuation/OWN_mask_reference.json").read_text())
    cfg = json.loads((ROOT / "exp/0184_l_s_own_instrument/config.yaml").read_text())
    out = dict(own_seed=cfg["seed"], own_fraction=cfg["random_fraction"], mask_sha={}, layers={})
    for pn in panels:
        out["layers"][pn] = {}
        for period in ("validation", "private"):
            out["mask_sha"][f"{pn}|{period}"] = mr["items"][f"{pn}|{period}"]["arrays"]
            sp, fl, vis, stations, dates = M.layer("own", pn) if period == "validation" else SA.layer_apr(pn)
            out["layers"][pn][period] = K.digest(sp, fl, vis, stations, [str(d) for d in dates])
            print(pn, period, out["layers"][pn][period]["shape"], out["layers"][pn][period]["n_vis"], flush=True)
    REF_ALL.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


def own_layer(period, panel):
    r = json.loads(REF_ALL.read_text())
    assert f"{panel}|{period}" in r["mask_sha"], "no mask reference for " + panel
    return K.own_layer(period, panel, r)


def layer(panels, out=None):
    r = json.loads(REF_ALL.read_text()); res = dict(ok=True, problems=[], layers={})
    for pn in panels.split(","):
        bad = 0
        for period in ("validation", "private"):
            sp, fl, vis, stations, dates, got = own_layer(period, pn)
            d = K.digest(sp, fl, vis, stations, dates); res["layers"].setdefault(pn, {})[period] = d
            for k, v in r["layers"][pn][period].items():
                if (abs(d[k] - v) > .01) if k == "x_tod_sum" else (d[k] != v):
                    res["problems"].append(f"{pn} {period} {k}: {d[k]} != {v}"); bad += 1
        print("layer", pn, "MISMATCH" if bad else "ok", flush=True)
    res["ok"] = not res["problems"]
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(ok=res["ok"], problems=res["problems"])), flush=True)
    sys.exit(0 if res["ok"] else 1)


def _flags():
    import torch
    torch.backends.cudnn.benchmark = False
    if DET:
        torch.backends.cudnn.deterministic = True; torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
    return torch


def _predict(M, model, sp, fl, vis, stations):
    """As the inner pred() of b6full_imp.train_predict."""
    import torch
    Xn, vf, cap, lanes = M.inputs(sp, fl, vis, stations)
    with torch.no_grad():
        Y = torch.cat([model(torch.tensor(Xn[i:i + 4], device=M.DEV)) for i in range(0, len(Xn), 4)]).cpu().numpy().astype(float)
    return np.clip(Y[:, 0] * vf, 3., 1.3 * vf).astype(np.float32), np.clip(Y[:, 1] * cap, 0., 1.3 * cap).astype(np.float32)


def _save(path, **arrays):
    tmp = path.with_name("tmp_" + path.name); np.savez_compressed(tmp, **arrays); os.replace(tmp, path)


def _train_fw(M, sp, fl, vis, stations, iters, seed, log, extra, fw, step=15.):
    """b6full_imp.train_predict with ONE change: the loss weight of a training-hidden cell is 1 + fw where the visible
    speed steps by `step` km/h or more into or out of the cell in time (weighted mean). Same random draws, optimiser,
    schedule, gradient clipping, clipping of the outputs and return values."""
    import torch
    DEV = M.DEV
    Xn, vf, cap, lanes = M.inputs(sp, fl, vis, stations)
    D, T, S = sp.shape
    d = np.nan_to_num(np.abs(np.diff(np.where(vis, sp, np.nan), axis=1)), nan=0.)
    jump = np.zeros((D, T, S), np.float32); jump[:, 1:] = d; jump[:, :-1] = np.maximum(jump[:, :-1], d)
    WF = torch.tensor(1. + fw * (jump >= step), device=DEV, dtype=torch.float32)
    log(f"  front weight {fw:g}: {float((jump >= step)[vis].mean()) * 100:.3f}% of the visible cells are front cells")
    X = torch.tensor(Xn, device=DEV)
    VIS = torch.tensor(vis, device=DEV)
    VF = torch.tensor(vf, device=DEV); CAP = torch.tensor(cap, device=DEV); LN = torch.tensor(lanes, device=DEV)
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = M.UNet(cin=X.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=iters)
    tt = torch.arange(T, device=DEV)[None, :, None]
    t0 = time.time(); B = min(4, D); run = 0.
    for it in range(iters):
        idx = torch.tensor(rng.choice(D, B, replace=False), device=DEV)
        x = X[idx].clone(); vm = VIS[idx]
        rate = torch.tensor(rng.uniform(.2, .5, B), device=DEV, dtype=torch.float32)[:, None, None]
        drop = (torch.rand(vm.shape, device=DEV) < rate) & vm
        blk = torch.tensor(rng.random(B) < .5, device=DEV)[:, None, None]
        t1 = torch.tensor(rng.integers(0, T - 18, B), device=DEV)[:, None, None]
        drop |= blk & (tt >= t1) & (tt < t1 + 18) & vm
        keep = (vm & ~drop).float()
        x[:, 0] *= keep; x[:, 1] *= keep; x[:, 2] = keep
        y = model(x)
        dv = (y[:, 0] - X[idx, 0]) * VF / 25; dq = (y[:, 1] - X[idx, 1]) * CAP / LN / 600
        wgt = drop * WF[idx]
        loss = ((.54 * dv ** 2 + .46 * dq ** 2) * wgt).sum() / wgt.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        run = .99 * run + .01 * float(loss) if it else float(loss)
        if it % 1000 == 0 or it == iters - 1:
            log(f"  it {it} loss {run:.5f} {time.time() - t0:.0f}s")
    model.eval()

    def pred(Xa):
        with torch.no_grad():
            Y = torch.cat([model(torch.tensor(Xa[i:i + 4], device=DEV)) for i in range(0, len(Xa), 4)]).cpu().numpy().astype(float)
        return (np.clip(Y[:, 0] * vf, 3., 1.3 * vf).astype(np.float32), np.clip(Y[:, 1] * cap, 0., 1.3 * cap).astype(np.float32))
    del X
    iv, iq = pred(Xn)
    ex = [pred(M.inputs(a, b_, c, stations)[0]) for a, b_, c in extra]
    del model
    torch.cuda.empty_cache()
    return iv, iq, float(run), ex


def trainf(period, panel, seed, iters, width, outroot, official="1"):
    torch = _flags(); M = K.b6(); seed, iters, width = int(seed), int(iters), int(width)
    out = Path(outroot) / (f"it{iters}_w{width}" + (f"_fw{FW:g}" if FW > 0 else "")); out.mkdir(parents=True, exist_ok=True)
    key = f"{TAG[period]}_{panel}_s{seed}"; done = out / f"run_{key}.json"
    if done.exists():
        print("skip", out.name, key, flush=True); return
    U = M.UNet; made = []

    def factory(cin=7):   # same class at the chosen width; keeps a handle so the weights can be saved after train_predict
        m = U(cin=cin, w=width); made.append(m); return m
    M.UNet = factory
    sp, fl, vis, stations, dates, got = own_layer(period, panel)
    extra = []
    if official == "1":
        ev = M.layer(MODE[period], panel)
        assert [str(d) for d in ev[4]] == dates; extra = [ev[:3]]
    lines = []; t0 = time.time()

    def log(s_):
        lines.append(s_.strip()); print(out.name, key, s_, flush=True)
    if FW > 0:
        iv, iq, loss, ex = _train_fw(M, sp, fl, vis, stations, iters, seed, log, extra, FW)
    else:
        iv, iq, loss, ex = M.train_predict(sp, fl, vis, stations, iters, seed, log=log, extra=extra)
    minutes = (time.time() - t0) / 60
    M.UNet = U
    assert len(made) == 1 and np.isfinite(iv).all() and np.isfinite(iq).all()
    _save(out / f"imp_{key}.npz", speed=iv, flow=iq, dates=np.array(dates))
    if ex:
        assert np.isfinite(ex[0][0]).all() and np.isfinite(ex[0][1]).all()
        _save(out / f"imp_{OTAG[period]}_{panel}_s{seed}.npz", speed=ex[0][0], flow=ex[0][1], dates=np.array(dates))
    sd = {k: v.detach().cpu() for k, v in made[0].state_dict().items()}
    torch.save(dict(state_dict=sd, width=width, cin=7, iters=iters, seed=seed, period=period, panel=panel), out / f"tmp_sd_{key}.pt")
    os.replace(out / f"tmp_sd_{key}.pt", out / f"sd_{key}.pt")
    dev = torch.cuda.get_device_name(0) if M.DEV == "cuda" else "cpu"
    done.write_text(json.dumps(dict(
        key=key, cfg=out.name, period=period, panel=panel, seed=seed, iters=iters, width=width, loss=loss, minutes=round(minutes, 2), device=dev,
        torch=torch.__version__, det=DET, shape=list(iv.shape), mask=got, official=bool(ex), log=lines[-3:],
        npz_sha256=sha((out / f"imp_{key}.npz").read_bytes()), sd_sha256=sha((out / f"sd_{key}.pt").read_bytes()),
        speed_sum=float(iv.astype(np.float64).sum()), flow_sum=float(iq.astype(np.float64).sum()), **(dict(front_w=FW) if FW > 0 else {})), indent=1))
    print("DONE", out.name, key, "loss", round(loss, 5), "min", round(minutes, 2), dev, "det", DET, flush=True)


def official(cfgdir, period, panel, seed):
    torch = _flags(); M = K.b6(); d = Path(cfgdir); key = f"{TAG[period]}_{panel}_s{int(seed)}"
    f = d / f"imp_{OTAG[period]}_{panel}_s{int(seed)}.npz"
    if f.exists():
        print("skip", f.name, flush=True); return
    z = torch.load(d / f"sd_{key}.pt", map_location="cpu", weights_only=True)
    model = M.UNet(cin=z["cin"], w=z["width"]).to(M.DEV); model.load_state_dict(z["state_dict"]); model.eval()
    sp, fl, vis, stations, dates = M.layer(MODE[period], panel)
    iv, iq = _predict(M, model, sp, fl, vis, stations)
    _save(f, speed=iv, flow=iq, dates=np.array([str(x) for x in dates]))
    print("official", f.name, flush=True)


def repeat(panel, iters):
    """Same-GPU repeatability and speed of the current kernel setting (B6LONG_DET): two identical runs in one process."""
    torch = _flags(); M = K.b6(); iters = int(iters)
    sp, fl, vis, stations, dates, got = own_layer("validation", panel)
    out = []
    for k in range(2):
        t0 = time.time()
        iv, iq, loss, _ = M.train_predict(sp, fl, vis, stations, iters, 0, log=lambda s_: None)
        out.append((K.logical(iv), K.logical(iq), loss, time.time() - t0))
        print("run", k, "sha", out[-1][0][:16], out[-1][1][:16], "loss", loss, "sec", round(out[-1][3], 1), flush=True)
    same = out[0][:2] == out[1][:2]
    print(json.dumps(dict(panel=panel, iters=iters, det=DET, device=torch.cuda.get_device_name(0) if M.DEV == "cuda" else "cpu",
                          identical=same, ms_per_it=round(1000 * out[1][3] / iters, 1))), flush=True)


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    dict(gpuinfo=K.gpuinfo, cache=cache, ref=ref_all, layer=layer, trainf=trainf, official=official, repeat=repeat)[cmd](*a)
