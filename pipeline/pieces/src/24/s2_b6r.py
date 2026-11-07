"""L_b6 A1 = S2 (advisor L7 card, LINE 3; formulation fixed in docs/research/advisor_L7/ADVICE.md): B6R = the B6 imputer with
released ramp and occupancy input channels, warm-started from a shipped B6 member with zero-initialised new input weights.

ONE input function for every view (I-S-OWN training/read view, official view, the build): s2_inputs(panel, split, sp, fl, vis,
stations) = b6full_imp.inputs (7 channels, unchanged) + 9 new channels, all from released data of the same split:
  7..12  on-ramp sensor (gap2_ramps.calib_fields form, ramp type OR): ZC1, ZC2, ZC4 clipped to [-1, 2] (nan -> 0) and
         log1p(ZW1), log1p(ZW2), log1p(ZW4) / 3   (= the 6 ramp channels of gap2w_cnn.build_inputs)
  13, 14 off-ramp sensor built the same way (ramp type FR), radius 2: ZC2 clipped to [-1, 2] (nan -> 0), log1p(ZW2) / 3
  15     occupancy / occ_ref at the view's visible cells (0 elsewhere), clipped to [0, 8]; occ_ref = the station's median
         visible occupancy in the view (panel median of those when a station has < 50 visible cells)
The per-ramp calibration (selection, congested / free mean of the ramp log-flow anomaly, weight) and occ_ref are fitted ONCE per
(panel, split) on the I-S-OWN view's visible cells (calib(); never an OWN-hidden cell) and used for every view, so the ramp channels
are identical in the trained and the served form (first skew run with per-view calibration: FR channels corr .6 between views ->
fixed this way before any training); the ramp flows themselves are released in every view (never masked). Task 1 (state): the
imputer sees the released day as B6 always did; no IDs, seeds or row order are features.
Training (= b6_warm.py S1 recipe, i.e. b6full_imp.train_predict loop, warm start): AdamW 3e-4 wd 1e-4 fresh, OneCycleLR(3e-4,
12000) torch defaults, seed stream seed + 24000, grad clip 1, fp32, deterministic kernels, same drop sampler. ONE extra line: the
occupancy channel is hidden with speed/flow on the training-dropped cells (x[:, 15] *= keep), so a dropped cell leaks nothing.
Model UNetS2 = b6full_imp.UNet(cin=7, w=32) + `ex` = Conv2d(9, 32, 3, padding=1, bias=False) added to the first conv's output
(before the GELU), ex zero-initialised: at step 0 the net equals the parent exactly; asserted bitwise on the training device
(OWN view and official view predictions of parent and B6R are np.array_equal) before any step.
  train PERIOD PANEL SEED [OUT]        OUT default artifacts/claude_ref/l_b6/s2 -> OUT/it36000_w32/{imp_<own|ownapr>_*.npz (OWN view),
                                       sd_*.pt, run_*.json LAST}
  serve --out RUNDIR [--root DIR]      CPU official-view members imp_<valown|privown>_<panel>_s<seed>.npz + receipt.json (asm2_b6l form)
  skew [OUT.json]                      served (official view) vs trained (OWN view) channel statistics, cold (first 7 dates) / warm dates
  time PANEL ITERS                     ms per iteration of the B6R loop (no file written)
  parents                              sha256 of the 60 parent weights vs their receipts
Env: B6R_PARENT = parent root holding it<PIT>_w32 (default artifacts/claude_ref/b6l, the shipped 24k set; or artifacts/claude_ref/l_b6/s1
for the S1 members, then B6R_PIT=36000), B6W_SMOKE_ITERS (smoke only)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json, sys, time, hashlib
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/reference_code"))
import c_b6long_full_step as F

PANELS = F.PANELS; TAG = F.TAG; OTAG = {"validation": "valown", "private": "privown"}; MODE = {"validation": "val", "private": "priv"}
PIT = int(os.environ.get("B6R_PIT", "24000"))
PROOT = ROOT / os.environ.get("B6R_PARENT", "artifacts/claude_ref/b6l")
PARENT = PROOT / f"it{PIT}_w32"
ADD = int(os.environ.get("B6W_SMOKE_ITERS", "12000"))     # smoke only; the S2 recipe is 12000
MAXLR, OFFSET, WIDTH, NEW = 3e-4, PIT, 32, 9
CFG = f"it{PIT + ADD}_w{WIDTH}"
OCC = 7 + 8
RECIPE = dict(model="UNetS2 = b6full_imp.UNet(cin=7, w=32) + ex Conv2d(9, 32, 3, pad 1, no bias) added before the first GELU, ex zero-init",
              channels="b6full_imp.inputs (7) + OR ZC1/2/4 clip[-1,2] + log1p(ZW1/2/4)/3 + FR ZC2 clip[-1,2] + log1p(ZW2)/3 + clip(occ/occ_ref, 0, 8) at vis",
              add_iters=ADD, max_lr=MAXLR, seed_offset=OFFSET, sched="OneCycleLR(max_lr, total_steps) torch defaults",
              optimizer="AdamW wd 1e-4 fresh state", drop="b6full_imp sampler; occupancy channel hidden with speed/flow", code="exp/L_b6/s2_b6r.py")


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ inputs (ONE function for every view)
def _anomaly(c):
    """Ramp log-flow anomaly (gap2_ramps.calib_fields form): released ramp flows only, identical in every view."""
    import warnings
    rf = np.asarray(c.arrays["ramp_flow"], float); re = np.asarray(c.arrays["ramp_elig"], bool)
    rf = np.where(re, rf, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = np.log(rf + 5.); an = lr - np.nanmedian(lr, 0, keepdims=True)
        an3 = np.nanmean(np.stack([np.roll(an, k, 1) for k in (-1, 0, 1)], 0), 0)
    return an3


_CONST = {}


def calib(panel, split):
    """Per-ramp calibration (gap2_ramps.calib_fields selection, mf, mc, weight) and occ_ref per station, fitted ONCE per (panel,
    split) on the I-S-OWN view's visible cells (a subset of the official view's visible cells: no OWN-hidden cell is used, and the
    official view and the build get exactly the same constants, so the ramp channels are identical in the trained and served form)."""
    if (panel, split) in _CONST:
        return _CONST[(panel, split)]
    sys.path.insert(0, str(ROOT / "src"))
    from tfb.data import load_cache
    c = load_cache(panel, split)
    sp, fl, vis, stations, dates, got = F.own_layer(split, panel)
    an3 = _anomaly(c)
    st = c.stations.sort_values("idx"); vc = st.v_cut.to_numpy(float); S = len(vc)
    vsp = np.where(vis, np.asarray(sp, float), np.nan)
    Cg = np.where(np.isfinite(vsp), vsp < vc[None, None, :], False); Vi = np.isfinite(vsp)
    ramps = {"OR": [], "FR": []}
    for r in c.ramps.itertuples():
        if r.ramp_type not in ramps or not np.isfinite(r.station_idx):
            continue
        s0 = int(r.station_idx); sl = slice(max(0, s0 - 1), min(S, s0 + 2))
        anyc = Cg[:, :, sl].any(2); anyv = Vi[:, :, sl].any(2)
        a = an3[:, :, int(r.idx)]; ok = np.isfinite(a)
        mc_ = ok & anyc; mf_ = ok & anyv & ~anyc
        if mc_.sum() < 20 or mf_.sum() < 100:
            continue
        mc, mf = a[mc_].mean(), a[mf_].mean()
        var = .5 * (a[mc_].var() + a[mf_].var()) + 1e-3
        if mc - mf > -.05:
            continue
        ramps[r.ramp_type].append((int(r.idx), s0, float(mf), float(mc), float((mc - mf) ** 2 / var)))
    oc = np.asarray(c.occ, float); ov = vis & np.isfinite(oc); ref = np.full(S, np.nan)
    for s in range(S):
        x = oc[:, :, s][ov[:, :, s]]
        if len(x) >= 50:
            ref[s] = np.median(x)
    ref = np.where(np.isfinite(ref) & (ref > 1e-4), ref, np.nanmedian(ref) if np.isfinite(ref).any() else .1)
    _CONST[(panel, split)] = k = dict(ramps=ramps, occ_ref=ref, station_id=list(st.station_id), an3=an3, occ=oc,
                                      own_mask=got)
    return k


def _fields(an3, rr, radii, shape):
    D, T, S = shape; out = {}
    zs = [(np.clip((an3[:, :, i] - mf) / (mc - mf), -1, 2), w, s0) for i, s0, mf, mc, w in rr]
    for nb in radii:
        num = np.zeros((D, T, S)); den = np.zeros((D, T, S))
        for zr, wr, s0 in zs:
            ok = np.isfinite(zr)
            for s in range(max(0, s0 - nb), min(S, s0 + nb + 1)):
                num[:, :, s] += np.where(ok, wr * zr, 0); den[:, :, s] += np.where(ok, wr, 0)
        out[f"ZC{nb}"] = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        out[f"ZW{nb}"] = den
    return out


def extra_channels(panel, split, sp, vis, stations, info=None):
    """(D, 9, T, S) float32 new channels of one view (vis = the view's visible cells; only the occupancy channel depends on it)."""
    k = calib(panel, split)
    assert k["station_id"] == list(stations.sort_values("idx").station_id), "station order differs from the view"
    oc = k["occ"]; shape = oc.shape; assert vis.shape == shape
    Zo = _fields(k["an3"], k["ramps"]["OR"], (1, 2, 4), shape); Zf = _fields(k["an3"], k["ramps"]["FR"], (2,), shape)
    zc = lambda Z, nb: np.nan_to_num(np.clip(Z[f"ZC{nb}"], -1, 2), nan=0.)
    ch = [zc(Zo, nb) for nb in (1, 2, 4)] + [np.log1p(Zo[f"ZW{nb}"]) / 3 for nb in (1, 2, 4)] + [zc(Zf, 2), np.log1p(Zf["ZW2"]) / 3]
    ov = vis & np.isfinite(oc)
    ch.append(np.where(ov, np.clip(oc / k["occ_ref"][None, None, :], 0, 8), 0.))
    if info is not None:
        info.update(n_or=len(k["ramps"]["OR"]), n_fr=len(k["ramps"]["FR"]), occ_ref_median=float(np.median(k["occ_ref"])))
    X = np.stack(ch, 1).astype(np.float32)
    assert X.shape[1] == NEW and np.isfinite(X).all()
    return X


def s2_inputs(panel, split, sp, fl, vis, stations, info=None):
    """THE input builder (own view, official view, build): b6full_imp.inputs + extra_channels."""
    M = F.K.b6()
    Xn, vf, cap, lanes = M.inputs(sp, fl, vis, stations)
    return np.concatenate([Xn, extra_channels(panel, split, sp, vis, stations, info)], 1), vf, cap, lanes


# ------------------------------------------------------------------ model
def make_model(torch, M, parent_sd=None):
    nn = torch.nn

    class UNetS2(M.UNet):
        def __init__(self, w=WIDTH):
            super().__init__(cin=7, w=w)
            self.ex = nn.Conv2d(NEW, w, 3, padding=1, bias=False)
            nn.init.zeros_(self.ex.weight)

        def forward(self, x):
            c = self.e1.c
            h = c[0](x[:, :7].contiguous()) + self.ex(x[:, 7:].contiguous())
            a = c[3](c[2](c[1](h)))
            Fn = torch.nn.functional
            b = self.e2(Fn.avg_pool2d(a, 2)); cc = self.e3(Fn.avg_pool2d(b, 2))
            b2 = self.d2(torch.cat([b, Fn.interpolate(cc, size=b.shape[2:])], 1))
            a2 = self.d1(torch.cat([a, Fn.interpolate(b2, size=a.shape[2:])], 1))
            return self.out(a2)
    m = UNetS2()
    if parent_sd is not None:
        missing, unexpected = m.load_state_dict(parent_sd, strict=False)
        assert list(missing) == ["ex.weight"] and not unexpected, (missing, unexpected)
    return m


def predict(torch, M, model, Xn, vf, cap):
    """As c_b6long_full_step._predict (batches of 4 days, clip), on a prebuilt input."""
    with torch.no_grad():
        Y = torch.cat([model(torch.tensor(Xn[i:i + 4], device=M.DEV)) for i in range(0, len(Xn), 4)]).cpu().numpy().astype(float)
    return np.clip(Y[:, 0] * vf, 3., 1.3 * vf).astype(np.float32), np.clip(Y[:, 1] * cap, 0., 1.3 * cap).astype(np.float32)


# ------------------------------------------------------------------ parents
def parent(period, panel, seed):
    key = f"{TAG[period]}_{panel}_s{seed}"
    w, r = PARENT / f"sd_{key}.pt", PARENT / f"run_{key}.json"
    run = json.loads(r.read_text()); h = fsha(w)
    assert h == run["sd_sha256"], f"parent weights do not match the receipt: {w}"
    assert (run["iters"], run["width"], run["seed"], run["period"], run["panel"]) == (PIT, WIDTH, seed, period, panel), r
    return w, run, h


def parents():
    n = 0
    for period in TAG:
        for pn in PANELS:
            for s in (0, 1, 2):
                parent(period, pn, s); n += 1
    print("PARENTS_OK", n, flush=True)


# ------------------------------------------------------------------ training
def _loop(torch, M, model, X, VIS, VF, CAP, LN, iters, seed, log):
    D, T = X.shape[0], X.shape[2]
    DEV = M.DEV
    torch.manual_seed(seed + OFFSET); rng = np.random.default_rng(seed + OFFSET)
    opt = torch.optim.AdamW(model.parameters(), MAXLR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, MAXLR, total_steps=iters)
    tt = torch.arange(T, device=DEV)[None, :, None]
    t0 = time.time(); B = min(4, D); run = 0.
    model.train()
    for it in range(iters):
        idx = torch.tensor(rng.choice(D, B, replace=False), device=DEV)
        x = X[idx].clone(); vm = VIS[idx]
        rate = torch.tensor(rng.uniform(.2, .5, B), device=DEV, dtype=torch.float32)[:, None, None]
        drop = (torch.rand(vm.shape, device=DEV) < rate) & vm
        blk = torch.tensor(rng.random(B) < .5, device=DEV)[:, None, None]
        t1 = torch.tensor(rng.integers(0, T - 18, B), device=DEV)[:, None, None]
        drop |= blk & (tt >= t1) & (tt < t1 + 18) & vm
        keep = (vm & ~drop).float()
        x[:, 0] *= keep; x[:, 1] *= keep; x[:, 2] = keep; x[:, OCC] *= keep
        y = model(x)
        dv = (y[:, 0] - X[idx, 0]) * VF / 25; dq = (y[:, 1] - X[idx, 1]) * CAP / LN / 600
        loss = ((.54 * dv ** 2 + .46 * dq ** 2) * drop).sum() / drop.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        run = .99 * run + .01 * float(loss) if it else float(loss)
        if it % 1000 == 0 or it == iters - 1:
            log(f"  it {it} loss {run:.5f} lr {sched.get_last_lr()[0]:.2e} |ex| {float(model.ex.weight.abs().mean()):.2e} {time.time() - t0:.0f}s")
    model.eval()
    return float(run), time.time() - t0


def _setup(period, panel, seed):
    torch = F._flags(); M = F.K.b6()
    w, prun, psha = parent(period, panel, seed)
    z = torch.load(w, map_location="cpu", weights_only=True)
    assert (z["iters"], z["width"], z["seed"], z["period"], z["panel"], z["cin"]) == (PIT, WIDTH, seed, period, panel, 7)
    pm = M.UNet(cin=7, w=WIDTH).to(M.DEV); pm.load_state_dict(z["state_dict"]); pm.eval()
    model = make_model(torch, M, z["state_dict"]).to(M.DEV); model.eval()
    sp, fl, vis, stations, dates, got = F.own_layer(period, panel)
    assert got == prun["mask"], "OWN mask digests differ from the parent's receipt"
    info = {}
    Xn, vf, cap, lanes = s2_inputs(panel, period, sp, fl, vis, stations, info)
    return torch, M, pm, model, (sp, fl, vis, stations, dates, got), (Xn, vf, cap, lanes), (w, prun, psha), info


def train(period, panel, seed, out=None):
    seed = int(seed)
    out = Path(out or ROOT / "artifacts/claude_ref/l_b6/s2") / CFG; out.mkdir(parents=True, exist_ok=True)
    key = f"{TAG[period]}_{panel}_s{seed}"; done = out / f"run_{key}.json"
    if done.exists():
        print("skip", key, flush=True); return
    torch, M, pm, model, (sp, fl, vis, stations, dates, got), (Xn, vf, cap, lanes), (w, prun, psha), info = _setup(period, panel, seed)
    # step-0 identity: parent (7 channels) and B6R (16 channels, ex = 0) give bitwise equal predictions on this device
    pv, pq = F._predict(M, pm, sp, fl, vis, stations)
    bv, bq = predict(torch, M, model, Xn, vf, cap)
    assert np.array_equal(pv, bv) and np.array_equal(pq, bq), "step 0: B6R != parent (OWN view)"
    ev = M.layer(MODE[period], panel); assert [str(d) for d in ev[4]] == dates
    Xo = s2_inputs(panel, period, *ev[:4])[0]
    ov, oq = F._predict(M, pm, *ev[:4]); bov, boq = predict(torch, M, model, Xo, vf, cap)
    assert np.array_equal(ov, bov) and np.array_equal(oq, boq), "step 0: B6R != parent (official view)"
    del Xo, ev, bov, boq, ov, oq
    rs = abs(float(pv.astype(np.float64).sum()) - prun["speed_sum"]) / prun["speed_sum"]
    rq = abs(float(pq.astype(np.float64).sum()) - prun["flow_sum"]) / prun["flow_sum"]
    assert rs < 1e-4 and rq < 1e-4, f"parent prediction far from its receipt sums: {rs:.2e} {rq:.2e}"
    lines = []

    def log(s_):
        lines.append(s_.strip()); print(CFG, key, s_, flush=True)
    log(f"parent {w.name} sha {psha[:12]} step-0 bitwise equal (OWN + official view); receipt sums rel {rs:.1e}, {rq:.1e}; {info}")
    X = torch.tensor(Xn, device=M.DEV); VIS = torch.tensor(vis, device=M.DEV)
    VF = torch.tensor(vf, device=M.DEV); CAP = torch.tensor(cap, device=M.DEV); LN = torch.tensor(lanes, device=M.DEV)
    loss, sec = _loop(torch, M, model, X, VIS, VF, CAP, LN, ADD, seed, log)
    del X
    iv, iq = predict(torch, M, model, Xn, vf, cap)
    assert np.isfinite(iv).all() and np.isfinite(iq).all()
    F._save(out / f"imp_{key}.npz", speed=iv, flow=iq, dates=np.array(dates))
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    wf = dict(file=str(w.relative_to(ROOT)).replace("\\", "/"), sha256=psha, iters=PIT)
    torch.save(dict(state_dict=sd, width=WIDTH, cin=7 + NEW, arch="UNetS2", iters=PIT + ADD, seed=seed, period=period, panel=panel,
                    warm_from=wf, recipe=RECIPE), out / f"tmp_sd_{key}.pt")
    os.replace(out / f"tmp_sd_{key}.pt", out / f"sd_{key}.pt")
    dev = torch.cuda.get_device_name(0) if M.DEV == "cuda" else "cpu"
    done.write_text(json.dumps(dict(
        key=key, cfg=CFG, period=period, panel=panel, seed=seed, iters=PIT + ADD, width=WIDTH, cin=7 + NEW, arch="UNetS2", loss=loss,
        minutes=round(sec / 60, 2), ms_per_it=round(1000 * sec / ADD, 2), device=dev, torch=torch.__version__, det=F.DET, shape=list(iv.shape),
        mask=got, official=False, channels_info=info, step0_bitwise=True, warm_from=dict(wf, receipt_sums_rel=[rs, rq]), recipe=RECIPE,
        log=lines[-3:], npz_sha256=fsha(out / f"imp_{key}.npz"), sd_sha256=fsha(out / f"sd_{key}.pt"),
        speed_sum=float(iv.astype(np.float64).sum()), flow_sum=float(iq.astype(np.float64).sum())), indent=1))
    print("DONE", CFG, key, "loss", round(loss, 5), "min", round(sec / 60, 2), dev, flush=True)


def time_it(panel, iters):
    torch, M, pm, model, L6, (Xn, vf, cap, lanes), _, info = _setup("validation", panel, 0)
    X = torch.tensor(Xn, device=M.DEV); VIS = torch.tensor(L6[2], device=M.DEV)
    T5 = (X, VIS, torch.tensor(vf, device=M.DEV), torch.tensor(cap, device=M.DEV), torch.tensor(lanes, device=M.DEV))
    loss, sec = _loop(torch, M, model, *T5, int(iters), 0, lambda s_: None)
    print(json.dumps(dict(SF1=True, arm="S2", panel=panel, stations=int(L6[0].shape[2]), iters=int(iters), ms_per_it=round(1000 * sec / int(iters), 2),
                          device=torch.cuda.get_device_name(0) if M.DEV == "cuda" else "cpu", info=info)), flush=True)


# ------------------------------------------------------------------ serving (official view, CPU) = the build step
def serve(out, root="artifacts/claude_ref/l_b6/s2", panels=None, seeds=(0, 1, 2)):
    import torch as _t
    torch = F._flags(); M = F.K.b6()
    d = ROOT / root / CFG; out = ROOT / out; out.mkdir(parents=True, exist_ok=True)
    rec = dict(iterations=PIT + ADD, width=WIDTH, arch="UNetS2", seeds=list(seeds), root=root, device=M.DEV, torch=_t.__version__,
               inputs="exp/L_b6/s2_b6r.py s2_inputs", files={}); t0 = time.time()
    for period in ("validation", "private"):
        for pn in (panels or PANELS):
            sp, fl, vis, stations, dates = M.layer(MODE[period], pn); dates = [str(x) for x in dates]
            Xn, vf, cap, lanes = s2_inputs(pn, period, sp, fl, vis, stations)
            for s in seeds:
                key = f"{TAG[period]}_{pn}_s{s}"; w, r = d / f"sd_{key}.pt", d / f"run_{key}.json"
                run = json.loads(r.read_text()); h = fsha(w)
                assert h == run["sd_sha256"], f"weights do not match their receipt: {w}"
                z = torch.load(w, map_location="cpu", weights_only=True)
                assert (z["iters"], z["width"], z["seed"], z["period"], z["panel"], z["arch"]) == (PIT + ADD, WIDTH, s, period, pn, "UNetS2"), w
                model = make_model(torch, M).to(M.DEV); model.load_state_dict(z["state_dict"]); model.eval()
                iv, iq = predict(torch, M, model, Xn, vf, cap); del model
                assert iv.shape == sp.shape and np.isfinite(iv).all() and np.isfinite(iq).all()
                f = out / f"imp_{OTAG[period]}_{pn}_s{s}.npz"; F._save(f, speed=iv, flow=iq, dates=np.array(dates))
                rec["files"][f.name] = dict(weights=str(w.relative_to(ROOT)).replace("\\", "/"), weights_sha256=h, trained_on=run.get("device"),
                                            speed_sha256=hashlib.sha256(np.ascontiguousarray(iv).tobytes()).hexdigest(),
                                            flow_sha256=hashlib.sha256(np.ascontiguousarray(iq).tobytes()).hexdigest(), shape=list(iv.shape))
            print("b6r", period, pn, "3 members", round(time.time() - t0), "s", flush=True)
    (out / "receipt.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("B6R_SERVE_OK", len(rec["files"]), "member files in", out, flush=True)


# ------------------------------------------------------------------ train-serve skew check
NAMES = ["or_zc1", "or_zc2", "or_zc4", "or_zw1", "or_zw2", "or_zw4", "fr_zc2", "fr_zw2", "occ_ratio"]


def skew(outp=None, panels=None):
    """Trained form = the OWN view (what the members fit on); served form = the official view (what serve / the build feed).
    Per split, panel and date group (cold = first 7 dates of the split, warm = the rest): mean, sd per channel on all cells (ramp
    channels) or on the view's visible cells (occupancy), the correlation of trained vs served values on the same cells, and the
    level / scale ratios. PASS per (split, group, channel), pooled over panels: |mean diff| <= .1 pooled sd, sd ratio in [.8, 1.25],
    corr >= .9; the occupancy channel compared on cells visible in both views."""
    M = F.K.b6(); res = dict(rule="pooled over panels: |dmean| <= .1 sd, sd ratio in [.8, 1.25], corr >= .9 (paired, same cells); occ_all unpaired (OWN-visible vs official-visible cells): |dmean| <= .1 sd, sd ratio in [.8, 1.25]", rows=[], ok=True)
    acc = {}
    for period in ("validation", "private"):
        for pn in (panels or PANELS):
            sp, fl, vis, stations, dates, got = F.own_layer(period, pn)
            Xt = extra_channels(pn, period, sp, vis, stations)
            ev = M.layer(MODE[period], pn); assert [str(d) for d in ev[4]] == dates
            Xs = extra_channels(pn, period, ev[0], ev[2], ev[3])
            both = vis & ev[2]
            D = len(dates); grp = {"cold": np.arange(D) < 7, "warm": np.arange(D) >= 7}
            for g, dm in grp.items():
                for j, nm in enumerate(NAMES):
                    a = Xt[dm, j]; b = Xs[dm, j]
                    if nm == "occ_ratio":
                        m = both[dm]; a = a[m]; b = b[m]
                    else:
                        a = a.ravel(); b = b.ravel()
                    k = (period, g, nm); s = acc.setdefault(k, [0., 0., 0., 0., 0., 0])
                    if nm == "occ_ratio":   # unpaired: the trained form on the OWN-visible cells vs the served form on all official-visible cells
                        u = acc.setdefault((period, g, "occ_all"), [0., 0., 0., 0., 0, 0])
                        ta = Xt[dm, j][vis[dm]].astype(np.float64); tb = Xs[dm, j][ev[2][dm]].astype(np.float64)
                        u[0] += ta.sum(); u[1] += tb.sum(); u[2] += (ta ** 2).sum(); u[3] += (tb ** 2).sum(); u[4] += ta.size; u[5] += tb.size
                    s[0] += a.astype(np.float64).sum(); s[1] += b.astype(np.float64).sum(); s[2] += (a.astype(np.float64) ** 2).sum()
                    s[3] += (b.astype(np.float64) ** 2).sum(); s[4] += (a.astype(np.float64) * b).sum(); s[5] += a.size
            print("skew", period, pn, flush=True)
    for (period, g, nm), (sa, sb, saa, sbb, sab, n) in sorted(acc.items()):
        if nm == "occ_all":
            na, nb_ = sab, n; ma, mb = sa / na, sb / nb_; sda, sdb = np.sqrt(saa / na - ma ** 2), np.sqrt(sbb / nb_ - mb ** 2); ratio = sdb / sda
            ok = abs(mb - ma) <= .1 * sda and .8 <= ratio <= 1.25
            res["rows"].append(dict(split=period, dates=g, channel=nm, n_trained=int(na), n_served=int(nb_), mean_trained=ma, mean_served=mb,
                                    sd_trained=sda, sd_served=sdb, sd_ratio=ratio, ok=bool(ok)))
            res["ok"] &= bool(ok)
            print(f"{period:10s} {g:4s} {nm:9s} mean {ma:+.4f}/{mb:+.4f} sd {sda:.4f}/{sdb:.4f} ratio {ratio:.3f} cells {int(na)}/{int(nb_)} {'ok' if ok else 'SKEW'}", flush=True)
            continue
        ma, mb = sa / n, sb / n; va, vb = saa / n - ma ** 2, sbb / n - mb ** 2
        sda, sdb = np.sqrt(max(va, 0)), np.sqrt(max(vb, 0)); cov = sab / n - ma * mb
        corr = cov / max(sda * sdb, 1e-12); ratio = sdb / max(sda, 1e-12)
        absent = sda < 1e-9 and sdb < 1e-9 and abs(mb - ma) < 1e-9   # channel identically constant in both views (no ramp passed the filter)
        ok = absent or (abs(mb - ma) <= .1 * max(sda, 1e-9) and .8 <= ratio <= 1.25 and corr >= .9)
        res["rows"].append(dict(split=period, dates=g, channel=nm, n=int(n), mean_trained=ma, mean_served=mb, sd_trained=sda, sd_served=sdb,
                                sd_ratio=ratio, corr=corr, absent=bool(absent), ok=bool(ok)))
        res["ok"] &= bool(ok)
        print(f"{period:10s} {g:4s} {nm:9s} mean {ma:+.4f}/{mb:+.4f} sd {sda:.4f}/{sdb:.4f} ratio {ratio:.3f} corr {corr:.4f} {'absent' if absent else 'ok' if ok else 'SKEW'}", flush=True)
    if outp:
        Path(outp).write_text(json.dumps(res, indent=1))
    print("S2_SKEW", "PASS" if res["ok"] else "FAIL", flush=True)
    return res["ok"]


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "serve":
        import argparse
        ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--root", default="artifacts/claude_ref/l_b6/s2")
        x = ap.parse_args(a); serve(x.out, x.root)
    elif cmd == "skew":
        ok = skew(a[0] if a else None, a[1].split(",") if len(a) > 1 else None); sys.exit(0 if ok else 1)
    else:
        dict(train=train, time=time_it, parents=parents)[cmd](*a)
