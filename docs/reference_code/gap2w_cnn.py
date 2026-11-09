"""gap2w (C build helper 2026-10-13): copy of gap2_cnn.py (gap2 conditional gap U-Net with on-ramp sensor channels, ledger
0477) that SAVES THE FITTED WEIGHTS, so the gap2 piece of C48 (exp/C_gap2c47) can be served from weights on a CPU box.
The training loop, the recipe and the random streams are gap2_cnn.py verbatim. Only additions:
- deterministic kernels asked for (cudnn.deterministic, cudnn.benchmark off, use_deterministic_algorithms warn_only,
  CUBLAS_WORKSPACE_CONFIG); the stored cnr0/1/2 runs of 2026-10-05 used the torch defaults, so a retrain is a new draw of
  the same recipe, not a bitwise copy;
- per panel: weights/<tag>_<kind>_<period>_<panel>.pt (torch state_dict on CPU, float32);
- per run: weights/receipt_<tag>_<kind>_<period>.json (recipe, seed used, input hashes, torch build, device, sha256 of
  every weights file and of the raw tensors, sha256 of the predictions, loss, seconds), rewritten after every panel;
- predictions of the fitted model on the official rows, written by the training device, in the gap2 file format:
  pred_gpu/pred_<tag>_<kind>_<period>.npz with <panel>_v and <panel>_q (float32, rows in the rows_off Mev order).
Output root: artifacts/claude_ref/c_gap2c47 (--outdir to change, e.g. for a smoke test). Serving: gap2w_serve.py.
Kind 'own' is not supported here (its ramp channels need artifacts/claude_ref/state6, deleted 2026-10-10).
Usage: python docs/reference_code/gap2w_cnn.py off <validation|private> <tag> [panels] [--iters N] [--seed S] [--outdir D]"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys, time, json, hashlib, argparse, platform
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
import gap_lib as G
import b6full_imp as BI
import gap2_ramps as GR
OUTW = G.ROOT / "artifacts/claude_ref/c_gap2c47"
RAMP = True

torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
DEV = BI.DEV
RECIPE = dict(model="b6full_imp.UNet(cin=18, w=32)", channels="7 b6full_imp.inputs + 6 on-ramp (ZC1/2/4 clipped to [-1, 2], log1p(ZW1/2/4)/3) + 5 condition maps",
              iters=4000, batch_days=4, optimizer="AdamW lr 1e-3 weight_decay 1e-4", schedule="OneCycleLR max_lr 1e-3 total_steps iters",
              grad_clip=1.0, block_classes_share={"onset": .4, "ongoing": .4, "free": .15, "other": .05}, extra_drop="uniform 10-30% of visible cells",
              loss=".54 (dv vf/25)^2 + .46 (dq cap/lanes/600)^2, weight 1 on block cells, .2 on other dropped cells", sim_origins_seed=1,
              divergence_guard="final running loss > .1 and seed < 1000: retrain once with seed + 1000", precision="fp32",
              predict="official windows of the day as condition maps (onset k = 6), clip v [3, 1.3 vf], q [0, 1.3 cap]")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_state(sd):
    """sha256 over the raw float32 tensors in key order (independent of the file container)."""
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(np.ascontiguousarray(sd[k].numpy()).tobytes())
    return h.hexdigest()


def cond_maps(D, T, S, blocks):
    """blocks: list of (d, t, cls, k). returns (D, 5, T, S) float32 condition channels."""
    C = np.zeros((D, 5, T, S), np.float32)
    for d, t, c, k in blocks:
        j = np.arange(1, 19)
        C[d, 0, t + 1:t + 19] = 1
        C[d, 4, t + 1:t + 19] = (j / 18.)[:, None]
        if c == "onset":
            C[d, 1, t + 1:t + 19] = (j >= k)[:, None]
            C[d, 2, t + 1:t + 19] = (j < k)[:, None]
        elif c in ("ongoing", "other"):
            C[d, 3, t + 1:t + 19] = 1
    return C


def build_inputs(panel, period, kind="off"):
    """Layer, model input Xn (D, 13, T, S) float32, vf, cap, lanes: exactly what gap2_cnn.run builds."""
    L = G.own_layer(panel, period) if kind == "own" else G.off_layer(panel, period)
    Xn, vf, cap, lanes = BI.inputs(L.sp, L.fl, L.K, L.stations)
    if RAMP:
        Z = GR.calib_fields(panel, period, kind, ("OR",))
        ch = [np.nan_to_num(np.clip(Z[f"ZC{nb}"], -1, 2), nan=0.) for nb in (1, 2, 4)] + [np.log1p(Z[f"ZW{nb}"]) / 3 for nb in (1, 2, 4)]
        Xn = np.concatenate([Xn, np.stack(ch, 1).astype(np.float32)], 1)
    return L, Xn, vf, cap, lanes


def run(kind, period, panel, tag, iters, seed, wdir):
    t0 = time.time()
    L = G.own_layer(panel, period) if kind == "own" else G.off_layer(panel, period)
    if (kind == "own" and not L.blocks) or (kind != "own" and not L.windows):
        print(kind, period, panel, "no eval blocks: skipped", flush=True)
        return np.zeros(0, np.float32), np.zeros(0, np.float32), dict(skipped="no official gap windows")
    rng = np.random.default_rng(seed)
    orig = G.sim_origins(L, np.random.default_rng(1))
    Xn, vf, cap, lanes = BI.inputs(L.sp, L.fl, L.K, L.stations)
    if RAMP:
        Z = GR.calib_fields(panel, period, kind, ("OR",))
        ch = [np.nan_to_num(np.clip(Z[f"ZC{nb}"], -1, 2), nan=0.) for nb in (1, 2, 4)] + [np.log1p(Z[f"ZW{nb}"]) / 3 for nb in (1, 2, 4)]
        Xn = np.concatenate([Xn, np.stack(ch, 1).astype(np.float32)], 1)
    D, T, S = L.sp.shape
    X = torch.tensor(Xn, device=DEV); VIS = torch.tensor(L.K, device=DEV)
    VF = torch.tensor(vf, device=DEV); CAP = torch.tensor(cap, device=DEV); LN = torch.tensor(lanes, device=DEV)
    torch.manual_seed(seed)
    model = BI.UNet(cin=Xn.shape[1] + 5).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=iters)
    cls = np.array([o[2] for o in orig])
    share = {"onset": .4, "ongoing": .4, "free": .15, "other": .05}
    w = np.array([share[c] / np.sum(cls == c) for c in cls]); w /= w.sum()
    B = 4; run_ = 0.
    for it in range(iters):
        pick = rng.choice(len(orig), B, p=w)
        days = [orig[i][0] for i in pick]
        C = torch.tensor(cond_maps(B, T, S, [(b, orig[i][1], orig[i][2], orig[i][3]) for b, i in enumerate(pick)]), device=DEV)
        idx = torch.tensor(days, device=DEV)
        x = X[idx].clone(); vm = VIS[idx]
        blk = (C[:, 0] > 0) & vm
        rate = torch.tensor(rng.uniform(.1, .3, B), device=DEV, dtype=torch.float32)[:, None, None]
        drop = ((torch.rand(vm.shape, device=DEV) < rate) & vm) | blk
        keep = (vm & ~drop).float()
        x[:, 0] *= keep; x[:, 1] *= keep; x[:, 2] = keep
        y = model(torch.cat([x, C], 1))
        dv = (y[:, 0] - X[idx, 0]) * VF / 25; dq = (y[:, 1] - X[idx, 1]) * CAP / LN / 600
        wt = blk.float() + .2 * (drop & ~blk).float()
        loss = ((.54 * dv ** 2 + .46 * dq ** 2) * wt).sum() / blk.float().sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        run_ = .99 * run_ + .01 * float(loss) if it else float(loss)
        if it % 1000 == 0 or it == iters - 1:
            print(panel, f"it {it} loss {run_:.5f} {time.time() - t0:.0f}s", flush=True)
    if run_ > 0.1 and seed < 1000:   # gap2 divergence guard: retrain once with another seed
        print(panel, f"final loss {run_:.4f} > .1: retrain with seed {seed + 1000}", flush=True)
        del model, X
        torch.cuda.empty_cache()
        return run(kind, period, panel, tag, iters, seed + 1000, wdir)
    model.eval()
    # ---- gap2w: save the fitted weights
    sd = {k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()}
    wf = wdir / f"{tag}_{kind}_{period}_{panel}.pt"
    torch.save(sd, wf)
    cc = {c: int(np.sum(cls == c)) for c in share}

    def pred_day(d, blocks):
        C = torch.tensor(cond_maps(1, T, S, [(0, t, c, k) for (t, c, k) in blocks]), device=DEV)
        with torch.no_grad():
            Y = model(torch.cat([X[d:d + 1], C], 1)).cpu().numpy().astype(float)[0]
        return np.clip(Y[0] * vf, 3., 1.3 * vf), np.clip(Y[1] * cap, 0., 1.3 * cap)

    out_v, out_q = [], []
    if kind == "own":
        z = np.load(G.OUT / f"rows_own_{period}_{panel}.npz"); M = z["Mev"]
        for bi, (d, t) in enumerate(L.blocks):
            c, k, _ = G.classify(L.kv_truth[d], L.el[d], L.vcut, t)
            v, q = pred_day(d, [(t, c, k)])
            m = M[:, 0] == bi
            jj, ss = M[m, 5], M[m, 6]
            assert (M[m, 2] == t).all()
            out_v.append(v[t + jj, ss]); out_q.append(q[t + jj, ss])
    else:
        z = np.load(G.OUT / f"rows_off_{period}_{panel}.npz"); M = z["Mev"]
        for bi, (d, t, c) in enumerate(L.windows):
            same = [(tt, cc_, 6 if cc_ == "onset" else 0) for (dd, tt, cc_) in L.windows if dd == d]
            v, q = pred_day(d, same)
            m = M[:, 0] == bi
            jj, ss = M[m, 5], M[m, 6]
            out_v.append(v[t + jj, ss]); out_q.append(q[t + jj, ss])
    del model, X
    torch.cuda.empty_cache()
    ov = np.concatenate(out_v).astype(np.float32); oq = np.concatenate(out_q).astype(np.float32)
    info = dict(seed_used=int(seed), final_running_loss=float(run_), seconds=round(time.time() - t0, 1), shape_D_T_S=[int(D), int(T), int(S)],
                sim_origins=len(orig), sim_origins_by_class=cc, windows=len(L.windows) if kind != "own" else len(L.blocks),
                inputs=dict(Xn_float32_sha256=sha_arr(Xn), visible_mask_sha256=sha_arr(L.K), rows_Mev_sha256=sha_arr(M),
                            windows_sha256=hashlib.sha256(json.dumps([list(map(str, x)) for x in (L.windows if kind != "own" else L.blocks)]).encode()).hexdigest()),
                weights_file=wf.name, weights_file_sha256=sha_file(wf), weights_bytes=wf.stat().st_size, weights_tensors_sha256=sha_state(sd),
                parameters=int(sum(v.numel() for v in sd.values())), rows=int(len(ov)), pred_v_sha256=sha_arr(ov), pred_q_sha256=sha_arr(oq))
    print(kind, period, panel, "done %.1f min" % ((time.time() - t0) / 60), "weights", wf.name, info["weights_file_sha256"][:16], flush=True)
    return ov, oq, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind"); ap.add_argument("period"); ap.add_argument("tag"); ap.add_argument("panels", nargs="?", default="all")
    ap.add_argument("--iters", type=int, default=4000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--noramp", action="store_true")
    ap.add_argument("--outdir", default=str(OUTW))
    a = ap.parse_args()
    assert a.kind == "off", "gap2w saves official-view members only"
    global RAMP
    RAMP = not a.noramp
    out = Path(a.outdir); wdir = out / "weights"; pdir = out / "pred_gpu"
    wdir.mkdir(parents=True, exist_ok=True); pdir.mkdir(parents=True, exist_ok=True)
    panels = G.PANELS if a.panels == "all" else a.panels.split(",")
    f = pdir / f"pred_{a.tag}_{a.kind}_{a.period}.npz"
    rf = wdir / f"receipt_{a.tag}_{a.kind}_{a.period}.json"
    res = dict(np.load(f)) if f.exists() else {}
    here = Path(__file__).resolve().parent
    rec = json.loads(rf.read_text()) if rf.exists() else dict(
        tag=a.tag, kind=a.kind, period=a.period, seed=a.seed, ramp=RAMP, recipe=dict(RECIPE, iters=a.iters),
        command="python docs/reference_code/gap2w_cnn.py " + " ".join(sys.argv[1:]),
        code_sha256={n: sha_file(here / n) for n in ("gap2w_cnn.py", "gap_lib.py", "b6full_imp.py", "gap2_ramps.py")},
        env=dict(torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version() if DEV == "cuda" else None, device=DEV,
                 device_name=torch.cuda.get_device_name(0) if DEV == "cuda" else platform.processor(), python=platform.python_version(),
                 numpy=np.__version__, platform=platform.platform(), torch_threads=torch.get_num_threads()),
        deterministic=dict(cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
                           use_deterministic_algorithms="True (warn_only)", CUBLAS_WORKSPACE_CONFIG=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                           seeds="numpy default_rng(seed) for origin picks and drop rates; torch.manual_seed(seed) for the initial weights and the CUDA drop masks; sim_origins default_rng(1)"),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), panels={})
    assert rec["seed"] == a.seed and rec["recipe"]["iters"] == a.iters, "receipt of another recipe is in the way"
    for pn in panels:
        if f"{pn}_v" in res and pn in rec["panels"]:
            print("skip", pn); continue
        v, q, info = run(a.kind, a.period, pn, a.tag, a.iters, a.seed, wdir)
        res[f"{pn}_v"] = v; res[f"{pn}_q"] = q
        np.savez_compressed(f, **res)
        rec["panels"][pn] = info; rec["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec["complete"] = all(p in rec["panels"] for p in G.PANELS)
        rf.write_text(json.dumps(rec, indent=1))
    print("RUN COMPLETE" if rec.get("complete") else "RUN PARTIAL", a.tag, a.kind, a.period, flush=True)


if __name__ == "__main__":
    main()
