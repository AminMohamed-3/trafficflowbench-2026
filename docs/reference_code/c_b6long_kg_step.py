"""K6 helper 2026-10-11 (ledger 0801, exp/C_b6long/B6LONG.md): B6 imputer length/width screen on I-S-OWN, Kaggle side.
Runs on the Kaggle notebook (TFB_ROOT set by c_b6long_kg_kernel.py) and on the laptop (smoke, reference digests).
The training code is b6full_imp.train_predict UNCHANGED (shipped recipe); only iters and the U-Net width are arguments.
The OWN layer is rebuilt from the raw release: tfb.data.build_cache -> OWN mask (producer logic of
src/tfb/lane_s/vm_s07_own_masks.py, logical sha256 checked against exp/0222_s07_vm_continuation/OWN_mask_reference.json)
-> visible cells exactly as b6full_imp.layer('own') / state5_imp_apr.layer_apr (official target | OWN hidden | gap hidden).
  gpuinfo
  cache PANEL                         validation + private caches of PANEL (disk guard reserve 2 GiB)
  ref PANEL                           LAPTOP ONLY: digests of the layer through the ORIGINAL code path -> exp/C_b6long/kg_ref.json
  layer PANEL [OUT.json]              rebuild masks + layer, compare with kg_ref.json; exit 1 on any mismatch
  train PERIOD PANEL SEED ITERS WIDTH [OUTDIR]   PERIOD validation | private; writes imp_<own|ownapr>_<PANEL>_it<ITERS>_w<WIDTH>_s<SEED>.npz + .json
Deterministic kernels (TFB_ASM2_DET=1 rule of asm2_b6imp.py) are always on here."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import hashlib, json, sys, time, types
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("TFB_ROOT") or HERE.parents[1])
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "src"))   # src first: docs/reference_code/tfb.py would shadow the package
import tfb.data   # bound now: b6full_imp puts docs/reference_code first on sys.path again
PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S", "D12_I405_N", "D12_I405_S"]
REF = ROOT / "exp/C_b6long/kg_ref.json"
OUT = ROOT / "artifacts/claude_ref/c_b6long/imp"
TAG = {"validation": "own", "private": "ownapr"}
sha = lambda b: hashlib.sha256(b).hexdigest()
logical = lambda a: sha(np.ascontiguousarray(a).tobytes())


def gpuinfo():
    import torch, pandas, pyarrow
    d = dict(torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__, pandas=pandas.__version__,
             pyarrow=pyarrow.__version__, gpu=torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        d.update(n_gpu=torch.cuda.device_count(), name=p.name, mem_gb=round(p.total_memory / 2**30, 1))
    print(json.dumps(d), flush=True)


def cache(panel):
    import importlib.util
    import tfb.data as D

    def guard(root, destination, expected_bytes=0):
        spec = importlib.util.spec_from_file_location("tfb_diskguard", Path(root) / "tools" / "diskguard.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        return m.check_free_space(destination, expected_bytes=expected_bytes, reserve_gb=2.0)
    D.guard = guard
    for split in ("validation", "private"):
        meta = D.build_cache(panel, split, root=ROOT)
        print(json.dumps({k: meta[k] for k in ("panel", "split", "shape", "target_rows", "runtime_seconds")}), flush=True)


def own_mask(c, panel, period, seed0, frac):
    """hidden, target of the frozen I-S-OWN population (vm_s07_own_masks.py, validation/private branch: all dates)."""
    v = np.array(c.speed, copy=True); q = np.array(c.flow, copy=True)
    original_target = np.array(c.target, copy=True); protected = np.array(c.hist | c.gap, copy=True); eligible = np.array(c.elig, copy=True)
    observed = eligible & np.isfinite(v) & np.isfinite(q) & (v >= 1) & (q >= 0) & ~original_target
    candidates = observed & ~protected
    seed = seed0 + int(hashlib.sha256((panel + "|" + period).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed); hidden = np.zeros(v.shape, bool); regimes = np.asarray(c.regimes)
    for regime in ("R1", "R2", "R3"):
        rows = np.flatnonzero((candidates & (regimes == regime)[:, None, None]).reshape(-1))
        number = int(np.floor(frac * len(rows)))
        selected = rng.choice(rows, size=number, replace=False); hidden.reshape(-1)[selected] = True
    for day in range(len(c.dates)):
        choices = [t for t in range(12, 270) if not protected[day, t + 1:t + 19].any()]
        assert choices
        origin = int(rng.choice(choices)); hidden[day, origin + 1:origin + 19] = True
    target = hidden & candidates
    assert not (target & original_target).any() and not (hidden & protected).any() and observed[target].all() and target.any()
    return hidden, target


def own_layer(period, panel, ref=None):
    """sp, fl (nan outside vis), vis, stations, dates, mask digests. Released observations only."""
    from tfb.data import load_cache
    ref = ref or json.loads(REF.read_text())
    c = load_cache(panel, period, root=ROOT)
    hidden, target = own_mask(c, panel, period, ref["own_seed"], ref["own_fraction"])
    dates = [str(d) for d in c.dates]
    got = dict(hidden=logical(hidden), target=logical(target), dates=logical(np.asarray(dates)))
    exp = ref["mask_sha"].get(f"{panel}|{period}")
    assert exp is None or got == exp, ("OWN mask differs from the frozen population", panel, period, got, exp)
    hide = np.asarray(c.target, bool) | hidden | np.asarray(c.gap, bool)
    sp = np.asarray(c.speed, float); fl = np.asarray(c.flow, float)
    vis = np.asarray(c.elig, bool) & ~hide & np.isfinite(sp) & np.isfinite(fl)
    return np.where(vis, sp, np.nan), np.where(vis, fl, np.nan), vis, c.stations, dates, got


def b6():
    """b6full_imp with a stub for phys_lwr_diag (only ROOT and the panel list are used by the functions called here)."""
    if "phys_lwr_diag" not in sys.modules:
        sys.modules["phys_lwr_diag"] = types.SimpleNamespace(ROOT=ROOT, P={"panels": PANELS})
    import b6full_imp as M
    return M


def digest(sp, fl, vis, stations, dates):
    """Content digest of the training input: channels 0,1,2,5,6 are exact arithmetic (bytes), 3,4 (sin/cos) by the sum of absolute values."""
    Xn, vf, cap, lanes = b6().inputs(sp, fl, vis, stations)
    return dict(shape=list(Xn.shape), n_vis=int(vis.sum()), vis=logical(vis), x_exact=logical(Xn[:, [0, 1, 2, 5, 6]]),
                x_tod_sum=round(float(np.abs(Xn[:, [3, 4]].astype(np.float64)).sum()), 3), x_sum=round(float(Xn[:, [0, 1]].astype(np.float64).sum()), 3),
                consts=logical(np.stack([vf, cap, lanes]).astype(np.float32)), dates=logical(np.asarray(dates)))


def ref(panel):
    """Laptop: the layer through the original code (flow_own + own_instrument.load_item), as the shipped members saw it."""
    import b6full_imp as M
    import state5_imp_apr as SA
    mr = json.loads((ROOT / "exp/0222_s07_vm_continuation/OWN_mask_reference.json").read_text())
    cfg = json.loads((ROOT / "exp/0184_l_s_own_instrument/config.yaml").read_text())
    out = dict(panel=panel, own_seed=cfg["seed"], own_fraction=cfg["random_fraction"],
               mask_sha={f"{panel}|{p}": mr["items"][f"{panel}|{p}"]["arrays"] for p in ("validation", "private")}, layer={})
    for period in ("validation", "private"):
        sp, fl, vis, stations, dates = M.layer("own", panel) if period == "validation" else SA.layer_apr(panel)
        out["layer"][period] = digest(sp, fl, vis, stations, [str(d) for d in dates])
        print(period, out["layer"][period], flush=True)
    REF.parent.mkdir(parents=True, exist_ok=True)
    REF.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")


def layer(panel, out=None):
    r = json.loads(REF.read_text()); assert r["panel"] == panel, (r["panel"], panel)
    res = dict(ok=True, problems=[], layer={}, mask={})
    for period in ("validation", "private"):
        sp, fl, vis, stations, dates, got = own_layer(period, panel, r)
        d = digest(sp, fl, vis, stations, dates); res["layer"][period] = d; res["mask"][period] = got
        for k, v in r["layer"][period].items():
            if k == "x_tod_sum":
                if abs(d[k] - v) > .01:
                    res["problems"].append(f"{period} {k}: {d[k]} != {v}")
            elif d[k] != v:
                res["problems"].append(f"{period} {k}: {d[k]} != {v}")
    res["ok"] = not res["problems"]
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(ok=res["ok"], problems=res["problems"], n_vis={p: res["layer"][p]["n_vis"] for p in res["layer"]})), flush=True)
    sys.exit(0 if res["ok"] else 1)


def train(period, panel, seed, iters, width, outdir=None):
    import torch
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    M = b6(); seed, iters, width = int(seed), int(iters), int(width)
    out = Path(outdir) if outdir else OUT; out.mkdir(parents=True, exist_ok=True)
    name = f"imp_{TAG[period]}_{panel}_it{iters}_w{width}_s{seed}"
    if (out / f"{name}.npz").exists():
        print("skip", name, flush=True); return
    if width != 32:   # train_predict builds UNet(cin=...) from the module global: same code, other width
        U = M.UNet
        M.UNet = lambda cin=7: U(cin=cin, w=width)
    sp, fl, vis, stations, dates, got = own_layer(period, panel)
    lines = []; t0 = time.time()

    def log(s):
        lines.append(s.strip()); print(name, s, flush=True)
    iv, iq, loss, _ = M.train_predict(sp, fl, vis, stations, iters, seed, log=log)
    minutes = (time.time() - t0) / 60
    assert np.isfinite(iv).all() and np.isfinite(iq).all()
    np.savez_compressed(out / f"{name}.npz", speed=iv, flow=iq, dates=np.array(dates))
    dev = torch.cuda.get_device_name(0) if M.DEV == "cuda" else "cpu"
    (out / f"{name}.json").write_text(json.dumps(dict(
        name=name, period=period, panel=panel, seed=seed, iters=iters, width=width, loss=loss, minutes=round(minutes, 2), device=dev,
        torch=torch.__version__, det=True, shape=list(iv.shape), vis=round(float(vis.mean()), 4), mask=got, log=lines,
        npz_sha256=sha((out / f"{name}.npz").read_bytes()), speed_sum=float(iv.astype(np.float64).sum()), flow_sum=float(iq.astype(np.float64).sum())), indent=1))
    print("DONE", name, "loss", round(loss, 5), "min", round(minutes, 2), dev, flush=True)


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    dict(gpuinfo=gpuinfo, cache=cache, ref=ref, layer=layer, train=train)[cmd](*a)
