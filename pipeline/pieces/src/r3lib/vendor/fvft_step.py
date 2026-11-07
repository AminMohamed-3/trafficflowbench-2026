"""L_fvft (ideas_L11 C.md idea 1, ADVICE A1): FV-FT nested instrument, host-agnostic steps (Kaggle T4, laptop GPU or CPU).

Question: does fine-tuning the S2b 48k B6R members on cells that are RELEASED in the official view but were never labels (the
OWN-hidden cells, +18.4% labels) improve the imputation of cells nobody saw? The S2b parents were trained on the OWN view, so the
only cells they never saw as labels or inputs are the OWN-hidden cells (OWN targets + OWN block). The nested hide set must sit
inside them, otherwise the score reads cells the parents memorised (an H drawn over the whole official view would put ~85% of H
on S2b training labels). Design (K-fold nested OWN, K = 3 by default):
  hid    = official-visible & ~OWN-visible  (= OWN-hidden cells released in the official view; every cell checked to be in the
           rebuilt OWN hidden mask)
  folds  = hid split into K folds the way OWN itself was drawn: the scattered OWN cells by regime (balanced random split, numpy
           default_rng(own_seed + 1 + h(panel|period))), the OWN 18-step block of a day assigned WHOLE to one fold (days split by
           regime). H_k = fold k: about 10%/K scattered + the block on 1/K of the days. No IDs, seeds or row order as features.
  arms   (all from the same S2b 48k parent, +ITERS (6000) with the S2/S2b schedule: fresh AdamW 3e-4 wd 1e-4, OneCycleLR(3e-4, ITERS)
          torch defaults, grad clip 1, fp32, deterministic kernels, seed stream seed + 48000, same drop sampler and occupancy hiding
          = s2_b6r._loop unchanged, inputs = s2_b6r.s2_inputs unchanged with the OWN-fitted ramp / occ_ref constants)
    ctl  : +ITERS on the OWN view (labels = OWN-visible; = official minus H for every k, plus minus the other folds)
    f<k> : +ITERS on the official view minus H_k (labels = OWN-visible + the other K-1 folds: dose (K-1)/K x 18.4%)
  serve  every arm predicts H_k from the view official minus H_k (the other folds visible as inputs, as the shipped S2b member sees
         the OWN-hidden cells as inputs on the official view). ctl serves all K views; f<k> serves only its own view; the parent
         (step 0, "par") is served on all K views too (reference). Scored cells: H_k only (OWN targets inside it, by the reader).
  leak asserts (every job): H_k never visible in the training view of f<k>; no OWN-hidden cell visible in ctl's training view;
         no official target / gap / non-released cell visible anywhere; the input channels speed, flow, mask, occupancy are 0 on
         every hidden cell of the training and serving views.
  step-0 parity to S2b (every job): the loaded parent on the OWN view reproduces the S2b receipt sums (rel < 1e-4, the S2b rule)
         and the laptop fingerprint of the stored S2b OWN-view prediction (256 fixed cells; tolerance in fvft_ref.json).
Commands (repo root on the laptop; TFB_ROOT on Kaggle):
  ref [P1,P2,..]                 LAPTOP ONLY: exp/L_fvft/fvft_ref.json = digests of official layers, OWN-view extra channels and
                                 calibration constants, fold assignments, counts; S2b fingerprints + CPU parity of the parents
  check PERIOD OUT.json [P,..]   rebuild layers, channels and folds, compare with fvft_ref.json; exit 1 on any mismatch
  parents PERIOD                 sha256 of the parents (FVFT_PARENT_DIR) vs their run_*.json receipts
  jobs OUTDIR PERIOD [P,..]      write OUTDIR/jobs.json (largest panel first; arms ctl, f0..f{K-1}; seeds 0,1,2)
  worker OUTDIR                  claim and run jobs until none is left or FVFT_STOP_MIN after FVFT_T0 (epoch seconds) is passed
  job OUTDIR PERIOD PANEL SEED ARM   one job (tests)
Env: FVFT_PARENT_DIR (default artifacts/claude_ref/l_b6/s2b/it48000_w32), FVFT_ITERS (6000), FVFT_K (3), FVFT_STOP_MIN, FVFT_T0.
Outputs per job: OUTDIR/<arm>/pred_<tag>_<panel>_s<seed>.npz (speed, flow float32 over the hid cells in C order, nan outside the
served folds; ctl also par_speed, par_flow), sd_<...>.pt (weights, kept in the kernel output), run_<...>.json LAST (= done)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["B6R_PIT"] = "48000"                       # s2_b6r.OFFSET = 48000 -> seed stream seed + 48000
import json, sys, time, hashlib, shutil
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("TFB_ROOT") or HERE.parents[1])
sys.path.insert(0, str(ROOT / "exp/L_b6"))
import s2_b6r as B
F = B.F; K = F.K
assert B.OFFSET == 48000 and B.MAXLR == 3e-4 and B.NEW == 9 and B.OCC == 15
PANELS = list(B.PANELS); TAG = B.TAG; MODE = B.MODE
ITERS = int(os.environ.get("FVFT_ITERS", "6000"))
KF = int(os.environ.get("FVFT_K", "3"))
PDIR = Path(os.environ.get("FVFT_PARENT_DIR") or ROOT / "artifacts/claude_ref/l_b6/s2b/it48000_w32")
REF = ROOT / "exp/L_fvft/fvft_ref.json"
NFP = 256
ARMS = lambda: ["ctl"] + [f"f{k}" for k in range(KF)]
RECIPE = dict(parent="S2b 48k UNetS2 member (artifacts/claude_ref/l_b6/s2b/it48000_w32, sha in s2b/RECEIPT.json)", iters=ITERS, folds=KF,
              loop="s2_b6r._loop unchanged: fresh AdamW 3e-4 wd 1e-4, OneCycleLR(3e-4, iters) torch defaults, clip 1, fp32, det, seed + 48000",
              inputs="s2_b6r.s2_inputs unchanged (OWN-fitted ramp / occ_ref constants)",
              views="ctl = OWN view; f<k> = official view minus fold k of the OWN-hidden cells; serve fold k from official minus fold k",
              code="exp/L_fvft/fvft_step.py")
sha = lambda b: hashlib.sha256(b).hexdigest()
logical = lambda a: sha(np.ascontiguousarray(a).tobytes())


def fsha(p):
    return B.fsha(p)


def _orig_mask_parts(c, panel, period, seed0, frac):
    """c_b6long_kg_step.own_mask with the SAME random draws, returning the block cells separately as well."""
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
    block = np.zeros(v.shape, bool)
    for day in range(len(c.dates)):
        choices = [t for t in range(12, 270) if not protected[day, t + 1:t + 19].any()]
        origin = int(rng.choice(choices)); hidden[day, origin + 1:origin + 19] = True; block[day, origin + 1:origin + 19] = True
    return hidden, hidden & candidates, block


def fold_assign(hid, block, regimes, panel, period, own_seed, k=None):
    k = k or KF
    rng = np.random.default_rng(own_seed + 1 + int(hashlib.sha256(f"FVFT|{panel}|{period}".encode()).hexdigest()[:8], 16))
    fold = np.full(hid.shape, -1, np.int8); regs = np.asarray(regimes)
    scat = hid & ~block; blk = hid & block
    for r in ("R1", "R2", "R3"):
        rows = np.flatnonzero((scat & (regs == r)[:, None, None]).reshape(-1))
        f = np.empty(len(rows), np.int8); f[rng.permutation(len(rows))] = np.arange(len(rows)) % k
        fold.reshape(-1)[rows] = f
    for r in ("R1", "R2", "R3"):
        dd = [d for d in range(hid.shape[0]) if regs[d] == r and blk[d].any()]
        for i, j in enumerate(rng.permutation(len(dd))):
            d = dd[j]; fold[d][blk[d]] = i % k
    assert (fold[hid] >= 0).all() and (fold[~hid] == -1).all()
    return fold


_V = {}


def views(period, panel):
    """Official and OWN layers, hid cells, folds (cached per process, one panel at a time)."""
    key = (period, panel)
    if key in _V:
        return _V[key]
    _V.clear(); B._CONST.clear()      # one panel in memory at a time
    from tfb.data import load_cache
    M = K.b6(); r = json.loads(F.REF_ALL.read_text())
    sp_w, fl_w, vis_w, stations, dates, got = F.own_layer(period, panel)
    c = load_cache(panel, period, root=K.ROOT)
    hidden, target, block = _orig_mask_parts(c, panel, period, r["own_seed"], r["own_fraction"])
    assert logical(hidden) == got["hidden"] and logical(target) == got["target"], "OWN mask rebuild differs from own_layer"
    sp_o, fl_o, vis_o, st_o, dates_o = M.layer(MODE[period], panel)
    assert [str(d) for d in dates_o] == dates and list(st_o.station_id) == list(stations.station_id)
    hide_off = np.asarray(c.target, bool) | np.asarray(c.gap, bool)
    assert not (vis_o & hide_off).any(), "official view shows a target or gap cell"
    assert not (vis_w & ~vis_o).any(), "OWN view is not a subset of the official view"
    hid = vis_o & ~vis_w
    assert not (hid & ~hidden).any(), "a cell visible in the official view only is not an OWN-hidden cell"
    assert not (hid & np.asarray(c.hist | c.gap, bool)).any()
    fold = fold_assign(hid, block, c.regimes, panel, period, r["own_seed"])
    v = dict(sp_o=sp_o, fl_o=fl_o, vis_o=vis_o, sp_w=sp_w, fl_w=fl_w, vis_w=vis_w, stations=stations, dates=dates, got=got,
             hid=hid, fold=fold, block=block, target=target, regimes=[str(x) for x in np.asarray(c.regimes)],
             hidx=np.flatnonzero(hid.reshape(-1)), hide_off=hide_off)
    _V[key] = v
    return v


def view_inputs(period, panel, vis, sp, fl, stations):
    sp_t = np.where(vis, sp, np.nan); fl_t = np.where(vis, fl, np.nan)
    Xn, vf, cap, lanes = B.s2_inputs(panel, period, sp_t, fl_t, vis, stations)
    hidden = ~vis
    for ch in (0, 1, 2, B.OCC):   # speed, flow, mask, occupancy carry nothing on a hidden cell
        assert not np.any(Xn[:, ch][hidden]), f"input channel {ch} is non-zero on a hidden cell"
    return Xn, vf, cap, lanes


def fold_view(v, k):
    return v["vis_o"] & (v["fold"] != k)


def digests(period, panel):
    """Laptop reference / Kaggle check: official layer digest, OWN-view extra channels and constants, folds."""
    v = views(period, panel)
    d_off = K.digest(np.where(v["vis_o"], v["sp_o"], np.nan), np.where(v["vis_o"], v["fl_o"], np.nan), v["vis_o"], v["stations"], v["dates"])
    k = B.calib(panel, period)
    Xe = B.extra_channels(panel, period, v["sp_w"], v["vis_w"], v["stations"])
    ch = [round(float(np.abs(Xe[:, j].astype(np.float64)).sum()), 2) for j in range(Xe.shape[1])]
    consts = dict(ramps={t: [[a, b, round(x, 6), round(y, 6), round(w, 6)] for a, b, x, y, w in k["ramps"][t]] for t in k["ramps"]},
                  occ_ref=logical(np.round(k["occ_ref"], 6)))
    cnt = dict(n_vis_o=int(v["vis_o"].sum()), n_vis_w=int(v["vis_w"].sum()), n_hid=int(v["hid"].sum()),
               extra_frac=float(v["vis_o"].sum() / v["vis_w"].sum() - 1),
               n_fold=[int((v["fold"] == j).sum()) for j in range(KF)],
               n_fold_target=[int(((v["fold"] == j) & v["target"]).sum()) for j in range(KF)],
               n_fold_block=[int(((v["fold"] == j) & v["block"]).sum()) for j in range(KF)])
    return dict(official=d_off, extra_abs_sums=ch, consts=consts, fold=logical(v["fold"]), hid=logical(v["hid"]), counts=cnt)


# ------------------------------------------------------------------ parents
def pkey(period, panel, seed):
    return f"{TAG[period]}_{panel}_s{int(seed)}"


def parent(period, panel, seed):
    key = pkey(period, panel, seed)
    w, r = PDIR / f"sd_{key}.pt", PDIR / f"run_{key}.json"
    run = json.loads(r.read_text()); h = fsha(w)
    assert h == run["sd_sha256"], f"parent weights do not match the receipt: {w}"
    assert (run["iters"], run["width"], run["seed"], run["period"], run["panel"], run["arch"]) == (48000, 32, int(seed), period, panel, "UNetS2"), r
    return w, run, h


def parents(period):
    n = 0
    for pn in PANELS:
        for s in (0, 1, 2):
            parent(period, pn, s); n += 1
    print("PARENTS_OK", period, n, flush=True)


def load_parent(torch, M, period, panel, seed):
    w, run, h = parent(period, panel, seed)
    z = torch.load(w, map_location="cpu", weights_only=True)
    assert (z["iters"], z["width"], z["seed"], z["period"], z["panel"], z["cin"], z["arch"]) == (48000, 32, int(seed), period, panel, 16, "UNetS2")
    model = B.make_model(torch, M).to(M.DEV); model.load_state_dict(z["state_dict"]); model.eval()
    return model, run, h, w


def fp_index(shape):
    return np.sort(np.random.default_rng(20261018).choice(int(np.prod(shape)), NFP, replace=False))


# ------------------------------------------------------------------ laptop reference
def ref(panels=None):
    torch = F._flags(); M = K.b6()
    panels = PANELS if not panels else panels.split(",")
    out = json.loads(REF.read_text()) if REF.exists() else dict(layers={}, members={})
    out.update(written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), k=KF, nfp=NFP, code="exp/L_fvft/fvft_step.py ref",
               rule=dict(sum_rel=1e-4, fp_speed_kmh=None, fp_flow_vph=None))
    for period in ("validation", "private"):
        for pn in panels:
            t0 = time.time()
            out["layers"][f"{pn}|{period}"] = digests(period, pn)
            v = views(period, pn)
            Xw, vf, cap, lanes = view_inputs(period, pn, v["vis_w"], v["sp_w"], v["fl_w"], v["stations"])
            idx = fp_index(v["vis_w"].shape)
            for s in (0, 1, 2):
                key = pkey(period, pn, s)
                z = np.load(PDIR / f"imp_{key}.npz")
                model, run, h, w = load_parent(torch, M, period, pn, s)
                pv, pq = B.predict(torch, M, model, Xw, vf, cap)
                rs = abs(float(pv.astype(np.float64).sum()) - run["speed_sum"]) / run["speed_sum"]
                rq = abs(float(pq.astype(np.float64).sum()) - run["flow_sum"]) / run["flow_sum"]
                sv, sq = z["speed"].reshape(-1)[idx], z["flow"].reshape(-1)[idx]
                out["members"][key] = dict(sd_sha256=h, fp_speed=[float(x) for x in sv], fp_flow=[float(x) for x in sq],
                                           cpu_sum_rel=[rs, rq], cpu_max_dv=float(np.abs(pv - z["speed"]).max()),
                                           cpu_max_dq=float(np.abs(pq - z["flow"]).max()),
                                           cpu_sums=[float(pv.astype(np.float64).sum()), float(pq.astype(np.float64).sum())],
                                           cpu_fp_speed=[float(x) for x in pv.reshape(-1)[idx]], cpu_fp_flow=[float(x) for x in pq.reshape(-1)[idx]])
                del model
            print("ref", period, pn, out["layers"][f"{pn}|{period}"]["counts"], round(time.time() - t0), "s", flush=True)
    mv = max(m["cpu_max_dv"] for m in out["members"].values()); mq = max(m["cpu_max_dq"] for m in out["members"].values())
    sr = max(max(m["cpu_sum_rel"]) for m in out["members"].values())
    # S2b members were trained and stored on an RTX 5070 (TF32 convolutions by default); the shipped build serves them on CPU (fp32).
    # vs the S2b receipt / stored prediction: TF32-level noise (CPU measured max sr, mv, mq) x 3; vs the laptop CPU serve: an fp32 GPU
    # without TF32 (T4, cc 7.5) must agree to 2e-5 (sums) and .02 km/h / 1 vph (fingerprint); a TF32 GPU (cc >= 8) gets the loose rule.
    out["rule"].update(sum_rel=max(1e-4, 3 * sr), fp_speed_kmh=max(.05, 3 * mv), fp_flow_vph=max(2., 3 * mq), cpu_max_dv=mv, cpu_max_dq=mq,
                       cpu_max_sum_rel=sr, cpu_sum_rel_fp32=2e-5, cpu_fp_speed_fp32=.02, cpu_fp_flow_fp32=1.)
    REF.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("FVFT_REF_OK", len(out["layers"]), "layers", len(out["members"]), "members; fingerprint tolerance", out["rule"], flush=True)


def _consts_close(a, b, tol=2e-6):
    if a["occ_ref"] != b["occ_ref"] or sorted(a["ramps"]) != sorted(b["ramps"]):
        return False
    for t_ in a["ramps"]:
        x, y = a["ramps"][t_], b["ramps"][t_]
        if len(x) != len(y) or any(r[:2] != s[:2] or any(abs(u - w) > tol for u, w in zip(r[2:], s[2:])) for r, s in zip(x, y)):
            return False
    return True


def check(period, outp, panels=None):
    r = json.loads(REF.read_text()); res = dict(ok=True, problems=[], period=period)
    for pn in (PANELS if not panels else panels.split(",")):
        d = digests(period, pn); e = r["layers"][f"{pn}|{period}"]
        for k_, v_ in e["official"].items():
            if (abs(d["official"][k_] - v_) > .01) if k_ in ("x_tod_sum", "x_sum") else d["official"][k_] != v_:
                res["problems"].append(f"{pn} official {k_}")
        if any(abs(a - b) > max(.05, 1e-6 * abs(b)) for a, b in zip(d["extra_abs_sums"], e["extra_abs_sums"])):
            res["problems"].append(f"{pn} extra channels {d['extra_abs_sums']} != {e['extra_abs_sums']}")
        for k_ in ("fold", "hid", "counts"):
            if json.dumps(d[k_], sort_keys=True) != json.dumps(e[k_], sort_keys=True):
                res["problems"].append(f"{pn} {k_}")
        if not _consts_close(d["consts"], e["consts"]):
            res["problems"].append(f"{pn} consts")
        print("check", period, pn, "MISMATCH" if res["problems"] and res["problems"][-1].startswith(pn) else "ok", flush=True)
    res["ok"] = not res["problems"]
    Path(outp).write_text(json.dumps(res, indent=1))
    print(json.dumps(dict(FVFT_CHECK=res["ok"], problems=res["problems"][:20])), flush=True)
    sys.exit(0 if res["ok"] else 1)


# ------------------------------------------------------------------ jobs
def jobs(outdir, period, panels=None):
    ps = PANELS if not panels else panels.split(",")
    S = {pn: json.loads(REF.read_text())["layers"][f"{pn}|{period}"]["official"]["shape"][3] if REF.exists() else 1 for pn in ps}
    ps = sorted(ps, key=lambda p: -S[p])
    js = [dict(period=period, panel=pn, seed=s, arm=a) for pn in ps for s in (0, 1, 2) for a in ARMS()]
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "jobs.json").write_text(json.dumps(js, indent=0))
    print("FVFT_JOBS", len(js), ps, flush=True)


def run_name(j):
    return f"{j['arm']}/run_{pkey(j['period'], j['panel'], j['seed'])}.json"


def job(outdir, period, panel, seed, arm, log=print):
    seed = int(seed); outdir = Path(outdir); od = outdir / arm; od.mkdir(parents=True, exist_ok=True)
    key = pkey(period, panel, seed); done = od / f"run_{key}.json"
    if done.exists():
        log(f"skip {arm} {key}"); return
    t0 = time.time()
    torch = F._flags(); M = K.b6()
    rr = json.loads(REF.read_text()); rule = rr["rule"]; fpr = rr["members"][key]
    v = views(period, panel)
    k = None if arm == "ctl" else int(arm[1:]); assert arm == "ctl" or 0 <= k < KF
    # ---- step-0 parity to S2b (OWN view)
    model, prun, psha, pw = load_parent(torch, M, period, panel, seed)
    assert psha == fpr["sd_sha256"], "parent sha differs from the laptop reference"
    Xw, vf, cap, lanes = view_inputs(period, panel, v["vis_w"], v["sp_w"], v["fl_w"], v["stations"])
    pv, pq = B.predict(torch, M, model, Xw, vf, cap)
    rs = abs(float(pv.astype(np.float64).sum()) - prun["speed_sum"]) / prun["speed_sum"]
    rq = abs(float(pq.astype(np.float64).sum()) - prun["flow_sum"]) / prun["flow_sum"]
    idx = fp_index(pv.shape)
    dv = float(np.abs(pv.reshape(-1)[idx] - np.asarray(fpr["fp_speed"])).max()); dq = float(np.abs(pq.reshape(-1)[idx] - np.asarray(fpr["fp_flow"])).max())
    assert rs < rule["sum_rel"] and rq < rule["sum_rel"], f"step 0: parent OWN-view sums far from the S2b receipt: {rs:.2e} {rq:.2e}"
    assert dv <= rule["fp_speed_kmh"] and dq <= rule["fp_flow_vph"], f"step 0: fingerprint differs from S2b: {dv:.4f} km/h {dq:.3f} vph"
    cs = [abs(float(pv.astype(np.float64).sum()) - fpr["cpu_sums"][0]) / fpr["cpu_sums"][0], abs(float(pq.astype(np.float64).sum()) - fpr["cpu_sums"][1]) / fpr["cpu_sums"][1]]
    cd = [float(np.abs(pv.reshape(-1)[idx] - np.asarray(fpr["cpu_fp_speed"])).max()), float(np.abs(pq.reshape(-1)[idx] - np.asarray(fpr["cpu_fp_flow"])).max())]
    tf32 = M.DEV == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
    # hard: within the TF32-level rule of the S2b receipt check on any device; recorded: the tight fp32 agreement with the laptop CPU
    # serve (expected on an fp32 GPU such as the T4 and on CPU; the reader reports it)
    assert max(cs) < rule["sum_rel"], f"step 0: parent OWN-view sums far from the laptop CPU serve: {cs}"
    tight = bool(max(cs) < rule["cpu_sum_rel_fp32"] and cd[0] <= rule["cpu_fp_speed_fp32"] and cd[1] <= rule["cpu_fp_flow_fp32"])
    del pv, pq
    lines = []

    def L(s_):
        lines.append(s_.strip()); log(f"{arm} {key} {s_}")
    L(f"parent {pw.name} sha {psha[:12]} step-0 vs S2b sums rel {rs:.1e} {rq:.1e} fp max {dv:.2e} km/h {dq:.2e} vph; vs laptop CPU sums {cs[0]:.1e} {cs[1]:.1e} fp {cd[0]:.1e} {cd[1]:.1e} (tf32 {tf32}, tight {tight})")
    # ---- views: training and serving
    folds_served = list(range(KF)) if arm == "ctl" else [k]
    serve_in = {j: view_inputs(period, panel, fold_view(v, j), v["sp_o"], v["fl_o"], v["stations"])[0] for j in folds_served}
    for j in folds_served:
        assert not (fold_view(v, j) & (v["fold"] == j)).any(), f"leak: fold {j} visible in its serving view"
    if arm == "ctl":
        vis_t = v["vis_w"]; Xt = Xw
        assert not (vis_t & v["hid"]).any(), "leak: an OWN-hidden cell is visible in the control's training view"
    else:
        vis_t = fold_view(v, k); Xt = serve_in[k]
        assert not (vis_t & (v["fold"] == k)).any(), "leak: H_k visible in the arm's training view"
    assert not (vis_t & ~v["vis_o"]).any() and not (vis_t & v["hide_off"]).any(), "leak: a non-released cell in the training view"
    n_lab = int(vis_t.sum())
    par = None
    if arm == "ctl":    # the parent served on the K fold views (reference arm 'par')
        ps_, pq_ = np.full(v["hid"].shape, np.nan, np.float32), np.full(v["hid"].shape, np.nan, np.float32)
        for j in folds_served:
            a, b = B.predict(torch, M, model, serve_in[j], vf, cap); m = v["fold"] == j; ps_[m] = a[m]; pq_[m] = b[m]
        par = (ps_.reshape(-1)[v["hidx"]], pq_.reshape(-1)[v["hidx"]])
    # ---- train
    X = torch.tensor(Xt, device=M.DEV); VIS = torch.tensor(vis_t, device=M.DEV)
    VF = torch.tensor(vf, device=M.DEV); CAP = torch.tensor(cap, device=M.DEV); LN = torch.tensor(lanes, device=M.DEV)
    loss, sec = B._loop(torch, M, model, X, VIS, VF, CAP, LN, ITERS, seed, L)
    del X, VIS
    # ---- serve
    sv, sq = np.full(v["hid"].shape, np.nan, np.float32), np.full(v["hid"].shape, np.nan, np.float32)
    for j in folds_served:
        a, b = B.predict(torch, M, model, serve_in[j], vf, cap); m = v["fold"] == j
        assert np.isfinite(a[m]).all() and np.isfinite(b[m]).all(); sv[m] = a[m]; sq[m] = b[m]
    arrs = dict(speed=sv.reshape(-1)[v["hidx"]], flow=sq.reshape(-1)[v["hidx"]], hid_sha=np.array(logical(v["hid"])), fold_sha=np.array(logical(v["fold"])))
    if par is not None:
        arrs.update(par_speed=par[0], par_flow=par[1])
    F._save(od / f"pred_{key}.npz", **arrs)
    sd = {kk: vv.detach().cpu() for kk, vv in model.state_dict().items()}
    torch.save(dict(state_dict=sd, width=32, cin=16, arch="UNetS2", iters=48000 + ITERS, seed=seed, period=period, panel=panel, arm=arm,
                    warm_from=dict(file=pw.name, sha256=psha, iters=48000), recipe=RECIPE), od / f"tmp_sd_{key}.pt")
    os.replace(od / f"tmp_sd_{key}.pt", od / f"sd_{key}.pt")
    dev = torch.cuda.get_device_name(0) if M.DEV == "cuda" else "cpu"
    done.write_text(json.dumps(dict(
        key=key, arm=arm, period=period, panel=panel, seed=seed, iters=ITERS, folds=KF, fold=k, loss=loss, minutes=round((time.time() - t0) / 60, 2),
        train_minutes=round(sec / 60, 2), ms_per_it=round(1000 * sec / max(ITERS, 1), 2), device=dev, torch=torch.__version__, det=F.DET,
        parent=dict(file=pw.name, sha256=psha, receipt_sums_rel=[rs, rq], fp_max=[dv, dq], cpu_sums_rel=cs, cpu_fp_max=cd, tf32_device=tf32, cpu_tight=tight), mask=v["got"], hid_sha=logical(v["hid"]),
        fold_sha=logical(v["fold"]), n_hid=int(v["hid"].sum()), n_labels_visible=n_lab, n_served=int(np.isfinite(arrs["speed"]).sum()),
        leak_asserts=["train view excludes " + ("all OWN-hidden cells" if arm == "ctl" else f"fold {k}"), "no target/gap/unreleased cell visible",
                      "speed/flow/mask/occ channels zero on hidden cells (train + serve views)"],
        recipe=RECIPE, log=lines[-3:], npz_sha256=fsha(od / f"pred_{key}.npz"), sd_sha256=fsha(od / f"sd_{key}.pt")), indent=1))
    log(f"DONE {arm} {key} loss {loss:.5f} min {(time.time() - t0) / 60:.2f} {dev}")


def worker(outdir):
    outdir = Path(outdir); js = json.loads((outdir / "jobs.json").read_text()); cl = outdir / "claims"; cl.mkdir(exist_ok=True)
    t0 = float(os.environ.get("FVFT_T0") or time.time()); stop = float(os.environ.get("FVFT_STOP_MIN") or 1e9)
    tag = os.environ.get("CUDA_VISIBLE_DEVICES", "x"); n = 0
    for j in js:
        if (outdir / run_name(j)).exists():
            continue
        if (time.time() - t0) / 60 > stop:
            print(f"FVFT_STOP time guard ({stop} min): no new job", flush=True); break
        c = cl / run_name(j).replace("/", "_").replace(".json", "")
        try:
            os.mkdir(c)
        except FileExistsError:
            continue
        try:
            job(outdir, j["period"], j["panel"], j["seed"], j["arm"], log=lambda s_: print(f"[w{tag}] {s_}", flush=True)); n += 1
        except Exception as ex:   # the job stays undone (no run json): the next session reruns it
            import traceback
            traceback.print_exc()
            print(f"FVFT_JOB_FAILED {run_name(j)} {type(ex).__name__}: {str(ex)[:300]}", flush=True)
            if "step 0" in str(ex) or "leak" in str(ex) or "differs" in str(ex):
                raise    # parity / leak failures stop the worker: the instrument would be invalid
    print("FVFT_WORKER_END", tag, "jobs run", n, flush=True)


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    dict(ref=ref, check=check, parents=parents, jobs=jobs, worker=worker, job=job)[cmd](*a)
