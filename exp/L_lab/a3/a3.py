"""L_lab A3 (BUILD helper, 2026-10-13): the method file of pool rule P1 on the parent C50 (c_gap2w_on_c47.csv, public .89813).

P1 = the per-date adaptation pool of the 10 shipped ongoing members keeps a pool window when the observed share of
t-12..t-1 is >= .30 AND the released eligible share is >= .70 (shipped: observed share >= .5). Nothing else changes: same
members, recipe mh5 on warm dates (5 ep, lr 6e-4, same day type), XSD on cold dates (all earlier dates 3 ep 3e-4, then the
same-day stage per hour checkpoint H, windows t' + 6 <= H), .75 adapted + .25 unadapted, seed mean, decode, ConvNeXt
member at .2 from its stored probabilities. For the sccnn members the t-1 gate inside the selection flag of a pool row uses
the same P1 gate (worker 2's tested form, a2s1.flags_p1). Onset rows, state rows and ODME rows are never touched.

Shared code is imported read-only; the two shipped fit loops (qcnn_predict_adapt.finetune, c_cold_split.ft) are re-stated
here as fit() because the pool and the flag rule must be passed in; `codeparity` proves fit() == both, bitwise, on the CPU.
Everything is written under artifacts/claude_ref/l_lab/a3 (OUT) and exp/L_lab/a3/logs.

  codeparity SPLIT                     (CPU) fit() == qcnn_predict_adapt.finetune and == c_cold_split.ft, 1 epoch, ctl and sccnn
  ref                                  (CPU) parent ongoing rows == stored C36 rows; shipped chain on the stored files == stored rows
  off KIND ARM SPLIT [--cpu=1 --dates=a,b --members=N --budget=MIN --tag=_x]
                                       official-form member job (KIND ctl|sc = 5 members, ARM P0|P1): one parquet per official
                                       date, finished dates skipped, stops before a new date once --budget minutes are used
  jf MEMBER [--arms=P1 --cpu=1 --maxdates=N --tag=_x]
                                       OFF250 member (Dec cutoff, Jan + Feb, mh5 per date) with the pool rule of the arm
  ship ARM                             (CPU) on/ files -> blend .75 -> seed mean -> decode -> ConvNeXt decode; parity (P0)
  noharm                               (CPU) OFF250 read of P1 against p8 on raw and denoised truth
  delta                                (CPU) rows of the candidate, counts, tools/probe.py make + expand, window guard
  result                               A3_RESULT.json
  drive                                the whole chain, every GPU job through gpu_lock.py, resumable
"""
import os, sys, json, time, copy, shutil, subprocess, filecmp, tempfile
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for _p in (ROOT / "src", ROOT / "docs/reference_code"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))
OUT = ROOT / "artifacts/claude_ref/l_lab/a3"
LOGS = HERE / "logs"
R = "docs/reference_code"
RUNS = {**{f"ctl_s{i}": f"artifacts/claude_ref/qcnn/run_full_s{i}_npt_e16" for i in range(5)},
        **{f"sc_s{i}": f"artifacts/claude_ref/ong2/run_sc_s{i}_e16" for i in range(5)}}
JF = [f"jft_{k}16_s{i}" for i in range(4) for k in ("sc", "ctl")]              # the 8 Dec-cutoff members of OFF250 (g_score.MEM)
M = "artifacts/claude_ref/asm2/e16/s01234_5_6e-4_none_out/rows"                # served mh5 on/ off/ (C35)
C36 = "artifacts/claude_ref/c_cold/official/ship_c35"                          # on/ = mh5 + XSD on cold windows; a0.75/ongoing_bb.parquet = shipped ongoing rows
COLD_MEMBERS = "artifacts/claude_ref/c_cold/official"                          # per member p_un, p_xsd on the cold windows (CPU build)
BB = "artifacts/claude_ref/c_cold/official/ship/bb_member.parquet"
CFG = f"{R}/asm2_bbmember_c28.json"
TYPES = "artifacts/claude_ref/ong2/official_types.csv"
KEYF = ROOT / "data/raw/kaggle_public/submission_key.csv"
PARENT = ROOT / "artifacts/claude_ref/probes/c_gap2w_on_c47.csv"
NAME = "c_llabp1_on_c50"
SPLITS = ("validation", "private")
K = ["window_id", "timestamp", "link_id"]
FL = {"A": (1., 1., 0.), "B": (1., 0., 1.), "C": (1., 0., 0.)}                 # ong2_sccnn_official.FL
MH5, XSD, MIX, COLD = (5, 6e-4), (3, 3e-4), "0.75", 2
MIN_FREE_GIB, MIN_RAM_GIB = 13., 1.5
GPU_HARD_MIN = 330.                                                            # no new GPU job once this many GPU minutes are used
CONTROL_MAX_CELLS = 20                                                         # P0 (GPU) rows may differ from the parent in at most this many of 87,000 cells
utc = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
weekend = lambda s: pd.Timestamp(str(s)).dayofweek >= 5
rp = lambda p: pd.read_parquet(ROOT / p if not Path(p).is_absolute() else p, use_threads=False)
MAKE_X = None


def free_gib():
    return shutil.disk_usage(str(ROOT)).free / 2 ** 30


def guard(wait_min=0):
    """C: free >= 13 GiB (hard stop) and >= 1.5 GiB of physical memory free (waits up to wait_min minutes for memory)."""
    import psutil
    fr = free_gib(); assert fr >= MIN_FREE_GIB, f"C: free {fr:.2f} GiB < {MIN_FREE_GIB}"
    t0 = time.time()
    while psutil.virtual_memory().available / 2 ** 30 < MIN_RAM_GIB:
        assert time.time() - t0 < wait_min * 60, f"free RAM {psutil.virtual_memory().available / 2 ** 30:.2f} GiB < {MIN_RAM_GIB}"
        time.sleep(30)
    return fr, psutil.virtual_memory().available / 2 ** 30


def peak_gib():
    import psutil
    return round(psutil.Process().memory_info().peak_wset / 2 ** 30, 2)


def gpu_note(job, minutes, dev):
    if dev == "cuda":
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / "gpu_minutes.jsonl", "a") as f:
            f.write(json.dumps(dict(utc=utc(), job=job, minutes=round(minutes, 2))) + "\n")


def gpu_used():
    f = OUT / "gpu_minutes.jsonl"
    return sum(json.loads(x)["minutes"] for x in f.read_text().splitlines() if x.strip()) if f.exists() else 0.


# ---------------------------------------------------------------- pools, flags, the fit
def windows_gate(obs, q, el, p1):
    """qcnn_shift.windows(step=1); p1: the first test (observed share of t-12..t-1 >= .5) becomes observed >= .30 and eligible >= .70."""
    out = []
    for d in range(obs.shape[0]):
        for t in range(24, 282):
            if p1:
                if obs[d, t - 12:t].mean() < .30 or el[d, t - 12:t].mean() < .70:
                    continue
            elif obs[d, t - 12:t].mean() < .5:
                continue
            if not ((q[d, t - 12:t].sum(0) >= 2).any() and (q[d, t + 1:t + 7] & obs[d, t + 1:t + 7]).any()):
                continue
            out.append((d, t))
    return out


def prep(data, shipped_w):
    """adds obs, q, W0 (shipped pool, from the shipped function) and W1 (P1 pool) to every panel; el must be present."""
    lost = 0
    for v in data.values():
        v["obs"] = v["c3"][:, :, 3] > .5; v["q"] = v["c3"][:, :, 4] > .5
        assert v["el"].shape == v["obs"].shape
        v["W0"] = [(int(d), int(t)) for d, t in shipped_w(v)]; v["W1"] = windows_gate(v["obs"], v["q"], v["el"], True)
        assert windows_gate(v["obs"], v["q"], v["el"], False) == v["W0"], "windows_gate(p1=False) != qcnn_shift.windows"
        lost += len(set(v["W0"]) - set(v["W1"]))
    return lost


def flags_p1(v, B, S):
    """a2s1.flags_p1 (worker 2): ong2_sccnn.make_flags for ongoing pool rows with the pre-T gate of t-1 = the P1 gate
    (observed >= .30, eligible >= .70, a visible queue). Same use of the dropout rng (one draw per row in train mode)."""
    out = []
    for d, t in B:
        obs, q, el = v["obs"][d], v["q"][d], v["el"][d]

        def pers(o):
            last = np.zeros(q.shape[1], bool)
            for s in range(o - 12, o):
                last = np.where(obs[s], q[s], last)
            val = obs[o + 1:o + 7]; fut = q[o + 1:o + 7] & val
            pm = np.broadcast_to(last, fut.shape) & val; u = (pm | fut).sum()
            return ((pm & fut).sum() / u if u else 1.), bool(fut.any())
        pt, _ = pers(t)
        gate = obs[t - 13:t - 1].mean() >= .30 and el[t - 13:t - 1].mean() >= .70 and bool(((q[t - 13:t - 1] & obs[t - 13:t - 1]).sum(0) >= 2).any())
        pa = False
        if gate:
            p1, f1 = pers(t - 1); pa = f1 and p1 > .9
        fh, fa = float(pt <= .9), float(pa)
        if S.MODE == "unknown" or (S.MODE == "train" and S._rng.random() < S.P_UNK):
            out.append((fh, 0., 1.))
        else:
            out.append((fh, fa, 0.))
    return out


def imports():
    """the unpatched make_x, taken before anything can call ong2_sccnn.enable()."""
    global MAKE_X
    import qcnn_train, qcnn_adapt
    if MAKE_X is None:
        assert not getattr(qcnn_train, "_sccnn", False), "ong2_sccnn.enable() ran before a3.imports()"
        MAKE_X = qcnn_adapt.make_x


def fit(base, groups, seed, sc, drop, dev, epochs, lr, p1):
    """qcnn_predict_adapt.finetune (cap None) = c_cold_split.ft on explicit (panel data, windows) groups in PANELS order:
    AdamW wd 1e-4, batch 32, the same rng use and batch order, labels and loss mask on the observed horizon cells.
    sccnn flags: shipped ong2_sccnn.make_flags (gate .5), or flags_p1 in arm P1. Returns (model, steps)."""
    import torch, torch.nn.functional as Fn, ong2_sccnn as S
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    if sum(len(W) for _, W in groups) == 0:
        return base, 0
    model = copy.deepcopy(base); model.train(); steps = 0
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        order = [(g, i) for g, (v, W) in enumerate(groups) for i in range(0, len(W), 32)]
        rng.shuffle(order)
        for g, i in order:
            v, W = groups[g]; B = [W[j] for j in rng.permutation(len(W))[:32]]
            x = MAKE_X(v["c3"], None, v["stat"], B, drop)
            if sc:
                x = S.fill(x, flags_p1(v, B, S) if p1 else S.make_flags(v["c3"], B, [True] * len(B), .5))
            if dev == "cuda":
                torch.backends.cudnn.deterministic = len(B) == 1     # fast cuDNN mode is not repeatable for a batch of one (c_adapt4o)
            obs = np.stack([v["obs"][d, t + 1:t + 7] for d, t in B]); q = np.stack([v["q"][d, t + 1:t + 7] for d, t in B])
            y = torch.tensor(q & obs, dtype=torch.float32, device=dev); m = torch.tensor(obs, dtype=torch.float32, device=dev)
            loss = (Fn.binary_cross_entropy_with_logits(model(x.to(dev)), y, reduction="none") * m).sum() / (m.sum() + 1)
            opt.zero_grad(); loss.backward(); opt.step(); steps += 1
    model.eval()
    return model, steps


def mdiff(m1, m2):
    return 0. if m1 is m2 else max(float((x.cpu() - y.cpu()).abs().max()) for x, y in zip(m1.state_dict().values(), m2.state_dict().values()))


def same_type_days(dates, k):
    return sum(1 for j in range(k) if weekend(dates[j]) == weekend(dates[k]))


def load_member(name, dev):
    import torch
    from qcnn_train import Net, CTX
    run = ROOT / RUNS[name]; cfg = json.loads((run / "config.json").read_text())
    assert cfg["drop"] == ["prior", "pos", "tod"] and cfg["epochs"] == 16 and not cfg.get("ramps", False) and bool(cfg.get("sccnn", False)) == name.startswith("sc"), cfg
    net = Net(cin=CTX * 5 + 16); net.load_state_dict(torch.load(run / "model.pt", map_location="cpu")); net.to(dev).eval()
    return net, cfg


def load_official(split):
    import qcnn_predict_adapt as qpa
    data = qpa.load_split(split, False)
    for v in data.values():
        v["el"] = np.asarray(v["c"]["elig"], bool)
        reg = v["c"].regimes if not callable(v["c"].regimes) else v["c"].regimes()
        v["reg"] = [str(x) for x in reg]; v["dates"] = np.array([str(x) for x in v["dates"]])
    lost = prep(data, lambda v: v["W"])
    return data, lost


def date_plan(data):
    alld = sorted({str(x) for v in data.values() for x in v["dates"]})
    offd = sorted({str(pd.Timestamp(o).date()) for v in data.values() for o in v["wi"].forecast_origin})
    return alld, offd, [dict(ko=ko, date=d, k=alld.index(d), pool=same_type_days(alld, alld.index(d))) for ko, d in enumerate(offd)]


def groups_mh5(data, key, ds):
    from qcnn_train import PANELS
    kind = weekend(ds)
    return [(data[p], [(d, t) for d, t in data[p][key] if data[p]["dates"][d] < ds and weekend(data[p]["dates"][d]) == kind]) for p in PANELS]


def groups_all(data, key, ds):
    from qcnn_train import PANELS
    return [(data[p], [(d, t) for d, t in data[p][key] if data[p]["dates"][d] < ds]) for p in PANELS]


def groups_sameday(data, key, ds, H):
    from qcnn_train import PANELS
    return [(data[p], [(d, t) for d, t in data[p][key] if data[p]["dates"][d] == ds and t + 6 <= H]) for p in PANELS]


# ---------------------------------------------------------------- official-form member job
def off(kind, arm, split, cpu=False, dates=None, nmem=5, budget=22., tag=""):
    import torch, qcnn_predict_adapt as qpa, ong2_sccnn as S
    imports(); guard(wait_min=5); t0 = time.time()
    assert kind in ("ctl", "sc") and arm in ("P0", "P1") and split in SPLITS
    dev = "cpu" if cpu else "cuda"; assert cpu or torch.cuda.is_available()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2"))); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = False
    pdir = OUT / f"off{tag}" / f"{kind}_{arm}_{split}"; pdir.mkdir(parents=True, exist_ok=True)
    if (pdir / "DONE.json").exists():
        print("exists", pdir); return 0
    sc = kind == "sc"; p1 = arm == "P1"; key = "W1" if p1 else "W0"
    names = [f"{kind}_s{i}" for i in range(nmem)]; bases = []
    for n in names:
        b, cfg = load_member(n, dev); bases.append(b)
    drop = cfg["drop"]; assert (qpa.BS, qpa.POOL_MAX_PERS) == (32, None)
    data, lost = load_official(split); alld, offd, plan = date_plan(data)
    types = pd.read_csv(ROOT / TYPES).set_index("window_id").typ.to_dict()
    print(json.dumps(dict(job=f"{kind}_{arm}_{split}", device=dev, members=names, shipped_pool_windows_lost_under_p1=lost,
                          pool_windows_month=dict(ship=sum(len(v["W0"]) for v in data.values()), p1=sum(len(v["W1"]) for v in data.values())),
                          plan=[(f["date"], f["pool"]) for f in plan], load_min=round((time.time() - t0) / 60, 2))), flush=True)
    assert lost == 0, "the P1 pool is not a superset of the shipped pool"

    def rs(seed):
        if sc:
            S.MODE = "train"; S._rng = np.random.default_rng(seed)

    def serve(model, x):
        if dev == "cuda":
            torch.backends.cudnn.deterministic = True                # predictions are batches of one window
        with torch.no_grad():
            return torch.sigmoid(model(x.to(dev)))[0].cpu().numpy().reshape(-1).astype(np.float32)
    todo = [f for f in plan if (dates is None or f["date"] in dates) and not (pdir / f"{f['date']}.parquet").exists()]
    more = False
    for n_done, f in enumerate(todo):
        if n_done and (time.time() - t0) / 60 > budget:
            more = True; break
        ds, ko, k, cold = f["date"], f["ko"], f["k"], f["pool"] <= COLD; t1 = time.time(); X = []
        for p, v in data.items():
            wi = v["wi"][[str(pd.Timestamp(o).date()) == ds for o in v["wi"].forecast_origin]]
            for w, t, x in qpa.official_x(v["c3"], v["rc"], v["stat"], v["st"], v["links"], wi, v["hist"], drop):
                if sc:
                    x = S.fill(x, [FL[types[w.window_id]]])
                X.append((p, w, t, x))
        Hs = sorted({12 * (t // 12) for _, _, t, _ in X})
        for attempt in range(4):                                     # a sporadic cuDNN error must not lose the job (c_adapt4o): refit the date, same seeds
            try:
                pa = [[None] * len(X) for _ in bases]; steps = 0; npool = 0
                if not cold:
                    G = groups_mh5(data, key, ds); npool = sum(len(W) for _, W in G)
                    rs(1000 + ko)                                    # shipped sccnn main: one rng per date, members fitted in order
                    for j, b in enumerate(bases):
                        m, n = fit(b, G, ko, sc, drop, dev, *MH5, p1); steps += n
                        for i, (_, _, _, x) in enumerate(X):
                            pa[j][i] = serve(m, x)
                        del m
                else:
                    G = groups_all(data, key, ds); npool = sum(len(W) for _, W in G)
                    for j, b in enumerate(bases):                    # c_cold_split.official: every member with its own reseeded rng
                        rs(10_000 + k); xall, n = fit(b, G, k, sc, drop, dev, *XSD, p1); steps += n
                        for H in Hs:
                            Gs = groups_sameday(data, key, ds, H)
                            rs(20_000 + 100 * k + H // 12); m, n = fit(xall, Gs, 1000 * k + H, sc, drop, dev, *XSD, p1); steps += n
                            for i, (_, _, t, x) in enumerate(X):
                                if 12 * (t // 12) == H:
                                    pa[j][i] = serve(m, x)
                            del m
                        del xall
                break
            except RuntimeError as e:
                if ("cuDNN" not in str(e) and "CUDA" not in str(e)) or attempt == 3:
                    raise
                print(json.dumps(dict(date=ds, kind="gpu_retry", attempt=attempt + 1, error=str(e)[:160])), flush=True)
                torch.cuda.empty_cache(); time.sleep(5)
        rows = []
        for i, (p, w, t, x) in enumerate(X):
            v = data[p]; L = len(v["links"]); T = pd.Timestamp(w.forecast_origin)
            row = {"split": split, "panel": p, "window_id": w.window_id, "date": ds, "t": t, "H": 12 * (t // 12), "ko": ko, "k": k, "pool": f["pool"], "cold": cold,
                   "reg": v["reg"][list(v["dates"]).index(ds)], "typ": types[w.window_id], "pool_windows": npool,
                   "timestamp": np.repeat([(T + pd.Timedelta(minutes=5 * (h + 1))).strftime("%Y-%m-%dT%H:%M:%SZ") for h in range(6)], L),
                   "link_id": np.tile(v["links"], 6)}
            for j, b in enumerate(bases):
                row[f"p_un_s{j}"] = serve(b, x); row[f"p_s{j}"] = pa[j][i]
            rows.append(pd.DataFrame(row))
        if dev == "cuda":
            torch.backends.cudnn.deterministic = False; torch.cuda.empty_cache()
        tmp = pdir / f"{ds}.tmp"; pd.concat(rows, ignore_index=True).to_parquet(tmp, index=False); os.replace(tmp, pdir / f"{ds}.parquet")
        info = dict(job=f"{kind}_{arm}_{split}", date=ds, ko=ko, k=k, pool_days=f["pool"], cold=cold, hours=len(Hs), pool_windows=npool, windows=len(X), steps=steps,
                    device=dev, members=nmem, fit_min=round((time.time() - t1) / 60, 2), made=utc())
        (pdir / f"{ds}.json").write_text(json.dumps(info)); print(json.dumps(info), flush=True)
    full = dates is None and nmem == 5 and all((pdir / f"{f['date']}.parquet").exists() for f in plan)
    mins = (time.time() - t0) / 60; gpu_note(f"off {kind}_{arm}_{split}{tag}", mins, dev)
    if full:
        js = [json.loads((pdir / f"{f['date']}.json").read_text()) for f in plan]
        (pdir / "DONE.json").write_text(json.dumps(dict(job=f"{kind}_{arm}_{split}", dates=len(plan), windows=sum(x["windows"] for x in js), steps=sum(x["steps"] for x in js),
                                                        fit_min=round(sum(x["fit_min"] for x in js), 2), torch=str(torch.__version__), made=utc()), indent=1))
    print("A3_OFF", "DONE" if full else ("MORE" if more else "PARTIAL"), json.dumps(dict(job=f"{kind}_{arm}_{split}", minutes=round(mins, 2), peak_ram_gib=peak_gib(),
          gpu_peak_mib=(round(torch.cuda.max_memory_allocated() / 2 ** 20) if dev == "cuda" else 0))), flush=True)
    return 0


def codeparity(split):
    """CPU, 1 epoch, members s0: fit() with the shipped pool == qcnn_predict_adapt.finetune (warm recipe) and == c_cold_split.ft
    (all-earlier stage and one same-day stage), bitwise, for the base CNN and (after ong2_sccnn.enable()) the sccnn."""
    import torch, qcnn_predict_adapt as qpa, ong2_sccnn as S, c_cold_split as CS
    imports(); guard(); torch.set_num_threads(2); t0 = time.time(); res = []
    data, lost = load_official(split); alld, offd, plan = date_plan(data)
    for v in data.values():
        v["drop"] = ["prior", "pos", "tod"]
    warm = [f for f in plan if f["pool"] > COLD][0]
    cold = [f for f in plan if f["pool"] <= COLD and sum(len(W) for _, W in groups_all(data, "W0", f["date"])) > 0][0]
    Hc = None
    for p, v in data.items():
        for o in v["wi"].forecast_origin:
            if str(pd.Timestamp(o).date()) == cold["date"]:
                t = pd.Timestamp(o).hour * 12 + pd.Timestamp(o).minute // 5; H = 12 * (t // 12)
                if sum(len(W) for _, W in groups_sameday(data, "W0", cold["date"], H)) > 0:
                    Hc = max(Hc or 0, H)
    for kind in ("ctl", "sc"):                                      # ctl first: enable() patches make_x for everything after it
        sc = kind == "sc"; base, cfg = load_member(f"{kind}_s0", "cpu"); drop = cfg["drop"]
        if sc:
            S.enable()

        def rs(seed):
            if sc:
                S.MODE = "train"; S._rng = np.random.default_rng(seed)
        e0, l0 = qpa.EPOCHS, qpa.LR; qpa.EPOCHS, qpa.LR = 1, MH5[1]
        rs(1000 + warm["ko"]); ref = qpa.finetune(base, data, warm["date"], drop, "cpu", seed=warm["ko"]); qpa.EPOCHS, qpa.LR = e0, l0
        rs(1000 + warm["ko"]); mine, n = fit(base, groups_mh5(data, "W0", warm["date"]), warm["ko"], sc, drop, "cpu", 1, MH5[1], False)
        res.append(dict(split=split, kind=kind, check="mh5 fit == qcnn_predict_adapt.finetune", date=warm["date"], steps=n, max_diff=mdiff(ref, mine)))
        print(json.dumps(res[-1]), flush=True)
        k = cold["k"]; ds = cold["date"]
        rs(10_000 + k); ref = CS.ft(base, CS.pool_groups(data, ds, None), k, 1, XSD[1])
        rs(10_000 + k); mine, n = fit(base, groups_all(data, "W0", ds), k, sc, drop, "cpu", 1, XSD[1], False)
        res.append(dict(split=split, kind=kind, check="XSD all-earlier fit == c_cold_split.ft", date=ds, steps=n, max_diff=mdiff(ref, mine))); print(json.dumps(res[-1]), flush=True)
        if Hc is not None:
            rs(20_000 + 100 * k + Hc // 12); ref2 = CS.ft(ref, CS.sameday_groups(data, ds, Hc), 1000 * k + Hc, 1, XSD[1])
            rs(20_000 + 100 * k + Hc // 12); mine2, n = fit(mine, groups_sameday(data, "W0", ds, Hc), 1000 * k + Hc, sc, drop, "cpu", 1, XSD[1], False)
            res.append(dict(split=split, kind=kind, check="XSD same-day fit == c_cold_split.ft", date=ds, H=Hc, steps=n, max_diff=mdiff(ref2, mine2))); print(json.dumps(res[-1]), flush=True)
    ok = all(r["max_diff"] == 0. and r["steps"] > 0 for r in res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"codeparity_{split}.json").write_text(json.dumps(dict(ok=ok, checks=res, p1_lost=lost, minutes=round((time.time() - t0) / 60, 1), peak_ram_gib=peak_gib(), made=utc()), indent=1))
    print("A3_CODEPARITY", "OK" if ok else "FAIL", split, flush=True)
    return 0 if ok else 3


# ---------------------------------------------------------------- OFF250 member job (Jan/Feb, Dec-cutoff members)
def month_data(month):
    """c_bb_raw.month_data without the raw cube r3 (not used here; it costs .2 GiB of the 1.5 GiB allowed)."""
    import qcnn_adapt
    from qcnn_data import cube
    from qcnn_train import PANELS
    from tfb.data import load_cache
    data = {}
    for pn in PANELS:
        c = load_cache(pn, "train"); st = c.stations.sort_values("idx")
        di = np.array([i for i, x in enumerate(c.dates) if str(x).startswith(month)])
        m3 = cube(*[np.asarray(c[x], np.float32)[di] for x in ("speed", "flow", "occ")], st)
        data[pn] = dict(c3=m3, el=np.asarray(c["elig"], bool)[di], rc=None, stat=qcnn_adapt.statics(st),
                        dates=np.array([str(c.dates[i]) for i in di]), links=st.link_id.to_numpy().astype(str))
        del c
    return data


def jf(member, arms=("P1",), cpu=False, maxdates=99, tag=""):
    import torch, c_adapt4 as A4, c_bb_raw, qcnn_predict_adapt as qpa, ong2_sccnn as S
    from qcnn_shift import windows
    from qcnn_train import PANELS
    imports(); guard(wait_min=5); t0 = time.time()
    dev = "cpu" if cpu else "cuda"; assert cpu or torch.cuda.is_available()
    out = OUT / f"jf{tag}" / f"{member}.npz"
    if out.exists():
        print("exists", out); return 0
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2"))); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = False
    base, cfg = A4.load(member, dev); drop = cfg["drop"]; sc = bool(cfg["sccnn"]); assert (qpa.BS, qpa.POOL_MAX_PERS) == (32, None) and all(a in ("P0", "P1") for a in arms)
    RR = [{kk: r[kk] for kk in ("month", "date", "in_legacy", "ty", "x_truth", "x_unk")} for r in torch.load(c_bb_raw.INPUTS, weights_only=False)["recs"]]   # only what c_adapt4.predict reads
    P = {}; steps_all = 0; log = []

    def rs(k):
        if sc:
            S.MODE = "train"; S._rng = np.random.default_rng(10_000 + k)      # c_adapt4.member
    for month in ("2031-01", "2031-02"):
        data = month_data(month); lost = prep(data, lambda v: windows(v["c3"], step=1)); assert lost == 0
        idx_m = [i for i, r in enumerate(RR) if r["month"] == month and r["in_legacy"]]; dates = sorted({RR[i]["date"] for i in idx_m})[:maxdates]
        for k, ds in enumerate(dates):
            idx = [i for i in idx_m if RR[i]["date"] == ds]
            for arm in arms:
                key = "W1" if arm == "P1" else "W0"; kind = weekend(ds)
                G = [(data[p], [(d, t) for d, t in data[p][key] if data[p]["dates"][d] < ds and weekend(data[p]["dates"][d]) == kind]) for p in PANELS]
                rs(k); m, n = fit(base, G, k, sc, drop, dev, *MH5, arm == "P1"); steps_all += n
                for i, p in A4.predict(m, RR, idx, sc, dev).items():
                    P[f"{arm}_{i}"] = p
                log.append(dict(month=month, date=ds, k=k, arm=arm, pool_windows=sum(len(W) for _, W in G), pool_windows_shipped=sum(len([w for w in data[p]["W0"] if data[p]["dates"][w[0]] < ds and weekend(data[p]["dates"][w[0]]) == kind]) for p in PANELS), steps=n, windows=len(idx)))
                del m
            if dev == "cuda":
                torch.cuda.empty_cache()
            print(json.dumps(dict(log[-1], min=round((time.time() - t0) / 60, 1))), flush=True)
        del data
    mins = (time.time() - t0) / 60; out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz"); np.savez_compressed(tmp, **{kk: vv.astype(np.float16) for kk, vv in P.items()}); os.replace(tmp, out)
    Path(str(out) + ".json").write_text(json.dumps(dict(run=member, sccnn=sc, arms=list(arms), device=dev, recipe=list(MH5), steps=steps_all, maxdates=maxdates,
                                                        torch=str(torch.__version__), minutes=round(mins, 2), peak_ram_gib=peak_gib(), fits=log), indent=1))
    gpu_note(f"jf {member}{tag}", mins, dev)
    print("A3_JF DONE", json.dumps(dict(member=member, steps=steps_all, minutes=round(mins, 2), peak_ram_gib=peak_gib())), flush=True)
    return 0


def noharm(tag=""):
    """OFF250: p = .75 x (mean of the family's 4 members adapted with the P1 pool) + .25 x unadapted, against p8, on the raw
    truth (g_score.cmd_score) and on the denoised truths (lab_score.score). Self-test: the same builder on the stored P0
    arm must give p8 exactly."""
    import pickle
    for p in (ROOT / "exp/G_inst", ROOT / "exp/L_lab", ROOT / "exp/F3_3"):
        if str(p) not in sys.path:
            sys.path.append(str(p))
    import g_score as GS, lab_score as LS
    guard(); inst = GS.INST; ND = OUT / f"noharm{tag}"; GS.G.OUT = ND; ND.mkdir(parents=True, exist_ok=True)     # the raw-truth score table lands here, not under g_inst
    df = pd.read_parquet(inst); T = pickle.load(open(ROOT / "artifacts/claude_ref/f3_2/inst_tm.pkl", "rb"))
    Z0 = {m: np.load(GS.A4 / f"{m}.npz") for m in GS.MEM}; Z1 = {m: np.load(OUT / f"jf{tag}" / f"{m}.npz") for m in JF}; assert sorted(JF) == sorted(GS.MEM)
    idx = sorted(int(k[3:]) for k in Z0[GS.MEM[0]].files if k.startswith("un_")); assert len(idx) == len(T) == 250
    pos = {k: g.index.to_numpy() for k, g in df.groupby(["panel", "date", "t"], sort=False)}
    p0 = np.full(len(df), np.nan, np.float32); p1 = p0.copy(); dmem = []
    for i, w in zip(idx, T):
        ii = pos[(w["panel"], w["date"], w["t"])]; ty = df.typ.to_numpy()[ii[0]]; grp = [m for m in GS.MEM if ("_sc" in m) == (ty == "A")]
        un = np.mean([Z0[m][f"un_{i}"].astype(np.float32) for m in grp], 0).reshape(-1)
        a0 = np.mean([Z0[m][f"P0_{i}"].astype(np.float32) for m in grp], 0).reshape(-1); a1 = np.mean([Z1[m][f"P1_{i}"].astype(np.float32) for m in grp], 0).reshape(-1)
        p0[ii] = .75 * a0 + .25 * un; p1[ii] = .75 * a1 + .25 * un; dmem.append(float(np.abs(a1 - a0).mean()))
    assert not np.isnan(p1).any() and float(np.abs(p0 - df.p8.to_numpy()).max()) == 0., "the builder does not reproduce p8 from the stored P0 arm"
    new = df[GS.KEY].copy(); new["p"] = p1; f = ND / f"a3_off250_p1{tag}.parquet"; new.to_parquet(f, index=False)
    print(f"OFF250 P1 built: 250 windows, {len(new)} cells, self-test p8 exact, mean |P1 - P0| adapted probability {np.mean(dmem):.5f}", flush=True)
    raw = GS.cmd_score(str(f), "p", "p8"); lab = LS.score(str(f), "p", "p8")
    r = pd.read_parquet(ND / f"score_a3_off250_p1{tag}_p.parquet"); reg = pd.read_csv(ROOT / "artifacts/claude_ref/l_lab/off250_windows.csv") if (ROOT / "artifacts/claude_ref/l_lab/off250_windows.csv").exists() else None
    by_reg = None
    if reg is not None and {"panel", "date", "t"} <= set(reg.columns) and ("reg" in reg.columns or "regime" in reg.columns):
        rc = "reg" if "reg" in reg.columns else "regime"; mm = r.merge(reg[["panel", "date", "t", rc]], on=["panel", "date", "t"], how="left")
        by_reg = {str(k): dict(n=int(len(g)), d=round(float(g.dl.mean()), 5), up=int((g.dl > 1e-12).sum()), down=int((g.dl < -1e-12).sum())) for k, g in mm.groupby(rc)}
    kill = raw["mix"] < -.002 and lab["kept"]["mix"] < -.002
    keep = lambda o: {k: (round(v, 6) if isinstance(v, float) else v) for k, v in o.items() if k in ("n", "changed", "mix", "lo", "hi", "winsor", "s_total", "wins", "losses", "verdict", "by_type", "pd_pos", "pd_neg")}
    res = dict(file=str(f.relative_to(ROOT)).replace("\\", "/"), base="p8", raw=keep(raw), denoised={k: keep(v) for k, v in lab.items()}, by_regime_raw=by_reg,
               kill_rule="below -.002 IoU on raw AND kept", kill=bool(kill), members=JF, gpu_minutes=round(sum(json.loads((OUT / f"jf{tag}" / f"{m}.npz.json").read_text())["minutes"] for m in JF), 1),
               pool_growth=round(float(np.mean([x["pool_windows"] / max(x["pool_windows_shipped"], 1) for m in JF[:1] for x in json.loads((OUT / f"jf{tag}" / f"{m}.npz.json").read_text())["fits"] if x["pool_windows_shipped"]])), 3),
               note="OFF250 has no XSD stage (p8 is mh5 on every date); it is a no-harm read only, blind to target-month gains", made=utc(), peak_ram_gib=peak_gib())
    (OUT / f"noharm{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print("A3_NOHARM", "KILL" if kill else "no-harm passed", f"raw {raw['mix']:+.5f} [{raw['lo']:+.5f},{raw['hi']:+.5f}] kept {lab['kept']['mix']:+.5f}", flush=True)
    return 0


# ---------------------------------------------------------------- ship: on/ files -> shipped chain -> ongoing rows
def chain_env():
    return dict(os.environ, TFB_ASM2_DET="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONPATH=str(ROOT / "src"),
                PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1")


def chain(on, Y, log):
    """shipped ongoing chain (asm2_run_all.sh lines 180-193, as c_adapt4o.chain): blend .75 adapted / .25 unadapted per member
    file (stored unadapted files), seed mean + eiou, sccnn on type A + noT, ConvNeXt member decode. Repo-relative posix paths."""
    def step(*args):
        subprocess.run([sys.executable, "-B", f"{R}/asm2_step.py", *[str(x) for x in args]], cwd=str(ROOT), env=chain_env(), stdout=log, stderr=subprocess.STDOUT, check=True)
    (ROOT / Y).mkdir(parents=True, exist_ok=True)
    step(f"{R}/c_noadapt_blend.py", on, f"{M}/off", MIX, Y)
    step(f"{R}/qcnn_avg_probs.py", f"{Y}/cnn_ens2.csv", *sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / Y).glob("cnn_s[0-9].parquet")))
    step(f"{R}/ong2_probe.py", "build", f"--sc={Y}/sccnn_official.parquet", "--noT", f"--base-probs={Y}/cnn_ens2.parquet", f"--ongoing-out={Y}/ongoing.parquet")
    step(f"{R}/asm2_bbmember.py", "decode", "--cfg", CFG, "--member", BB, "--base", f"{Y}/cnn_ens2.parquet", "--sc", f"{Y}/sccnn_official.parquet",
         "--parent-ongoing", f"{Y}/ongoing.parquet", "--out", f"{Y}/ongoing_bb.parquet")


def ongoing_key():
    """submission_id, split, window_id, timestamp, link_id of the 87,000 ongoing rows + type, regime, date, panel (cached)."""
    f = OUT / "ongoing_key.parquet"
    if f.exists():
        return pd.read_parquet(f)
    ty = pd.read_csv(ROOT / TYPES); ty = ty[ty.split != "train"]; wins = set(ty.window_id)
    key = pd.concat([x[x.window_id.isin(wins)] for x in pd.read_csv(KEYF, usecols=["submission_id", "task", "split", "panel", "timestamp", "link_id", "mask_regime", "window_id"],
                                                                     dtype=str, chunksize=250000)], ignore_index=True)
    assert len(key) == 87000 and (key.task == "queue").all()
    key["submission_id"] = key.submission_id.astype(np.int64)
    key = key.merge(ty[["window_id", "typ", "date", "t"]], on="window_id", how="left", validate="many_to_one")
    key["date"] = key.date.astype(str); OUT.mkdir(parents=True, exist_ok=True); key.to_parquet(f, index=False)
    return key


def table(ref, new, label, reg=None):
    """changed ongoing cells new vs ref (87,000 rows by submission_id) per split, type and mask regime of the day."""
    key = ongoing_key().set_index("submission_id")
    r = ref.set_index("submission_id").queue_pred; n = new.set_index("submission_id").queue_pred.reindex(r.index)
    assert len(r) == 87000 and n.notna().all() and set(r.index) == set(key.index)
    d = key.loc[r.index].copy(); d["old"] = r.to_numpy(float); d["new"] = n.to_numpy(float); d["ch"] = d.old != d.new
    if reg is not None:
        d["reg"] = d.window_id.map(reg)
    else:
        d["reg"] = d.mask_regime
    w = d.groupby(["split", "window_id", "typ", "reg", "date", "panel"]).agg(cells=("ch", "size"), changed=("ch", "sum"), to1=("new", lambda s: 0), pos_old=("old", "sum"), pos_new=("new", "sum")).reset_index()
    w["to1"] = d[d.ch & (d.new > d.old)].groupby("window_id").size().reindex(w.window_id).fillna(0).to_numpy(int); w["to0"] = w.changed - w.to1
    cw = w[w.changed > 0]
    cnt = lambda cols: {"/".join(map(str, k if isinstance(k, tuple) else (k,))): dict(windows=int(len(g)), windows_changed=int((g.changed > 0).sum()), cells_changed=int(g.changed.sum()),
                                                                                    to1=int(g.to1.sum()), to0=int(g.to0.sum())) for k, g in w.groupby(cols)}
    res = dict(label=label, changed_cells=int(d.ch.sum()), changed_windows=int(len(cw)), by_split=cnt(["split"]), by_split_type=cnt(["split", "typ"]), by_split_regime=cnt(["split", "reg"]),
               by_split_regime_type=cnt(["split", "reg", "typ"]), by_split_date=cnt(["split", "date"]))
    print(f"== {label}: changed {res['changed_cells']} of 87000 cells in {res['changed_windows']} windows | " +
          " | ".join(f"{k}: {v['windows_changed']} of {v['windows']} windows, {v['cells_changed']} cells (+{v['to1']} / -{v['to0']})" for k, v in res["by_split"].items()), flush=True)
    return res, w


def parts(kind, arm, split, tag=""):
    d = OUT / f"off{tag}" / f"{kind}_{arm}_{split}"; assert (d / "DONE.json").exists(), f"job {kind}_{arm}_{split} is not finished"
    return pd.concat([pd.read_parquet(f, use_threads=False) for f in sorted(d.glob("2031-*.parquet"))], ignore_index=True)


def ref_check():
    """(1) the parent file's ongoing rows == the stored C36 rows (c_cold/official/ship_c35/a0.75/ongoing_bb.parquet);
    (2) the shipped chain on the stored member files gives those rows again."""
    guard(); res = {}
    r = rp(f"{C36}/a0.75/ongoing_bb.parquet").set_index("submission_id").queue_pred
    p = pd.read_csv(PARENT, usecols=["submission_id", "task", "queue_pred"], float_precision="round_trip").set_index("submission_id")
    assert len(p) == 6_980_503 and len(r) == 87000 and (p.loc[r.index, "task"] == "queue").all()
    nd = int((p.loc[r.index, "queue_pred"].to_numpy(float) != r.to_numpy(float)).sum()); res["parent_ongoing_rows_differ_from_stored"] = nd; del p
    key = ongoing_key(); assert set(key.submission_id) == set(r.index)
    Y = (OUT / "chain_parity").relative_to(ROOT).as_posix(); LOGS.mkdir(exist_ok=True)
    with open(LOGS / "chain_parity.log", "w") as log:
        chain(f"{C36}/on", Y, log)
    n = rp(f"{Y}/ongoing_bb.parquet").set_index("submission_id").queue_pred.reindex(r.index)
    res["chain_on_stored_files_cells_differ"] = int((n.to_numpy(float) != r.to_numpy(float)).sum())
    res["chain_on_stored_files_bytes_equal"] = bool(filecmp.cmp(ROOT / Y / "ongoing_bb.parquet", ROOT / C36 / "a0.75/ongoing_bb.parquet", shallow=False))
    res["ok"] = nd == 0 and res["chain_on_stored_files_cells_differ"] == 0; res["made"] = utc(); res["peak_ram_gib"] = peak_gib()
    (OUT / "ref_check.json").write_text(json.dumps(res, indent=1)); print("A3_REF", "OK" if res["ok"] else "FAIL", json.dumps(res), flush=True)
    return 0 if res["ok"] else 3


def ship(arm, tag=""):
    import torch
    from qcnn_shift import eiou_mask
    guard(); X = (OUT / f"ship_{arm}{tag}").relative_to(ROOT).as_posix(); (ROOT / X / "on").mkdir(parents=True, exist_ok=True)
    ctl = pd.concat([parts("ctl", arm, s, tag) for s in SPLITS], ignore_index=True); scd = pd.concat([parts("sc", arm, s, tag) for s in SPLITS], ignore_index=True)
    assert len(ctl) == len(scd) == 87000 and (ctl[K].to_numpy() == scd[K].to_numpy()).all() and ctl.window_id.nunique() == 80
    cold = ctl.cold.to_numpy(); par = {}; f8 = lambda x: np.asarray(x, np.float64)
    wins = ctl.drop_duplicates("window_id")[["split", "panel", "window_id", "date", "t", "H", "ko", "k", "pool", "cold", "reg", "typ", "pool_windows"]]
    wins.to_csv(ROOT / X / "windows.csv", index=False)
    stored_cold = set(pd.read_csv(ROOT / "artifacts/claude_ref/c_cold/official/ship/cold_windows.csv").window_id)
    assert set(wins.window_id[wins.cold]) == stored_cold, "my cold windows != the shipped cold windows"

    def put(fname, key_df, p_new, p_un, decide):
        on = rp(f"{C36}/on/{fname}"); of = rp(f"{M}/off/{fname}")
        assert (on[K].to_numpy() == of[K].to_numpy()).all()
        idx = pd.Series(np.arange(len(on)), index=pd.MultiIndex.from_frame(on[K])).reindex(pd.MultiIndex.from_frame(key_df[K]))
        assert idx.notna().all() and idx.nunique() == len(on) == len(key_df), f"rows of {fname} not covered one to one"; idx = idx.to_numpy(np.int64)
        du = np.abs(f8(of.prob.to_numpy()[idx]) - f8(p_un)); dn = np.abs(f8(on.prob.to_numpy()[idx]) - f8(p_new))
        st = lambda a: [float(a.max()), float(a.mean()), float(np.quantile(a, .999))]
        par[fname] = dict(un_vs_stored_off=st(du), adapted_vs_stored_on_warm=st(dn[~cold]), adapted_vs_stored_on_cold=st(dn[cold]))
        new = on.copy(); pr = new.prob.to_numpy().copy(); pr[idx] = np.asarray(p_new).astype(pr.dtype); new["prob"] = pr
        if decide:
            q = new.queue_pred.to_numpy().copy()
            for w, ii in on.groupby("window_id", sort=False).indices.items():
                assert np.array_equal(eiou_mask(on.prob.to_numpy()[ii]).astype(q.dtype), on.queue_pred.to_numpy()[ii]), "decoder parity on the stored file"
                q[ii] = eiou_mask(pr[ii]).astype(q.dtype)
            new["queue_pred"] = q
        new.to_parquet(ROOT / X / "on" / fname, index=False)
        v = par[fname]; print(f"{arm} {fname}: un vs stored off max {v['un_vs_stored_off'][0]:.2e} mean {v['un_vs_stored_off'][1]:.2e} | adapted vs stored on: warm max {v['adapted_vs_stored_on_warm'][0]:.4f} mean {v['adapted_vs_stored_on_warm'][1]:.6f}, cold max {v['adapted_vs_stored_on_cold'][0]:.4f} mean {v['adapted_vs_stored_on_cold'][1]:.6f}", flush=True)
    for i in range(5):
        put(f"cnn_s{i}.parquet", ctl, ctl[f"p_s{i}"].to_numpy(), ctl[f"p_un_s{i}"].to_numpy(), True)
    ens = lambda pre: torch.stack([torch.from_numpy(scd[f"{pre}{j}"].to_numpy(np.float32).copy()) for j in range(5)]).mean(0).numpy()   # shipped seed-mean formula
    put("sccnn_official.parquet", scd, ens("p_s"), ens("p_un_s"), False)
    mem = {}
    if arm == "P0":                                                 # per member against the stored cold-window member files (CPU build, p_xsd)
        for kind, df in (("ctl", ctl), ("sc", scd)):
            for j in range(5):
                s = rp(f"{COLD_MEMBERS}/{kind}_s{j}.parquet"); mine = df[df.cold].set_index(K)[f"p_s{j}"].reindex(pd.MultiIndex.from_frame(s[K]))
                assert mine.notna().all(); dd = np.abs(f8(mine.to_numpy()) - f8(s.p_xsd.to_numpy())); mem[f"{kind}_s{j}"] = [float(dd.max()), float(dd.mean())]
        print("P0 cold windows, per member vs the stored CPU XSD probabilities (max, mean):", json.dumps({k: [round(v[0], 4), round(v[1], 6)] for k, v in mem.items()}), flush=True)
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / f"chain_{arm}{tag}.log", "w") as log:
        chain(f"{X}/on", f"{X}/a0.75", log)
    refrows = rp(f"{C36}/a0.75/ongoing_bb.parquet"); new = rp(f"{X}/a0.75/ongoing_bb.parquet")
    res, w = table(refrows, new, f"{arm} path vs the parent's ongoing rows", wins.set_index("window_id").reg.to_dict()); w.to_csv(ROOT / X / "windows_vs_parent.csv", index=False)
    bad = []
    for fn, v in par.items():
        if v["un_vs_stored_off"][0] > 2e-2 or v["un_vs_stored_off"][1] > 1e-4:
            bad.append(f"un parity {fn} {v['un_vs_stored_off']}")
        if arm == "P0" and max(v["adapted_vs_stored_on_warm"][1], v["adapted_vs_stored_on_cold"][1]) > 5e-4:
            bad.append(f"adapted parity {fn} warm {v['adapted_vs_stored_on_warm']} cold {v['adapted_vs_stored_on_cold']}")
    if arm == "P0" and res["changed_cells"] > CONTROL_MAX_CELLS:
        bad.append(f"control: {res['changed_cells']} decoded cells differ from the parent (limit {CONTROL_MAX_CELLS})")
    res.update(arm=arm, parity=par, cold_members_vs_stored_xsd=mem, bad=bad, ok=not bad, made=utc(), peak_ram_gib=peak_gib(),
               control=("EXACT" if res["changed_cells"] == 0 else "NEAR") if arm == "P0" and not bad else None)
    (ROOT / X / "ship.json").write_text(json.dumps(res, indent=1))
    for b in bad:
        print("CHECK_FAIL", b, flush=True)
    print(f"A3_SHIP {arm}", "OK" if not bad else "FAIL", (f"CONTROL {res['control']}: {res['changed_cells']} of 87000 cells differ from the parent" if arm == "P0" else ""), flush=True)
    return 0 if not bad else 3


# ---------------------------------------------------------------- delta on the parent, checks, window guard
def run_tool(args, logf, check=True):
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONPATH=str(ROOT / "src"))
    p = subprocess.run([sys.executable, "-B", *[str(a) for a in args]], cwd=str(ROOT), env=env, capture_output=True, text=True)
    with open(logf, "a", encoding="utf-8") as f:
        f.write(f"[{utc()}] $ {' '.join(str(a) for a in args)}\n{p.stdout}\n{p.stderr}\nrc={p.returncode}\n")
    if check:
        assert p.returncode == 0, f"{args[0]} {args[1] if len(args) > 1 else ''} failed rc={p.returncode}: {(p.stdout + p.stderr)[-400:]}"
    return p


def stream_check(parent, full, seed_ids):
    """line by line: FULL equals PARENT except queue_pred on exactly the seed ids (all queue-task rows)."""
    n = diff = 0; seen = set()
    with open(parent, encoding="utf-8", newline="") as a, open(full, encoding="utf-8", newline="") as b:
        ha = next(a).rstrip("\r\n").split(","); hb = next(b).rstrip("\r\n").split(","); assert ha == hb; qi = ha.index("queue_pred"); ti = ha.index("task")
        for la, lb in zip(a, b):
            n += 1
            if la == lb:
                continue
            va, vb = la.rstrip("\r\n").split(","), lb.rstrip("\r\n").split(",")
            assert va[0] == vb[0] and va[ti] == vb[ti] == "queue" and all(x == y for i, (x, y) in enumerate(zip(va, vb)) if i != qi) and float(va[qi]) != float(vb[qi]), f"unexpected difference at id {va[0]}"
            seen.add(int(va[0])); diff += 1
        assert next(a, None) is None and next(b, None) is None
    assert n == 6_980_503 and seen == set(seed_ids), (n, len(seen), len(seed_ids))
    return diff


def delta(new_rows=None, name=NAME, tmpdir=None, outdir=None, tag=""):
    """candidate ongoing rows = the parent's rows with the cells where the P1 path differs from the P0 path (both from this
    GPU path; equal to the plain P1 rows when the control is exact). Counts by month, regime and type; window rule (>= 3
    March windows or none); tools/probe.py expand + make + expand (sha check); tools/window_guard.py --no-record; a line by
    line check that nothing but queue_pred on the changed ongoing rows differs. new_rows: test hook (a rows parquet)."""
    assert (new_rows is None and not tag) == (outdir is None), "a test run (--rows or --tag) must write outside the project (--out)"
    guard(); D = OUT / "delta" if outdir is None else Path(outdir); D.mkdir(parents=True, exist_ok=True); LOGS.mkdir(exist_ok=True); logf = (LOGS if outdir is None else D) / "delta.log"
    ref = rp(f"{C36}/a0.75/ongoing_bb.parquet"); r = ref.set_index("submission_id").queue_pred
    if new_rows is None:
        a0 = rp((OUT / f"ship_P0{tag}/a0.75/ongoing_bb.parquet")).set_index("submission_id").queue_pred.reindex(r.index); a1 = rp((OUT / f"ship_P1{tag}/a0.75/ongoing_bb.parquet")).set_index("submission_id").queue_pred.reindex(r.index)
        reg = pd.read_csv(OUT / f"ship_P1{tag}/windows.csv").set_index("window_id").reg.to_dict()
    else:
        a0 = r.copy(); a1 = rp(new_rows).set_index("submission_id").queue_pred.reindex(r.index); reg = None
    assert a0.notna().all() and a1.notna().all()
    eff = a1.to_numpy(float) != a0.to_numpy(float); v = r.to_numpy(float).copy(); v[eff] = a1.to_numpy(float)[eff]
    iso = pd.DataFrame({"submission_id": r.index.to_numpy(), "queue_pred": v}); iso.to_parquet(D / "ongoing_candidate.parquet", index=False)
    res = dict(name=name, parent=str(PARENT.relative_to(ROOT)).replace("\\", "/"), control_cells=int((a0.to_numpy(float) != r.to_numpy(float)).sum()), p1_vs_p0_cells=int(eff.sum()),
               p1_vs_parent_cells=int((a1.to_numpy(float) != r.to_numpy(float)).sum()))
    res["candidate"], w = table(ref, iso, "CANDIDATE rows (parent + P1 minus P0) vs the parent", reg); w.to_csv(D / "windows_changed.csv", index=False)
    res["plain_p1_rows"], _ = table(ref, pd.DataFrame({"submission_id": r.index.to_numpy(), "queue_pred": a1.to_numpy(float)}), "plain P1 rows vs the parent", reg)
    nmar = res["candidate"]["by_split"].get("validation", {}).get("windows_changed", 0); napr = res["candidate"]["by_split"].get("private", {}).get("windows_changed", 0)
    res["march_windows_changed"] = nmar; res["april_windows_changed"] = napr
    res["window_rule"] = "PASS (none)" if nmar == 0 else ("PASS (>= 3 March windows)" if nmar >= 3 else f"REFUSE: {nmar} March window(s) changed; a board read would reveal them. No file is built.")
    if int(eff.sum()) == 0 or 0 < nmar < 3:
        res["file"] = None; res["made"] = utc(); (D / "delta.json").write_text(json.dumps(res, indent=1)); print("A3_DELTA", "NO FILE:", "no ongoing cell changes" if int(eff.sum()) == 0 else res["window_rule"], flush=True)
        return 0
    seed = iso[v != r.to_numpy(float)]; seed.to_csv(D / "seed.csv", index=False)
    tmp = Path(tmpdir) if tmpdir else Path(tempfile.gettempdir()) / "tfb_a3_tmp"; tmp.mkdir(parents=True, exist_ok=True)   # outside the project: the full files are never kept
    assert free_gib() >= MIN_FREE_GIB + .5, f"C: free {free_gib():.2f} GiB: no room for the streamed check"
    full1, full2 = tmp / f"{name}.build.csv", tmp / f"{name}.csv"; dl = D / f"{name}.delta.csv.gz"; t0 = time.time()
    try:
        run_tool([ROOT / "tools/probe.py", "expand", PARENT, D / "seed.csv", full1], logf)
        p = run_tool([ROOT / "tools/probe.py", "make", PARENT, full1, dl], logf); meta = json.loads((D / f"{name}.json").read_text())
        full1.unlink()
        p = run_tool([ROOT / "tools/probe.py", "expand", PARENT, dl, full2], logf); res["expand"] = p.stdout.strip().splitlines()[-1]; assert "(MATCH)" in res["expand"], res["expand"]
        res["stream_check_rows_differ"] = stream_check(PARENT, full2, seed.submission_id.to_numpy()); assert res["stream_check_rows_differ"] == len(seed) == meta["rows_changed"]
        g = run_tool([ROOT / "tools/window_guard.py", full2, "--against", PARENT, "--no-record"], logf, check=False)
        res["window_guard"] = dict(rc=g.returncode, output=g.stdout.strip().splitlines(), verdict="PASS" if g.returncode == 0 else "REFUSE")
        (D / "window_guard.txt").write_text(g.stdout + g.stderr)
    finally:
        for f in (full1, full2):                                    # my own temporary full files (about 411 MB each); never kept
            if f.exists():
                f.unlink()
    res.update(file=(str(dl.relative_to(ROOT)) if outdir is None else str(dl)).replace("\\", "/"), delta_sha256=meta["delta_sha256"], full_sha256=meta["full_sha256"], parent_sha256=meta["parent_sha256"], rows_changed=meta["rows_changed"],
               columns=meta["columns"], rows_by_task=meta["rows_by_task"], minutes=round((time.time() - t0) / 60, 1), made=utc(), peak_ram_gib=peak_gib())
    (D / "delta.json").write_text(json.dumps(res, indent=1))
    print("A3_DELTA OK", json.dumps(dict(file=res["file"], rows=res["rows_changed"], march_windows=nmar, april_windows=napr, guard=res["window_guard"]["verdict"], full_sha256=res["full_sha256"][:8])), flush=True)
    return 0


# ---------------------------------------------------------------- result and driver
def result(status, note=""):
    rd = lambda f: json.loads(f.read_text()) if f.exists() else None
    sp0, sp1, dl, nh = rd(OUT / "ship_P0/ship.json"), rd(OUT / "ship_P1/ship.json"), rd(OUT / "delta/delta.json"), rd(OUT / "noharm.json")
    jobs = {d.name: rd(d / "DONE.json") for d in sorted((OUT / "off").glob("*")) if d.is_dir()} if (OUT / "off").exists() else {}
    worst = lambda sp, key: max(v[key][0] for v in sp["parity"].values()) if sp else None
    res = dict(status=status, note=note, made=utc(), method="pool rule P1: observed share >= .30 and eligible share >= .70 instead of history coverage >= .5; 10 members, both months, mh5 on warm dates, XSD on cold dates, .75/.25 mix, shipped blend and decode; sccnn pool-row flag gate of t-1 = the P1 gate (worker 2's form)",
               parent=str(PARENT.relative_to(ROOT)).replace("\\", "/"),
               parity=dict(code_path_cpu_bitwise={s: rd(OUT / f"codeparity_{s}.json") for s in SPLITS}, reference=rd(OUT / "ref_check.json"), smoke=rd(OUT / "smoke.json"),
                           gpu_P0=None if not sp0 else dict(control=sp0["control"], decoded_cells_differ_from_parent=sp0["changed_cells"], windows=sp0["changed_windows"], ok=sp0["ok"], bad=sp0["bad"],
                                                            max_abs_dp_unadapted=worst(sp0, "un_vs_stored_off"), max_abs_dp_adapted_warm=worst(sp0, "adapted_vs_stored_on_warm"),
                                                            max_abs_dp_adapted_cold=worst(sp0, "adapted_vs_stored_on_cold"), per_file=sp0["parity"], cold_members_vs_stored_xsd=sp0["cold_members_vs_stored_xsd"])),
               noharm_off250=nh, p1_official=None if not sp1 else dict(rows_vs_parent=dict(cells=sp1["changed_cells"], windows=sp1["changed_windows"], by_split=sp1["by_split"]), ok=sp1["ok"], bad=sp1["bad"]),
               delta=dl, jobs=jobs, gpu_minutes=round(gpu_used(), 1), gpu_hours=round(gpu_used() / 60, 2), disk_free_gib=round(free_gib(), 2),
               files_mb=round(sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 2 ** 20, 1),
               how_to_read="status DONE = every stage ran. Upload only if: parity.gpu_P0.control is EXACT or NEAR, noharm_off250.kill is false, delta.window_rule starts with PASS and delta.window_guard.verdict is PASS. The candidate is delta.file on the parent (tools/probe.py expand PARENT DELTA OUT.csv; sha in delta.full_sha256). Nothing was uploaded.")
    (HERE / "A3_RESULT.json").write_text(json.dumps(res, indent=1, default=float)); print(f"[{utc()}] A3_RESULT {status} {note}", flush=True)


def drive():
    """the chain. Every GPU job goes through gpu_lock.py and gives the lock back; finished pieces are skipped."""
    py = str(ROOT / ".venv_tfb/Scripts/python.exe"); me = str(Path(__file__).resolve()); lock = str(ROOT / R / "gpu_lock.py"); LOGS.mkdir(exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONPATH=str(ROOT / "src"))
    say = lambda s: print(f"[{utc()}] {s}", flush=True)

    def stop(why):
        say(f"STOP: {why}"); result("STOPPED", why); sys.exit(1)

    def cpu(name, *args, done=None, ok_rc=(0,)):
        if done is not None and done.exists():
            say(f"skip {name} (done)"); return 0
        try:
            guard(wait_min=30)
        except AssertionError as e:
            stop(str(e))
        say(f"start {name}")
        with open(LOGS / f"{name}.log", "w") as f:
            rc = subprocess.call([py, "-B", me, *args], cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
        say(f"done {name} rc {rc}")
        if rc not in ok_rc:
            stop(f"{name} failed rc {rc} (see exp/L_lab/a3/logs/{name}.log)")
        return rc

    def gpu(name, args, done, count):
        tries = 0
        while not done.exists():
            if gpu_used() > GPU_HARD_MIN:
                stop(f"GPU minutes used {gpu_used():.0f} > {GPU_HARD_MIN:.0f} before {name}")
            try:
                guard(wait_min=30)
            except AssertionError as e:
                stop(str(e))
            before = count(); say(f"queue {name} behind the GPU lock (GPU minutes used {gpu_used():.1f}, C: free {free_gib():.2f} GiB)")
            with open(LOGS / f"{name}.log", "a") as f:
                rc = subprocess.call([py, "-B", lock, "a3_" + name, "--", py, "-B", me, *args], cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
            say(f"back {name} rc {rc} pieces {count()}")
            time.sleep(35)                                           # let a job that waits behind the lock (it polls every 15 s) take its turn
            if rc != 0 or (count() == before and not done.exists()):
                tries += 1
                if tries > 2:
                    stop(f"{name} failed 3 times (see exp/L_lab/a3/logs/{name}.log)")
                time.sleep(20)
    say(f"A3 chain start | C: free {free_gib():.2f} GiB | GPU minutes used so far {gpu_used():.1f}")
    for s in SPLITS:
        cpu(f"codeparity_{s}", "codeparity", s, done=OUT / f"codeparity_{s}.json")
        if not json.loads((OUT / f"codeparity_{s}.json").read_text())["ok"]:
            stop(f"code-path parity failed on {s}")
    cpu("ref", "ref", done=OUT / "ref_check.json")
    if not json.loads((OUT / "ref_check.json").read_text())["ok"]:
        stop("reference check failed (parent ongoing rows or the chain on stored files)")

    def off_jobs(arm):
        for s in SPLITS:
            for kind in ("ctl", "sc"):
                d = OUT / "off" / f"{kind}_{arm}_{s}"
                gpu(f"off_{kind}_{arm}_{s}", ["off", kind, arm, s], d / "DONE.json", lambda d=d: len(list(d.glob("2031-*.parquet"))) if d.exists() else 0)
    off_jobs("P0")                                                  # 1. PARITY FIRST
    cpu("ship_P0", "ship", "P0", done=None, ok_rc=(0, 3))
    sp0 = json.loads((OUT / "ship_P0/ship.json").read_text())
    if not sp0["ok"]:
        stop("P0 parity failed: " + "; ".join(sp0["bad"])[:300])
    say(f"PARITY {sp0['control']}: P0 path differs from the parent's ongoing rows in {sp0['changed_cells']} of 87000 cells")
    for m in JF:                                                    # 3. OFF250 no-harm before the P1 fits (a kill saves their GPU time)
        gpu(f"jf_{m}", ["jf", m], OUT / "jf" / f"{m}.npz", lambda m=m: int((OUT / "jf" / f"{m}.npz").exists()))
    cpu("noharm", "noharm", done=OUT / "noharm.json")
    nh = json.loads((OUT / "noharm.json").read_text())
    say(f"OFF250 no-harm: raw {nh['raw']['mix']:+.5f} [{nh['raw']['lo']:+.5f},{nh['raw']['hi']:+.5f}], kept {nh['denoised']['kept']['mix']:+.5f} -> {'KILL' if nh['kill'] else 'passed'}")
    if nh["kill"]:
        result("KILLED", "OFF250 no-harm read below -.002 IoU on raw and on denoised truth: the card's kill line. No P1 file was built."); say("A3_CHAIN_DONE killed by the no-harm read"); return 0
    off_jobs("P1")                                                  # 2. P1 on the official windows
    cpu("ship_P1", "ship", "P1", done=None, ok_rc=(0, 3))
    if not json.loads((OUT / "ship_P1/ship.json").read_text())["ok"]:
        stop("ship P1 failed its checks")
    cpu("delta", "delta", done=None)                                # 4. delta, streamed check, window guard
    result("DONE"); say(f"A3_CHAIN_DONE ok | GPU minutes {gpu_used():.1f} | C: free {free_gib():.2f} GiB")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]; opt = dict(x[2:].split("=", 1) for x in sys.argv[2:] if x.startswith("--")); pos = [x for x in sys.argv[2:] if not x.startswith("--")]
    rc = 0
    if cmd == "off":
        rc = off(pos[0], pos[1], pos[2], opt.get("cpu", "0") == "1", set(opt["dates"].split(",")) if opt.get("dates") else None, int(opt.get("members", 5)), float(opt.get("budget", 22)), opt.get("tag", ""))
    elif cmd == "jf":
        rc = jf(pos[0], tuple(opt.get("arms", "P1").split(",")), opt.get("cpu", "0") == "1", int(opt.get("maxdates", 99)), opt.get("tag", ""))
    elif cmd == "codeparity":
        rc = codeparity(pos[0])
    elif cmd == "ref":
        rc = ref_check()
    elif cmd == "ship":
        rc = ship(pos[0], opt.get("tag", ""))
    elif cmd == "noharm":
        rc = noharm(opt.get("tag", ""))
    elif cmd == "delta":
        rc = delta(opt.get("rows"), opt.get("name", NAME), opt.get("tmp"), opt.get("out"), opt.get("tag", ""))
    elif cmd == "result":
        result(pos[0] if pos else "MANUAL")
    elif cmd == "drive":
        rc = drive()
    else:
        raise SystemExit(__doc__)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(rc or 0)
