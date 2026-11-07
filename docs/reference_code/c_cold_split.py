"""C semisup helper 2026-10-09 (Exp 1 of exp/C_semisup/SEMISUP.md): cold-start per-date adaptation pools on I-Q-SPLIT.

Official ongoing windows sit on days 1-5; their same-day-type adaptation pool (qcnn_predict_adapt.finetune: earlier dates of
the split with the same weekend flag) is often empty (base returned) or 1-2 days. On the I-Q-SPLIT window set of
c_bb_tune_split (days <= maxday, selector persistence <= .9, grid + type A/C + replica picks), for the COLD dates only
(same-type pool <= --cold days; other dates are unchanged by every arm), per member and date d (seed = date index k):
  un     base member
  serve  qcnn_predict_adapt.finetune as shipped (cap None, 3 ep, lr 3e-4, same day type)
  xall   same recipe, pool = ALL earlier dates of the split (other day type included)
  sd     serve, then same-day pre-T stage: hourly checkpoint H = 12*floor(t/12) fine-tuned (same recipe) on pseudo windows
         (d, t') of day d with t' + 6 <= H (labels end at or before H <= t); a window at t uses checkpoint H(t)
  xsd    xall, then the same same-day stage
  mar    (private only, hedge B, D-0062) pool = March days of the same day type + April earlier same-type days
Online rule: every pool uses dates < d, or day d with label timestamps <= H <= t; poison check per split (one cold date):
dates > d become noise (and day d after H for the same-day checkpoints, windows recomputed from the poisoned cube); every
refit model must equal the original bitwise.
export: python c_cold_split.py export <run_dir> <out.parquet> [--splits=validation,private] [--maxday=12] [--cold=2]
score:  python c_cold_split.py score --sc a.parquet,b.parquet --ctl c.parquet,d.parquet --name N [--mix .75]
        shipped form (A from sc, M/C from ctl, decoder v2, a=.75 mix with un for every adapted arm), contrasts vs mserve by
        split x pool bucket x type, official type mix, and official-date-weighted (window_index day-of-month counts)."""
import sys, json, time, copy, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
OUT = ROOT / "artifacts/claude_ref/c_cold"
ARMS = ("un", "serve", "xall", "sd", "xsd", "mar")


def ft(base, groups, seed, epochs=3, lr=3e-4):
    """qcnn_predict_adapt.finetune (cap None) generalised to explicit (v, windows) groups, same rng use and batch order."""
    import qcnn_predict_adapt as qpa, qcnn_adapt
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    groups = [(v, W) for v, W in groups]
    if sum(len(W) for _, W in groups) == 0:
        return base
    model = copy.deepcopy(base); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        order = [(g, i) for g, (v, W) in enumerate(groups) for i in range(0, len(W), qpa.BS)]
        rng.shuffle(order)
        for g, i in order:
            v, W = groups[g]; B = [W[j] for j in rng.permutation(len(W))[:qpa.BS]]
            x = qcnn_adapt.make_x(v["c3"], v["rc"], v["stat"], B, v["drop"])
            obs = np.stack([v["c3"][d, t + 1:t + 7, 3] > .5 for d, t in B]); q = np.stack([v["c3"][d, t + 1:t + 7, 4] > .5 for d, t in B])
            y = torch.tensor(q & obs, dtype=torch.float32); m = torch.tensor(obs, dtype=torch.float32)
            loss = (F.binary_cross_entropy_with_logits(model(x), y, reduction="none") * m).sum() / (m.sum() + 1)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def same_type_days(dates, k):
    import qcnn_predict_adapt as qpa
    return sum(1 for j in range(k) if qpa.weekend(dates[j]) == qpa.weekend(dates[k]))


def pool_groups(data, before, kind=None):
    """Per panel (PANELS order) windows of dates < before; kind None = any day type, else weekend flag must equal kind."""
    import qcnn_predict_adapt as qpa
    from qcnn_train import PANELS
    return [(data[p], [(d, t) for d, t in data[p]["W"] if data[p]["dates"][d] < before
                       and (kind is None or qpa.weekend(data[p]["dates"][d]) == kind)]) for p in PANELS]


def sameday_groups(data, d_str, H, recompute=False):
    from qcnn_shift import windows
    from qcnn_train import PANELS
    out = []
    for p in PANELS:
        v = data[p]; di = [j for j, x in enumerate(v["dates"]) if x == d_str]
        if not di:
            out.append((v, [])); continue
        W = windows(v["c3"][di[0]:di[0] + 1], step=1) if recompute else [(0, t) for d, t in v["W"] if d == di[0]]
        out.append((v, [(di[0], t) for _, t in W if t + 6 <= H]))
    return out


def export(a):
    import qcnn_predict_adapt as qpa, qcnn_adapt, ong2_sccnn as S
    import c_bb_tune_split as CT
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "2")))
    dev = "cpu"
    base, cfg = CT.load_member(a.run, dev); drop = cfg["drop"]; sc = bool(cfg.get("sccnn", False))
    assert cfg.get("fit_end", "2031-02-28") <= "2031-02-28"
    if sc:
        S.enable()
    qpa.EPOCHS, qpa.LR = 3, 3e-4
    parts, checks, t0 = [], [], time.time()
    mar = None
    for split in a.splits.split(","):
        data = qpa.load_split(split, False); CT.select(data, split, a.maxday)
        for v in data.values():
            v["drop"] = drop
        if split == "private" and a.march:
            mar = qpa.load_split("validation", False)
            for v in mar.values():
                v["drop"] = drop; v.pop("hist", None); v.pop("sample", None)
        dates = sorted({str(x) for v in data.values() for x in v["dates"]})
        cold = [(k, d) for k, d in enumerate(dates) if int(d[8:10]) <= a.maxday and same_type_days(dates, k) <= a.cold]
        print(json.dumps(dict(split=split, cold=[(d, same_type_days(dates, k)) for k, d in cold])), flush=True)
        poison_k = cold[len(cold) // 2][0]

        def rs(seed):
            if sc:
                S.MODE = "train"; S._rng = np.random.default_rng(seed)

        def fit_all(k, d_str, Hs, recompute=False):
            kind = qpa.weekend(d_str); ms = {"un": base}
            rs(10_000 + k); ms["serve"] = ft(base, pool_groups(data, d_str, kind), k)
            rs(10_000 + k); ms["xall"] = ft(base, pool_groups(data, d_str, None), k)
            if split == "private" and mar is not None:
                rs(10_000 + k); ms["mar"] = ft(base, pool_groups(mar, "9999", kind) + pool_groups(data, d_str, kind), k)
            for arm, m0 in (("sd", ms["serve"]), ("xsd", ms["xall"])):
                for H in Hs:
                    rs(20_000 + 100 * k + H // 12); ms[(arm, H)] = ft(m0, sameday_groups(data, d_str, H, recompute), 1000 * k + H)
            return ms

        for k, d_str in cold:
            Bs = {p: [(d, t) for d, t, *_ in v["Wx"] if v["dates"][d] == d_str] for p, v in data.items()}
            Hs = sorted({12 * (t // 12) for B in Bs.values() for _, t in B})
            ms = fit_all(k, d_str, Hs)
            if same_type_days(dates, k) > 0 and not any(c["split"] == split and c["kind"] == "serve_parity" for c in checks):   # my ft == shipped finetune
                rs(10_000 + k); ref = qpa.finetune(base, data, d_str, drop, dev, seed=k, cap=None, daytype=True)
                dmax = max(float((x - y).abs().max()) for x, y in zip(ref.state_dict().values(), ms["serve"].state_dict().values()))
                checks.append(dict(split=split, date=d_str, kind="serve_parity", max_diff=dmax)); print(json.dumps(checks[-1]), flush=True)
                assert dmax == 0., "ft differs from qcnn_predict_adapt.finetune"
            if a.poison and k == poison_k:                     # future poison: later dates, and day d after each H
                saved = {p: v["c3"].copy() for p, v in data.items()}; g = np.random.default_rng(1)
                for p, v in data.items():
                    di = [j for j, x in enumerate(v["dates"]) if str(x) > d_str]
                    v["c3"][di] = g.random(v["c3"][di].shape).astype(v["c3"].dtype)
                worst = 0.
                for H in Hs:
                    for p, v in data.items():
                        di = [j for j, x in enumerate(v["dates"]) if x == d_str]
                        if di:
                            v["c3"][di[0], H + 1:] = g.random(v["c3"][di[0], H + 1:].shape).astype(v["c3"].dtype)
                    mz = fit_all(k, d_str, [H], recompute=True)
                    for key in [kk for kk in mz if kk != "un" and (not isinstance(kk, tuple) or kk[1] == H)]:
                        worst = max(worst, max(float((x - y).abs().max()) for x, y in zip(mz[key].state_dict().values(), ms[key].state_dict().values())))
                    for p, v in data.items():
                        v["c3"][:] = saved[p]
                        di = [j for j, x in enumerate(v["dates"]) if str(x) > d_str]
                        v["c3"][di] = np.random.default_rng(1).random(v["c3"][di].shape).astype(v["c3"].dtype)
                for p, v in data.items():
                    v["c3"] = saved[p]
                checks.append(dict(split=split, date=d_str, kind="poison", hours=len(Hs), max_diff=worst)); print(json.dumps(checks[-1]), flush=True)
                assert worst == 0., "future poison changed a fitted model"
            for p, v in data.items():
                B = [w for w in v["Wx"] if v["dates"][w[0]] == d_str]
                if not B:
                    continue
                L = len(v["links"]); res = {}
                for arm in ARMS:
                    if arm == "mar" and "mar" not in ms:
                        res[arm] = res["serve"]; continue
                    S.MODE = "truth"; ps = []
                    for d, t, *_ in B:
                        m = ms[(arm, 12 * (t // 12))] if arm in ("sd", "xsd") else ms[arm]
                        x = qcnn_adapt.make_x(v["c3"], v["rc"], v["stat"], [(d, t)], drop)
                        with torch.no_grad():
                            ps.append(torch.sigmoid(m(x)).numpy()[0])
                    res[arm] = ps
                for i, (d, t, ty, gr, rp) in enumerate(B):
                    obs, qq = v["c3"][d, :, 3] > .5, v["c3"][d, :, 4] > .5
                    row = {"split": split, "panel": p, "date": d_str, "t": t, "typ": ty, "grid": gr, "rep": rp,
                           "pool": same_type_days(dates, k), "h": np.repeat(np.arange(1, 7), L).astype(np.int8),
                           "link_id": np.tile(v["links"], 6), "label": (qq[t + 1:t + 7] & obs[t + 1:t + 7]).reshape(-1),
                           "valid": obs[t + 1:t + 7].reshape(-1)}
                    for arm in ARMS:
                        row[f"p_{arm}"] = res[arm][i].reshape(-1).astype(np.float32)
                    parts.append(pd.DataFrame(row))
            print(json.dumps(dict(split=split, date=d_str, hours=len(Hs), min=round((time.time() - t0) / 60, 1))), flush=True)
        del data
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(out)
    Path(str(out) + ".json").write_text(json.dumps(dict(run=a.run, checks=checks, minutes=round((time.time() - t0) / 60, 1)), indent=1))
    print(json.dumps(dict(saved=str(out), minutes=round((time.time() - t0) / 60, 1), checks=checks)), flush=True)


def official(a):
    """official mode (C 2026-10-10, ship arm XSD, ledger 0778): per-window probabilities of the OFFICIAL ongoing windows on
    cold official dates (same-type split-day pool <= --cold), for ONE member (ctl: RUN; sccnn: all sc RUNS in build order,
    --member J). Same production inputs as the shipped serve (qcnn_predict_adapt.load_split / official_x, sccnn flags by
    official type, TFB_ASM2_DET=1, CPU). Arms per window: un (base), serve (production qcnn_predict_adapt.finetune, seed =
    official-date index, sccnn: S._rng = default_rng(1000 + k) and members 0..J fitted in order, exactly as the shipped
    sccnn main; with J = last also p_serve_mean = the shipped ensemble formula over members 0..J), xall and xsd (export's recipe and seeds: split-date index k; xsd checkpoint H = 12*floor(t/12)).
    Checks: ft == qcnn_predict_adapt.finetune (one date per split, bitwise); poison (one cold date per split): dates > d and
    the whole day d poisoned -> serve and xall refits identical; then per H, dates > d and day d after H poisoned (same-day
    windows recomputed from the poisoned cube) -> every xsd checkpoint identical. Out: parquet (one row per window x
    horizon x link) + .json (checks). The serve/un probabilities are compared to the shipped on/off files by c_cold_ship.py."""
    import os
    import qcnn_predict_adapt as qpa, ong2_sccnn as S
    from qcnn_train import Net, CTX
    assert os.environ.get("TFB_ASM2_DET") == "1" and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "run as the shipped serve: DET, CPU"
    qpa.configure_determinism(); dev = "cpu"
    assert (qpa.EPOCHS, qpa.LR, qpa.BS, qpa.POOL_MAX_PERS) == (3, 3e-4, 32, None)
    # 2026-10-11 A3 (ledger 0800, exp/C_adapt3/ADAPT3.md "XSD strength"): the assert above is the SERVE arm's recipe
    # (qpa.finetune, parity/info only) and stays. The xall stage and the same-day stage take --xsd-recipe
    # E1,LR1[,E2,LR2] (stage 1[, stage 2]; default 3,3e-4 = the C36 form, same bytes as before this change).
    rx = [float(z) for z in a.xsd_recipe.split(",")]; rx = rx + rx if len(rx) == 2 else rx
    assert len(rx) == 4 and rx[0] == int(rx[0]) and rx[2] == int(rx[2]), "--xsd-recipe E1,LR1[,E2,LR2]"
    e1, l1, e2, l2 = int(rx[0]), rx[1], int(rx[2]), rx[3]
    runs = a.run.split(","); J = a.member; cfg = json.loads((Path(runs[J]) / "config.json").read_text())
    sc = bool(cfg.get("sccnn", False)); drop = cfg["drop"]
    assert "prior" in drop and not cfg.get("ramps", False) and (sc or len(runs) == 1)

    def load(r):
        m = Net(cin=CTX * 5 + 16).to(dev); m.load_state_dict(torch.load(Path(r) / "model.pt", map_location=dev)); m.eval()
        return m
    bases = [load(r) for r in runs[:J + 1]]; base = bases[J]
    if sc:
        S.enable()
        import ong2_sccnn_official as O
        types = pd.read_csv(ROOT / "artifacts/claude_ref/ong2/official_types.csv").set_index("window_id").typ.to_dict()
    only = set(a.dates.split(",")) if a.dates else None
    parts, checks, t0 = [], [], time.time()

    def rs(seed):
        if sc:
            S.MODE = "train"; S._rng = np.random.default_rng(seed)

    def mdiff(m1, m2):
        return max(float((x - y).abs().max()) for x, y in zip(m1.state_dict().values(), m2.state_dict().values()))
    for split in a.splits.split(","):
        data = qpa.load_split(split, False)
        for v in data.values():
            v["drop"] = drop
        alld = sorted({str(x) for v in data.values() for x in v["dates"]})
        offd = sorted({str(pd.Timestamp(o).date()) for v in data.values() for o in v["wi"].forecast_origin})
        cold = [(ko, d, alld.index(d)) for ko, d in enumerate(offd) if same_type_days(alld, alld.index(d)) <= a.cold]
        print(json.dumps(dict(split=split, member=runs[J], cold=[(d, ko, k, same_type_days(alld, k)) for ko, d, k in cold])), flush=True)
        poison_d = cold[len(cold) // 2][1]
        if only is not None:
            cold = [c for c in cold if c[1] in only]

        def fit(ko, d_str, k, Hs, recompute=False, xall=None):
            kind = qpa.weekend(d_str); ms = {"un": base}
            if xall is None:
                rs(1000 + ko)                                              # shipped serve: members 0..J in build order
                ms["serve_all"] = [qpa.finetune(b, data, d_str, drop, dev, seed=ko) for b in bases]; ms["serve"] = ms["serve_all"][J]
                rs(10_000 + k); ms["xall"] = ft(base, pool_groups(data, d_str, None), k, e1, l1)
            else:
                ms["xall"] = xall
            for H in Hs:
                rs(20_000 + 100 * k + H // 12); ms[("xsd", H)] = ft(ms["xall"], sameday_groups(data, d_str, H, recompute), 1000 * k + H, e2, l2)
            return ms

        def poison(d_str, H, g):
            for p, v in data.items():
                di = [j for j, x in enumerate(v["dates"]) if str(x) > d_str]
                v["c3"][di] = g.random(v["c3"][di].shape).astype(v["c3"].dtype)
                di = [j for j, x in enumerate(v["dates"]) if str(x) == d_str]
                if di:
                    v["c3"][di[0], H + 1:] = g.random(v["c3"][di[0], H + 1:].shape).astype(v["c3"].dtype)

        for ko, d_str, k in cold:
            wis = {p: v["wi"][[str(pd.Timestamp(o).date()) == d_str for o in v["wi"].forecast_origin]] for p, v in data.items()}
            Hs = sorted({12 * ((pd.Timestamp(o).hour * 12 + pd.Timestamp(o).minute // 5) // 12) for w in wis.values() for o in w.forecast_origin})
            ms = fit(ko, d_str, k, Hs)
            kind = qpa.weekend(d_str)
            if same_type_days(alld, k) > 0 and not any(c["split"] == split and c["kind"] == "ft_parity" for c in checks):
                rs(77); ref = qpa.finetune(base, data, d_str, drop, dev, seed=ko)
                rs(77); mine = ft(base, pool_groups(data, d_str, kind), ko)
                checks.append(dict(split=split, date=d_str, kind="ft_parity", max_diff=mdiff(ref, mine))); print(json.dumps(checks[-1]), flush=True)
                assert checks[-1]["max_diff"] == 0., "ft differs from qcnn_predict_adapt.finetune"
            if a.poison and d_str == poison_d:
                saved = {p: v["c3"].copy() for p, v in data.items()}
                poison(d_str, -1, np.random.default_rng(1))                  # later dates and ALL of day d
                mz = fit(ko, d_str, k, [])
                w0 = max(mdiff(mz["serve"], ms["serve"]), mdiff(mz["xall"], ms["xall"]))
                worst = 0.
                for H in Hs:
                    for p, v in data.items():
                        v["c3"][:] = saved[p]
                    poison(d_str, H, np.random.default_rng(2 + H))
                    mz = fit(ko, d_str, k, [H], recompute=True, xall=ms["xall"])
                    worst = max(worst, mdiff(mz[("xsd", H)], ms[("xsd", H)]))
                for p, v in data.items():
                    v["c3"] = saved[p]
                checks.append(dict(split=split, date=d_str, kind="poison", hours=len(Hs), serve_xall=w0, xsd=worst))
                print(json.dumps(checks[-1]), flush=True)
                assert w0 == 0. and worst == 0., "future poison changed a fitted model"
            n = 0
            for p, v in data.items():
                for w, t, x in qpa.official_x(v["c3"], v["rc"], v["stat"], v["st"], v["links"], wis[p], v["hist"], drop):
                    if sc:
                        x = S.fill(x, [O.FL[types[w.window_id]]])
                    H = 12 * (t // 12); L = len(v["links"]); T = pd.Timestamp(w.forecast_origin)
                    row = {"split": split, "panel": p, "window_id": w.window_id, "date": d_str, "t": t, "H": H, "ko": ko, "k": k,
                           "pool": same_type_days(alld, k),
                           "timestamp": np.repeat([(T + pd.Timedelta(minutes=5 * (i + 1))).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(6)], L),
                           "link_id": np.tile(v["links"], 6)}
                    for arm, m in (("un", ms["un"]), ("serve", ms["serve"]), ("xall", ms["xall"]), ("xsd", ms[("xsd", H)])):
                        with torch.no_grad():
                            row[f"p_{arm}"] = torch.sigmoid(m(x.to(dev)))[0].cpu().numpy().reshape(-1)
                    if sc and J == len(runs) - 1:                          # shipped sccnn ensemble formula (parity)
                        with torch.no_grad():
                            row["p_serve_mean"] = torch.stack([torch.sigmoid(m(x.to(dev)))[0] for m in ms["serve_all"]]).mean(0).cpu().numpy().reshape(-1)
                    parts.append(pd.DataFrame(row)); n += 1
            print(json.dumps(dict(split=split, date=d_str, ko=ko, k=k, pool=same_type_days(alld, k), hours=len(Hs), windows=n,
                                  min=round((time.time() - t0) / 60, 1))), flush=True)
        del data
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(out)
    Path(str(out) + ".json").write_text(json.dumps(dict(runs=runs, member=J, sccnn=sc, xsd_recipe=[e1, l1, e2, l2], checks=checks, torch=str(torch.__version__),
                                                        threads=torch.get_num_threads(), minutes=round((time.time() - t0) / 60, 1)), indent=1))
    print("COLD_OFFICIAL_DONE", json.dumps(dict(saved=str(out), minutes=round((time.time() - t0) / 60, 1), checks=checks)), flush=True)


def official_dom():
    out = {}
    for f in (ROOT / "data/raw/kaggle_public/task2").glob("*/*/window_index.csv"):
        sp = f.parent.name
        if sp in ("validation", "private"):
            w = pd.read_csv(f); w = w[w.condition == "queue_ongoing"]
            for o in w.forecast_origin:
                k = (sp, int(str(o)[8:10])); out[k] = out.get(k, 0) + 1
    return out


def score(a):
    from tfb.data import load_cache
    from qcnn_data import cube
    from gapres_selcond_rule import iou
    from ong3_eval import chain
    from ong3_types_diag import last_state
    al = float(a.mix); K = ["split", "panel", "date", "t"]
    R = [x for x in ARMS if x != "un"]

    def ens(paths):
        P = [pd.read_parquet(x) for x in paths.split(",") if x]
        df = P[0].copy()
        for q in P[1:]:
            assert (q[K + ["h", "link_id"]].to_numpy() == df[K + ["h", "link_id"]].to_numpy()).all()
        for c in [f"p_{x}" for x in ARMS]:
            df[c] = np.mean([q[c].to_numpy(np.float64) for q in P], 0)
        for r in R:
            df[f"p_m{r}"] = al * df[f"p_{r}"] + (1 - al) * df["p_un"]
        return df
    KINDS = ["un"] + R + [f"m{r}" for r in R]
    Sx, Cx = ens(a.sc), ens(a.ctl); rows = []
    for (split, pn), gs in Sx.groupby(["split", "panel"], sort=False):
        cc = load_cache(pn, split); st = cc.stations.sort_values("idx"); links = st.link_id.astype(str).tolist()
        c3 = cube(*[np.asarray(cc[x], np.float32) for x in ("speed", "flow", "occ")], st); OBS, Q = c3[:, :, 3] > .5, c3[:, :, 4] > .5
        gc = Cx[(Cx.split == split) & (Cx.panel == pn)]
        for ((date, t), g), (kc, h) in zip(gs.groupby(["date", "t"], sort=False), gc.groupby(["date", "t"], sort=False)):
            assert kc == (date, t)
            typ = g.typ.iloc[0]
            if typ not in ("A", "M", "C"):
                continue
            src = g if typ == "A" else h
            L = len(g) // 6; d = cc.day_index(str(date)); idx = np.array([links.index(l) for l in g.link_id.to_numpy()[:L]])
            pm = np.broadcast_to(last_state(OBS[d], Q[d], t - 12, t)[idx], (6, L)).ravel()
            pmT = np.broadcast_to(last_state(OBS[d], Q[d], t - 12, t + 1)[idx], (6, L)).ravel()
            V = g.valid.to_numpy(); Y = g.label.to_numpy() & V
            r = dict(split=split, panel=pn, date=str(date), t=int(t), typ=typ, pool=int(g["pool"].iloc[0]))
            for kn in KINDS:
                r[kn] = iou(chain(src[f"p_{kn}"].to_numpy(np.float64), pm, pmT, typ == "A", "v2") & V, Y)
            rows.append(r)
    df = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True); df.to_parquet(OUT / f"score_{a.name}.parquet")
    df["dom"] = df.date.str[8:10].astype(int)
    W = {"A": .55, "M": .25, "C": .20}; off = official_dom(); res = {}; rng = np.random.default_rng(0)
    arms = {"X": lambda g: np.where(g["pool"] == 0, g["mxall"], g["mserve"]), "X2": lambda g: g["mxall"],
            "SD": lambda g: g["msd"], "XSD0": lambda g: np.where(g["pool"] == 0, g["mxsd"], g["msd"]), "XSD": lambda g: g["mxsd"],
            "MAR": lambda g: g["mmar"], "un": lambda g: g["un"], "pure_serve": lambda g: g["serve"]}
    for split, gs in df.groupby("split"):
        gs = gs.copy()
        doms = sorted(k for (s, k) in off if s == split and k <= 12); wt = {k: off[(split, k)] for k in doms}
        for name, f in arms.items():
            if name == "MAR" and split != "private":
                continue
            gs["dlt"] = f(gs) - gs["mserve"]
            line = {}
            for b, gb in [("pool0", gs[gs["pool"] == 0]), ("pool1-2", gs[gs["pool"] > 0]), ("all", gs)]:
                line[b] = {t: dict(n=int((gb.typ == t).sum()), d=round(float(gb.dlt[gb.typ == t].mean()), 5)) for t in W}
                line[b]["MIX"] = round(sum(W[t] * (line[b][t]["d"] if line[b][t]["n"] else 0.) for t in W), 5)
            # official-date-weighted: delta per day of month (type-pooled; days not cold contribute 0), window bootstrap within day
            per = {k: gs.dlt[gs.dom == k].to_numpy() for k in doms}
            est = sum(wt[k] * (per[k].mean() if len(per[k]) else 0.) for k in doms) / sum(wt.values())
            bs = [sum(wt[k] * (rng.choice(per[k], len(per[k])).mean() if len(per[k]) else 0.) for k in doms) / sum(wt.values()) for _ in range(2000)]
            line["official_weighted"] = dict(d=round(float(est), 5), lo=round(float(np.quantile(bs, .025)), 5), hi=round(float(np.quantile(bs, .975)), 5),
                                             cold_days={k: round(float(per[k].mean()), 4) for k in doms if len(per[k])},
                                             weight_cold=int(sum(wt[k] for k in doms if len(per[k]))), weight_all=int(sum(wt.values())))
            res[f"{split}/{name}-mserve"] = line
            o = line["official_weighted"]
            print(f"{split[:3]} {name:>10}-mserve OFFW {o['d']:+.4f} [{o['lo']:+.4f},{o['hi']:+.4f}] | pool0 MIX {line['pool0']['MIX']:+.4f} | pool1-2 MIX {line['pool1-2']['MIX']:+.4f} | days {o['cold_days']}", flush=True)
    (OUT / f"score_{a.name}.json").write_text(json.dumps(res, indent=1)); print("saved", OUT / f"score_{a.name}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["export", "score", "official"]); ap.add_argument("run", nargs="?"); ap.add_argument("out", nargs="?")
    ap.add_argument("--member", type=int, default=0, help="official: index of the fitted member in RUN (sccnn: comma list in build order)")
    ap.add_argument("--dates", default="", help="official: only these cold dates (smoke test)")
    ap.add_argument("--xsd-recipe", default="3,3e-4", help="official: EPOCHS,LR of the xall stage and the same-day stage, or E1,LR1,E2,LR2 (C36: 3,3e-4)")
    ap.add_argument("--splits", default="validation,private"); ap.add_argument("--maxday", type=int, default=12)
    ap.add_argument("--cold", type=int, default=2); ap.add_argument("--no-march", dest="march", action="store_false")
    ap.add_argument("--no-poison", dest="poison", action="store_false", help="skip the poison refits (run once per member kind)")
    ap.add_argument("--sc", default=""); ap.add_argument("--ctl", default=""); ap.add_argument("--mix", default=".75"); ap.add_argument("--name", default="cold")
    a = ap.parse_args()
    {"export": export, "score": score, "official": official}[a.cmd](a)
    if a.cmd == "official":
        sys.stdout.flush(); __import__("os")._exit(0)   # pyarrow teardown abort (asm2_step.py note)
