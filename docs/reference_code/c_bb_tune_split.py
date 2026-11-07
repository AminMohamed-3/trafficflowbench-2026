"""C-tune 2026-10-09: adaptation recipe on I-Q-SPLIT (coordinator: RAW/F4T/DecAll are train-month scenarios; the board
months are March/April). For one serving member (fit <= 2031-02-28), on the I-Q-SPLIT pseudo windows of ong3_export.py
(validation = March, private = April, days <= 12; grid + all step-1 type A/C origins + organizer replica picks,
selector persistence <= .9), per date d (same-split earlier days of the same day type, seed = date index, sccnn flag
RNG reset per date as in serving):
  un    unadapted member
  serve qcnn_predict_adapt.finetune as shipped: cap None, 3 epochs, lr 3e-4
  g3    cap 1500, 3 epochs, lr 1e-4 (recipe G); g5 = the same run continued to 5 epochs (constant lr, the first 3
        epochs of a 5-epoch run are the 3-epoch run exactly: same rng draws), snapshot after epoch 3
--strong (coordinator 10-09: adaptation decisions use I-Q-SPLIT): serve = epoch-3 snapshot of a 5-epoch serving run
(checked equal to the shipped finetune on the poison date), s5 = its epoch 5; h3/h5 = cap None, lr 6e-4, 3/5 epochs.
Mixes are formed at scoring time (a * adapted + (1 - a) * un).
Future poison: one date per split, every day after d replaced by noise before the fine-tunes; outputs for d must not move.
export: python c_bb_tune_split.py export <run_dir> <out.parquet> [--splits=validation,private] [--maxday=12]
score:  python c_bb_tune_split.py score --sc a.parquet,b.parquet --ctl c.parquet,d.parquet --name <name> [--mix .75]
        shipped form: type A from the sc ensemble, types M (official B analog) and C from the ctl ensemble; decoder v2
        (ong3_eval.chain) on p_truth; per split x type and the official mix .55 A + .25 M + .20 C; date-bootstrap CIs."""
import sys, json, time, copy, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
OUT = ROOT / "artifacts/claude_ref/c_bb/split"
G = dict(cap=1500, lr=1e-4, epochs=5, snap=3)


def load_member(run, dev):
    cfg = json.loads((Path(run) / "config.json").read_text())
    if "spec" in cfg:
        import c_bb_tune_arch
        net = c_bb_tune_arch.make(cfg["spec"])
    else:
        from qcnn_train import Net, CTX
        net = Net(cin=CTX * 5 + 16)
    net.load_state_dict(torch.load(Path(run) / "model.pt", map_location=dev)); net.to(dev).eval()
    return net, cfg


def finetune_snap(base, data, before, drop, dev, seed, cap, lr, epochs, snap):
    """qcnn_predict_adapt.finetune (daytype pool, AdamW wd 1e-4, bs 32, same rng use) with a snapshot after `snap` epochs."""
    import qcnn_predict_adapt as qpa
    from qcnn_train import PANELS
    from qcnn_adapt import make_x
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    kind = qpa.weekend(before)
    pool = {p: [(d, t) for d, t in v["W"] if v["dates"][d] < before and qpa.weekend(v["dates"][d]) == kind] for p, v in data.items()}
    if sum(len(w) for w in pool.values()) == 0:
        return {snap: base, epochs: base}
    model = copy.deepcopy(base); model.train(); out = {}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        sub = pool
        if cap:
            n = sum(len(w) for w in pool.values()); keep = rng.random(n) < min(1., cap / n); it = iter(keep)
            sub = {p: [w for w in pool[p] if next(it)] for p in PANELS}
        order = [(p, i) for p in PANELS for i in range(0, len(sub[p]), qpa.BS)]
        rng.shuffle(order)
        for p, i in order:
            v = data[p]; W = sub[p]; B = [W[j] for j in rng.permutation(len(W))[:qpa.BS]]
            x = make_x(v["c3"], v["rc"], v["stat"], B, drop).to(dev)
            obs = np.stack([v["c3"][d, t + 1:t + 7, 3] > .5 for d, t in B]); q = np.stack([v["c3"][d, t + 1:t + 7, 4] > .5 for d, t in B])
            y = torch.tensor(q & obs, dtype=torch.float32, device=dev); m = torch.tensor(obs, dtype=torch.float32, device=dev)
            loss = (F.binary_cross_entropy_with_logits(model(x), y, reduction="none") * m).sum() / (m.sum() + 1)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep + 1 == snap:
            out[snap] = copy.deepcopy(model).eval()
    out[epochs] = model.eval()
    return out


def select(data, split, maxday):
    from qcnn_shift import windows
    from ong3_types_diag import pers_at, wtype
    from ong3_replica import replica_day
    from tfb.data import load_cache
    for pn, v in data.items():
        grid = set(windows(v["c3"]))
        el = np.asarray(load_cache(pn, split)["elig"], bool); rep = {}
        for d in range(v["c3"].shape[0]):
            if int(str(v["dates"][d])[8:10]) <= maxday:
                rep.update({(d, t): ty for t, ty in replica_day(v["c3"][d, :, 4] > .5, v["c3"][d, :, 3] > .5, el[d])})
        sel = []
        for d, t in sorted(set(windows(v["c3"], step=1)) | set(rep)):
            if int(str(v["dates"][d])[8:10]) > maxday:
                continue
            obs, q = v["c3"][d, :, 3] > .5, v["c3"][d, :, 4] > .5
            if pers_at(obs, q, t)[0] > .9:
                continue
            ty = wtype(obs, q, t)
            if (d, t) in grid or ty in ("A", "C") or (d, t) in rep:
                sel.append((d, t, ty, (d, t) in grid, rep.get((d, t), "")))
        v["Wx"] = sel


def export(a):
    import qcnn_predict_adapt as qpa, qcnn_adapt, ong2_sccnn as S
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    base, cfg = load_member(a.run, dev); drop = cfg["drop"]; sc = bool(cfg.get("sccnn", False))
    assert cfg.get("fit_end", "2031-02-28") <= "2031-02-28", cfg.get("fit_end")
    if sc:
        S.enable()
    modes = ("truth",)          # shipped form uses sc on type A only, with truth flags
    parts, pois, t0 = [], [], time.time()
    for split in a.splits.split(","):
        data = qpa.load_split(split, False); select(data, split, a.maxday)
        dates = sorted({str(x) for v in data.values() for x in v["dates"]})
        days = [(k, d) for k, d in enumerate(dates) if int(d[8:10]) <= a.maxday]

        def models_for(k, d_str):
            if sc:
                S.MODE = "train"; S._rng = np.random.default_rng(10_000 + k)
            if a.strong:        # serve = epoch-3 snapshot of a 5-epoch serving run (identical draws); s5 = epoch 5
                ss = finetune_snap(base, data, d_str, drop, dev, k, None, 3e-4, 5, 3)
                ms = {"un": base, "serve": ss[3], "s5": ss[5]}
                if sc:
                    S.MODE = "train"; S._rng = np.random.default_rng(10_000 + k)
                hh = finetune_snap(base, data, d_str, drop, dev, k, None, 6e-4, 5, 3)   # lr up x2: h3, h5
                ms["h3"], ms["h5"] = hh[3], hh[5]
            else:
                qpa.EPOCHS, qpa.LR = 3, 3e-4
                ms = {"un": base, "serve": qpa.finetune(base, data, d_str, drop, dev, seed=k, cap=None, daytype=True)}
            if sc:
                S.MODE = "train"; S._rng = np.random.default_rng(10_000 + k)
            gg = finetune_snap(base, data, d_str, drop, dev, k, G["cap"], G["lr"], G["epochs"], G["snap"])
            ms["g3"], ms["g5"] = gg[G["snap"]], gg[G["epochs"]]
            return ms

        def predict(ms, d_str):
            res = {}
            for p, v in data.items():
                BB = [(d, t) for d, t, *_ in v["Wx"] if v["dates"][d] == d_str]
                if not BB:
                    continue
                for kn, m in ms.items():
                    for mode in modes:
                        S.MODE = mode
                        ps = []
                        for i0 in range(0, len(BB), 256):
                            x = qcnn_adapt.make_x(v["c3"], v["rc"], v["stat"], BB[i0:i0 + 256], drop).to(dev)
                            with torch.no_grad():
                                ps.append(torch.sigmoid(m(x)).cpu().numpy())
                        res[(p, kn, mode)] = np.concatenate(ps)
            return res
        for k, d_str in days:
            ms = models_for(k, d_str); res = predict(ms, d_str)
            if a.strong and k == days[len(days) // 2][0]:     # check (poison date): the snapshot equals the shipped finetune
                if sc:
                    S.MODE = "train"; S._rng = np.random.default_rng(10_000 + k)
                qpa.EPOCHS, qpa.LR = 3, 3e-4
                r0 = predict({"serve": qpa.finetune(base, data, d_str, drop, dev, seed=k, cap=None, daytype=True)}, d_str)
                dmax = max(float(np.abs(r0[key] - res[key]).max()) for key in r0)
                print("SNAPCHECK", json.dumps(dict(split=split, date=d_str, max_diff=dmax)), flush=True)
                assert dmax <= 1e-5, "serve snapshot differs from qcnn_predict_adapt.finetune"
            if k == days[len(days) // 2][0]:           # future poison: days after d become noise
                saved = {p: v["c3"].copy() for p, v in data.items()}; g = np.random.default_rng(1)
                for p, v in data.items():
                    di = [j for j, x in enumerate(v["dates"]) if str(x) > d_str]
                    v["c3"][di] = g.random(v["c3"][di].shape).astype(v["c3"].dtype)
                rz = predict(models_for(k, d_str), d_str)
                for p, v in data.items():
                    v["c3"] = saved[p]
                pois.append(dict(split=split, date=d_str, max_diff=max(float(np.abs(rz[key] - res[key]).max()) for key in res)))
                print("POISON", json.dumps(pois[-1]), flush=True)
                assert pois[-1]["max_diff"] <= 1e-6, "future poison changed predictions"
            for p, v in data.items():
                B = [w for w in v["Wx"] if v["dates"][w[0]] == d_str]
                L = len(v["links"])
                for i, (d, t, ty, gr, rp) in enumerate(B):
                    obs, qq = v["c3"][d, :, 3] > .5, v["c3"][d, :, 4] > .5
                    row = {"split": split, "panel": p, "date": d_str, "t": t, "typ": ty, "grid": gr, "rep": rp,
                           "h": np.repeat(np.arange(1, 7), L).astype(np.int8), "link_id": np.tile(v["links"], 6),
                           "label": (qq[t + 1:t + 7] & obs[t + 1:t + 7]).reshape(-1), "valid": obs[t + 1:t + 7].reshape(-1)}
                    for kn in ("un", "serve", "g3", "g5") + (("s5", "h3", "h5") if a.strong else ()):
                        row[f"p_{kn}"] = res[(p, kn, "truth")][i].reshape(-1).astype(np.float32)
                    parts.append(pd.DataFrame(row))
            print(json.dumps(dict(split=split, date=d_str, min=round((time.time() - t0) / 60, 1))), flush=True)
        del data
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(out)
    print(json.dumps(dict(saved=str(out), minutes=round((time.time() - t0) / 60, 1), poison=pois)), flush=True)


def score(a):
    from tfb.data import load_cache
    from qcnn_data import cube
    from gapres_selcond_rule import iou
    from ong3_eval import chain, boot
    from ong3_types_diag import last_state
    al = float(a.mix)
    K = ["split", "panel", "date", "t"]

    P_all = [pd.read_parquet(x) for x in (a.sc + "," + a.ctl).split(",") if x]
    R = ["serve", "g3", "g5"] + [x for x in ("s5", "h3", "h5") if all(f"p_{x}" in q for q in P_all)]
    del P_all

    def ens(paths):
        P = [pd.read_parquet(x) for x in paths.split(",") if x]
        df = P[0].copy()
        for c in ["p_un"] + [f"p_{r}" for r in R]:
            df[c] = np.mean([q[c].to_numpy(np.float64) for q in P], 0)
        for q in P[1:]:
            assert (q[K + ["h", "link_id"]].to_numpy() == df[K + ["h", "link_id"]].to_numpy()).all()
        for r in R:
            df[f"p_m{r}"] = al * df[f"p_{r}"] + (1 - al) * df["p_un"]
        return df
    KINDS = ["un"] + R + [f"m{r}" for r in R]
    S, C = ens(a.sc), ens(a.ctl)
    rows, caches = [], {}
    for (split, pn), gs in S.groupby(["split", "panel"], sort=False):
        cc = load_cache(pn, split); st = cc.stations.sort_values("idx"); links = st.link_id.astype(str).tolist()
        c3 = cube(*[np.asarray(cc[x], np.float32) for x in ("speed", "flow", "occ")], st); OBS, Q = c3[:, :, 3] > .5, c3[:, :, 4] > .5
        gc = C[(C.split == split) & (C.panel == pn)]
        for ((date, t), g), (kc, h) in zip(gs.groupby(["date", "t"], sort=False), gc.groupby(["date", "t"], sort=False)):
            assert kc == (date, t), (kc, date, t)
            typ = g.typ.iloc[0]
            if typ not in ("A", "M", "C"):
                continue
            src = g if typ == "A" else h
            L = len(g) // 6; d = cc.day_index(str(date)); idx = np.array([links.index(l) for l in g.link_id.to_numpy()[:L]])
            pm = np.broadcast_to(last_state(OBS[d], Q[d], t - 12, t)[idx], (6, L)).ravel()
            pmT = np.broadcast_to(last_state(OBS[d], Q[d], t - 12, t + 1)[idx], (6, L)).ravel()
            V = g.valid.to_numpy(); Y = g.label.to_numpy() & V
            r = dict(split=split, panel=pn, date=str(date), t=int(t), typ=typ)
            for kn in KINDS:
                r[kn] = iou(chain(src[f"p_{kn}"].to_numpy(np.float64), pm, pmT, typ == "A", "v2") & V, Y)
            rows.append(r)
    df = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True); df.to_parquet(OUT / f"score_{a.name}.parquet")
    W = {"A": .55, "M": .25, "C": .20}
    # official ongoing window counts by day of month (window_index.csv, exp/C_semisup/SEMISUP.md): the official-date-
    # weighted contrast gives each day of month its share of official windows (coordinator 10-09: this one decides)
    OFF = {"validation": {1: 8, 2: 4, 3: 8, 4: 7, 5: 3, 6: 3, 8: 1, 9: 1, 10: 1, 11: 2, 12: 2},
           "private": {1: 4, 2: 6, 3: 9, 4: 6, 5: 4, 6: 1, 7: 3, 8: 2, 10: 1, 12: 1}}
    df["dom"] = df.date.str[8:10].astype(int)

    def offw(g, x, split, nb=2000):
        o = OFF[split]; dd = g.assign(x=x).groupby("dom").x.mean()
        dd = dd[[k for k in dd.index if k in o]]
        w = np.array([o[k] for k in dd.index], float); v = dd.to_numpy()
        e = float((w * v).sum() / w.sum()); r = np.random.default_rng(0); bs = []
        for _ in range(nb):
            i = r.integers(0, len(v), len(v)); bs.append((w[i] * v[i]).sum() / w[i].sum())
        return e, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(w.sum())
    res = {}
    for split, gs in df.groupby("split"):
        X = [k for k in KINDS if k not in ("un", "mserve")]
        for new, old in [(k, "mserve") for k in ["un"] + X] + [(k, "un") for k in ["mserve"] + X]:
            line = {}
            for typ in ("A", "M", "C"):
                g = gs[gs.typ == typ]; e, lo, hi = boot((g[new] - g[old]).to_numpy(), g.date.tolist())
                line[typ] = dict(n=len(g), new=round(float(g[new].mean()), 5), old=round(float(g[old].mean()), 5), d=round(float(e), 5), lo=round(float(lo), 5), hi=round(float(hi), 5))
            line["MIX"] = round(sum(W[t] * line[t]["d"] for t in W), 5)
            for typ in ("A", "M", "C"):
                g = gs[gs.typ == typ]; e, lo, hi, nw = offw(g, (g[new] - g[old]).to_numpy(), split)
                line[typ].update(dw=round(e, 5), dw_lo=round(lo, 5), dw_hi=round(hi, 5), off_n=nw)
            line["MIXW"] = round(sum(W[t] * line[t]["dw"] for t in W), 5)
            res[f"{split}/{new}-{old}"] = line
            print(f"{split[:3]} {new:>6}-{old:<6} MIX {line['MIX']:+.4f} | " + " | ".join(f"{t} {line[t]['d']:+.4f} [{line[t]['lo']:+.4f},{line[t]['hi']:+.4f}] n{line[t]['n']}" for t in ("A", "M", "C")), flush=True)
            print(f"{'':>17} OFFW {line['MIXW']:+.4f} | " + " | ".join(f"{t} {line[t]['dw']:+.4f} [{line[t]['dw_lo']:+.4f},{line[t]['dw_hi']:+.4f}]" for t in ("A", "M", "C")), flush=True)
    (OUT / f"score_{a.name}.json").write_text(json.dumps(res, indent=1)); print("saved", OUT / f"score_{a.name}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["export", "score"]); ap.add_argument("run", nargs="?"); ap.add_argument("out", nargs="?")
    ap.add_argument("--splits", default="validation,private"); ap.add_argument("--maxday", type=int, default=12)
    ap.add_argument("--strong", action="store_true", help="add s5 (serving recipe to 5 epochs), h3/h5 (lr 6e-4, 3/5 epochs)")
    ap.add_argument("--sc", default=""); ap.add_argument("--ctl", default=""); ap.add_argument("--mix", default=".75"); ap.add_argument("--name", default="split")
    a = ap.parse_args()
    {"export": export, "score": score}[a.cmd](a)
