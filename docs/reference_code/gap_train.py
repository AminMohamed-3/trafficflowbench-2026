"""Claude gap helper 2026-10-04: train the pooled gap specialist (LightGBM on simulated selector-like blocks of the split's
own visible data) and score it on I-S-OWN March artificial blocks of selector-like class (onset / ongoing), alone and
blended with C12 (= C13 on gap cells). Blend weights per class (v and q separately, grid .05) fit on odd calendar days
(dev), scored on even days (confirm); date bootstrap (1000) of (a) gap-cell rmse and (b) board-mix S_state
(own_instrument.paired_state rows: OWN non-block C12 sums + block sums reweighted to each panel's official gap&target
share; D12_I405 has no official gaps).
Usage: python docs/reference_code/gap_train.py <tag> [--base lin|an|none] [--rounds N] [--leaves N] [--nok] [--sub sel|sel9]"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, argparse, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import gap_lib as G
import lightgbm as lgb
import flow_own as FO

GRID = np.round(np.linspace(0, 1, 21), 2)


def load(kind, period, panels, maxrows=2_400_000):
    """Eval parts per panel; training rows preallocated into one float32 matrix (memory). If the total exceeds maxrows,
    a fixed-seed uniform row subsample per panel (same fraction) is kept."""
    R = {}; n = 0
    for pn in panels:
        with np.load(G.OUT / f"rows_{kind}_{period}_{pn}.npz") as z:
            R[pn] = {k: z[k] for k in z.files if k not in ("Xtr",)}
            n += len(R[pn]["Ytr"]); nf = len(R[pn]["names"])
    frac = min(1., maxrows / n)
    sel = {}
    rng = np.random.default_rng(7)
    for pn in panels:
        m = len(R[pn]["Ytr"])
        sel[pn] = np.sort(rng.choice(m, int(round(m * frac)), replace=False)) if frac < 1 else np.arange(m)
        R[pn]["Ytr"] = R[pn]["Ytr"][sel[pn]]; R[pn]["Mtr"] = R[pn]["Mtr"][sel[pn]]
    X = np.empty((sum(len(v) for v in sel.values()), nf), np.float32); i = 0
    for pn in panels:
        with np.load(G.OUT / f"rows_{kind}_{period}_{pn}.npz") as z:
            x = z["Xtr"][sel[pn]]; X[i:i + len(x)] = x; i += len(x)
    print(kind, period, "training rows", len(X), "of", n, flush=True)
    return R, X


def fit(R, X, names, args, drop=()):
    keep = [i for i, n in enumerate(names) if n not in drop]
    if len(keep) < X.shape[1]:
        X = X[:, keep]
    Y = np.concatenate([R[p]["Ytr"] for p in R])
    Mt = np.concatenate([R[p]["Mtr"] for p in R])
    sw = np.ones(len(Y))
    if getattr(args, "wpers", 1.) != 1.:   # up-weight official-like hard ongoing blocks (persistence IoU <= .9)
        sw = np.where((Mt[:, 3] == 1) & (Mt[:, 7] >= 0) & (Mt[:, 7] <= 900), args.wpers, 1.)
    base = _base(X, [names[i] for i in keep], args.base)
    models = []
    for c in range(2):
        y = Y[:, c] - base[c]
        ok = np.isfinite(y)
        ds = lgb.Dataset(X if ok.all() else X[ok], y if ok.all() else y[ok], weight=sw if ok.all() else sw[ok], feature_name=[names[i] for i in keep], free_raw_data=True)
        prm = dict(objective="l2", learning_rate=.05, num_leaves=args.leaves, min_data_in_leaf=200, feature_fraction=.8,
                   bagging_fraction=.8, bagging_freq=1, lambda_l2=1., num_threads=3, verbose=-1, seed=c)
        models.append(lgb.train(prm, ds, args.rounds))
    return models, keep


def _base(X, names, kind):
    if kind == "none":
        return 0., 0.
    a, b = ("lin_v", "lin_q") if kind == "lin" else ("an_v", "an_q")
    return X[:, names.index(a)], X[:, names.index(b)]


def predict(models, keep, names, Xe, args):
    X = Xe[:, keep]; nm = [names[i] for i in keep]
    b = _base(X, nm, args.base)
    return models[0].predict(X, num_threads=3) + b[0], models[1].predict(X, num_threads=3) + b[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("--base", default="an"); ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--leaves", type=int, default=63); ap.add_argument("--nok", action="store_true")
    ap.add_argument("--sub", default="sel"); ap.add_argument("--drop", default="")
    ap.add_argument("--evalonly", action="store_true"); ap.add_argument("--halves", default="odd"); ap.add_argument("--cnn", default=""); ap.add_argument("--smooth", type=int, default=0)
    ap.add_argument("--period", default="validation"); ap.add_argument("--wpers", type=float, default=1.); ap.add_argument("--wjson", default="")
    args = ap.parse_args()
    t0 = time.time()
    if args.evalonly:
        return eval_saved(args)
    R, X = load("own", args.period, G.PANELS)
    names = [str(n) for n in R[G.PANELS[0]]["names"]]
    drop = ("krel", "onset_pre") if args.nok else ()
    if args.drop:
        drop = drop + tuple(n for n in names if any(n.startswith(p) for p in args.drop.split(",")))
    models, keep = fit(R, X, names, args, drop); del X
    print("trained %.0fs" % (time.time() - t0), flush=True)
    imp = sorted(zip(models[0].feature_importance("gain"), [names[i] for i in keep]), reverse=True)[:15]
    print("top v features", [(n, round(g / 1e3)) for g, n in imp])
    nb = json.loads((G.OUT / f"nb_{args.period}.json").read_text())
    ev = {}
    for pn in G.PANELS:
        r = R[pn]; M = r["Mev"]; E = r["Eev"]
        sv, sq = predict(models, keep, names, r["Xev"], args)
        s = M[:, 6]
        vs = sv * r["vf"][s]; qs = sq * r["cap"][s]
        ev[pn] = dict(M=M, tv=E[:, 0], tq=E[:, 1], c12=(E[:, 2], E[:, 3]), c11=(E[:, 4], E[:, 5]), imp=(E[:, 6], E[:, 7]),
                      spec=(np.clip(vs, 3, 1.3 * r["vf"][s]), np.clip(qs, 0, 1.3 * r["cap"][s])), la=r["lanes"][s])
    res = evaluate(ev, nb, args)
    res["args"] = vars(args)
    (G.OUT / f"eval_{args.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(G.OUT / f"pred_{args.tag}.npz", **{f"{pn}_v": ev[pn]["spec"][0].astype(np.float32) for pn in ev},
                        **{f"{pn}_q": ev[pn]["spec"][1].astype(np.float32) for pn in ev})


def eval_saved(args):
    R = {}
    for pn in G.PANELS:
        with np.load(G.OUT / f"rows_own_{args.period}_{pn}.npz") as z:
            R[pn] = {k: z[k] for k in ("Mev", "Eev", "lanes")}
    z = np.load(G.OUT / f"pred_{args.tag}.npz")
    ev = {}
    for pn in G.PANELS:
        M = R[pn]["Mev"]; E = R[pn]["Eev"]; s = M[:, 6]
        ev[pn] = dict(M=M, tv=E[:, 0], tq=E[:, 1], c12=(E[:, 2], E[:, 3]), c11=(E[:, 4], E[:, 5]), imp=(E[:, 6], E[:, 7]),
                      spec=(z[f"{pn}_v"].astype(float), z[f"{pn}_q"].astype(float)), la=R[pn]["lanes"][s])
    if args.cnn:
        zc = np.load(G.OUT / f"pred_{args.cnn}.npz")
        for pn in G.PANELS:
            ev[pn]["cnn"] = (zc[f"{pn}_v"].astype(float), zc[f"{pn}_q"].astype(float))
    if args.smooth:
        for pn in G.PANELS:
            for k in ("spec", "cnn"):
                if k in ev[pn]:
                    ev[pn][k] = tuple(G.smooth_rows(ev[pn]["M"], x, args.smooth) for x in ev[pn][k])
    nb = json.loads((G.OUT / f"nb_{args.period}.json").read_text())
    res = evaluate(ev, nb, args)
    (G.OUT / f"eval_{args.tag}{'_' + args.period if args.period != 'validation' else ''}{'_' + args.cnn if args.cnn else ''}{'_sm%d' % args.smooth if args.smooth else ''}_{args.sub}_{args.halves}.json").write_text(json.dumps(res, indent=1, default=float))


def sub_mask(M, sub):
    cls = M[:, 3]; pers = M[:, 7]
    m = (cls == 0) | (cls == 1)
    if sub == "sel9":
        m &= (cls == 0) | (pers <= 900)
    if sub == "all":
        m = np.ones(len(M), bool)
    return m


def evaluate(ev, nb, args):
    p = G.P.P; panels = p["panels"]; fam = p["families"]
    groups = [[i for i, x in enumerate(panels) if fam[x] == f] for f in sorted(set(fam.values()))]
    dates = [r["date"] for r in nb[panels[0]]["rows"]]
    odd = np.array([int(d[-2:]) % 2 == 1 for d in dates])
    # pooled per class simplex-grid blend weights (members c12, spec[, cnn]) on odd (dev) days, via Gram matrices
    mem = ["c12", "spec"] + (["cnn"] if "cnn" in next(iter(ev.values())) else [])
    k = len(mem)
    grid = np.array([w for w in np.ndindex(*(21,) * k) if sum(w) == 20], float) / 20.
    W = {}
    for c in (0, 1):
        Wc = []
        for ch in (0, 1):
            Gm = np.zeros((k, k)); h = np.zeros(k)
            for pn, e in ev.items():
                M = e["M"]; m = sub_mask(M, args.sub) & (M[:, 3] == c) & odd[M[:, 1]]
                sc = 1. if ch == 0 else e["la"][m]
                X_ = np.stack([e[n][ch][m] / sc for n in mem], 1); y_ = (e["tv"] if ch == 0 else e["tq"])[m] / sc
                Gm += X_.T @ X_; h += X_.T @ y_
            sse = np.einsum("gi,ij,gj->g", grid, Gm, grid) - 2 * grid @ h
            Wc.append([float(x) for x in grid[np.argmin(sse)]])
        W[c] = Wc
    if args.wjson:
        Wf = json.loads((G.OUT / args.wjson).read_text())
        assert Wf["members"] == mem
        W = {int(k): v for k, v in Wf["weights"].items()}
    print("blend weights", mem, "by class onset/ongoing [v, q], fit odd days:" if not args.wjson else "FIXED from " + args.wjson, W)
    for pn, e in ev.items():
        M = e["M"]; c = np.clip(M[:, 3], 0, 1)
        e["blend"] = tuple(sum(np.array([W[0][ch][i], W[1][ch][i]])[c] * e[n][ch] for i, n in enumerate(mem)) for ch in (0, 1))
    res = dict(weights=W, members=mem, rmse={}, board={})
    # gap-cell rmse by class and day half
    HALVES = [h for h in (("odd", odd), ("even", ~odd)) if h[0] in args.halves.split(",")]
    for half, hm in HALVES:
        for cname, cset in (("onset", (0,)), ("ongoing", (1,)), ("sel", (0, 1))):
            acc = {k: [0., 0., 0] for k in ("c12", "c11", "imp", "spec", "blend") + tuple(m_ for m_ in mem if m_ == "cnn")}
            for pn, e in ev.items():
                M = e["M"]; m = sub_mask(M, args.sub) & np.isin(M[:, 3], cset) & hm[M[:, 1]]
                for k in acc:
                    acc[k][0] += np.sum((e[k][0][m] - e["tv"][m]) ** 2); acc[k][1] += np.sum(((e[k][1][m] - e["tq"][m]) / e["la"][m]) ** 2); acc[k][2] += m.sum()
            res["rmse"][f"{half}_{cname}"] = {k: (round(np.sqrt(a[0] / max(a[2], 1)), 3), round(np.sqrt(a[1] / max(a[2], 1)), 2), a[2]) for k, a in acc.items()}
            print(half, cname, res["rmse"][f"{half}_{cname}"], flush=True)
    # board-mix S_state, paired by date
    own = FO.own
    for cand, mix in (("spec", "own"), ("blend", "own"), ("blend", "bal"), ("spec", "bal")) + ((("cnn", "bal"),) if "cnn" in mem else ()):
        for half, hm in HALVES + ([("all", np.ones(len(dates), bool))] if "even" in args.halves else []):
            ref_rows, new_rows = [], []
            for pn in panels:
                e = ev[pn]; M = e["M"]; m = sub_mask(M, args.sub)
                nbr = nb[pn]["rows"]; share = nb[pn]["share"]
                nnb = sum(r["n"] for r in nbr); nbl = m.sum()
                f = np.full(len(M), share / (1 - share) * nnb / max(nbl, 1))
                if mix == "bal":   # official mix: 5 onset + 5 ongoing windows -> half of the gap cells each
                    non, nog = (m & (M[:, 3] == 0)).sum(), (m & (M[:, 3] == 1)).sum()
                    if non and nog:
                        f = np.where(M[:, 3] == 0, share / 2 / (1 - share) * nnb / non, share / 2 / (1 - share) * nnb / nog)
                for d, r in enumerate(nbr):
                    if not hm[d]:
                        continue
                    md = m & (M[:, 1] == d)
                    for rows, k in ((ref_rows, "c12"), (new_rows, cand)):
                        sv = np.sum(f[md] * (e[k][0][md] - e["tv"][md]) ** 2); sq = np.sum(f[md] * ((e[k][1][md] - e["tq"][md]) / e["la"][md]) ** 2)
                        rows.append(dict(panel=pn, date=r["date"], regime=r["regime"], n=r["n"] + f[md].sum(), sq_v=r["sq_v"] + sv, sq_q=r["sq_q"] + sq))
            ds = [d for d, h in zip(dates, hm) if h]
            x = own.paired_state(ref_rows, new_rows, panels, ds, groups)
            res["board"][f"{cand}_{mix}_{half}"] = dict(delta=x["delta_S_state"], ci=x["ci95"], verdict=x["verdict"])
            print("board-mix S_state", cand, mix, half, "%+.6f" % x["delta_S_state"], [round(c, 6) for c in x["ci95"]], x["verdict"], flush=True)
    return res


if __name__ == "__main__":
    main()
