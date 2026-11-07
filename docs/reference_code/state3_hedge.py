"""Claude helper (state3, ledger 0368) 2026-10-04: congestion-probability hedged speed on the reference (F4 + 0360 common mode).
Candidate target cells C (class uncertain): |v_ref - v_cut| < 30, or the nearest observed same-station neighbour before / after
(within 6 steps) has the other class than v_ref; and >= 3 observed stations at the step (blocks excluded).
On C: p = P(v < v_cut | x) (LightGBM binary), mu_c = v_ref + g_c(x), mu_f = v_ref + g_f(x) (LightGBM L2 on residual,
fitted on truly congested / truly free cells of C). Hedged speed v = p mu_c + (1 - p) mu_f where EPS <= p <= 1 - EPS, else v_ref.
Flow unchanged. Features: state3_feat.FEATS (observed cells of the blinded view + v_ref + F4 flow).
OPT: cross-fit leave-one-panel-out x other day parity (as state2), 10 panels, paired date bootstrap of delta J_S vs reference.
OWN: models fitted on all OPT panels and days (train split), applied to March (cross-split transfer).
Usage: python docs/reference_code/state3_hedge.py opt [alpha [eps]]   |   python docs/reference_code/state3_hedge.py own [alpha [eps]]
alpha (default 1) shrinks the change: v = v_ref + alpha (v_hedge - v_ref)."""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phys_lwr_diag as P
import state2_cm as CM
import state3_feat as F

OUT = F.OUT
PANELS = P.P["panels"]
EPS = 0.02
BAND = 30.
FI = {f: i for i, f in enumerate(F.FEATS)}
PRM = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=1.0, num_threads=4, verbose=-1, seed=0)
ROUNDS = 400


def candidate(X, cut):
    vr = X[:, FI["vr"]]; rc = vr < cut
    nb = ((X[:, FI["db"]] <= 6) & ((X[:, FI["vb"]] < cut) != rc)) | ((X[:, FI["da"]] <= 6) & ((X[:, FI["va"]] < cut) != rc))
    return ((np.abs(vr - cut) < BAND) | nb) & (X[:, FI["nobs_t"]] >= 3)


def fit(X, y, cut):
    yc = y < cut; r = y - X[:, FI["vr"]]
    clf = lgb.train(dict(PRM, objective="binary"), lgb.Dataset(X, yc.astype(float)), ROUNDS)
    gc = lgb.train(dict(PRM, objective="regression"), lgb.Dataset(X[yc], r[yc]), ROUNDS)
    gf = lgb.train(dict(PRM, objective="regression"), lgb.Dataset(X[~yc], r[~yc]), ROUNDS)
    return clf, gc, gf


def hedge(models, X, raw=False):
    """Hedged speed and p; raw=True returns the mixture without the EPS keep rule."""
    clf, gc, gf = models
    vr = X[:, FI["vr"]]
    p = clf.predict(X); mc = vr + gc.predict(X); mf = vr + gf.predict(X)
    v = p * mc + (1 - p) * mf
    if raw:
        return v, p
    keep = (p < EPS) | (p > 1 - EPS)
    return np.where(keep, vr, v), p


def load(panel):
    z = np.load(OUT / f"opt_{panel}.npz")
    return z["X"].astype(float), z["ix"], z["y"].astype(float), z["vcut"].astype(float)


def cls_of(X, cut):
    cb = X[:, FI["vb"]] < cut; ca = X[:, FI["va"]] < cut
    close = (X[:, FI["db"]] <= 3) & (X[:, FI["da"]] <= 3)
    return np.where(~close, "nob", np.where(cb != ca, "front", np.where(cb, "cong", "free")))


def opt(alpha=1.0, nboot=1000):
    data = {p: load(p) for p in PANELS}
    cand = {p: candidate(d[0], d[3]) for p, d in data.items()}
    for p in PANELS:
        print(p, "candidates", int(cand[p].sum()), round(float(cand[p].mean()), 4), flush=True)
    D = len(P.P["OPT_held_dates"]); par = np.arange(D) % 2
    pred = {}
    for panel in PANELS:
        X, ix, y, cut = data[panel]
        out = X[:, FI["vr"]].copy(); pp = np.full(len(y), np.nan)
        for f in (0, 1):
            tr = [(d[0][(par[d[1][:, 0]] != f) & cand[q]], d[2][(par[d[1][:, 0]] != f) & cand[q]], d[3][(par[d[1][:, 0]] != f) & cand[q]])
                  for q, d in data.items() if q != panel]
            Xt = np.concatenate([a[0] for a in tr]); yt = np.concatenate([a[1] for a in tr]); ct = np.concatenate([a[2] for a in tr])
            models = fit(Xt, yt, ct)
            sel = (par[ix[:, 0]] == f) & cand[panel]
            v, p = hedge(models, X[sel], raw=True)
            out[sel] = v; pp[sel] = p
        np.savez(OUT / f"opt_pred_{panel}.npz", v=out.astype(np.float32), p=pp.astype(np.float32), cand=cand[panel])
        keep = ~((pp >= EPS) & (pp <= 1 - EPS))
        out = np.where(keep, X[:, FI["vr"]], out)
        pred[panel] = (out, pp)
        print(panel, "fitted", flush=True)
    rows, comps, cls_rows = [], {}, []
    for panel in PANELS:
        X, ix, y, cut = data[panel]
        held, tv, tq, pv, pq = P.held_panel(panel)
        vr_full = F.ref_opt(panel, held, pv)
        vrf = vr_full[ix[:, 0], ix[:, 1], ix[:, 2]]
        vh = vrf + alpha * (pred[panel][0] - X[:, FI["vr"]])
        nv = vr_full.copy(); nv[ix[:, 0], ix[:, 1], ix[:, 2]] = vh
        ref = P.score(held, tv, tq, vr_full, pq); new = P.score(held, tv, tq, nv, pq)
        Cf, reg = CM.day_components(held, tv, tq, vr_full, pq); Cn, _ = CM.day_components(held, tv, tq, nv, pq)
        comps[panel] = (Cf, Cn, reg)
        rows.append(dict(panel=panel, **{f"ref_{k}": v for k, v in ref.items()}, **{f"new_{k}": v for k, v in new.items()}))
        print(panel, "ref", ref, "new", new, flush=True)
        c = cls_of(X, cut); mis = (vrf < cut) != (y < cut)
        p = pred[panel][1]
        for name, m in (("misclassified", mis), ("front", c == "front"), ("nob", c == "nob"), ("free", c == "free"), ("cong", c == "cong"),
                        ("candidate", cand[panel]), ("changed", np.abs(vh - vrf) > 1e-9), ("all", np.ones(len(y), bool))):
            cls_rows.append(dict(panel=panel, cls=name, n=int(m.sum()), se_ref=float(((vrf - y) ** 2)[m].sum()), se_new=float(((vh - y) ** 2)[m].sum())))
        # calibration of p on candidates
        ok = np.isfinite(p)
        if ok.any():
            b = pd.cut(p[ok], [0, .02, .1, .3, .5, .7, .9, .98, 1.0001], include_lowest=True)
            cal = pd.DataFrame(dict(b=b, y=(y < cut)[ok])).groupby("b", observed=False).y.agg(["size", "mean"])
            print(cal.round(3).to_string(), flush=True)
    rng = np.random.default_rng(0); boots = []
    for _ in range(nboot):
        wts = np.bincount(rng.integers(0, D, D), minlength=D).astype(float)
        dl = {k: [] for k in ("J", "S_state", "S_phys", "S_LWR", "S_FD")}
        for panel, (Cf, Cn, reg) in comps.items():
            a, b = CM.aggregate(Cf, reg, wts), CM.aggregate(Cn, reg, wts)
            for k in dl: dl[k].append(b[k] - a[k])
        boots.append({k: np.mean(v) for k, v in dl.items()})
    B = pd.DataFrame(boots); df = pd.DataFrame(rows)
    summary = {k: dict(mean_delta=float((df[f"new_{k}"] - df[f"ref_{k}"]).mean()),
                       ci=[float(B[k].quantile(.025)), float(B[k].quantile(.975))],
                       panels_pos=int(((df[f"new_{k}"] - df[f"ref_{k}"]) > 0).sum())) for k in ("J", "S_state", "S_phys", "S_LWR", "S_FD")}
    cr = pd.DataFrame(cls_rows)
    agg = cr.groupby("cls")[["n", "se_ref", "se_new"]].sum()
    agg["rmse_ref"] = np.sqrt(agg.se_ref / agg.n); agg["rmse_new"] = np.sqrt(agg.se_new / agg.n)
    print(json.dumps(summary, indent=1)); print(agg[["n", "rmse_ref", "rmse_new"]].round(3).to_string())
    tag = f"a{alpha:g}_e{EPS:g}"
    df.to_csv(OUT / f"opt_screen_{tag}.csv", index=False); cr.to_csv(OUT / f"opt_class_{tag}.csv", index=False)
    (OUT / f"opt_summary_{tag}.json").write_text(json.dumps(dict(summary=summary, class_rmse=agg[["n", "rmse_ref", "rmse_new"]].to_dict("index"),
                                                                  eps=EPS, band=BAND, alpha=alpha, rounds=ROUNDS, params=PRM), indent=1))


def own(alpha=1.0):
    own_m, bind = CM.own_modules()
    coef = json.loads((P.ROOT / "artifacts/claude_ref/state2/opt_coef.json").read_text()); cf = (coef["b_free"], coef["b_cong"])
    data = [load(p) for p in PANELS]
    cs = [candidate(d[0], d[3]) for d in data]
    models = fit(np.concatenate([d[0][c] for d, c in zip(data, cs)]), np.concatenate([d[2][c] for d, c in zip(data, cs)]),
                 np.concatenate([d[3][c] for d, c in zip(data, cs)]))
    print("fitted on all OPT candidates", sum(int(c.sum()) for c in cs), flush=True)
    items = [i for i in json.loads(Path(P.P["OWN_metadata_binding"]["manifest"]).read_text())["items"] if i["period"] == "validation"]
    ref_rows, new_rows, ref_nb, new_nb, per = [], [], [], [], []
    for item in items:
        item = bind.bound_own_item(P.P, item)
        view, tv, tq, score_target, hidden = own_m.load_item(P.ROOT, item)
        f = P.D / "F4_predictions/OWN" / f"{item['panel']}_validation"
        pv, pq = (np.load(f / f"{n}.npy").astype(float) for n in ("speed", "flow"))
        speed = np.asarray(view.speed, float); gap = np.asarray(view.gap, bool)
        observed = np.asarray(view.elig, bool) & ~np.asarray(view.target, bool) & ~gap & np.isfinite(speed)
        gr, n = CM.common_factor(speed, observed, view.stations)
        vcut = view.stations.sort_values("idx").v_cut.to_numpy(float)
        st = np.asarray(score_target, bool)
        vr = np.where(st, CM.apply(pv, gr, vcut, cf), pv)
        X, ix = F.build(speed, observed, view.stations, vr, pq, st)
        X = X.astype(float); cut = vcut[ix[:, 2]]
        c = candidate(X, cut)
        vh = vr[ix[:, 0], ix[:, 1], ix[:, 2]].copy()
        if c.any():
            vh[c] += alpha * (hedge(models, X[c])[0] - X[c, FI["vr"]])
        nv = vr.copy(); nv[ix[:, 0], ix[:, 1], ix[:, 2]] = vh
        a, _ = own_m.state_rows(view, tv, tq, vr, pq, score_target); b, _ = own_m.state_rows(view, tv, tq, nv, pq, score_target)
        ref_rows += a; new_rows += b
        nbt = st & ~gap
        ref_nb += own_m.state_rows(view, tv, tq, vr, pq, nbt)[0]; new_nb += own_m.state_rows(view, tv, tq, nv, pq, nbt)[0]
        m = st & np.isfinite(tv); mis = m & ((vr < vcut) != (tv < vcut))
        r = lambda x, mm: float(np.sqrt(((x - tv)[mm] ** 2).mean())) if mm.any() else None
        per.append(dict(panel=item["panel"], n=int(m.sum()), cand_share=float(c.mean()), changed=int((np.abs(nv - vr)[m] > 1e-9).sum()),
                        rmse_ref=r(vr, m), rmse_new=r(nv, m), rmse_ref_nonblock=r(vr, m & ~gap), rmse_new_nonblock=r(nv, m & ~gap),
                        n_mis=int(mis.sum()), rmse_ref_mis=r(vr, mis), rmse_new_mis=r(nv, mis)))
        print(json.dumps(per[-1]), flush=True)
    fam = P.P["families"]
    groups = [[i for i, p in enumerate(PANELS) if fam[p] == f] for f in sorted(set(fam.values()))]
    dates = sorted({r["date"] for r in ref_rows})
    ci = own_m.paired_state(ref_rows, new_rows, PANELS, dates, groups)
    ci_nb = own_m.paired_state(ref_nb, new_nb, PANELS, dates, groups)
    print("all OWN targets", json.dumps(ci, indent=1)); print("non-block OWN targets", json.dumps(ci_nb, indent=1))
    (OUT / f"own_march_a{alpha:g}_e{EPS:g}.json").write_text(json.dumps(dict(paired_state=ci, paired_state_nonblock=ci_nb, per_panel=per,
                                                                   eps=EPS, band=BAND, alpha=alpha), indent=1))


if __name__ == "__main__":
    a = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    if len(sys.argv) > 3:
        EPS = float(sys.argv[3])
    {"opt": opt, "own": own}[sys.argv[1]](a)
