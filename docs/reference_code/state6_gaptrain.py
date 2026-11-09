"""state6 R (ledger 0474): gap specialist v5 = gap v3 recipe + released ramp-flow features inside the 90-min blackout.
Rows: artifacts/claude_ref/gap/rows_<kind>_<period>_<pn>.npz (gap_build; Mtr/Mev = block, d, t0, cls, k, j, s, pers).
Ramp features per row (d, t0, j, s), for on-ramps (OR) and off-ramps (FR), radius nb in (1, 2) stations:
  log ramp flow anomaly vs the split's time-of-day median (elig only), mean over ramps attached within s+-nb:
  a (at t0+j), a3 (+-1 step mean), pre (t0-5..t0), post (t0+19..t0+24), d3 = a3 - linear pre->post interpolation;
  plus corridor means of OR/FR anomaly at t0+j minus their pre/post interpolation.
Model: gap_train.fit recipe (base an, 255 leaves, lr .05, min leaf 200, ff/bf .8, l2 1), targets v/vf and q/cap.
Usage: python state6_gaptrain.py TAG KIND PERIOD [--rounds N] [--leaves N] [--noramp]
  KIND own -> eval on OWN blocks (saves pred_<TAG>_<period>.npz); KIND off -> official windows (saves offpred).
Threads 3."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, argparse, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gap_lib as G
import lightgbm as lgb

OUT = G.ROOT / "artifacts/claude_ref/state6"
RNAMES = None


def ramp_fields(pn, split):
    sys.path.insert(0, str(G.ROOT / "src"))
    from tfb.data import load_cache
    c = load_cache(pn, split); A = c.arrays
    rf = np.asarray(A["ramp_flow"], float); re = np.asarray(A["ramp_elig"], bool)
    rf = np.where(re, rf, np.nan)
    lr = np.log(rf + 5.); an = lr - np.nanmedian(lr, 0, keepdims=True)
    S = len(c.stations); D, T, _ = rf.shape
    out = {}
    for typ in ("OR", "FR"):
        sel = [r for r in c.ramps.itertuples() if r.ramp_type == typ and np.isfinite(r.station_idx)]
        allm = np.nanmean(np.stack([an[:, :, int(r.idx)] for r in sel], 2), 2) if sel else np.full((D, T), np.nan)
        out[f"{typ}_corr"] = allm
        for nb in (1, 2):
            num = np.zeros((D, T, S)); den = np.zeros((D, T, S))
            for r in sel:
                a = an[:, :, int(r.idx)]; ok = np.isfinite(a)
                for s in range(max(0, int(r.station_idx) - nb), min(S, int(r.station_idx) + nb + 1)):
                    num[:, :, s] += np.where(ok, a, 0); den[:, :, s] += ok
            out[f"{typ}{nb}"] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    return out


def ramp_calib_fields(pn, split, kind):
    """Per-ramp calibrated congestion sensor (v6): for each on-ramp r attached at s0, from the LAYER's visible mainline
    cells at s0-1..s0+1 (OWN view for kind own, official view for kind off) learn the mean +-1-step log-flow anomaly when the
    attached area is congested (any of the 3 stations visible and below v_cut) vs free (all visible ones free):
    z_r = (a3_r - mf_r) / (mc_r - mf_r) clipped to [-1, 2], weight w_r = (mc_r - mf_r)^2 / var_r (needs >= 20 congested).
    Station fields: weighted mean of z over ramps within s+-nb, nb in (1, 2, 4)."""
    sys.path.insert(0, str(G.ROOT / "src"))
    from tfb.data import load_cache
    c = load_cache(pn, split); A = c.arrays
    rf = np.asarray(A["ramp_flow"], float); re = np.asarray(A["ramp_elig"], bool)
    rf = np.where(re, rf, np.nan)
    lr = np.log(rf + 5.); an = lr - np.nanmedian(lr, 0, keepdims=True)
    D, T, _ = rf.shape
    an3 = np.nanmean(np.stack([np.roll(an, k, 1) for k in (-1, 0, 1)], 0), 0)
    if kind == "own":
        z = np.load(OUT / f"base_{'mar' if split == 'validation' else 'apr'}_{pn}.npz")
        vis = np.where(z["target"], np.nan, z["vobs"].astype(float))
    else:
        vis = np.where(np.asarray(A["target"], bool) | np.asarray(A["gap"], bool), np.nan, np.asarray(A["speed"], float))
    st = c.stations.sort_values("idx"); vc = st.v_cut.to_numpy(float); S = len(vc)
    Cg = np.where(np.isfinite(vis), vis < vc[None, None, :], False); Vi = np.isfinite(vis)
    zs, ws, ss = [], [], []
    for r in c.ramps.itertuples():
        if r.ramp_type != "OR" or not np.isfinite(r.station_idx):
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
        zs.append(np.clip((a - mf) / (mc - mf), -1, 2)); ws.append((mc - mf) ** 2 / var); ss.append(s0)
    out = {}
    for nb in (1, 2, 4):
        num = np.zeros((D, T, S)); den = np.zeros((D, T, S))
        for zr, wr, s0 in zip(zs, ws, ss):
            ok = np.isfinite(zr)
            for s in range(max(0, s0 - nb), min(S, s0 + nb + 1)):
                num[:, :, s] += np.where(ok, wr * zr, 0); den[:, :, s] += np.where(ok, wr, 0)
        out[f"ZC{nb}"] = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        out[f"ZW{nb}"] = den
    return out


def ramp_feats(F, M):
    d, t0, j, s = M[:, 1].astype(int), M[:, 2].astype(int), M[:, 5].astype(int), M[:, 6].astype(int)
    T = next(iter(F.values())).shape[1]
    cols = {}

    def at(x, tt, spatial=True):
        tt = np.clip(tt, 0, T - 1)
        return x[d, tt, s] if spatial else x[d, tt]

    def mean_over(x, ks, spatial=True):
        with np.errstate(all="ignore"):
            import warnings
            warnings.simplefilter("ignore")
            return np.nanmean(np.stack([at(x, t0 + k, spatial) for k in ks], 1), 1)

    for key, x in F.items():
        if key.startswith("ZW"):
            cols[f"r_{key}"] = at(x, t0 + j); continue
        sp = not key.endswith("_corr")
        a = at(x, t0 + j, sp)
        with np.errstate(all="ignore"):
            import warnings
            warnings.simplefilter("ignore")
            a3 = np.nanmean(np.stack([at(x, t0 + j + k, sp) for k in (-1, 0, 1)], 1), 1)
        pre = mean_over(x, range(-5, 1), sp); post = mean_over(x, range(19, 25), sp)
        lin = pre + (post - pre) * (j / 19.)
        if sp:
            cols[f"r_{key}_a"] = a; cols[f"r_{key}_a3"] = a3; cols[f"r_{key}_pre"] = pre; cols[f"r_{key}_post"] = post
        cols[f"r_{key}_d3"] = a3 - lin
    names = sorted(cols)
    return names, np.stack([cols[n] for n in names], 1).astype(np.float32)


def load(kind, period, panels, maxrows=2_400_000, ramp=True, calib=False):
    """As gap_train.load (same fixed-seed subsample), plus ramp columns appended to Xtr and Xev."""
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
    names = [str(x) for x in R[panels[0]]["names"]]
    rn = []
    if ramp:
        for pn in panels:
            Fd = ramp_fields(pn, period)
            if calib:
                Fd.update(ramp_calib_fields(pn, period, kind))
            rn, xr = ramp_feats(Fd, R[pn]["Mtr"]); R[pn]["Rtr"] = xr
            _, xe = ramp_feats(Fd, R[pn]["Mev"]) if len(R[pn]["Mev"]) else (rn, np.zeros((0, len(rn)), np.float32))
            R[pn]["Xev"] = np.concatenate([R[pn]["Xev"], xe], 1)
    X = np.empty((sum(len(v) for v in sel.values()), nf + len(rn)), np.float32); i = 0
    for pn in panels:
        with np.load(G.OUT / f"rows_{kind}_{period}_{pn}.npz") as z:
            x = z["Xtr"][sel[pn]]
        X[i:i + len(x), :nf] = x
        if ramp:
            X[i:i + len(x), nf:] = R[pn].pop("Rtr")
        i += len(x)
    print(kind, period, "training rows", len(X), "of", n, "features", X.shape[1], flush=True)
    return R, X, names + rn


def fit(R, X, names, rounds, leaves):
    Y = np.concatenate([R[p]["Ytr"] for p in R])
    ia, iq = names.index("an_v"), names.index("an_q")
    base = (X[:, ia], X[:, iq])
    models = []
    for c in range(2):
        y = Y[:, c] - base[c]
        ok = np.isfinite(y)
        ds = lgb.Dataset(X[ok], y[ok], feature_name=names, free_raw_data=True)
        prm = dict(objective="l2", learning_rate=.05, num_leaves=leaves, min_data_in_leaf=200, feature_fraction=.8,
                   bagging_fraction=.8, bagging_freq=1, lambda_l2=1., num_threads=3, verbose=-1, seed=c)
        models.append(lgb.train(prm, ds, rounds))
        print("trained channel", c, flush=True)
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag"); ap.add_argument("kind"); ap.add_argument("period")
    ap.add_argument("--rounds", type=int, default=1500); ap.add_argument("--leaves", type=int, default=255)
    ap.add_argument("--noramp", action="store_true"); ap.add_argument("--calib", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    R, X, names = load(a.kind, a.period, G.PANELS, ramp=not a.noramp, calib=a.calib)
    models = fit(R, X, names, a.rounds, a.leaves); del X
    imp = sorted(zip(models[0].feature_importance("gain"), names), reverse=True)[:20]
    print("top v features", [(n, round(g / 1e3)) for g, n in imp], flush=True)
    ia, iq = names.index("an_v"), names.index("an_q")
    out = {}
    for pn in G.PANELS:
        r = R[pn]; Xe = r["Xev"]
        if len(Xe) == 0:
            out[f"{pn}_v"] = np.zeros(0, np.float32); out[f"{pn}_q"] = np.zeros(0, np.float32); continue
        s = r["Mev"][:, 6]
        sv = models[0].predict(Xe, num_threads=3) + Xe[:, ia]; sq = models[1].predict(Xe, num_threads=3) + Xe[:, iq]
        vs = sv * r["vf"][s]; qs = sq * r["cap"][s]
        out[f"{pn}_v"] = np.clip(vs, 3, 1.3 * r["vf"][s]).astype(np.float32); out[f"{pn}_q"] = np.clip(qs, 0, 1.3 * r["cap"][s]).astype(np.float32)
    f = OUT / f"pred_{a.tag}_{a.kind}_{a.period}.npz"
    np.savez_compressed(f, **out)
    for i, m in enumerate(models):
        m.save_model(str(OUT / f"model_{a.tag}_{a.kind}_{a.period}_{'vq'[i]}.txt"))
    (OUT / f"names_{a.tag}.json").write_text(json.dumps(names))
    print("saved", f, "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
