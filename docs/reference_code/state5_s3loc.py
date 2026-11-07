"""Claude helper 2026-10-04 (state5): feasibility test of a SPLIT-LOCAL stage 3 on the final stack (test-time residual learning on
the scored month's own observed cells). On I-S-OWN, the hidden OWN cells are released observations of that month, so a residual
model fitted on them is available for the official file too (fit on all OWN cells of the month, apply to official targets).
Test: LightGBM residual model per output (y_v = tv - v, y_q = (tq - q) / lanes) on non-block OWN targets, fitted on one day parity
and applied to the other (cross-fit), shrunk by a factor k; pooled over panels (panel one-hot not used; station-level features
are relative). Features: stack v/q, F4, S08 (if given), imputer, nearest observed same-station speed/flow before/after and
distance, upstream/downstream observed same-step speed/flow, time of day, weekday, v_cut, capacity, lanes, v - v_cut.
Kill rule: cross-fit nb gain < +.0002 S_state or CI touching 0.
Usage: python docs/reference_code/state5_s3loc.py <split validation|private> <npz pattern with {pn}> [vkey qkey]"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json
from pathlib import Path
from datetime import date
import numpy as np
import lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import state5_lib as L
import state3_feat as F3

PRM = dict(objective="regression", learning_rate=.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=.8, bagging_fraction=.8,
           bagging_freq=1, lambda_l2=10., num_threads=3, verbose=-1, seed=0)
ROUNDS = 300


def nearest(a, axis_t=True):
    """For each cell: nearest observed value before/after along time (axis 1) and distance (capped 24)."""
    D, T, S = a.shape; ok = np.isfinite(a)
    vb = np.full(a.shape, np.nan); db = np.full(a.shape, 24.); va = vb.copy(); da = db.copy()
    last = np.full((D, S), np.nan); lt = np.full((D, S), -999)
    for t in range(T):
        vb[:, t] = last; db[:, t] = np.minimum(t - lt, 24)
        last = np.where(ok[:, t], a[:, t], last); lt = np.where(ok[:, t], t, lt)
    nxt = np.full((D, S), np.nan); nt = np.full((D, S), 10 ** 6)
    for t in range(T - 1, -1, -1):
        va[:, t] = nxt; da[:, t] = np.minimum(nt - t, 24)
        nxt = np.where(ok[:, t], a[:, t], nxt); nt = np.where(ok[:, t], t, nt)
    return vb, db, va, da


def spatial(a):
    """Nearest observed upstream/downstream station at the same step (by idx order): value and index distance (cap 10)."""
    D, T, S = a.shape; ok = np.isfinite(a)
    up = np.full(a.shape, np.nan); du = np.full(a.shape, 10.); dn = up.copy(); dd = du.copy()
    last = np.full((D, T), np.nan); ls = np.full((D, T), -999)
    for s in range(S):
        up[:, :, s] = last; du[:, :, s] = np.minimum(s - ls, 10)
        last = np.where(ok[:, :, s], a[:, :, s], last); ls = np.where(ok[:, :, s], s, ls)
    nxt = np.full((D, T), np.nan); ns = np.full((D, T), 10 ** 6)
    for s in range(S - 1, -1, -1):
        dn[:, :, s] = nxt; dd[:, :, s] = np.minimum(ns - s, 10)
        nxt = np.where(ok[:, :, s], a[:, :, s], nxt); ns = np.where(ok[:, :, s], s, ns)
    return up, du, dn, dd


def feats(z, v, q, la, vc, cap, extra):
    D, T, S = v.shape
    vo, qo = z["vobs"].astype(float), z["qobs"].astype(float)
    vb, db, va, da = nearest(vo); qb, _, qa, _ = nearest(qo)
    vu, du, vd, dd = spatial(vo); qu, _, qd, _ = spatial(qo)
    tod = np.broadcast_to(np.arange(T)[None, :, None], v.shape)
    wd = np.array([date.fromisoformat(str(x)).weekday() for x in z["dates"]])
    W = np.broadcast_to(wd[:, None, None], v.shape)
    cols = dict(v=v, q=q / la, dvc=v - vc, vc=np.broadcast_to(vc, v.shape), cap=np.broadcast_to(cap / la, v.shape),
                la=np.broadcast_to(la, v.shape), tod=tod, wd=W,
                vb=vb - v, db=db, va=va - v, da=da, qb=(qb - q) / la, qa=(qa - q) / la,
                vu=vu - v, du=du, vd=vd - v, dd=dd, qu=(qu - q) / la, qd=(qd - q) / la)
    for k, a in extra.items():
        cols[k] = a
    return cols


def main(split, pat, vk="v13", qk="q13"):
    sh = L.share_array(split)
    D0 = None; ref = {}; new = {k: {} for k in (.5, 1., "1P")}; data = {}
    for pn in L.PANELS:
        z = dict(np.load(L.OUT / pat.format(pn=pn)))
        la, vc, cap, _ = L.lanes_of(pn, split)
        v, q = z[vk].astype(float), z[qk].astype(float)
        extra = {"f4v": z["f4v"].astype(float) - v, "f4q": (z["f4q"].astype(float) - q) / la}
        cols = feats(z, v, q, la, vc, cap, extra)
        tv, tq = z["tv"].astype(float), z["tq"].astype(float)
        m = z["target"] & ~z["gap"] & np.isfinite(tv) & np.isfinite(tq)
        X = np.stack([np.asarray(a, float)[m] for a in cols.values()], 1).astype(np.float32)
        ix = np.argwhere(m)
        data[pn] = (z, v, q, la, X, ix, (tv - v)[m], ((tq - q) / la)[m])
        print(pn, X.shape, flush=True)
    names = list(cols)
    D = len(data[L.PANELS[0]][0]["dates"]); par = np.arange(D) % 2
    pred = {pn: (np.zeros(len(d[6])), np.zeros(len(d[6]))) for pn, d in data.items()}
    imp = {}
    for f in (0, 1):
        Xt = np.concatenate([d[4][par[d[5][:, 0]] != f] for d in data.values()])
        for j, k in ((0, 6), (1, 7)):
            y = np.concatenate([d[k][par[d[5][:, 0]] != f] for d in data.values()])
            mdl = lgb.train(PRM, lgb.Dataset(Xt, y), ROUNDS)
            imp.setdefault(j, np.zeros(len(names)))
            imp[j] += mdl.feature_importance("gain")
            for pn, d in data.items():
                sel = par[d[5][:, 0]] == f
                pred[pn][j][sel] = mdl.predict(d[4][sel])
        print("fold", f, "fitted", flush=True)
    for pn, (z, v, q, la, X, ix, yv, yq) in data.items():
        ref[pn] = L.stats(pn, z, v, q, la)
        I = (ix[:, 0], ix[:, 1], ix[:, 2])
        for k in new:
            kk = 1. if k == "1P" else k
            nv = v.copy(); nq = q.copy()
            nv[I] = np.maximum(v[I] + kk * pred[pn][0], 3.); nq[I] = np.clip(q[I] + kk * la[ix[:, 2]] * pred[pn][1], 50., 1.3 * L.lanes_of(pn, split)[2][ix[:, 2]])
            if k == "1P":  # then the C13 final soft phys3 + TV on the corrected values
                import polish_eval as E, research2_tv as TVM
                stn = L.lanes_of(pn, split)[3]
                c2 = E.Ctx(stn, z["target"], z["vobs"].astype(float), None, nv, nq, z["gap"])
                nv, nq = E.project_soft(c2, dict(r=3., a=.9, g=1., h=.1))
                nv, nq = TVM.smooth(nv, nq, z["target"] & ~z["gap"], stn, .004, 0.)
            new[k][pn] = L.stats(pn, z, nv, nq, la)
    res = {str(k): L.compare(ref, new[k], sh) for k in new}
    tag0 = Path(pat).stem.replace("{pn}", "").strip("_")
    if os.environ.get("S3L_SAVE"): np.savez_compressed(L.OUT / f"s3loc_pred_{split}_{tag0}.npz", **{f"{pn}_{a}": b for pn, d in data.items() for a, b in
                        (("ix", d[5].astype(np.int16)), ("pv", pred[pn][0].astype(np.float32)), ("pq", pred[pn][1].astype(np.float32)),
                         ("db", d[4][:, names.index("db")]), ("da", d[4][:, names.index("da")]))})
    res["importance_v"] = dict(zip(names, np.round(imp[0] / imp[0].sum(), 3).tolist()))
    res["importance_q"] = dict(zip(names, np.round(imp[1] / imp[1].sum(), 3).tolist()))
    print(json.dumps(res, indent=0), flush=True)
    tag = Path(pat).stem.replace("{pn}", "").strip("_")
    (L.OUT / f"s3loc_{split}_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
