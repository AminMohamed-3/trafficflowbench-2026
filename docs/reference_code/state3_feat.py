"""Claude helper (state3, ledger 0368) 2026-10-04: features for a congestion-probability model on target speed cells.
All features come from the blinded view (observed cells only) plus the reference prediction (F4 + 0360 common-mode fix).
observed = elig & ~target & ~gap & finite(speed). Stations are ordered by idx (= order_index, milepost increasing).
Usage: python docs/reference_code/state3_feat.py            (builds OPT caches artifacts/claude_ref/state3/opt_<panel>.npz)"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phys_lwr_diag as P
import state2_cm as CM

OUT = P.ROOT / "artifacts/claude_ref/state3"
BIG = 99
FEATS = ["vr", "vr_cut", "q_cap", "vcut", "tod", "db", "da", "vb", "va", "vb_cut", "va_cut",
         "du", "dd", "vu", "vd", "vu_cut", "vd_cut", "dtu", "dtd", "vr_m1", "vr_p1", "nobs_t", "cong_share_t",
         "vb2", "va2", "db2", "da2"]


def nearest_axis(obs, axis):
    """Index of nearest observed cell before / after along axis (-1 if none). obs: (D, T, L) bool."""
    o = np.moveaxis(obs, axis, 0)
    n = o.shape[0]
    ib = np.full(o.shape, -1, np.int32); ia = np.full(o.shape, -1, np.int32)
    last = np.full(o.shape[1:], -1, np.int32)
    for i in range(n):
        ib[i] = last; last = np.where(o[i], i, last)
    nxt = np.full(o.shape[1:], -1, np.int32)
    for i in range(n - 1, -1, -1):
        ia[i] = nxt; nxt = np.where(o[i], i, nxt)
    return np.moveaxis(ib, 0, axis), np.moveaxis(ia, 0, axis)


def second_obs(obs, ib, ia):
    """Second nearest observed before / after along time (axis 1)."""
    D, T, L = obs.shape
    di = np.arange(D)[:, None, None]; li = np.arange(L)[None, None, :]
    ib2 = np.where(ib >= 0, ib[di, np.maximum(ib, 0), li], -1)  # nearest before the previous observation
    ia2 = np.where(ia >= 0, ia[di, np.maximum(ia, 0), li], -1)
    return ib2, ia2


def build(speed, observed, stations, vr, pq, mask):
    """Feature matrix (n, len(FEATS)) for cells in mask, plus their (d, t, l) indices."""
    st = stations.sort_values("idx")
    vcut = st.v_cut.to_numpy(float); cap = st.capacity_vph.to_numpy(float); mp = st.milepost.to_numpy(float)
    D, T, L = speed.shape
    di = np.arange(D)[:, None, None]; ti = np.arange(T)[None, :, None]; li = np.arange(L)[None, None, :]
    v = np.where(observed, speed, np.nan)
    ib, ia = nearest_axis(observed, 1)
    ib2, ia2 = second_obs(observed, ib, ia)

    def tv_at(idx):
        return np.where(idx >= 0, v[di, np.maximum(idx, 0), li], np.nan)

    def tdist(idx, before):
        return np.where(idx >= 0, (ti - idx) if before else (idx - ti), BIG).astype(float)

    # spatial neighbours at the same step (nearest observed station up / downstream)
    su, sd = nearest_axis(observed, 2)

    def sv_at(idx):
        return np.where(idx >= 0, v[di, ti, np.maximum(idx, 0)], np.nan)

    def sdist(idx):
        return np.where(idx >= 0, np.abs(mp[np.maximum(idx, 0)] - mp[None, None, :]), BIG).astype(float)

    vu, vd = sv_at(su), sv_at(sd)
    vcut_u = np.where(su >= 0, vcut[np.maximum(su, 0)], np.nan); vcut_d = np.where(sd >= 0, vcut[np.maximum(sd, 0)], np.nan)
    # nearest time distance to a congested observed reading at upstream / downstream neighbour is costly; use class at t instead
    nobs = observed.sum(2, keepdims=True).astype(float)
    cong_obs = observed & (v < vcut[None, None, :])
    cshare = np.where(nobs > 0, cong_obs.sum(2, keepdims=True) / np.maximum(nobs, 1), np.nan)
    vr_m1 = np.concatenate([vr[:, :1], vr[:, :-1]], 1); vr_p1 = np.concatenate([vr[:, 1:], vr[:, -1:]], 1)
    vb, va = tv_at(ib), tv_at(ia)
    cols = dict(
        vr=vr, vr_cut=vr - vcut, q_cap=pq / cap, vcut=np.broadcast_to(vcut, vr.shape), tod=np.broadcast_to(ti, vr.shape) / T,
        db=tdist(ib, True), da=tdist(ia, False), vb=vb, va=va, vb_cut=vb - vcut, va_cut=va - vcut,
        du=sdist(su), dd=sdist(sd), vu=vu, vd=vd, vu_cut=vu - vcut_u, vd_cut=vd - vcut_d,
        dtu=np.zeros_like(vr), dtd=np.zeros_like(vr), vr_m1=vr_m1, vr_p1=vr_p1,
        nobs_t=np.broadcast_to(nobs, vr.shape), cong_share_t=np.broadcast_to(cshare, vr.shape),
        vb2=tv_at(ib2), va2=tv_at(ia2), db2=tdist(ib2, True), da2=tdist(ia2, False))
    # dtu / dtd: steps to the nearest congested observed cell at the nearest upstream / downstream station (+-12 steps), else BIG
    for name, sidx in (("dtu", su), ("dtd", sd)):
        out = np.full(vr.shape, BIG, float)
        for k in range(12, -1, -1):
            for s in ((k, -k) if k else (0,)):
                tt = np.clip(ti + s, 0, T - 1)
                c = (sidx >= 0) & cong_obs[di, tt, np.maximum(sidx, 0)] & (ti + s >= 0) & (ti + s < T)
                out = np.where(c, k, out)
        cols[name] = out
    idx = np.nonzero(mask)
    X = np.stack([np.asarray(cols[f])[idx] for f in FEATS], 1).astype(np.float32)
    return X, np.stack(idx, 1).astype(np.int32)


def ref_opt(panel, held, pv):
    """Reference speed on OPT: F4 + 0360 common-mode fix with the leave-panel-out, other-parity coefficients of state2."""
    z = np.load(P.ROOT / "artifacts/claude_ref/state2" / f"opt_{panel}_cm.npz"); gr = z["gr"].astype(float)
    import pandas as pd
    row = pd.read_csv(P.ROOT / "artifacts/claude_ref/state2/opt_screen.csv").set_index("panel").loc[panel]
    coefs = [json.loads(row.coef_fold0), json.loads(row.coef_fold1)]
    vcut = held.stations.sort_values("idx").v_cut.to_numpy(float)
    par = np.arange(pv.shape[0]) % 2; vr = pv.copy()
    for f in (0, 1):
        s = par == f
        vr[s] = CM.apply(pv[s], gr[s], vcut, coefs[f])
    return vr


def main(*panels):
    OUT.mkdir(parents=True, exist_ok=True)
    for panel in panels or P.P["panels"]:
        held, tv, tq, pv, pq = P.held_panel(panel)
        tgt = np.asarray(held.target, bool); sp = np.asarray(held.speed, float)
        observed = np.asarray(held.elig, bool) & ~tgt & ~np.asarray(held.gap, bool) & np.isfinite(sp)
        vr = ref_opt(panel, held, pv)
        m = tgt & np.isfinite(tv)
        X, ix = build(sp, observed, held.stations, vr, pq, m)
        y = tv[m].astype(np.float32)
        vcut = held.stations.sort_values("idx").v_cut.to_numpy(float)
        np.savez(OUT / f"opt_{panel}.npz", X=X, ix=ix, y=y, vcut=vcut[ix[:, 2]].astype(np.float32))
        mis = (vr[m] >= vcut[ix[:, 2]]) != (y >= vcut[ix[:, 2]])
        print(panel, X.shape, "mis share", round(float(mis.mean()), 5), "mis mse share",
              round(float(((vr[m] - y) ** 2)[mis].sum() / ((vr[m] - y) ** 2).sum()), 3), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
