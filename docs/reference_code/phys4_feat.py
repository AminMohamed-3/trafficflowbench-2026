"""Claude helper phys4 (2026-10-04): features for a density-targeted stage 3 on top of C13 (target cells only).
Field F = observed truth where observed, C13 elsewhere (log v, log q, log k = log q - log v). Features relative to C13 at the
cell: nearest observed before/after in time (value diff, distance), F at t-1/t+1 and t-2/t+2 minus C13, the cell's own C13
anomaly vs its time neighbours, spatial nearest observed up/down at t: their anomaly (x(t) - mean(x(t-1), x(t+1)) of F), the
corridor median anomaly of observed cells at t, context (v/vcut, q/cap, 0368 p, run length, position in run, link length,
lanes, time of day)."""
import numpy as np
import state3_feat as F

BIG = 99.
FEATS = []


def runinfo(T):
    D, n, S = T.shape; L = np.zeros(T.shape, np.float32); pos = np.zeros(T.shape, np.float32)
    for d in range(D):
        for s in range(S):
            x = T[d, :, s].astype(np.int8); dx = np.diff(np.concatenate([[0], x, [0]]))
            for a, b in zip(np.flatnonzero(dx == 1), np.flatnonzero(dx == -1)):
                L[d, a:b, s] = b - a
                pos[d, a:b, s] = np.minimum(np.arange(b - a), np.arange(b - a)[::-1])
    return L, pos


def shift(x, s, axis=1, fill=np.nan):
    out = np.full_like(x, fill)
    if axis == 1:
        if s > 0: out[:, s:] = x[:, :-s]
        else: out[:, :s] = x[:, -s:]
    else:
        if s > 0: out[:, :, s:] = x[:, :, :-s]
        else: out[:, :, :s] = x[:, :, -s:]
    return out


def build(obs_v, obs_q, T, v13, q13, p, stations, length, mask):
    """obs_v/obs_q: observed truth (nan elsewhere, target cells nan). Returns X [n, f], names, index [n, 3]."""
    st = stations.sort_values("idx")
    vcut = st.v_cut.to_numpy(float); cap = st.capacity_vph.to_numpy(float); mp = st.milepost.to_numpy(float)
    lanes = np.maximum(st.lanes.to_numpy(float), 1.)
    D, n, S = T.shape
    di = np.arange(D)[:, None, None]; ti = np.arange(n)[None, :, None]; li = np.arange(S)[None, None, :]
    ob = np.isfinite(obs_v) & np.isfinite(obs_q) & (obs_q > 0) & (obs_v > 0)
    lv13 = np.log(np.maximum(v13, 1.)); lq13 = np.log(np.maximum(q13, 1.))
    Fv = np.where(ob, np.log(np.maximum(np.nan_to_num(obs_v), 1.)), lv13)
    Fq = np.where(ob, np.log(np.maximum(np.nan_to_num(obs_q), 1.)), lq13)
    Fk = Fq - Fv
    cols = {}
    ib, ia = F.nearest_axis(ob, 1)
    su, sd = F.nearest_axis(ob, 2)
    for nm, Fx in (("v", Fv), ("q", Fq), ("k", Fk)):
        ref = Fx  # at target cells Fx is C13
        b = np.where(ib >= 0, Fx[di, np.maximum(ib, 0), li], np.nan) - ref
        a = np.where(ia >= 0, Fx[di, np.maximum(ia, 0), li], np.nan) - ref
        cols[f"{nm}_b"] = b; cols[f"{nm}_a"] = a
        for s in (1, 2):
            cols[f"{nm}_m{s}"] = shift(Fx, s) - ref; cols[f"{nm}_p{s}"] = shift(Fx, -s) - ref
        anom = Fx - .5 * (shift(Fx, 1) + shift(Fx, -1))
        oa = np.where(ob & shift(ob, 1, fill=False) & shift(ob, -1, fill=False), anom, np.nan)
        with np.errstate(all="ignore"):
            cm = np.nanmedian(oa, axis=2)
        cols[f"{nm}_cm"] = np.broadcast_to(cm[:, :, None], T.shape)
        cols[f"{nm}_self_anom"] = anom
        u = np.where(su >= 0, oa[di, ti, np.maximum(su, 0)], np.nan)
        d = np.where(sd >= 0, oa[di, ti, np.maximum(sd, 0)], np.nan)
        cols[f"{nm}_up_anom"] = u; cols[f"{nm}_dn_anom"] = d
        # spatial level diff of observed neighbour at t vs its own value at t-1 relative to ours
    cols["db"] = np.where(ib >= 0, ti - ib, BIG); cols["da"] = np.where(ia >= 0, ia - ti, BIG)
    cols["du"] = np.where(su >= 0, np.abs(mp[np.maximum(su, 0)] - mp[None, None, :]), BIG)
    cols["dd"] = np.where(sd >= 0, np.abs(mp[np.maximum(sd, 0)] - mp[None, None, :]), BIG)
    RL, pos = runinfo(np.asarray(T, bool))
    cols.update(v_cut=v13 / vcut, q_cap=q13 / cap, p=np.where(np.isfinite(p), p, -1.), run=RL, pos=pos,
                L=np.broadcast_to(length, T.shape), lanes=np.broadcast_to(lanes, T.shape), tod=np.broadcast_to(ti / n, T.shape),
                lk13=lq13 - lv13)
    names = list(cols)
    idx = np.nonzero(mask)
    X = np.stack([np.asarray(np.broadcast_to(cols[f], T.shape))[idx] for f in names], 1).astype(np.float32)
    return X, names, np.stack(idx, 1).astype(np.int32)
