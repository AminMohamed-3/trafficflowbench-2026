"""Claude helper 2026-10-04 (ledger 0389, idea S2): features for a learned stage 2 on top of the shipped Task 1 stack.
Inputs per cell: F4 (pv, pq); 0360 speed rv, common shock gr and its support ncm; 0358 kriged log flow N and support den;
0368 hedge p (nan off candidates) and hedged speed hv; 0358 flow on hv (hq); phys3 output (sv, sq) = the stack.
Observed = elig & ~target & ~gap & finite (separately for speed and flow) of the blinded view. Relative features (to the stack)
are used where possible so that level shifts between splits do not enter.
Branch distance br = log hq - log q_FD,cong(hv) (published FD constants, as phys3_proj)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import state3_feat as F
import phys3_proj as PJ

BIG = 99.
FEATS = ["vS", "vS_cut", "qSl", "q_cap", "dv_cm", "gr", "ncm", "hp", "dv_h", "dlq_k", "dN", "den", "dv_p", "dlq_p", "br",
         "dvb", "dva", "db", "da", "dlqb", "dlqa", "dvu", "dvd", "du", "dd", "dlqu", "dlqd", "vcut", "lanes", "tod"]


def build(speed, flow, elig, target, gap, stations, pv, pq, rv, gr, ncm, N, den, hp, hv, hq, sv, sq, mask):
    st = stations.sort_values("idx")
    vcut = st.v_cut.to_numpy(float); cap = st.capacity_vph.to_numpy(float); mp = st.milepost.to_numpy(float)
    lanes = np.maximum(st.lanes.to_numpy(float), 1.)
    D, T, L = speed.shape
    di = np.arange(D)[:, None, None]; ti = np.arange(T)[None, :, None]; li = np.arange(L)[None, None, :]
    base = np.asarray(elig, bool) & ~np.asarray(target, bool) & ~np.asarray(gap, bool)
    ov = base & np.isfinite(speed); oq = base & np.isfinite(flow) & (flow > 0)
    v = np.where(ov, speed, np.nan); lq = np.where(oq, np.log(np.maximum(np.nan_to_num(flow), 1.)), np.nan)
    lsq = np.log(np.maximum(sq, 1.))
    cols = {}
    for name, obs, val, ref in (("v", ov, v, sv), ("lq", oq, lq, lsq)):
        ib, ia = F.nearest_axis(obs, 1)
        b = np.where(ib >= 0, val[di, np.maximum(ib, 0), li], np.nan) - ref
        a = np.where(ia >= 0, val[di, np.maximum(ia, 0), li], np.nan) - ref
        su, sd = F.nearest_axis(obs, 2)
        u = np.where(su >= 0, val[di, ti, np.maximum(su, 0)], np.nan) - ref
        d = np.where(sd >= 0, val[di, ti, np.maximum(sd, 0)], np.nan) - ref
        if name == "v":
            cols.update(dvb=b, dva=a, dvu=u, dvd=d,
                        db=np.where(ib >= 0, ti - ib, BIG), da=np.where(ia >= 0, ia - ti, BIG),
                        du=np.where(su >= 0, np.abs(mp[np.maximum(su, 0)] - mp[None, None, :]), BIG),
                        dd=np.where(sd >= 0, np.abs(mp[np.maximum(sd, 0)] - mp[None, None, :]), BIG))
        else:
            cols.update(dlqb=b, dlqa=a, dlqu=u, dlqd=d)
    w, kj, _, _ = PJ._consts(stations)
    qfd = np.maximum(hv, 1.) * w * kj / (np.maximum(hv, 1.) + w)
    lpq = np.log(np.maximum(pq, 1.))
    cols.update(vS=sv, vS_cut=sv - vcut, qSl=sq / lanes, q_cap=sq / cap, dv_cm=rv - pv, gr=np.broadcast_to(gr[:, :, None], sv.shape),
                ncm=np.broadcast_to(ncm[:, :, None].astype(float), sv.shape), hp=hp, dv_h=hv - rv,
                dlq_k=np.log(np.maximum(hq, 1.)) - lpq, dN=N - lpq, den=den, dv_p=sv - hv, dlq_p=lsq - np.log(np.maximum(hq, 1.)),
                br=np.log(np.maximum(hq, 1.)) - np.log(qfd), vcut=np.broadcast_to(vcut, sv.shape), lanes=np.broadcast_to(lanes, sv.shape),
                tod=np.broadcast_to(ti / T, sv.shape))
    idx = np.nonzero(mask)
    X = np.stack([np.asarray(np.broadcast_to(cols[f], sv.shape))[idx] for f in FEATS], 1).astype(np.float32)
    return X, np.stack(idx, 1).astype(np.int32)
