"""phys5 (C helper, 2026-10-05): same-split recurring-profile features (leave-one-day-out over the split's other days,
released observed cells only; Task 1 is offline, so other days of the same month are legal inputs).
Per station x step: flow profile P_q (mean over other days of observed free log q minus its station-day mean), speed
profile P_v (mean of observed log v minus station-day mean of free log v... all states), congestion share P_c (share of
other days with observed v < v_cut). Features at a target cell relative to the nearest observed same-station cells:
pq_dev = (P_q(t) - mean(P_q(t_b), P_q(t_a))) - (lq(t) - mean(lq_obs(t_b), lq_obs(t_a))) style, curvatures, P_c.
Saves artifacts/claude_ref/phys5/feat2_<pn>.npz aligned with feat_<pn>.npz rows.
Usage: python phys5_feat2.py [panels]"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import phys4_common as C
import phys5_diag as DG

OUT = DG.OUT
NAMES2 = ["pq_curv", "pq_nbr", "pq_tb", "pq_ta", "pv_curv", "pv_nbr", "pv_tb", "pv_ta", "pc_t", "pc_nbr"]


def loo_profile(x):
    """x [D, T, S] with nan; leave-one-day-out mean over days."""
    S_ = np.nansum(x, 0); N_ = np.isfinite(x).sum(0)
    own = np.isfinite(x)
    return np.where(N_[None] - own > 0, (S_[None] - np.nan_to_num(x)) / np.maximum(N_[None] - own, 1), np.nan)


def feats(held, tv, tq, v, q, ix, Xa, names):
    T = np.asarray(held.target, bool); el = np.asarray(held.elig, bool); vf = C.consts(held)["vf"]
    vc = held.stations.sort_values("idx").v_cut.to_numpy(float)
    ob = el & ~T & np.isfinite(tv) & np.isfinite(tq) & (tq > 0) & (tv > 0)
    xq = np.log(np.where(ob & (tv > .9 * vf), tq, np.nan)); xq = xq - np.nanmean(xq, axis=1, keepdims=True)
    xv = np.log(np.where(ob, tv, np.nan)); xv = xv - np.nanmean(np.where(tv > .9 * vf, xv, np.nan), axis=1, keepdims=True)
    xc = np.where(ob, (tv < vc).astype(float), np.nan)
    Pq, Pv, Pc = loo_profile(xq), loo_profile(xv), loo_profile(xc)
    d, t, s = ix.T.astype(int); Tn = T.shape[1]
    nm = [str(x) for x in names]
    db = Xa[:, nm.index("db")].astype(int); da = Xa[:, nm.index("da")].astype(int)
    tb = np.clip(t - db, 0, Tn - 1); ta = np.clip(t + da, 0, Tn - 1); hb = db < 99; ha = da < 99
    out = []
    for P, key in ((Pq, "q"), (Pv, "v")):
        pt = P[d, t, s]; pb = np.where(hb, P[d, tb, s], np.nan); pa = np.where(ha, P[d, ta, s], np.nan)
        prev = P[d, np.clip(t - 1, 0, Tn - 1), s]; nxt = P[d, np.clip(t + 1, 0, Tn - 1), s]
        curv = pt - .5 * (prev + nxt)
        xb = Xa[:, nm.index(f"{key}_b")]; xa = Xa[:, nm.index(f"{key}_a")]     # observed neighbour minus C14 at the cell
        nbr = (pt - np.nanmean(np.stack([pb, pa]), 0)) + np.nanmean(np.stack([xb, xa]), 0)
        out += [curv, nbr, pt - pb, pt - pa]
    pct = Pc[d, t, s]; pcn = np.nanmean(np.stack([np.where(hb, Pc[d, tb, s], np.nan), np.where(ha, Pc[d, ta, s], np.nan)]), 0)
    out += [pct, pct - pcn]
    return np.stack(out, 1).astype(np.float32)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    for pn in sys.argv[1:] or C.PANELS:
        f = OUT / f"feat2_{pn}.npz"
        if f.exists():
            continue
        t0 = time.time()
        held, tv, tq, v, q, p = DG.load(pn)
        z = np.load(OUT / f"feat_{pn}.npz")
        X2 = feats(held, tv, tq, v, q, z["ix"], z["X"], z["names"])
        np.savez(f, X=X2, names=np.array(NAMES2))
        print(pn, X2.shape, round(time.time() - t0), "s", flush=True)
