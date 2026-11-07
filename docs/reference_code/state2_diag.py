"""Claude helper (state2, ledger 0360) 2026-10-04: where F4 target speed error sits on OPT, and the common-mode headroom.
Classes per target cell from the nearest observed same-station time neighbours (before b, after a):
  free / cong (both within 3 steps, same class), front (both within 3, class differs), nob (no observed neighbour within 3 on a side).
Common mode g(d,t) = mean over observed free cells of (v - station free constant) at the same step (split-local constants from
observed cells of the held dates). Reports OLS coefficient of F4 free-cell residual on g (2-fold by day parity).
Usage: python docs/reference_code/state2_diag.py [panel ...]"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phys_lwr_diag as P

OUT = P.ROOT / "artifacts/claude_ref/state2"
K = 3


def nearest_obs(obs):
    """Distance (steps) to the nearest observed cell before / after along time, per (day, t, station); large if none."""
    D, T, L = obs.shape
    big = 999
    db = np.full(obs.shape, big, np.int32); da = np.full(obs.shape, big, np.int32)
    ib = np.full(obs.shape, -1, np.int32); ia = np.full(obs.shape, -1, np.int32)
    last = np.full((D, L), -1, np.int32)
    for t in range(T):
        ok = last >= 0
        db[:, t] = np.where(ok, t - last, big); ib[:, t] = last
        last = np.where(obs[:, t], t, last)
    nxt = np.full((D, L), -1, np.int32)
    for t in range(T - 1, -1, -1):
        ok = nxt >= 0
        da[:, t] = np.where(ok, nxt - t, big); ia[:, t] = nxt
        nxt = np.where(obs[:, t], t, nxt)
    return db, da, ib, ia


def classes(held, vobs, obs):
    vcut = held.stations.sort_values("idx").v_cut.to_numpy(float)
    db, da, ib, ia = nearest_obs(obs)
    D, T, L = obs.shape
    di = np.arange(D)[:, None, None]; li = np.arange(L)[None, None, :]
    vb = np.where(ib >= 0, vobs[di, np.maximum(ib, 0), li], np.nan)
    va = np.where(ia >= 0, vobs[di, np.maximum(ia, 0), li], np.nan)
    close = (db <= K) & (da <= K)
    fb = vb >= vcut; fa = va >= vcut
    cls = np.full(obs.shape, "nob", dtype="<U5")
    cls[close & fb & fa] = "free"; cls[close & ~fb & ~fa] = "cong"; cls[close & (fb != fa)] = "front"
    return cls


def common_mode(held, vobs, obs):
    st = held.stations.sort_values("idx")
    vcut = st.v_cut.to_numpy(float)
    free = obs & (vobs >= vcut + 5)
    c = np.array([np.median(vobs[:, :, l][free[:, :, l]]) if free[:, :, l].sum() > 20 else np.nan for l in range(vobs.shape[2])])
    r = np.where(free & np.isfinite(c), vobs - c, np.nan)
    n = np.isfinite(r).sum(2)
    g = np.where(n > 0, np.nansum(r, 2) / np.maximum(n, 1), 0.)
    return g, n, c


def main(*panels):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for panel in panels or P.P["panels"]:
        held, tv, tq, pv, pq = P.held_panel(panel)
        tgt = np.asarray(held.target, bool)
        vobs = np.asarray(held.speed, float)
        obs = np.asarray(held.elig, bool) & ~tgt & np.isfinite(vobs)
        cls = classes(held, vobs, obs)
        m = tgt & np.isfinite(tv)
        e2 = (pv - tv) ** 2
        tot = e2[m].sum()
        res = dict(panel=panel, n=int(m.sum()), rmse=float(np.sqrt(e2[m].mean())))
        for c in ("free", "cong", "front", "nob"):
            mc = m & (cls == c)
            res[c] = dict(share=round(float(mc.sum() / m.sum()), 4), rmse=round(float(np.sqrt(e2[mc].mean())), 3),
                          mse_share=round(float(e2[mc].sum() / tot), 4))
        # common mode on cells F4 predicts free
        g, n, c = common_mode(held, vobs, obs)
        vcut = held.stations.sort_values("idx").v_cut.to_numpy(float)
        G = np.broadcast_to(g[:, :, None], pv.shape)
        fm = m & (pv >= vcut + 5) & (n[:, :, None] >= 5)
        resid = (tv - pv)
        D = pv.shape[0]
        par = (np.arange(D) % 2)[:, None, None] * np.ones_like(pv, dtype=int)
        newv = pv.copy(); coefs = []
        for f in (0, 1):
            tr = fm & (par != f)
            b = float((resid[tr] * G[tr]).sum() / (G[tr] ** 2).sum()); coefs.append(b)
            ap = fm & (par == f)
            newv[ap] = pv[ap] + b * G[ap]
        res["cm"] = dict(coef=[round(x, 3) for x in coefs], share=round(float(fm.sum() / m.sum()), 4),
                         rmse_free_F4=round(float(np.sqrt(e2[fm].mean())), 4),
                         rmse_free_new=round(float(np.sqrt(((newv - tv) ** 2)[fm].mean())), 4),
                         corr=round(float(np.corrcoef(resid[fm], G[fm])[0, 1]), 3))
        ref = P.score(held, tv, tq, pv, pq); new = P.score(held, tv, tq, newv, pq)
        res["cm"]["dS_state"] = round(new["S_state"] - ref["S_state"], 5); res["cm"]["dJ"] = round(new["J"] - ref["J"], 5)
        res["F4"] = ref
        print(json.dumps(res), flush=True)
        rows.append(res)
    (OUT / "diag_classes.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
