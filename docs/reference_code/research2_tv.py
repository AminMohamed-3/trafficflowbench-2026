"""Research2 helper (2026-10-04): L1 (total-variation) density smoothing inside runs of consecutive Task 1 target cells,
on top of our S2 stack (F4 + 0360 -> 0368 -> 0358 -> phys3), OPT replica (train held dates, local truth-projected S_LWR).
Idea from a public competitor log (Hisernberg kraggle-3-in-one trafficflow/docs/T3_SMOOTHING.md, board +.0006); own code.
Per station-day, maximal runs of consecutive target cells (length 2..17; >= 18 treated as gap blocks, untouched):
  x = argmin 1/2 |x - k|^2 + lam sum |x_{i+1} - x_i|, lam = tau * mean(k_run), k = q / v (per station).
Per cell: tau_free if v >= v_cut else tau_gate (each run solved with both). r = x / k clipped |log r| <= .5;
free cells: q *= r (speed unchanged); gate cells: v *= r^-a, q *= r^(1-a), a = .75.
Tier 0 screen. Usage: python docs/reference_code/research2_tv.py [panel ...]"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phys_lwr_diag as P
import flow_screen as FS
import phys3_proj as PJ
import phys3_screen as S3
import phys3_stack as ST

OUT = P.ROOT / "artifacts/claude_ref/research2"
MAXRUN = 18


def runs(T):
    """T [D,T,S] bool -> list of (d, s, start, length) for runs along time."""
    D_, n, S = T.shape; out = []
    for d in range(D_):
        for s in range(S):
            x = T[d, :, s].astype(np.int8)
            dx = np.diff(np.concatenate([[0], x, [0]]))
            st = np.flatnonzero(dx == 1); en = np.flatnonzero(dx == -1)
            for a, b in zip(st, en):
                out.append((d, s, a, b - a))
    return out


def tv_batch(K, lam, iters=300):
    """K [m,n], lam [m] -> TV-denoised rows (dual projected gradient)."""
    m, n = K.shape
    if n < 2:
        return K.copy()
    u = np.zeros((m, n - 1)); L = lam[:, None]
    for _ in range(iters):
        DTu = np.zeros((m, n)); DTu[:, 1:] += u; DTu[:, :-1] -= u      # (D^T u)_j = u_{j-1} - u_j
        x = K - DTu
        u = np.clip(u + .25 * (x[:, 1:] - x[:, :-1]), -L, L)
    DTu = np.zeros((m, n)); DTu[:, 1:] += u; DTu[:, :-1] -= u
    return K - DTu


def smooth(v, q, T, stations, tau_free, tau_gate, a=.75):
    vcut = stations.sort_values("idx").v_cut.to_numpy(float)
    k = q / np.maximum(v, 1.)
    gate = v < vcut[None, None, :]
    newk = k.copy()
    R = [r for r in runs(T) if 2 <= r[3] < MAXRUN]
    by_len = {}
    for r in R:
        by_len.setdefault(r[3], []).append(r)
    for n, rs in by_len.items():
        d = np.array([r[0] for r in rs]); s = np.array([r[1] for r in rs]); t0 = np.array([r[2] for r in rs])
        tt = t0[:, None] + np.arange(n)[None, :]
        K = k[d[:, None], tt, s[:, None]]; G = gate[d[:, None], tt, s[:, None]]
        mk = K.mean(1)
        xf = tv_batch(K, tau_free * mk) if tau_free > 0 else K
        xg = tv_batch(K, tau_gate * mk) if tau_gate > 0 else K
        newk[d[:, None], tt, s[:, None]] = np.where(G, xg, xf)
    r = np.exp(np.clip(np.log(np.maximum(newk, 1e-9) / np.maximum(k, 1e-9)), -.5, .5))
    nv = np.where(gate, v * r ** (-a), v); nq = np.where(gate, q * r ** (1 - a), q * r)
    m = T & np.isfinite(nv) & np.isfinite(nq)
    return np.where(m, nv, v), np.where(m, nq, q)


GRID = [(0, 0), (.002, 0), (.004, 0), (.004, .005), (.004, .01), (.002, .005)]


def main(*panels):
    panels = panels or ("D12_I5_N", "D7_I10_W")
    D = len(P.P["OPT_held_dates"]); res = {}
    for panel in panels:
        held, tv, tq, pv, pq = P.held_panel(panel)
        T = np.asarray(held.target, bool)
        rv, rq = S3.reference(panel, held, pv, pq, D)
        z = np.load(P.ROOT / "artifacts/claude_ref/state3" / f"opt_pred_{panel}.npz"); ix = np.load(P.ROOT / "artifacts/claude_ref/state3" / f"opt_{panel}.npz")["ix"]
        hv = rv.copy(); hv[ix[:, 0], ix[:, 1], ix[:, 2]] = np.where(z["cand"], z["v"].astype(float), rv[ix[:, 0], ix[:, 1], ix[:, 2]])
        hq = ST.blend_folds(panel, held, hv, pq, D)
        sv, sq = PJ.project(hv, hq, T, held.stations, ST.R, ST.A, ST.G)[:2]
        res[panel] = {}
        base = None
        for tf, tg in GRID:
            nv, nq = smooth(sv, sq, T, held.stations, tf, tg) if (tf or tg) else (sv, sq)
            C, regs = FS.day_components(held, tv, tq, nv, nq)
            agg = FS.aggregate(C, regs, np.ones(D))
            if base is None:
                base = (C, regs, agg)
            # paired date bootstrap of J delta
            rng = np.random.default_rng(0); bs = []
            for _ in range(300):
                w = np.bincount(rng.integers(0, D, D), minlength=D).astype(float)
                bs.append(FS.aggregate(C, regs, w)["J"] - FS.aggregate(base[0], base[1], w)["J"])
            key = f"free{tf}_gate{tg}"
            res[panel][key] = dict(J=agg["J"], dJ=agg["J"] - base[2]["J"], ci=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                                   dS_state=agg["S_state"] - base[2]["S_state"], dS_LWR=agg["S_LWR"] - base[2]["S_LWR"], S_LWR=agg["S_LWR"])
            print(panel, key, {k: (round(x, 6) if isinstance(x, float) else [round(c, 6) for c in x]) for k, x in res[panel][key].items()}, flush=True)
    (OUT / "tv_opt.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
