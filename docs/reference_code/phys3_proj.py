"""Claude helper 2026-10-04 (ledger 0369): joint (v, q) error-weighted projection onto the congested FD branch.
Congested branch (triangular FD, station totals): q_FD(v) = v w kj / (v + w), w = cap / (kj - kc).
In log space x = log v, f(x) = log q_FD(e^x); slope s = w / (v + w). For a predicted pair (x0, y0) = (log v, log q) the
projection point x* minimises (x - x0)^2 + r^2 (f(x) - y0)^2 with r = sv / sq (error sd ratio of log speed to log flow):
r = 0 keeps v (q := q_FD(v), the 0362 move), r = inf keeps q (v := v_FD(q), the 0354 direction).
Then both move a fraction a of the way in log space: v' = exp(x0 + a (x* - x0)), q' = exp(y0 + a (f(x*) - y0)).
Applied on target cells with v < g v_cut, excluding gap cells (official 90-min gaps / OWN blocks) when gap is given.
Clips: projection point v* in [max(3, v e^-.5), max(v, min(v_cut, v e^.5))]; q' in [50, 1.3 cap].
OPT pick (ledger 0369): r = 3, a = .75, g = 1 (gap cells excluded on official/OWN)."""
import numpy as np

MAXSTEP = .5


def _consts(stations):
    s = stations.sort_values("idx")
    cap = s.capacity_vph.to_numpy(float); kj = s.k_jam.to_numpy(float); kc = s.critical_density.to_numpy(float)
    return cap / (kj - kc), kj, s.v_cut.to_numpy(float), cap


def project(pv, pq, target, stations, r, a, g, gap=None, iters=25):
    w, kj, vcut, cap = _consts(stations)
    m = np.asarray(target, bool) & (pv < g * vcut) & np.isfinite(pv) & np.isfinite(pq) & (pv > 0) & (pq > 0)
    if gap is not None:
        m &= ~np.asarray(gap, bool)
    if a == 0 or not m.any():
        return pv.copy(), pq.copy(), m
    W = np.broadcast_to(w, pv.shape)[m]; K = np.broadcast_to(kj, pv.shape)[m]; VC = np.broadcast_to(vcut, pv.shape)[m]
    x0 = np.log(pv[m]); y0 = np.log(pq[m])
    lo = np.minimum(np.maximum(np.log(3.), x0 - MAXSTEP), x0); hi = np.maximum(np.minimum(np.log(VC), x0 + MAXSTEP), x0)
    f = lambda x: x + np.log(W * K) - np.log(np.exp(x) + W)
    x = np.clip(x0, lo, hi)
    if r > 0:
        for _ in range(iters):
            v = np.exp(x); s = W / (v + W); e = f(x) - y0
            if np.isinf(r):
                dx = -e / s
            else:
                dx = -((x - x0) + r * r * s * e) / (1 + r * r * s * s)
            x = np.clip(x + np.clip(dx, -.2, .2), lo, hi)
    xs = x0 + a * (x - x0); ys = y0 + a * (f(x) - y0)
    nv = pv.copy(); nq = pq.copy()
    nv[m] = np.exp(xs)
    C = np.broadcast_to(cap, pv.shape)[m]
    nq[m] = np.clip(np.exp(ys), 50., 1.3 * C)
    return nv, nq, m
