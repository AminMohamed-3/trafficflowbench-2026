"""Claude helper 2026-10-04: spatial same-time kriging of log flow for Task 1 target cells, and a gated blend with F4.
Method (few parameters, physics: at 5-min resolution free-flow vehicles cross 8+ km, so same-time flow is near-conserved
along the corridor; temporal neighbours carry white noise and hurt):
  log q(d,t,s) = A_s(30-min bin) + c(d,t) + R(d,t,s); A, c by alternating means on released observed cells of the split
  (split-local, no labels); R at a target cell = sum_k w_k R_k / (sum_k w_k + lam) over observed stations within K
  indices at the same timestep, w = exp(-|dx_km|/ell).
Blend: q = (1-w) q_F4 + w exp(N), w = w_b(F4 flow-per-lane band) * den/(den+SHRINK), w = 0 where F4 speed < gate * v_cut.
"""
import numpy as np

ELL, LAM, K, BIN = .5, .05, 10, 6
BANDS = np.array([0, 450, 950, 1250, 1500, 1e9])
GATE = 1.15
SHRINK = .5  # repair 1 (2026-10-04): N weight scaled by den/(den+SHRINK), den = summed kernel weight of same-time observed neighbours


def positions(stations):
    s = stations.sort_values("idx")
    return s.milepost.to_numpy(float) + s.length_km.to_numpy(float) / 2


def krige_log_flow(flow_obs, x, ell=ELL, lam=LAM, k=K, b=BIN):
    """flow_obs: [D,T,S] released flow (nan = blank). Returns log-flow estimate N [D,T,S] (leave-cell-out at observed cells
    is not applied; only target cells are used downstream, and those are blank). Also returns den (kernel support)."""
    O = np.isfinite(flow_obs)
    Lo = np.where(O, np.log(np.maximum(np.nan_to_num(flow_obs), 1.)), 0.)
    D, NT, NS = Lo.shape
    bins = np.arange(NT) // b
    starts = np.arange(0, NT, b)
    A = np.zeros((bins.max() + 1, NS))
    for _ in range(8):
        r = np.where(O, Lo - A[bins][None], 0.)
        cc = r.sum(2) / np.maximum(O.sum(2), 1)
        r = np.where(O, Lo - cc[:, :, None], 0.)
        A = np.add.reduceat(r.sum(0), starts, axis=0) / np.maximum(np.add.reduceat(O.sum(0), starts, axis=0), 1)
    base = A[bins][None] + cc[:, :, None]
    R = np.where(O, Lo - base, 0.)
    num = np.zeros_like(R); den = np.zeros_like(R)
    s = np.arange(NS)
    for off in range(-k, k + 1):
        if off == 0:
            continue
        j = s + off; ok = (j >= 0) & (j < NS); jj = np.clip(j, 0, NS - 1)
        w = np.exp(-np.abs(x[jj] - x[s]) / ell) * ok
        num += w[None, None] * R[:, :, jj]
        den += w[None, None] * O[:, :, jj]
    return base + num / (den + lam), den


def band_index(pq, lanes):
    return np.clip(np.searchsorted(BANDS, pq / lanes, side="right") - 1, 0, len(BANDS) - 2)


def blend(pv, pq, N, target, stations, weights, gate=GATE, den=None, shrink=SHRINK):
    """weights: array len(BANDS)-1 of N-weights. Returns new flow (F4 kept off target and in gated cells)."""
    s = stations.sort_values("idx")
    lanes = s.lanes.to_numpy(float); vcut = s.v_cut.to_numpy(float); cap = s.capacity_vph.to_numpy(float)
    w = np.asarray(weights, float)[band_index(pq, lanes)]
    w = np.where(pv >= gate * vcut, w, 0.)
    if den is not None:
        w = w * den / (den + shrink)
    q = (1 - w) * pq + w * np.exp(N)
    q = np.clip(q, 50., 1.3 * cap)
    return np.where(target, q, pq)
