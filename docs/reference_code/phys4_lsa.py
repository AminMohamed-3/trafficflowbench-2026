"""Claude helper phys4 (2026-10-04): LSA = local same-time spatial average estimator of log flow (and log speed).
Model on free-flow observed cells: x(d,t,s) = R_s(d,t) + c_s(d,t) + e, e iid per cell (sigma_q ~ .022-.028 from clean
adjacent-station pairs, ramps add nothing at 5-min resolution). R_s(d,t): the station's day-specific relative level, smooth
in time (exp kernel scale K steps, two-sided, on x - c). c_s(d,t): local same-time common term = weighted mean over observed
stations j != s within W indices of (x_j(t) - R_j(d,t)) (+ optional gamma weight on t-1, t+1). Alternating updates (5 its).
Differs from 0358 kriging: day-specific smooth levels (0358: 30-min bins pooled over days + corridor-wide common) and a wide
local window (0358: exp kernel ell .5 km, nearest 1-2 stations). Split-local, uses released observations only."""
import numpy as np
from scipy.ndimage import convolve1d


def tsmooth(x, w, K):
    kern = np.exp(-np.abs(np.arange(-3 * K, 3 * K + 1)) / K)
    num = convolve1d(np.where(w, x, 0), kern, axis=1, mode="constant"); den = convolve1d(w.astype(float), kern, axis=1, mode="constant")
    return num / np.maximum(den, 1e-9), den


def ssum(x, w, W, shape="flat", gamma=0., scale=None):
    if shape == "flat":
        kern = np.ones(2 * W + 1)
    else:
        kern = np.exp(-np.abs(np.arange(-W, W + 1)) / (scale or W / 2))
    kern[W] = 0
    xw = np.where(w, x, 0); wf = w.astype(float)
    num = convolve1d(xw, kern, axis=2, mode="constant"); den = convolve1d(wf, kern, axis=2, mode="constant")
    if gamma > 0:
        k2 = kern.copy(); k2[W] = 1
        for s in (1, -1):
            xs = np.roll(xw, s, axis=1); ws = np.roll(wf, s, axis=1)
            if s > 0: xs[:, :s] = 0; ws[:, :s] = 0
            else: xs[:, s:] = 0; ws[:, s:] = 0
            num += gamma * convolve1d(xs, k2, axis=2, mode="constant"); den += gamma * convolve1d(ws, k2, axis=2, mode="constant")
    return num, den


def ssum_km(x, w, M):
    """M [S, S] station weight matrix (zero diagonal)."""
    return np.einsum("dts,sr->dtr", np.where(w, x, 0), M), np.einsum("dts,sr->dtr", w.astype(float), M)


def lsa(X, ob, W=12, K=72, shape="flat", gamma=0., iters=5, scale=None, mp=None, km=None):
    if km is not None:  # flat window in km (|milepost difference| <= km), overrides W
        M = (np.abs(mp[:, None] - mp[None, :]) <= km).astype(float); np.fill_diagonal(M, 0.)
        sfun = lambda x, w: ssum_km(x, w, M)
    else:
        sfun = lambda x, w: ssum(x, w, W, shape, gamma, scale)
    """X: log values [D,T,S] (any where not ob); ob: cells used for estimation. Returns estimate [D,T,S], support n."""
    with np.errstate(all="ignore"):
        R = np.nanmean(np.where(ob, X, np.nan), axis=1, keepdims=True)
    R = np.broadcast_to(np.where(np.isfinite(R), R, np.nanmean(np.where(ob, X, np.nan))), X.shape).copy()
    for _ in range(iters):
        s, n = sfun(X - R, ob); cl = s / np.maximum(n, 1e-9)
        R, _ = tsmooth(X - cl, ob, K)
    s, n = sfun(X - R, ob); cl = s / np.maximum(n, 1e-9)
    _, rden = tsmooth(X, ob, K)
    return R + cl, np.where(rden > 1., n, 0.)  # support 0 where the station has (almost) no own observations near t
