"""gen: travel-time-aligned half-hour demand block statistics (generator rule: corridor demand steps at :00/:30 at the
upstream end and travels downstream at free-flow speed; inside a block a station's free-flow latent flow is ~constant).
block_stats(lq_obs, mp, vf_mean, delta) -> per cell: mean of OTHER observed clean cells of the same station-block, count."""
import numpy as np


def block_index(n_t, mp, vf_mean, delta=.5):
    lag = mp / (vf_mean / 12.0)
    cc = np.arange(n_t)[:, None] + .5 - lag[None, :] - delta
    blk = np.floor(cc / 6).astype(int); pos = cc - 6 * blk
    return blk, pos


def block_stats(x, blk, pos, margin=.5):
    """x [D, T, S] log values with nan for unobserved/unusable. Returns (mean of other same-block clean cells, count)."""
    D, Tn, S = x.shape
    clean = (pos > margin) & (pos < 6 - margin)
    xc = np.where(clean[None] & np.isfinite(x), x, np.nan)
    b0 = blk.min(); nb = blk.max() - b0 + 1
    sums = np.zeros((D, nb, S)); cnts = np.zeros((D, nb, S))
    bi = blk - b0
    for t in range(Tn):
        v = xc[:, t]; ok = np.isfinite(v)
        np.add.at(sums, (slice(None), bi[t], np.arange(S)), 0) if False else None
        for s_ in range(S):
            pass
    # vectorised accumulate over t
    for t in range(Tn):
        v = xc[:, t]; ok = np.isfinite(v)
        sums[:, bi[t], np.arange(S)] += np.where(ok, v, 0)
        cnts[:, bi[t], np.arange(S)] += ok
    S_ = sums[:, bi, np.arange(S)[None, :]]  # [D, T, S] via fancy: bi [T,S]
    C_ = cnts[:, bi, np.arange(S)[None, :]]
    own = np.isfinite(xc)
    S_ = S_ - np.where(own, xc, 0); C_ = C_ - own
    m = np.where(C_ > 0, S_ / np.maximum(C_, 1), np.nan)
    return m, C_, clean
