"""phys5 (C helper, 2026-10-05): local demand-step features (block alignment as phys5_s3.step_feats): median over
stations j in s-3..s+3 (j != s) of the aligned block step M_j(b+1) - M_j(b) and M_j(b) - M_j(b-1), their support, and the
local block level difference (median over neighbours of M_j(b) minus own-station M_s(b) is NOT used: shares differ).
Saves feat3_<pn>.npz aligned with feat_<pn>.npz rows."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import phys4_common as C
import phys5_diag as DG
import gen_blk as B

OUT = DG.OUT
NAMES3 = ["lstep_next", "lstep_prev", "lstep_n", "ostep_next", "ostep_prev"]


def feats(held, tv, tq, ix, W=3):
    T = np.asarray(held.target, bool); el = np.asarray(held.elig, bool); vf = C.consts(held)["vf"]
    mp = held.stations.sort_values("idx").milepost.to_numpy(float)
    D, Tn, S = T.shape
    obs = el & ~T & np.isfinite(tv) & np.isfinite(tq) & (tq > 0)
    x = np.where(obs & (tv > .9 * vf), np.log(np.maximum(np.nan_to_num(tq), 1)), np.nan)
    blk, pos = B.block_index(Tn, mp, vf.mean(), .5)
    b0 = blk.min(); nb = blk.max() - b0 + 1; bi = blk - b0 + 1
    clean = (pos > .5) & (pos < 5.5)
    sums = np.zeros((D, nb + 2, S)); cnts = np.zeros((D, nb + 2, S))
    for t in range(Tn):
        ok = np.isfinite(x[:, t]) & clean[t][None]
        sums[:, bi[t], np.arange(S)] += np.where(ok, x[:, t], 0); cnts[:, bi[t], np.arange(S)] += ok
    M = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    dM = M[:, 1:] - M[:, :-1]                                   # [D, nb+1, S]: j = step from block j to j+1
    stack = []
    for k in range(-W, W + 1):
        if k == 0: continue
        sh = np.full_like(dM, np.nan)
        if k > 0: sh[:, :, :-k] = dM[:, :, k:]
        else: sh[:, :, -k:] = dM[:, :, :k]
        stack.append(sh)
    stack = np.stack(stack)
    with np.errstate(all="ignore"):
        L = np.nanmedian(stack, 0); Ln = np.isfinite(stack).sum(0)
    d, t, s = ix.T.astype(int)
    jn = np.clip(bi, 0, nb)[t, s]; jp = np.clip(bi - 1, 0, nb)[t, s]
    return np.stack([L[d, jn, s], L[d, jp, s], Ln[d, jn, s], dM[d, jn, s], dM[d, jp, s]], 1).astype(np.float32)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    for pn in sys.argv[1:] or C.PANELS:
        f = OUT / f"feat3_{pn}.npz"
        if f.exists(): continue
        t0 = time.time()
        held, tv, tq, v, q, p = DG.load(pn)
        X3 = feats(held, tv, tq, np.load(OUT / f"feat_{pn}.npz")["ix"])
        np.savez(f, X=X3, names=np.array(NAMES3)); print(pn, X3.shape, round(time.time() - t0), "s", flush=True)
