"""phys5 (C helper, 2026-10-05): per-station demand-step timing offset features. phys5_phase.py found the aligned step
boundary (gen_blk, corridor-mean vf) is off by a station-specific delta_s (sd .08-.20 steps, odd/even-day corr .92-.99,
same from observed-only cells as from truth). delta_s is fitted per station on the released view (observed free cells,
all days of the panel; no target values), then: phase_adj = pos - delta_s, delta_s, and ramp_ex = expected log q under a
1.5-step linear ramp centred at the shifted boundaries, minus the base log q (same block means as phys5_s3.step_feats).
Saves feat4_<pn>.npz aligned with feat_<pn>.npz rows. Arm F = B + these."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import phys4_common as C
import phys5_diag as DG
import gen_blk as B

OUT = DG.OUT
NAMES4 = ["phase_adj", "delta_s", "ramp_ex"]
GRID = np.round(np.arange(-1.5, 1.51, .05), 2)


def block_means(x, bi, pos, nb):
    D, Tn, S = x.shape
    clean = (pos > .5) & (pos < 5.5)
    sums = np.zeros((D, nb + 2, S)); cnts = np.zeros((D, nb + 2, S))
    for t in range(Tn):
        ok = np.isfinite(x[:, t]) & clean[t][None]
        sums[:, bi[t], np.arange(S)] += np.where(ok, x[:, t], 0); cnts[:, bi[t], np.arange(S)] += ok
    return np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)


def deltas(x, M, bi, pos):
    """Per-station ramp offset from cells within 1.5 steps of a boundary with own step > .08 (both side means present)."""
    D, Tn, S = x.shape; Sx = np.arange(S)
    R, Y, SS = [], [], []
    for t in range(Tn):
        for side in (0, -1):
            jl = bi[t] + side                          # boundary between padded blocks jl and jl+1
            rel = pos[t] - 6 * (side + 1)              # side 0: pos-6 ; side -1: pos
            near = np.abs(rel) <= 1.5
            if not near.any(): continue
            Ml = M[:, np.clip(jl, 0, M.shape[1] - 1), Sx]; Mr = M[:, np.clip(jl + 1, 0, M.shape[1] - 1), Sx]
            st = Mr - Ml
            with np.errstate(all="ignore"):
                y = (x[:, t] - Ml) / st
            m = near[None] & np.isfinite(y) & (np.abs(st) > .08)
            dd, ss = np.nonzero(m)
            R.append(rel[ss]); Y.append(y[dd, ss]); SS.append(ss)
    R, Y, SS = np.concatenate(R), np.clip(np.concatenate(Y), -.5, 1.5), np.concatenate(SS)
    out = np.zeros(S)
    for s in range(S):
        k = SS == s
        if k.sum() < 30: continue
        r, y = R[k], Y[k]
        sse = [np.mean((y - np.clip(.5 + (r - dl) / 1.5, 0, 1)) ** 2) for dl in GRID]
        out[s] = GRID[int(np.argmin(sse))]
    return out


def feats_from(so, fo, T, vf, mp, lq_cells, ix):
    """so/fo released observed speed/flow (nan elsewhere). lq_cells: base log q at ix rows. Returns [n, 3] float32."""
    D, Tn, S = T.shape
    obs = np.isfinite(so) & np.isfinite(fo) & (fo > 0)
    x = np.where(obs & (so > .9 * vf), np.log(np.maximum(np.nan_to_num(fo), 1)), np.nan)
    blk, pos = B.block_index(Tn, mp, vf.mean(), .5)
    b0 = blk.min(); nb = blk.max() - b0 + 1; bi = blk - b0 + 1
    M = block_means(x, bi, pos, nb)
    dl = deltas(x, M, bi, pos)
    d, t, s = ix.T.astype(int)
    p = pos[t, s]; b = bi[t, s]
    Mp, Mc, Mn = M[d, np.clip(b - 1, 0, nb + 1), s], M[d, b, s], M[d, np.clip(b + 1, 0, nb + 1), s]
    yp = np.clip(.5 + (p - dl[s]) / 1.5, 0, 1); yn = np.clip(.5 + (p - 6 - dl[s]) / 1.5, 0, 1)
    with np.errstate(all="ignore"):
        ex = Mc + (yp - 1) * np.nan_to_num(Mc - Mp) + yn * np.nan_to_num(Mn - Mc)
    return np.stack([p - dl[s], dl[s], ex - lq_cells], 1).astype(np.float32), dl


if __name__ == "__main__":
    import os, warnings; warnings.filterwarnings("ignore")
    BASE = os.environ.get("PHYS5_BASE", "")      # stage-4 mode: ramp_ex relative to C14 x exp(pred_BASE)
    for pn in sys.argv[1:] or C.PANELS:
        f = OUT / (f"feat4_{BASE}_{pn}.npz" if BASE else f"feat4_{pn}.npz")
        if f.exists(): continue
        if BASE:
            import phys5_s3 as P5
            z = dict(np.load(OUT / f"feat4_{pn}.npz")); ix = np.load(OUT / f"feat_{pn}.npz")["ix"]
            held, tv, tq, v, q, p = DG.load(pn); d, t, s = ix.T.astype(int)
            _, qb = P5.apply(pn, BASE, v, q, ix)
            z["X"][:, 2] = z["X"][:, 2] + np.log(np.maximum(q[d, t, s], 1.)) - np.log(np.maximum(qb[d, t, s], 1.))
            np.savez(f, **z); print(pn, "base", BASE, flush=True); continue
        t0 = time.time()
        held, tv, tq, v, q, p = DG.load(pn)
        T = np.asarray(held.target, bool); el = np.asarray(held.elig, bool); vf = C.consts(held)["vf"]
        mp = held.stations.sort_values("idx").milepost.to_numpy(float)
        ok = el & ~T & np.isfinite(tv) & np.isfinite(tq)
        so = np.where(ok, tv, np.nan); fo = np.where(ok, tq, np.nan)
        ix = np.load(OUT / f"feat_{pn}.npz")["ix"]; d, t, s = ix.T.astype(int)
        X4, dl = feats_from(so, fo, T, vf, mp, np.log(np.maximum(q[d, t, s], 1.)), ix)
        np.savez(f, X=X4, names=np.array(NAMES4), delta=dl)
        print(pn, X4.shape, "delta mean %.2f sd %.2f" % (dl.mean(), dl.std()), round(time.time() - t0), "s", flush=True)
