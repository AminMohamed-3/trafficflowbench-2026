"""Claude helper 2026-10-04 (state5): shared loaders and I-S-OWN scoring for state5 experiments.
Scoring replicates own_instrument.reduce_state (S_state = .54 S_speed + .46 S_flow per panel/regime, regime mean, panel mean
within family, family mean) on three populations: nb (non-block OWN targets), all, and mix (official-mix: per panel/regime
MSE = g * MSE_block + (1 - g) * MSE_nonblock, g = share of official target cells inside official gaps for that split;
geometry only, no labels). Date bootstrap (1000, seed 4701) of the paired delta."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phys_lwr_diag as P

ROOT = P.ROOT
OUT = ROOT / "artifacts/claude_ref/state5"
S08 = ROOT / "artifacts/lane_s/0398_s08"
PANELS = P.P["panels"]
FAM = P.P["families"]
GROUPS = [[i for i, x in enumerate(PANELS) if FAM[x] == f] for f in sorted(set(FAM.values()))]
REG = ("R1", "R2", "R3")
_SH = {}


def official_shares(split):
    """Per panel, per regime: share of official target cells that lie in official gap cells."""
    if split in _SH:
        return _SH[split]
    sys.path.insert(0, str(ROOT / "src"))
    from tfb.data import load_cache
    out = {}
    for pn in PANELS:
        c = load_cache(pn, split); T = np.asarray(c.target, bool); G = np.asarray(c.gap, bool) & T
        reg = np.asarray(c.regimes)
        out[pn] = {r: float(G[reg == r].sum() / max(T[reg == r].sum(), 1)) for r in REG}
    _SH[split] = out
    return out


def lanes_of(pn, split="validation"):
    sys.path.insert(0, str(ROOT / "src"))
    from tfb.data import load_cache
    c = load_cache(pn, split)
    st = c.stations.sort_values("idx")
    return np.maximum(st.lanes.to_numpy(float), 1.), st.v_cut.to_numpy(float), st.capacity_vph.to_numpy(float), c.stations


def stats(pn, z, v, q, lanes, mask=None):
    """Per (regime, date, class) [n, sq_v, sq_q]; class 0 = non-block, 1 = block. z: cache dict (tv, tq, target, gap, regimes)."""
    tv, tq = z["tv"].astype(float), z["tq"].astype(float)
    T = z["target"] & np.isfinite(tv) & np.isfinite(tq)
    if mask is not None:
        T = T & mask
    D = tv.shape[0]; out = np.zeros((3, D, 2, 3))
    ev = (v - tv) ** 2; eq = ((q - tq) / lanes[None, None, :]) ** 2
    for d in range(D):
        r = REG.index(str(z["regimes"][d]))
        for c, m in ((0, T[d] & ~z["gap"][d]), (1, T[d] & z["gap"][d])):
            out[r, d, c] = (m.sum(), ev[d][m].sum(), eq[d][m].sum())
    return out


def reduce(data, w, pop, shares):
    """data [P, 3, D, 2, 3]; w date weights; pop nb | all | mix."""
    t = np.einsum("prdcx,d->prcx", data, w)
    if pop == "nb":
        n, sv, sq = np.moveaxis(t[:, :, 0], -1, 0)
        mv = np.divide(sv, n, out=np.zeros_like(n), where=n > 0); mq = np.divide(sq, n, out=np.zeros_like(n), where=n > 0)
    elif pop == "all":
        a = t.sum(2); n, sv, sq = np.moveaxis(a, -1, 0)
        mv = np.divide(sv, n, out=np.zeros_like(n), where=n > 0); mq = np.divide(sq, n, out=np.zeros_like(n), where=n > 0)
    else:
        n0, s0v, s0q = np.moveaxis(t[:, :, 0], -1, 0); n1, s1v, s1q = np.moveaxis(t[:, :, 1], -1, 0)
        f = lambda s, n: np.divide(s, n, out=np.zeros_like(n), where=n > 0)
        g = shares
        mv = g * f(s1v, n1) + (1 - g) * f(s0v, n0); mq = g * f(s1q, n1) + (1 - g) * f(s0q, n0)
        n = n0 + n1
    sc = .54 * np.maximum(0, 1 - np.sqrt(mv) / 25) + .46 * np.maximum(0, 1 - np.sqrt(mq) / 600)
    pp = np.where(n > 0, sc, 0).mean(1)
    return float(np.mean([pp[g].mean() for g in GROUPS]))


def compare(ref, new, shares, pops=("nb", "mix", "all"), nboot=1000):
    """ref/new: dict panel -> stats array. Returns {pop: (ref, delta, lo, hi, n_pos_panels)}."""
    A = np.stack([ref[p] for p in PANELS]); B = np.stack([new[p] for p in PANELS]); D = A.shape[2]
    rng = np.random.default_rng(4701); W = [np.bincount(rng.integers(0, D, D), minlength=D).astype(float) for _ in range(nboot)]
    res = {}
    for pop in pops:
        base = reduce(A, np.ones(D), pop, shares); dl = reduce(B, np.ones(D), pop, shares) - base
        bs = np.array([reduce(B, w, pop, shares) - reduce(A, w, pop, shares) for w in W])
        npos = 0
        for i, p in enumerate(PANELS):
            a1 = A[i:i + 1]; b1 = B[i:i + 1]
            g = [[0]]
            t = lambda x: reduce_single(x, pop, shares[i:i + 1])
            npos += int(t(b1) > t(a1))
        res[pop] = dict(ref=round(base, 6), delta=round(dl, 6), ci=[round(float(np.quantile(bs, .025)), 6), round(float(np.quantile(bs, .975)), 6)], n_pos=npos)
    return res


def reduce_single(data, pop, shares):
    global GROUPS
    g0 = GROUPS; GROUPS = [[0]]
    try:
        return reduce(data, np.ones(data.shape[2]), pop, shares)
    finally:
        GROUPS = g0


def share_array(split):
    sh = official_shares(split)
    return np.array([[sh[p][r] for r in REG] for p in PANELS])


def load_own(pn, split="validation"):
    z = dict(np.load(OUT / f"own_{pn}.npz" if split == "validation" else OUT / f"own_apr_{pn}.npz", allow_pickle=False))
    return z


def load_s08(group, pn):
    """S08 arrays (speed, flow full grid, target) from S's 0400 delivery; group own_validation | own_private | opt."""
    f = S08 / group / f"{pn}.npz"
    if not f.exists():
        return None
    return dict(np.load(f, allow_pickle=False))
