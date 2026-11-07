"""Claude gap helper 2026-10-04: shared code for the 90-min gap (blackout) specialist.
Layer = one split view with only released visible cells (K mask). Two kinds:
  own   : I-S-OWN blinded view (March 'validation' or April 'private' period); eval blocks = OWN artificial blocks
          (hidden, truth = OWN labels), classes from truth with the selector rules.
  off   : official view (load_cache validation/private); eval blocks = official gap windows (window_index.csv,
          condition known; onset => first queued step k = 6 by the selector rule).
Simulated training blocks: origins in the layer's own visible data (block t+1..t+18 mostly visible), classed with the
organizer selector rules on visible cells: onset-like (no link with >= 2 queued steps in t-12..t-1, first queued block
step k), ongoing-like (visible queue in t-12..t-1 and a queued cell in t+1..t+6; persistence IoU recorded), free, other.
Features per block cell (j, s): edges (pre/post window means, last/next values, ages, trends), linear interpolation,
neighbour edges (+-1, 2, 4 stations; idx increases downstream), leave-one-day-out time-of-day profile and profile-anomaly
interpolation, queue geometry (distance to nearest queued station pre/post, both directions), generator-rule template
(fixed head, linear tail; onset starts at k), corridor congestion shares, class and k.
Targets: v / vf and q / cap."""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "2")
import sys, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phys_lwr_diag as P

ROOT = P.ROOT
OUT = ROOT / "artifacts/claude_ref/gap"
PANELS = P.P["panels"]
REL = ROOT / "data/raw/kaggle_public/task2"
NB = (-4, -2, -1, 1, 2, 4)
CLS = {"onset": 0, "ongoing": 1, "free": 2, "other": 3}


# ------------------------------------------------------------------ layers
def own_layer(panel, period="validation"):
    import flow_own as FO
    p = P.P
    item = [i for i in json.loads(Path(p["OWN_metadata_binding"]["manifest"]).read_text())["items"]
            if i["period"] == period and i["panel"] == panel][0]
    view, tv, tq, st_, hidden = FO.own.load_item(ROOT, FO.binding.bound_own_item(p, item))
    st_ = np.asarray(st_, bool); hidden = np.asarray(hidden, bool)
    hide = np.asarray(view.target, bool) | np.asarray(view.gap, bool) | hidden
    sp = np.asarray(view.speed, float); fl = np.asarray(view.flow, float); el = np.asarray(view.elig, bool)
    K = el & ~hide & np.isfinite(sp) & np.isfinite(fl)
    L = _mk(panel, view.stations, list(view.dates), np.asarray(view.regimes), np.where(K, sp, np.nan), np.where(K, fl, np.nan), K, el)
    L.score_target = st_; L.tv = np.asarray(tv, float); L.tq = np.asarray(tq, float)
    L.hidden = hidden
    # known speed for eval-block classes: visible or hidden truth
    L.kv_truth = np.where(hidden, L.tv, L.sp)
    L.blocks = [(view.dates.index(b["date"]), int(b["origin_step"])) for b in item["blocks"]]
    L.gapmask = np.asarray(view.gap, bool)
    return L


def off_layer(panel, split):
    from tfb.data import load_cache
    c = load_cache(panel, split)
    tgt = np.asarray(c.target, bool); gap = np.asarray(c.gap, bool)
    sp = np.asarray(c.speed, float); fl = np.asarray(c.flow, float); el = np.asarray(c.elig, bool)
    hist = np.asarray(c.hist, bool)
    K = el & ~tgt & ~gap & np.isfinite(sp) & np.isfinite(fl)
    L = _mk(panel, c.stations, list(c.dates), np.asarray(c.regimes), np.where(K, sp, np.nan), np.where(K, fl, np.nan), K, el)
    L.target = tgt; L.gapmask = gap
    w_path = REL / panel / split / "window_index.csv"
    L.windows = []
    if w_path.exists():
        w = pd.read_csv(w_path)
        for r in w.itertuples(index=False):
            ts = pd.Timestamp(r.forecast_origin); d = L.dates.index(ts.strftime("%Y-%m-%d")); t = ts.hour * 12 + ts.minute // 5
            assert gap[d, t + 1:t + 19].all(), (panel, split, d, t)
            L.windows.append((d, t, "onset" if r.condition == "queue_onset" else "ongoing"))
    return L


def _mk(panel, stations, dates, regimes, sp, fl, K, el):
    s = stations.sort_values("idx")
    L = SimpleNamespace(panel=panel, stations=stations, dates=dates, regimes=regimes, sp=sp, fl=fl, K=K, el=el)
    L.vf = s.free_speed_kmh.to_numpy(float); L.vcut = s.v_cut.to_numpy(float); L.cap = s.capacity_vph.to_numpy(float)
    L.lanes = np.maximum(s.lanes.to_numpy(float), 1.); L.mp = s.milepost.to_numpy(float)
    L.len = s.length_km.to_numpy(float); L.onr = s.has_on_ramp.to_numpy(float); L.offr = s.has_off_ramp.to_numpy(float)
    D, T, S = sp.shape
    # leave-one-day-out profiles (mean of visible values at (t, s) over the other days)
    vn = sp / L.vf; qn = fl / L.cap; cg = (sp <= L.vcut).astype(float)
    cnt = K.sum(0).astype(float)
    sv = np.where(K, vn, 0).sum(0); sq = np.where(K, qn, 0).sum(0); sc = np.where(K, cg, 0).sum(0)
    L.prof = (cnt, sv, sq, sc)
    return L


def loo_prof(L, d):
    """(T, S) leave-day-d-out profile of v/vf, q/cap, congestion share, with smoothing fallbacks."""
    cnt, sv, sq, sc = L.prof
    K = L.K[d]
    c = cnt - K
    v = sv - np.where(K, L.sp[d] / L.vf, 0); q = sq - np.where(K, L.fl[d] / L.cap, 0); g = sc - np.where(K, (L.sp[d] <= L.vcut), 0)
    with np.errstate(all="ignore"):
        pv, pq, pg = v / c, q / c, g / c
    # fallback: station mean over the day
    for a in (pv, pq, pg):
        m = ~np.isfinite(a)
        if m.any():
            fill = np.nanmean(a, 0)
            a[m] = np.broadcast_to(fill, a.shape)[m]
            a[~np.isfinite(a)] = np.nanmean(a) if np.isfinite(np.nanmean(a)) else 0
    return pv, pq, pg


# ------------------------------------------------------------------ classes
def classify(kv, el, vcut, t):
    """Selector-rule class of block t+1..t+18 given a (T, S) known speed (nan unknown). Returns (cls, k, pers)."""
    q = np.isfinite(kv) & (kv <= vcut[None, :])
    obs = np.isfinite(kv)
    hist = q[t - 12:t]
    vis = ((hist & el[t - 12:t]).sum(0) >= 2).any()
    blk = q[t + 1:t + 19]
    anyq = blk.any(1)
    if not anyq.any():
        return "free", 0, np.nan
    k = int(np.argmax(anyq)) + 1
    if not (hist.sum(0) >= 2).any():
        return "onset", k, np.nan
    if vis and q[t + 1:t + 7].any():
        last = np.zeros(q.shape[1], bool)
        for s in range(t - 12, t):
            last = np.where(obs[s], q[s], last)
        val = obs[t + 1:t + 7]; fut = q[t + 1:t + 7] & val
        pp = np.broadcast_to(last, fut.shape) & val; u = (pp | fut).sum()
        return "ongoing", k, float((pp & fut).sum() / u) if u else 1.
    return "other", k, np.nan


def sim_origins(L, rng, stride_on=1, stride_og=3, n_free=4, cover=.5):
    """Training origins from visible data. Returns list of (d, t, cls, k, pers)."""
    D, T, S = L.sp.shape
    out = []
    for d in range(D):
        kv = L.sp[d]; el = L.el[d]
        cov = (L.K[d] .sum(1) / np.maximum(el.sum(1), 1))     # per step visible share of eligible
        frees = []
        for t in range(12, 270):
            if cov[t + 1:t + 19].mean() < cover or cov[max(t - 5, 0):t + 1].mean() < cover or cov[t + 19:t + 25].mean() < cover:
                continue
            if L.gapmask[d, t - 12:t + 25].any():
                continue
            c, k, pers = classify(kv, el, L.vcut, t)
            if c == "onset" and t % stride_on == 0:
                out.append((d, t, c, k, pers))
            elif c in ("ongoing", "other") and t % stride_og == 0:
                out.append((d, t, c, k, pers))
            elif c == "free":
                frees.append(t)
        if frees:
            for t in rng.choice(frees, min(n_free, len(frees)), replace=False):
                out.append((d, int(t), "free", 0, np.nan))
    return out


# ------------------------------------------------------------------ features
def _wmean(x, a, b):
    a = max(a, 0); b = min(b, x.shape[0])
    if b <= a:
        return np.full(x.shape[1], np.nan)
    with np.errstate(all="ignore"):
        return np.nanmean(x[a:b], 0)


def _last(x, t, back=24):
    """last finite value at or before t (within back steps) and its age."""
    v = np.full(x.shape[1], np.nan); age = np.full(x.shape[1], np.nan)
    for a in range(0, back):
        s = t - a
        if s < 0:
            break
        m = np.isnan(v) & np.isfinite(x[s])
        v[m] = x[s][m]; age[m] = a
    return v, age


def _next(x, t, fwd=24):
    v = np.full(x.shape[1], np.nan); age = np.full(x.shape[1], np.nan)
    for a in range(0, fwd):
        s = t + a
        if s >= x.shape[0]:
            break
        m = np.isnan(v) & np.isfinite(x[s])
        v[m] = x[s][m]; age[m] = a
    return v, age


def _shift(a, o):
    """a[s + o] (nan outside)."""
    out = np.full_like(a, np.nan)
    S = a.shape[-1]
    if o > 0:
        out[..., :S - o] = a[..., o:]
    elif o < 0:
        out[..., -o:] = a[..., :S + o]
    else:
        out = a.copy()
    return out


def _fill_edge(a, b):
    """fill nan of a from b, then from a station-neighbour interpolation."""
    a = np.where(np.isfinite(a), a, b)
    if np.isnan(a).any() and np.isfinite(a).any():
        idx = np.arange(len(a)); ok = np.isfinite(a)
        a = np.interp(idx, idx[ok], a[ok])
    return a


def _dist_to(mask, mp):
    """distance (km) from each station to the nearest station in mask downstream (higher idx) and upstream; 99 if none."""
    S = len(mp); dn = np.full(S, 99.); up = np.full(S, 99.)
    last = None
    for s in range(S - 1, -1, -1):
        if mask[s]:
            last = s
        if last is not None:
            dn[s] = mp[last] - mp[s]
    last = None
    for s in range(S):
        if mask[s]:
            last = s
        if last is not None:
            up[s] = mp[s] - mp[last]
    return np.minimum(dn, 99.), np.minimum(up, 99.)


def _runs(mask):
    out = []; s = 0; S = len(mask)
    while s < S:
        if mask[s]:
            a = s
            while s < S and mask[s]:
                s += 1
            out.append((a, s - 1))
        else:
            s += 1
    return out


def template(pre_c, post_c, k, cls, mp):
    """Generator-rule queue indicator (18, S): queues = runs of congested stations (holes <= 1 merged).
    onset: each post run grows from its head (most downstream station) starting at block step k to the post extent
    at step 19 (linear in distance). otherwise: tail (upstream end) moves linearly between the pre and post run that
    share the head region; runs present only pre recede (tail -> head), only post grow from head over the whole block."""
    S = len(mp); out = np.zeros((18, S))
    def merge(m):
        m = m.copy()
        for s in range(1, S - 1):
            if not m[s] and m[s - 1] and m[s + 1]:
                m[s] = True
        return m
    pre_c = merge(pre_c); post_c = merge(post_c)
    pr = _runs(pre_c); po = _runs(post_c)
    j = np.arange(1, 19)
    used = set()
    for (a, b) in po:
        # find matching pre run overlapping [a, b]
        match = [(c, e) for (c, e) in pr if not (e < a or c > b)]
        head = b
        if cls == "onset" or not match:
            k0 = k if cls == "onset" and k >= 1 else 1
            f = np.clip((j - k0 + 1) / max(19 - k0 + 1, 1), 0, 1)
            tail_post = mp[head] - mp[a]
            ids = np.arange(S)
            for jj in range(18):
                if j[jj] < k0:
                    continue
                ext = f[jj] * tail_post
                sel = (ids >= a) & (ids <= head) & (mp >= mp[head] - ext - 1e-9)
                out[jj, sel] = 1
                out[jj, head] = 1
        else:
            c, e = match[0]; used.add((c, e))
            head = max(b, e)
            t_pre = mp[head] - mp[c]; t_post = mp[head] - mp[a]
            for jj in range(18):
                w = j[jj] / 19.
                ext = (1 - w) * t_pre + w * t_post
                sel = (mp <= mp[head]) & (mp >= mp[head] - ext - 1e-9)
                out[jj, sel] = 1
    if cls != "onset":
        for (c, e) in pr:
            if (c, e) in used or any(not (e < a or c > b) for (a, b) in po):
                continue
            t_pre = mp[e] - mp[c]
            for jj in range(18):
                w = j[jj] / 19.
                ext = (1 - w) * t_pre
                if w > .999:
                    continue
                sel = (mp <= mp[e]) & (mp >= mp[e] - ext - 1e-9)
                out[jj, sel] = 1
    return out


FEATS = None


def block_features(L, d, t, cls, k, prof=None):
    """Returns dict of (18, S) float32 feature arrays for block t+1..t+18 on day d (uses only L.sp/L.fl visible)."""
    S = L.sp.shape[2]; vf, vcut, cap = L.vf, L.vcut, L.cap
    x = L.sp[d]; y = L.fl[d]
    pv, pq, pg = prof if prof is not None else loo_prof(L, d)
    pre_v = _wmean(x, t - 5, t + 1); pre_v2 = _wmean(x, t - 11, t - 5)
    post_v = _wmean(x, t + 19, t + 25); post_v2 = _wmean(x, t + 25, t + 31)
    pre_q = _wmean(y, t - 5, t + 1); post_q = _wmean(y, t + 19, t + 25)
    lv, la = _last(x, t); nv, na = _next(x, t + 19)
    lq, _ = _last(y, t); nq, _ = _next(y, t + 19)
    tc = min(t + 21, 287); tp = max(t - 2, 0)
    pre_v = _fill_edge(pre_v, lv); post_v = _fill_edge(post_v, nv)
    pre_q = _fill_edge(pre_q, lq); post_q = _fill_edge(post_q, nq)
    pre_vn = pre_v / vf; post_vn = post_v / vf
    j = np.arange(1, 19)[:, None].astype(float); w = j / 19.
    tt = t + np.arange(1, 19)
    F = {}
    one = np.ones((18, S))
    F["j"] = j * one; F["w"] = w * one
    F["cls"] = CLS[cls] * one
    F["krel"] = (j - k) * one if cls == "onset" else -99 * one
    F["tod"] = tt[:, None] * one
    F["vf"] = vf * one; F["vcutn"] = (vcut / vf) * one; F["caplane"] = (cap / L.lanes) * one; F["lanes"] = L.lanes * one
    F["len"] = L.len * one; F["onr"] = L.onr * one; F["offr"] = L.offr * one; F["pos"] = (np.arange(S) / S) * one
    F["pre_v"] = pre_vn * one; F["post_v"] = post_vn * one
    F["pre_v2"] = (pre_v2 / vf) * one; F["post_v2"] = (post_v2 / vf) * one
    F["last_v"] = (lv / vf) * one; F["next_v"] = (nv / vf) * one; F["age_l"] = la * one; F["age_n"] = na * one
    F["pre_q"] = (pre_q / cap) * one; F["post_q"] = (post_q / cap) * one
    F["lin_v"] = pre_vn + w * (post_vn - pre_vn)
    F["lin_q"] = pre_q / cap + w * (post_q / cap - pre_q / cap)
    for o in NB:
        F[f"pre_v{o:+d}"] = _shift(pre_vn, o) * one; F[f"post_v{o:+d}"] = _shift(post_vn, o) * one
    for o in (-1, 1):
        F[f"pre_q{o:+d}"] = _shift(pre_q / cap, o) * one; F[f"post_q{o:+d}"] = _shift(post_q / cap, o) * one
    F["prof_v"] = pv[tt]; F["prof_q"] = pq[tt]; F["prof_g"] = pg[tt]
    F["an_v"] = pv[tt] + (1 - w) * (pre_vn - pv[tp]) + w * (post_vn - pv[tc])
    F["an_q"] = pq[tt] + (1 - w) * (pre_q / cap - pq[tp]) + w * (post_q / cap - pq[tc])
    F["prof_dpre"] = (pre_vn - pv[tp]) * one; F["prof_dpost"] = (post_vn - pv[tc]) * one
    pre_c = pre_v <= vcut; post_c = post_v <= vcut
    a, b = _dist_to(pre_c, L.mp); F["d_dn_pre"] = a * one; F["d_up_pre"] = b * one
    a, b = _dist_to(post_c, L.mp); F["d_dn_post"] = a * one; F["d_up_post"] = b * one
    F["share_pre"] = pre_c.mean() * one; F["share_post"] = post_c.mean() * one
    tm = template(pre_c, post_c, k, cls, L.mp)
    F["tmpl"] = tm
    vq = np.where(post_c, post_vn, np.where(pre_c, pre_vn, vcut / vf * .7))
    vfree = np.where(~pre_c, pre_vn, np.where(~post_c, post_vn, 1.))
    F["tmpl_v"] = np.where(tm > 0, vq, vfree)
    if cls == "onset":
        F["onset_pre"] = (j < k) * one
    else:
        F["onset_pre"] = 0 * one
    if FRONT:
        sp_, sa_ = _switch(x, vcut, t, -1); so_, sb_ = _switch(x, vcut, t + 19, +1)
        F["sw_pre"] = sp_ * one; F["sw_post"] = so_ * one; F["st_pre"] = sa_ * one; F["st_post"] = sb_ * one
        for o in (-2, -1, 1, 2, 4):
            F[f"sw_pre{o:+d}"] = _shift(sp_, o) * one; F[f"sw_post{o:+d}"] = _shift(so_, o) * one
            F[f"st_pre{o:+d}"] = _shift(sa_, o) * one; F[f"st_post{o:+d}"] = _shift(sb_, o) * one
        c12 = _wmean(x, t - 13, t - 9) <= vcut; c31 = _wmean(x, t + 29, t + 33) <= vcut
        F["share_pre12"] = np.nanmean(c12) * one; F["share_post12"] = np.nanmean(c31) * one
        # time-relative position inside the gap w.r.t. the front estimates
        F["j_after_pre_sw"] = j + sp_[None, :]; F["j_before_post_sw"] = (19 - j) + so_[None, :]
    return {n: v.astype(np.float32) for n, v in F.items()}


FRONT = True
SWW = 24


def _switch(x, vcut, t, side):
    """side -1: age (steps before t) of the last state change at or before t within SWW steps, and the current state
    (1 congested / 0 free / nan unknown); side +1: steps after t until the first state change within SWW steps, and the
    state at t (first visible at/after t). No change found -> SWW + 1."""
    T, S = x.shape
    if side < 0:
        a, b = max(t - SWW, 0), t + 1
        W = x[a:b][::-1]
    else:
        a, b = t, min(t + SWW + 1, T)
        W = x[a:b]
    vis = np.isfinite(W)
    st = (W <= vcut[None, :]).astype(float)
    if W.shape[0] == 0:
        return np.full(S, SWW + 1.), np.full(S, np.nan)
    first = np.argmax(vis, 0); has = vis.any(0)
    cur = np.where(has, st[first, np.arange(S)], np.nan)
    diff = vis & (st != cur[None, :])
    idx = np.argmax(diff, 0).astype(float)
    age = np.where(diff.any(0), idx, SWW + 1.)
    return age, cur


def stack(blocks_feats):
    names = list(blocks_feats[0].keys())
    return names, np.stack([np.concatenate([f[n].ravel() for f in blocks_feats]) for n in names], 1)


def smooth_rows(M, x, h):
    """Nan-aware centered moving average (half-window h steps) along block step j of row values x, per block and
    station, using only the rows themselves (target cells of the block). M columns: block, d, t, cls, k, j, s, pers."""
    if h <= 0:
        return x
    out = x.copy()
    for bi in np.unique(M[:, 0]):
        m = np.flatnonzero(M[:, 0] == bi)
        jj, ss = M[m, 5] - 1, M[m, 6]
        S = int(ss.max()) + 1
        g = np.full((18, S), np.nan); g[jj, ss] = x[m]
        num = np.zeros_like(g); den = np.zeros_like(g)
        for o in range(-h, h + 1):
            sh = np.full_like(g, np.nan)
            if o >= 0:
                sh[:18 - o] = g[o:]
            else:
                sh[-o:] = g[:18 + o]
            ok = np.isfinite(sh); num += np.where(ok, sh, 0); den += ok
        out[m] = (num / np.maximum(den, 1))[jj, ss]
    return out
