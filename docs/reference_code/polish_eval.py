"""Claude helper 2026-10-04 (ledger 0402, polish): small post-processors on top of the C12 Task 1 stack (polish_ref arrays).
Chain (applied in this order, each optional): S3 = final phys3 with retuned (r, a, g, soft gate h) on the pre-projection
blend (replaces the C12 final phys3 r=3 a=.75 g=1) -> TV (C2) free-flow density TV in target runs (research2_tv.smooth,
gate tau 0) -> S4 station-day free-flow speed offset -> S5 class-hedged flow on 0368 candidates.
S4: prof(t,s) = median over the split's days of observed free speeds (v >= v_cut); obs offset o = kernel mean over the same
station-day of (v_obs - prof) on observed free cells (exp kernel scale H steps, or whole day); target free cell (v >= v_cut):
e = clip(o - (v - prof), +-15); v += beta e n/(n + 6). beta pooled OLS of (tv - v) on e n/(n+6).
S5: on 0368 candidate cells (p finite, not gap): r = (tq - q)/lanes ~ X b, X = [p, 1-p, p dv, (1-p) dv, p ql, (1-p) ql],
dv = (v - v_cut)/v_cut, ql = q / lanes / cap_lane; q += lanes X b, clipped [50, 1.3 cap]. b pooled OLS.
OPT: parameters chosen / fitted on one day parity (pooled 10 panels), applied to the other (cross-fit); official mode A
J_S via flow_screen.day_components; day bootstrap (1000) of the mean panel delta; panels up.
OWN (I-S-OWN March): parameters fitted on all OPT days, applied to all March days (cross-split transfer), non-block and all
targets, own_instrument.paired_state. Base = C12 with the already accepted chain (--base).
Usage: python docs/reference_code/polish_eval.py <idea tv|s3|s4|s5|none> [--base JSON]   (base: accepted chain params)"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phys_lwr_diag as P
import flow_screen as FS
import phys3_proj as PJ
import phys3_stack as ST
import research2_tv as TVM

OUT = P.ROOT / "artifacts/claude_ref/polish"
PANELS = P.P["panels"]
M = ("J", "S_state", "S_phys", "S_LWR", "S_FD")
REF_S3 = dict(r=3., a=.75, g=1., h=0.)
S3_GRID = [dict(r=r, a=a, g=g, h=0.) for r in (1., 3., 10.) for a in (.6, .75, .9) for g in (.95, 1., 1.05)] + \
          [dict(r=3., a=a, g=1., h=h) for a in (.75, .9) for h in (.05, .1)]
TV_GRID = [dict(tau=t) for t in (0., .002, .004, .006)]
S4_CFG = [dict(H=12.), dict(H=36.), dict(H=0.)]  # H=0: whole station-day
K0 = 6.


# ---------------------------------------------------------------- context
class Ctx:
    def __init__(s, stations, T, vobs, p, bl_v, bl_q, gap):
        s.stations = stations; s.T = T; s.vobs = vobs; s.p = p; s.bl_v = bl_v; s.bl_q = bl_q; s.gap = gap
        st = stations.sort_values("idx")
        s.vcut = st.v_cut.to_numpy(float); s.cap = st.capacity_vph.to_numpy(float)
        s.lanes = np.maximum(st.lanes.to_numpy(float), 1.)
        s._prof = None

    def prof(s):
        if s._prof is None:
            x = np.where(s.vobs >= s.vcut[None, None, :], s.vobs, np.nan)
            with np.errstate(all="ignore"):
                pr = np.nanmedian(x, axis=0)                     # [T, S]
                cnt = np.isfinite(x).sum(0)
                sm = np.nanmedian(x.reshape(-1, x.shape[2]), axis=0)  # station median
            pr = np.where((cnt >= 3) & np.isfinite(pr), pr, sm[None, :])
            s._prof = np.where(np.isfinite(pr), pr, s.vcut[None, :] + 20.)
        return s._prof


def project_soft(ctx, prm):
    if prm["h"] == 0:
        return PJ.project(ctx.bl_v, ctx.bl_q, ctx.T, ctx.stations, prm["r"], prm["a"], prm["g"], gap=ctx.gap)[:2]
    pv, pq = ctx.bl_v, ctx.bl_q; r, a, g, h = prm["r"], prm["a"], prm["g"], prm["h"]
    w, kj, vcut, cap = PJ._consts(ctx.stations)
    m = ctx.T & ~ctx.gap & (pv < g * (1 + h) * vcut) & np.isfinite(pv) & np.isfinite(pq) & (pv > 0) & (pq > 0)
    W = np.broadcast_to(w, pv.shape)[m]; K = np.broadcast_to(kj, pv.shape)[m]; VC = np.broadcast_to(vcut, pv.shape)[m]
    x0 = np.log(pv[m]); y0 = np.log(pq[m])
    al = a * np.clip((g * (1 + h) * VC - pv[m]) / (2 * h * g * VC), 0, 1)
    lo = np.minimum(np.maximum(np.log(3.), x0 - PJ.MAXSTEP), x0); hi = np.maximum(np.minimum(np.log(VC), x0 + PJ.MAXSTEP), x0)
    f = lambda x: x + np.log(W * K) - np.log(np.exp(x) + W)
    x = np.clip(x0, lo, hi)
    for _ in range(25):
        v = np.exp(x); sl = W / (v + W); e = f(x) - y0
        dx = -((x - x0) + r * r * sl * e) / (1 + r * r * sl * sl)
        x = np.clip(x + np.clip(dx, -.2, .2), lo, hi)
    nv = pv.copy(); nq = pq.copy()
    nv[m] = np.exp(x0 + al * (x - x0))
    nq[m] = np.clip(np.exp(y0 + al * (f(x) - y0)), 50., 1.3 * np.broadcast_to(cap, pv.shape)[m])
    return nv, nq


def s4_feat(ctx, v, cfg):
    """Returns (mask of applied cells, x = e n/(n+K0))."""
    pr = ctx.prof()[None]
    of = (ctx.vobs >= ctx.vcut[None, None, :]) & np.isfinite(ctx.vobs)
    val = np.where(of, ctx.vobs - pr, 0.); ind = of.astype(float)
    if cfg["H"] == 0:
        n = np.broadcast_to(ind.sum(1, keepdims=True), v.shape); sv = np.broadcast_to(val.sum(1, keepdims=True), v.shape)
    else:
        L = int(4 * cfg["H"]); kern = np.exp(-np.abs(np.arange(-L, L + 1)) / cfg["H"])
        n = convolve1d(ind, kern, axis=1, mode="constant"); sv = convolve1d(val, kern, axis=1, mode="constant")
    with np.errstate(all="ignore"):
        o = sv / n
    m = ctx.T & ~ctx.gap & (v >= ctx.vcut[None, None, :]) & (n > .5) & np.isfinite(v)
    e = np.clip(o - (v - pr), -15, 15)
    x = np.where(m, e * n / (n + K0), 0.)
    return m, x


def s5_feat(ctx, v, q):
    m = ctx.T & ~ctx.gap & np.isfinite(ctx.p) & np.isfinite(v) & np.isfinite(q)
    p = ctx.p[m].astype(float); dv = (v / ctx.vcut[None, None, :] - 1)[m]
    ql = (q / ctx.cap[None, None, :])[m]
    X = np.stack([p, 1 - p, p * dv, (1 - p) * dv, p * ql, (1 - p) * ql], 1)
    return m, X


def apply_chain(chain, ctx, v=None, q=None):
    """chain: dict with optional keys s3, tv, s4, s5 (params). Returns v, q (C12 base if empty)."""
    if "s3" in chain:
        v, q = project_soft(ctx, chain["s3"])
    if "tv" in chain and chain["tv"]["tau"] > 0:
        v, q = TVM.smooth(v, q, ctx.T & ~ctx.gap, ctx.stations, chain["tv"]["tau"], 0.)
    if "s4" in chain and chain["s4"]["beta"] != 0:
        m, x = s4_feat(ctx, v, chain["s4"])
        v = np.where(m, v + chain["s4"]["beta"] * x, v)
    if "s5" in chain:
        m, X = s5_feat(ctx, v, q)
        q = q.copy(); la = np.broadcast_to(ctx.lanes, q.shape)[m]; cp = np.broadcast_to(ctx.cap, q.shape)[m]
        q[m] = np.clip(q[m] + la * (X @ np.asarray(chain["s5"]["b"])), 50., 1.3 * cp)
    return v, q


# ---------------------------------------------------------------- OPT
def opt_load(pn):
    held, tv, tq, _, _ = P.held_panel(pn)
    z = np.load(OUT / f"c12_opt_{pn}.npz")
    T = np.asarray(held.target, bool)
    vobs = np.where(np.asarray(held.elig, bool) & ~T & np.isfinite(tv), tv, np.nan)
    ctx = Ctx(held.stations, T, vobs, z["p"].astype(float), z["bl_v"].astype(float), z["bl_q"].astype(float), np.zeros_like(T))
    return held, tv, tq, ctx, z["v"].astype(float), z["q"].astype(float)


def boot(cn, cr, D, nboot=1000):
    regs = {p: cn[p][1] for p in cn}
    per = {p: {m: FS.aggregate(cn[p][0], regs[p], np.ones(D))[m] - FS.aggregate(cr[p][0], regs[p], np.ones(D))[m] for m in M} for p in cn}
    rng = np.random.default_rng(0); B = []
    for _ in range(nboot):
        w = np.bincount(rng.integers(0, D, D), minlength=D).astype(float)
        B.append({m: np.mean([FS.aggregate(cn[p][0], regs[p], w)[m] - FS.aggregate(cr[p][0], regs[p], w)[m] for p in cn]) for m in M})
    B = pd.DataFrame(B)
    summ = {m: dict(mean=float(np.mean([per[p][m] for p in cn])), ci=[float(B[m].quantile(.025)), float(B[m].quantile(.975))],
                    n_pos=int(sum(per[p][m] > 0 for p in cn))) for m in M}
    return summ, {p: {m: round(x, 6) for m, x in d.items()} for p, d in per.items()}


def ols(XtX, Xty, ridge=1e-6):
    return np.linalg.solve(XtX + ridge * np.trace(XtX) / len(XtX) * np.eye(len(XtX)), Xty)


def opt(idea, base):
    D = len(P.P["OPT_held_dates"]); par = np.arange(D) % 2
    dsel = lambda f: np.broadcast_to((par == f)[:, None, None], (D, 1, 1))
    grid = dict(tv=TV_GRID, s3=S3_GRID).get(idea)
    comps = {}; cref = {}; fitd = {0: [], 1: []}
    t0 = time.time()
    for pn in PANELS:  # pass 1
        held, tv, tq, ctx, v0, q0 = opt_load(pn)
        bv, bq = apply_chain(base, ctx, v0, q0)
        cref[pn] = FS.day_components(held, tv, tq, bv, bq)
        if grid:
            comps[pn] = []
            for g in grid:
                ch = dict(base); ch[idea] = g
                comps[pn].append(FS.day_components(held, tv, tq, *apply_chain(ch, ctx, v0, q0))[0])
        elif idea == "s4":
            st = {}
            for ci, cfg in enumerate(S4_CFG):
                m, x = s4_feat(ctx, bv, cfg); y = np.where(m & np.isfinite(tv), tv - bv, 0.); x = np.where(np.isfinite(tv), x, 0.)
                for f in (0, 1):
                    sel = np.broadcast_to((par == f)[:, None, None], x.shape)
                    st[(ci, f)] = (float(np.sum((x * x)[sel])), float(np.sum((x * y)[sel])), float(np.sum((y * y)[sel & m])))
            fitd[0].append(st)
        elif idea == "s5":
            m, X = s5_feat(ctx, bv, bq)
            y = ((tq - bq) / ctx.lanes[None, None, :])[m]; ok = np.isfinite(y)
            dd = np.nonzero(m)[0]
            for f in (0, 1):
                s = ok & (par[dd] == f)
                fitd[f].append((X[s].T @ X[s], X[s].T @ y[s], int(s.sum())))
        print(pn, "pass1", round(time.time() - t0), "s", flush=True)
    # choose / fit per fold on the other parity
    chosen = {}
    for f in (0, 1):
        o = 1 - f; w = (par == o).astype(float)
        if grid:
            sc = [np.mean([FS.aggregate(comps[pn][i], cref[pn][1], w)["J"] for pn in PANELS]) for i in range(len(grid))]
            chosen[f] = dict(grid[int(np.argmax(sc))])
        elif idea == "s4":
            best = None
            for ci, cfg in enumerate(S4_CFG):
                xx = sum(st[(ci, o)][0] for st in fitd[0]); xy = sum(st[(ci, o)][1] for st in fitd[0])
                beta = xy / xx; gain = xy * xy / xx
                if best is None or gain > best[2]:
                    best = (ci, beta, gain)
            chosen[f] = dict(S4_CFG[best[0]], beta=float(best[1]), sse_gain=float(best[2]))
        elif idea == "s5":
            XtX = sum(a[0] for a in fitd[o]); Xty = sum(a[1] for a in fitd[o])
            chosen[f] = dict(b=ols(XtX, Xty).tolist(), n=int(sum(a[2] for a in fitd[o])))
    # all-days fit (for OWN and the probe)
    grid_scores = None
    if grid:
        sc = [np.mean([FS.aggregate(comps[pn][i], cref[pn][1], np.ones(D))["J"] for pn in PANELS]) for i in range(len(grid))]
        full = dict(grid[int(np.argmax(sc))]); ri = 0 if idea == "tv" else S3_GRID.index(REF_S3)
        grid_scores = {json.dumps(g): round(x - sc[ri], 7) for g, x in zip(grid, sc)}
    elif idea == "s4":
        best = None
        for ci, cfg in enumerate(S4_CFG):
            xx = sum(st[(ci, f)][0] for st in fitd[0] for f in (0, 1)); xy = sum(st[(ci, f)][1] for st in fitd[0] for f in (0, 1))
            if best is None or xy * xy / xx > best[2]:
                best = (ci, xy / xx, xy * xy / xx)
        full = dict(S4_CFG[best[0]], beta=float(best[1]))
    elif idea == "s5":
        full = dict(b=ols(sum(a[0] for f in (0, 1) for a in fitd[f]), sum(a[1] for f in (0, 1) for a in fitd[f])).tolist())
    print("chosen by fold", chosen, "all-days", full, flush=True)
    # pass 2: spliced cross-fit prediction
    cnew = {}
    for pn in PANELS:
        held, tv, tq, ctx, v0, q0 = opt_load(pn)
        nv = np.zeros_like(v0); nq = np.zeros_like(q0)
        for f in (0, 1):
            ch = dict(base); ch[idea] = chosen[f]
            a, b = apply_chain(ch, ctx, v0, q0)
            nv[par == f] = a[par == f]; nq[par == f] = b[par == f]
        cnew[pn] = FS.day_components(held, tv, tq, nv, nq)
    summ, per = boot(cnew, cref, D)
    print(idea, "OPT cross-fit vs base", {m: (round(v["mean"], 6), [round(c, 6) for c in v["ci"]], v["n_pos"]) for m, v in summ.items()}, flush=True)
    return dict(chosen={str(f): c for f, c in chosen.items()}, all_days=full, grid_scores_all_days=grid_scores, summary=summ, per_panel=per)


# ---------------------------------------------------------------- OWN
def own(base, new, tag):
    import flow_own as FO
    own_m, binding = FO.own, FO.binding
    p = P.P
    items = [i for i in json.loads(Path(p["OWN_metadata_binding"]["manifest"]).read_text())["items"] if i["period"] == "validation"]
    rows = {}; info = {}
    for item in items:
        pn = item["panel"]
        view, tv, tq, st_, hidden = own_m.load_item(P.ROOT, binding.bound_own_item(p, item))
        st_ = np.asarray(st_, bool); gap = np.asarray(view.gap, bool)
        z = np.load(OUT / f"c12_own_{pn}.npz"); assert (z["target"] == st_).all()
        speed = np.asarray(view.speed, float)
        observed = np.asarray(view.elig, bool) & ~np.asarray(view.target, bool) & ~gap & np.isfinite(speed)
        ctx = Ctx(view.stations, st_, np.where(observed, speed, np.nan), z["p"].astype(float), z["bl_v"].astype(float), z["bl_q"].astype(float), gap)
        v0, q0 = z["v"].astype(float), z["q"].astype(float)
        bv, bq = apply_chain(base, ctx, v0, q0)
        nv, nq = apply_chain(new, ctx, v0, q0)
        for name, (v, q) in (("ref", (bv, bq)), ("new", (nv, nq))):
            for part, m in (("all", st_), ("nonblock", st_ & ~gap)):
                rows.setdefault((name, part), []).extend(own_m.state_rows(view, tv, tq, v, q, m)[0])
        ok = st_ & ~gap & np.isfinite(tv) & np.isfinite(tq)
        info[pn] = dict(rmse_v_ref=float(np.sqrt(np.mean((bv - tv)[ok] ** 2))), rmse_v_new=float(np.sqrt(np.mean((nv - tv)[ok] ** 2))),
                        rmse_ql_ref=float(np.sqrt(np.mean(((bq - tq) / ctx.lanes)[ok] ** 2))), rmse_ql_new=float(np.sqrt(np.mean(((nq - tq) / ctx.lanes)[ok] ** 2))),
                        changed=float(np.mean((np.abs(nv - bv) > 1e-6) | (np.abs(nq - bq) > 1e-6))))
        print(pn, {k: round(x, 4) for k, x in info[pn].items()}, flush=True)
    panels = p["panels"]; fam = p["families"]
    groups = [[i for i, x in enumerate(panels) if fam[x] == f] for f in sorted(set(fam.values()))]
    dates = sorted({r["date"] for r in rows[("ref", "all")]})
    res = dict(per_panel=info)
    for part in ("nonblock", "all"):
        x = own_m.paired_state(rows[("ref", part)], rows[("new", part)], panels, dates, groups)
        bp = {pn: round(own_m.paired_state([r for r in rows[("ref", part)] if r["panel"] == pn], [r for r in rows[("new", part)] if r["panel"] == pn], [pn], dates, [[0]])["delta_S_state"], 6) for pn in panels}
        res[part] = dict(delta=x["delta_S_state"], ci=x["ci95"], verdict=x["verdict"], ref=x["incumbent_S_state"], by_panel=bp,
                         n_pos=int(sum(v > 0 for v in bp.values())))
        print(tag, "OWN", part, round(x["delta_S_state"], 6), [round(c, 6) for c in x["ci95"]], x["verdict"], flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("idea"); ap.add_argument("--base", default="{}"); ap.add_argument("--skip-opt", action="store_true")
    ap.add_argument("--prm", default=None); ap.add_argument("--tag", default="")
    a = ap.parse_args(); base = json.loads(a.base)
    res = dict(idea=a.idea, base=base)
    if a.idea == "combo":  # combined chain vs bare C12; base = {"_fold": {"0": chain, "1": chain}, "_all": chain}
        res["opt"] = opt_combined(base); res["own"] = own({}, base["_all"], "combo")
    else:
        if not a.skip_opt:
            res["opt"] = opt(a.idea, base)
        prm = json.loads(a.prm) if a.prm else res["opt"]["all_days"]
        new = dict(base); new[a.idea] = prm
        res["own"] = own(base, new, a.idea); res["own_prm"] = prm
    (OUT / f"eval_{a.idea}{a.tag}.json").write_text(json.dumps(res, indent=1, default=float))


def opt_combined(chain):
    """Combined chain vs C12 on OPT: chain parameters per fold given as chain['_fold'] = {0: {...}, 1: {...}}."""
    D = len(P.P["OPT_held_dates"]); par = np.arange(D) % 2; cref = {}; cnew = {}
    for pn in PANELS:
        held, tv, tq, ctx, v0, q0 = opt_load(pn)
        cref[pn] = FS.day_components(held, tv, tq, v0, q0)
        nv = np.zeros_like(v0); nq = np.zeros_like(q0)
        for f in (0, 1):
            a, b = apply_chain(chain["_fold"][str(f)], ctx, v0, q0); nv[par == f] = a[par == f]; nq[par == f] = b[par == f]
        cnew[pn] = FS.day_components(held, tv, tq, nv, nq)
        print(pn, "combined", flush=True)
    summ, per = boot(cnew, cref, D)
    print("COMBINED OPT vs C12", {m: (round(v["mean"], 6), [round(c, 6) for c in v["ci"]], v["n_pos"]) for m, v in summ.items()}, flush=True)
    return dict(summary=summ, per_panel=per)


if __name__ == "__main__":
    main()
