"""Claude helper 2026-10-04 (ledger 0369): OPT screen of the joint (v, q) congested-branch projection (phys3_proj.project)
on top of the reference F4 + 0360 (common-mode speed, LOPO/parity-fold coefs from state2 cm stats) + 0358 (flow kriging,
LOPO/parity-fold band weights, gate on the 0360 speed). Second reference: ref + 0362 (r = 0, a = 1, g = 1).
Grid (r, a, g); selection per panel and parity fold by J on the other 9 panels and other-fold days (as flow2_screen);
variant 'safe': best J among combos whose training state delta vs ref >= 0.
Usage: python docs/reference_code/phys3_screen.py stage1 [panel ...]   (per-panel components -> artifacts/claude_ref/phys3/opt_<panel>.npz)
       python docs/reference_code/phys3_screen.py stage2                (selection, bootstrap, summary)"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phys_lwr_diag as P
import flow_krige as FK
import flow_screen as FS
import flow2_screen as F2
import state2_cm as CM
import phys3_proj as PJ

OUT = P.ROOT / "artifacts/claude_ref/phys3"
PANELS = P.P["panels"]
R_GRID = (0., .5, 1., 1.5, 2., 3., 5., np.inf)
A_GRID = (.25, .5, .75, 1.)
G_GRID = (.9, 1., 1.15)
KEYS = ["ref", "ref362"] + [f"r{r}_a{a}_g{g}" for r in R_GRID for a in A_GRID for g in G_GRID]


def reference(panel, held, pv, pq, D):
    """F4 + 0360 (fold coefs) + 0358 (fold weights) for OPT held dates."""
    z = {p: np.load(CM.OUT / f"opt_{p}_cm.npz") for p in PANELS}
    gr = z[panel]["gr"].astype(float)
    vcut = held.stations.sort_values("idx").v_cut.to_numpy(float)
    par = np.arange(D) % 2; nv = pv.copy()
    for f in (0, 1):
        tr = par != f
        coef = CM.solve(sum(z[p]["XX"][tr].sum(0) for p in PANELS if p != panel), sum(z[p]["Xy"][tr].sum(0) for p in PANELS if p != panel))
        nv[par == f] = CM.apply(pv[par == f], gr[par == f], vcut, coef)
    T = np.asarray(held.target, bool)
    nv = np.where(T, nv, pv)
    stats = {p: dict(np.load(FS.OUT / f"{p}_stats.npz")) for p in PANELS}
    N = np.load(FS.OUT / f"{panel}_N.npy").astype(float); den = np.load(FS.OUT / f"{panel}_den.npy").astype(float)
    nq = pq.copy()
    for f in (0, 1):
        fq = FK.blend(nv, pq, N, T, held.stations, FS.fit_weights(stats, panel, f, D), den=den)
        nq[par == f] = fq[par == f]
    return nv, nq


def diag(held, tv, tq, v, q, m):
    w, kj, vcut, cap = PJ._consts(held.stations)
    mm = m & np.isfinite(tv) & np.isfinite(tq) & (tv > 0) & (tq > 0)
    ev = np.log(v[mm] / tv[mm]); eq = np.log(q[mm] / tq[mm])
    W = np.broadcast_to(w, tv.shape)[mm]; K = np.broadcast_to(kj, tv.shape)[mm]
    res_t = np.log(tq[mm]) - np.log(tv[mm] * W * K / (tv[mm] + W))
    res_p = np.log(q[mm]) - np.log(v[mm] * W * K / (v[mm] + W))
    return dict(n=int(mm.sum()), sd_ev=float(ev.std()), sd_eq=float(eq.std()), corr=float(np.corrcoef(ev, eq)[0, 1]),
                truth_branch_res_mean=float(res_t.mean()), truth_branch_res_sd=float(res_t.std()),
                pred_branch_res_mean=float(res_p.mean()), pred_branch_res_sd=float(res_p.std()),
                mean_slope=float(np.mean(W / (tv[mm] + W))))


def stage1(*panels):
    OUT.mkdir(parents=True, exist_ok=True)
    D = len(P.P["OPT_held_dates"])
    for panel in panels or PANELS:
        held, tv, tq, pv, pq = P.held_panel(panel)
        T = np.asarray(held.target, bool)
        rv, rq = reference(panel, held, pv, pq, D)
        C = {}
        C["ref"], reg = FS.day_components(held, tv, tq, rv, rq)
        chk = P.score(held, tv, tq, rv, rq); agg = FS.aggregate(C["ref"], reg, np.ones(D))
        assert abs(chk["J"] - agg["J"]) < 2e-4, (chk, agg)
        q362 = F2.shrink(rv, rq, T, held.stations, 1., 1.)
        C["ref362"] = FS.day_components(held, tv, tq, rv, q362)[0]
        _, _, m1 = PJ.project(rv, rq, T, held.stations, 0., 1., 1.)
        dg = diag(held, tv, tq, rv, rq, m1)
        for r in R_GRID:
            for a in A_GRID:
                for g in G_GRID:
                    nv, nq, _ = PJ.project(rv, rq, T, held.stations, r, a, g)
                    C[f"r{r}_a{a}_g{g}"] = FS.day_components(held, tv, tq, nv, nq)[0]
        assert np.allclose(C["ref362"], C["r0.0_a1.0_g1.0"]), "r=0,a=1,g=1 must equal 0362"
        np.savez(OUT / f"opt_{panel}.npz", regimes=np.array(reg), **{k: C[k] for k in KEYS})
        (OUT / f"opt_{panel}_diag.json").write_text(json.dumps(dg, indent=1))
        J = {k: FS.aggregate(C[k], reg, np.ones(D))["J"] - agg["J"] for k in KEYS}
        best = max(J, key=J.get)
        print(panel, "diag", {k: round(v, 4) for k, v in dg.items()}, "| dJ 362", round(J["ref362"], 5), "best", best, round(J[best], 5), flush=True)


def stage2(nboot=1000):
    D = len(P.P["OPT_held_dates"])
    Z = {p: np.load(OUT / f"opt_{p}.npz") for p in PANELS}
    comps = {p: {k: Z[p][k] for k in KEYS} for p in PANELS}; regs = {p: list(Z[p]["regimes"]) for p in PANELS}
    cand = KEYS[2:]
    agg = lambda p, k, w: FS.aggregate(comps[p][k], regs[p], w)

    def chosen(panel, fold, safe):
        days = (np.arange(D) % 2 != fold).astype(float)
        others = [p for p in PANELS if p != panel]
        base = {p: agg(p, "ref", days) for p in others}
        sc = {}
        for k in cand:
            res = [agg(p, k, days) for p in others]
            dJ = np.mean([x["J"] - base[p]["J"] for x, p in zip(res, others)])
            dS = np.mean([x["S_state"] - base[p]["S_state"] for x, p in zip(res, others)])
            sc[k] = (dJ, dS)
        pool = [k for k in cand if not safe or sc[k][1] >= 0] or ["ref"]
        return max(pool, key=lambda k: sc[k][0] if k in sc else 0.)

    out = {}
    for safe in (False, True):
        sel = {p: {f: chosen(p, f, safe) for f in (0, 1)} for p in PANELS}
        newc = {}
        for p in PANELS:
            Cn = comps[p]["ref"].copy()
            for f in (0, 1):
                rows = np.arange(D) % 2 == f; Cn[rows] = comps[p][sel[p][f]][rows]
            newc[p] = Cn
        name = "safe" if safe else "maxJ"
        out[name] = dict(chosen={p: [sel[p][0], sel[p][1]] for p in PANELS})
        for basek in ("ref", "ref362"):
            per = []
            for p in PANELS:
                a = agg(p, basek, np.ones(D)); b = FS.aggregate(newc[p], regs[p], np.ones(D))
                per.append({k: b[k] - a[k] for k in ("J", "S_state", "S_phys", "S_LWR", "S_FD")})
            rng = np.random.default_rng(0); boots = []
            for _ in range(nboot):
                w = np.bincount(rng.integers(0, D, D), minlength=D).astype(float)
                boots.append({k: np.mean([FS.aggregate(newc[p], regs[p], w)[k] - agg(p, basek, w)[k] for p in PANELS])
                              for k in ("J", "S_state", "S_phys", "S_LWR")})
            B = pd.DataFrame(boots); dfp = pd.DataFrame(per, index=PANELS)
            out[name][f"vs_{basek}"] = dict(
                summary={k: dict(mean=float(dfp[k].mean()), ci=[float(B[k].quantile(.025)), float(B[k].quantile(.975))], n_pos=int((dfp[k] > 0).sum())) for k in B.columns},
                per_panel=dfp.round(6).to_dict(orient="index"))
        print(name, json.dumps({k: v for k, v in out[name].items() if k != "chosen"}, indent=0)[:3000], flush=True)
        print(name, "chosen", out[name]["chosen"], flush=True)
    # ref362 vs ref, and in-sample grid means
    rng = np.random.default_rng(0); boots = []
    for _ in range(nboot):
        w = np.bincount(rng.integers(0, D, D), minlength=D).astype(float)
        boots.append({k: np.mean([agg(p, "ref362", w)[k] - agg(p, "ref", w)[k] for p in PANELS]) for k in ("J", "S_state", "S_phys", "S_LWR")})
    B = pd.DataFrame(boots)
    out["ref362_vs_ref"] = {k: dict(mean=float(np.mean([agg(p, "ref362", np.ones(D))[k] - agg(p, "ref", np.ones(D))[k] for p in PANELS])),
                                    ci=[float(B[k].quantile(.025)), float(B[k].quantile(.975))]) for k in B.columns}
    grid = {}
    for k in cand:
        d = [{m: agg(p, k, np.ones(D))[m] - agg(p, "ref", np.ones(D))[m] for m in ("J", "S_state", "S_phys")} for p in PANELS]
        grid[k] = {m: float(np.mean([x[m] for x in d])) for m in ("J", "S_state", "S_phys")}
    out["grid_in_sample"] = grid
    top = sorted(grid, key=lambda k: -grid[k]["J"])[:12]
    print("ref362 vs ref", out["ref362_vs_ref"]); print("top in-sample", {k: {m: round(v, 5) for m, v in grid[k].items()} for k in top})
    (OUT / "opt_summary.json").write_text(json.dumps(out, indent=1))


def fixed(*keys, nboot=1000):
    """Fixed settings (no per-fold selection) vs ref and ref362, paired date bootstrap; also best pure options in-sample."""
    D = len(P.P["OPT_held_dates"])
    Z = {p: np.load(OUT / f"opt_{p}.npz") for p in PANELS}; regs = {p: list(Z[p]["regimes"]) for p in PANELS}
    agg = lambda p, k, w: FS.aggregate(Z[p][k], regs[p], w)
    ones = np.ones(D); M = ("J", "S_state", "S_phys", "S_LWR")
    g = {k: np.mean([agg(p, k, ones)["J"] - agg(p, "ref", ones)["J"] for p in PANELS]) for k in KEYS[2:]}
    pure0 = max((k for k in g if k.startswith("r0.0_")), key=g.get); pinf = max((k for k in g if k.startswith("rinf_")), key=g.get)
    out = {}
    for k in list(keys) + [pure0, pinf]:
        out[k] = {}
        for b in ("ref", "ref362"):
            per = {p: {m: agg(p, k, ones)[m] - agg(p, b, ones)[m] for m in M} for p in PANELS}
            rng = np.random.default_rng(0); boots = []
            for _ in range(nboot):
                w = np.bincount(rng.integers(0, D, D), minlength=D).astype(float)
                boots.append({m: np.mean([agg(p, k, w)[m] - agg(p, b, w)[m] for p in PANELS]) for m in M})
            B = pd.DataFrame(boots)
            out[k][f"vs_{b}"] = {m: dict(mean=float(np.mean([per[p][m] for p in PANELS])), ci=[float(B[m].quantile(.025)), float(B[m].quantile(.975))],
                                         n_pos=int(sum(per[p][m] > 0 for p in PANELS))) for m in M}
            out[k][f"per_panel_vs_{b}"] = {p: {m: round(per[p][m], 6) for m in M} for p in PANELS}
            print(k, b, {m: (round(v["mean"], 5), [round(c, 5) for c in v["ci"]], v["n_pos"]) for m, v in out[k][f"vs_{b}"].items()}, flush=True)
    out["best_pure"] = dict(r0=pure0, rinf=pinf)
    (OUT / "opt_fixed.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"stage1": stage1, "stage2": stage2, "fixed": fixed}[sys.argv[1]](*sys.argv[2:])
