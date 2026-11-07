"""Claude helper phys4 (2026-10-04, ledger 0405): LSA blend on top of C13 (free-flow target cells).
On target cells with C13 v > G vf and LSA support: log q += wq (LSA_q - log q), log v += wv (LSA_v - log v).
LSA (phys4_lsa.lsa, W 12 flat, K 72) estimated from observed free cells (released v > .9 vf) of the same split.
OPT: J_S per config on the grid; config chosen on the other day parity (pooled 10 panels, mean panel J), spliced, day
bootstrap vs C13 (cross-fit as polish_eval). OWN: I-S-OWN March, all-days config, non-block and all targets vs C13 (_all chain).
Usage: python docs/reference_code/phys4_lsa_eval.py opt | own --prm JSON"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import phys4_common as C
import phys4_lsa as LS
import phys_lwr_diag as P
import polish_eval as PE
import flow_screen as FS
from tfb import metric

OUT = C.OUT
LSA_CFG = dict(W=12, K=72)
GRID = [dict(wq=a, cq=cq, wv=b, cv=cv, G=.9) for a in (0., .3, .5, .7) for cq in (.015, .03) for b in (0., .3, .5) for cv in (.01, .02)]
GRID = [json.loads(x) for x in dict.fromkeys(json.dumps({k: (v if (k[1:] != 'q' or g['wq']) and (k[1:] != 'v' or g['wv']) else 0) for k, v in g.items()}) for g in GRID)]


def vf_of(view):
    return metric._parameters(view, P.ROOT)[0].free_speed_kmh.to_numpy(float)


def lsa_fields(speed_obs, flow_obs, vf, mp=None):
    """speed_obs/flow_obs: released observed values (nan elsewhere). Returns (Nq, nq, Nv, nv)."""
    ob = np.isfinite(speed_obs) & np.isfinite(flow_obs) & (flow_obs > 0) & (speed_obs > .9 * vf[None, None, :])
    Xq = np.log(np.maximum(np.nan_to_num(flow_obs, nan=1.), 1.)); Xv = np.log(np.maximum(np.nan_to_num(speed_obs, nan=1.), 1.))
    Nq, nq = LS.lsa(Xq, ob, mp=mp, **LSA_CFG); Nv, nv = LS.lsa(Xv, ob, mp=mp, **LSA_CFG)
    return Nq, nq, Nv, nv


def apply(v, q, T, vf, F, prm, gap=None):
    Nq, nq, Nv, nv = F
    m = T & np.isfinite(v) & np.isfinite(q) & (v > prm["G"] * vf[None, None, :]) & (nq > 0) & (q > 0)
    if prm.get("xgap") and gap is not None:
        m &= ~np.asarray(gap, bool)
    nv_, nq_ = v.copy(), q.copy()
    lq = np.log(np.maximum(q, 1.)); lv = np.log(np.maximum(v, 1.))
    cq, cv = prm.get("cq", 9.), prm.get("cv", 9.)
    nq_ = np.where(m, np.exp(lq + prm["wq"] * np.clip(Nq - lq, -cq, cq)), q)
    nv_ = np.where(m & (nv > 0), np.exp(lv + prm["wv"] * np.clip(Nv - lv, -cv, cv)), v)
    return nv_, nq_


def opt():
    D = C.D_OPT; par = C.PAR
    comps = {}; cref = {}
    t0 = time.time()
    for pn in C.PANELS:
        held, tv, tq, ctx, v, q = C.c13_opt(pn)
        T = np.asarray(held.target, bool); el = np.asarray(held.elig, bool); vf = vf_of(held)
        so = np.where(el & ~T, tv, np.nan); fo = np.where(el & ~T, tq, np.nan)
        F = lsa_fields(so, fo, vf)
        cref[pn] = FS.day_components(held, tv, tq, v, q)
        comps[pn] = [FS.day_components(held, tv, tq, *apply(v, q, T, vf, F, g))[0] for g in GRID]
        print(pn, round(time.time() - t0), "s", flush=True)
    chosen = {}
    for f in (0, 1):
        w = (par == 1 - f).astype(float)
        sc = [np.mean([FS.aggregate(comps[pn][i], cref[pn][1], w)["J"] for pn in C.PANELS]) for i in range(len(GRID))]
        chosen[f] = int(np.argmax(sc))
    sc_all = [np.mean([FS.aggregate(comps[pn][i], cref[pn][1], np.ones(D))["J"] - FS.aggregate(cref[pn][0], cref[pn][1], np.ones(D))["J"] for pn in C.PANELS]) for i in range(len(GRID))]
    full = int(np.argmax(sc_all))
    print("chosen by fold", {f: GRID[i] for f, i in chosen.items()}, "all days", GRID[full], flush=True)
    cnew = {}
    for pn in C.PANELS:
        Cn = cref[pn][0].copy()
        for f in (0, 1):
            Cn[par == f] = comps[pn][chosen[f]][par == f]
        cnew[pn] = (Cn, cref[pn][1])
    summ, per = PE.boot(cnew, cref, D)
    print("LSA OPT cross-fit vs C13", {m: (round(x["mean"], 6), [round(c, 6) for c in x["ci"]], x["n_pos"]) for m, x in summ.items()}, flush=True)
    # also each config all-days (in-sample) J delta for the record
    res = dict(grid=GRID, chosen={str(f): GRID[i] for f, i in chosen.items()}, all_days=GRID[full],
               grid_J_all_days={json.dumps(g): round(float(s), 7) for g, s in zip(GRID, sc_all)}, summary=summ, per_panel=per)
    (OUT / "eval_lsa_opt.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def own(prm, tag="lsa"):
    import flow_own as FO
    own_m, binding = FO.own, FO.binding
    p = P.P
    items = [i for i in json.loads(Path(p["OWN_metadata_binding"]["manifest"]).read_text())["items"] if i["period"] == "validation"]
    rows = {}; info = {}
    for item in items:
        pn = item["panel"]
        view, tv, tq, st_, hidden = own_m.load_item(P.ROOT, binding.bound_own_item(p, item))
        st_ = np.asarray(st_, bool); gap = np.asarray(view.gap, bool)
        z = np.load(PE.OUT / f"c12_own_{pn}.npz"); assert (z["target"] == st_).all()
        speed = np.asarray(view.speed, float); flow = np.asarray(view.flow, float)
        observed = np.asarray(view.elig, bool) & ~np.asarray(view.target, bool) & ~gap & np.isfinite(speed)
        ctx = PE.Ctx(view.stations, st_, np.where(observed, speed, np.nan), z["p"].astype(float), z["bl_v"].astype(float), z["bl_q"].astype(float), gap)
        bv, bq = PE.apply_chain(C.CHAIN["_all"], ctx, z["v"].astype(float), z["q"].astype(float))
        vf = vf_of(view)
        F = lsa_fields(np.where(observed, speed, np.nan), np.where(observed & np.isfinite(flow), flow, np.nan), vf)
        nv, nq = apply(bv, bq, st_, vf, F, prm, gap)
        for name, (v, q) in (("ref", (bv, bq)), ("new", (nv, nq))):
            for part, m in (("all", st_), ("nonblock", st_ & ~gap)):
                rows.setdefault((name, part), []).extend(own_m.state_rows(view, tv, tq, v, q, m)[0])
        ok = st_ & ~gap & np.isfinite(tv) & np.isfinite(tq)
        la = np.maximum(view.stations.sort_values("idx").lanes.to_numpy(float), 1.)
        info[pn] = dict(rmse_v_ref=float(np.sqrt(np.mean((bv - tv)[ok] ** 2))), rmse_v_new=float(np.sqrt(np.mean((nv - tv)[ok] ** 2))),
                        rmse_ql_ref=float(np.sqrt(np.mean(((bq - tq) / la)[ok] ** 2))), rmse_ql_new=float(np.sqrt(np.mean(((nq - tq) / la)[ok] ** 2))),
                        changed=float(np.mean((np.abs(nv - bv) > 1e-6) | (np.abs(nq - bq) > 1e-6))))
        print(pn, {k: round(x, 4) for k, x in info[pn].items()}, flush=True)
    panels = p["panels"]; fam = p["families"]
    groups = [[i for i, x in enumerate(panels) if fam[x] == f] for f in sorted(set(fam.values()))]
    dates = sorted({r["date"] for r in rows[("ref", "all")]})
    res = dict(prm=prm, per_panel=info)
    for part in ("nonblock", "all"):
        x = own_m.paired_state(rows[("ref", part)], rows[("new", part)], panels, dates, groups)
        bp = {pn: round(own_m.paired_state([r for r in rows[("ref", part)] if r["panel"] == pn], [r for r in rows[("new", part)] if r["panel"] == pn], [pn], dates, [[0]])["delta_S_state"], 6) for pn in panels}
        res[part] = dict(delta=x["delta_S_state"], ci=x["ci95"], verdict=x["verdict"], ref=x["incumbent_S_state"], by_panel=bp, n_pos=int(sum(v > 0 for v in bp.values())))
        print(tag, "OWN", part, round(x["delta_S_state"], 6), [round(c, 6) for c in x["ci95"]], x["verdict"], flush=True)
    (OUT / f"eval_{tag}_own.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--prm", default=None); ap.add_argument("--tag", default="lsa")
    a = ap.parse_args()
    if a.cmd == "opt":
        opt()
    else:
        own(json.loads(a.prm), a.tag)
