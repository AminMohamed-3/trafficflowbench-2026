"""Claude helper 2026-10-04 (ledger 0403, B6 v2): paired comparison of imputer variants, each blended with the C11 stack and
re-projected with phys3 (BLP), exactly as b6full_eval (blend, weight grid, shrinkage .5 to pooled, phys3 r=3 a=.75 g=1).
Arms: name=dir:prefix:seeds:suffix  -> files artifacts/claude_ref/<dir>/imp_<prefix><mode>_<panel>_s<k>.npz, fields
speed<suffix>/flow<suffix>, seeds averaged. The FIRST arm is the reference (normally B6 v1: b6full::0,1,2:).
OWN: weights fit on odd days, scored on even days; own_instrument.paired_state (date bootstrap, family groups),
non-block and all targets; each arm vs C11 and vs the reference arm.
OPT: weights cross-fit by day parity; flow_screen.day_components (official mode A incl. S_LWR); date bootstrap over 10
panels; each arm vs C11 and vs the reference arm.
Usage: python docs/reference_code/b6v2_eval.py own|opt <out_name> <arm> [<arm> ...]"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phys_lwr_diag as P
import phys3_proj as PJ
import phys3_stack as ST
import b6full_eval as EV

CR = P.ROOT / "artifacts/claude_ref"
OUT = CR / "b6v2"
PANELS = P.P["panels"]


def parse(a):
    name, rest = a.split("=", 1); d, pre, seeds, suf = rest.split(":")
    return dict(name=name, dir=d, pre=pre, seeds=[int(s) for s in seeds.split(",")], suf=suf)


def find(arm, name):
    """arm dir and prefix may list alternatives joined by '+' (e.g. dir b6v2+b6v2_pull: laptop and VM-built files;
    prefix priv_+base_priv_: v1 file names and v2 file names); first existing (dir, prefix) wins."""
    pres = arm["pre"].split("+")
    rest = name[len("imp_") + len(arm["pre"]):] if name.startswith("imp_" + arm["pre"]) else None
    for d in arm["dir"].split("+"):
        for pre in (pres if rest is not None else [None]):
            n = name if pre is None else f"imp_{pre}{rest}"
            if (CR / d / n).exists():
                return CR / d / n
    raise FileNotFoundError(f"{name} in {arm['dir']}")


def load_imp(arm, mode, panel):
    zs = [np.load(find(arm, f"imp_{arm['pre']}{mode}_{panel}_s{k}.npz")) for k in arm["seeds"]]
    return (np.mean([z["speed" + arm["suf"]] for z in zs], 0).astype(float),
            np.mean([z["flow" + arm["suf"]] for z in zs], 0).astype(float))


def own(name, arms):
    import flow_own as FO
    own_m, binding = FO.own, FO.binding
    p = P.P
    items = {i["panel"]: i for i in json.loads(Path(p["OWN_metadata_binding"]["manifest"]).read_text())["items"] if i["period"] == "validation"}

    def panel_data(pn):
        view, tv, tq, st_, hidden = own_m.load_item(P.ROOT, binding.bound_own_item(p, items[pn]))
        st_ = np.asarray(st_, bool); gap = np.asarray(view.gap, bool)
        r = np.load(EV.OUT / f"ref_own_{pn}.npz"); assert (r["target"] == st_).all()
        return view, np.asarray(tv, float), np.asarray(tq, float), st_, gap, r["c11_v"].astype(float), r["c11_q"].astype(float)
    stats_p = {par: {a["name"]: {} for a in arms} for par in (0, 1)}
    for pn in PANELS:  # pass 1: odd-day SSE grids per arm
        view, tv, tq, st_, gap, cv, cq = panel_data(pn)
        la = np.broadcast_to(np.maximum(view.stations.sort_values("idx").lanes.to_numpy(float), 1.), tv.shape)
        day = np.broadcast_to(np.arange(tv.shape[0])[:, None, None], tv.shape)
        ok0 = st_ & np.isfinite(tv) & np.isfinite(tq)
        for a in arms:
            iv, iq = load_imp(a, "own", pn)
            for par in (1, 0):  # odd days (fit for the even-day score) and even days (for all-days probe weights)
                m = ok0 & (day % 2 == par)
                stats_p[par][a["name"]][pn] = dict(v_nb=EV.sse_grid(cv, iv, tv, m & ~gap), v_blk=EV.sse_grid(cv, iv, tv, m & gap),
                                                   q_nb=EV.sse_grid(cq, iq, tq, m & ~gap, la), q_blk=EV.sse_grid(cq, iq, tq, m & gap, la))
        print("pass1", pn, flush=True)
    W = {k: EV.fit_weights(v) for k, v in stats_p[1].items()}
    WALL = {k: EV.fit_weights({pn: {q: v[pn][q] + stats_p[0][k][pn][q] for q in v[pn]} for pn in v})[0] for k, v in stats_p[1].items()}
    rows = {}; per = {}
    for pn in PANELS:  # pass 2: blend with odd-day weights, phys3, rows
        view, tv, tq, st_, gap, cv, cq = panel_data(pn)
        rows.setdefault(("C11", "all"), []).extend(own_m.state_rows(view, tv, tq, cv, cq, st_)[0])
        rows.setdefault(("C11", "nonblock"), []).extend(own_m.state_rows(view, tv, tq, cv, cq, st_ & ~gap)[0])
        ok = st_ & np.isfinite(tv) & np.isfinite(tq); per[pn] = {}
        for a in arms:
            iv, iq = load_imp(a, "own", pn)
            bv, bq = EV.blend(cv, cq, iv, iq, gap, W[a["name"]][0][pn])
            pv_, pq_, _ = PJ.project(bv, bq, st_, view.stations, ST.R, ST.A, ST.G, gap=gap)
            for part, m in (("all", st_), ("nonblock", st_ & ~gap)):
                rows.setdefault((a["name"], part), []).extend(own_m.state_rows(view, tv, tq, pv_, pq_, m)[0])
            per[pn][a["name"]] = dict(w=W[a["name"]][0][pn], weights_all_days=WALL[a["name"]][pn], rmse_v_nb_imp=float(np.sqrt(np.mean((iv - tv)[ok & ~gap] ** 2))),
                                      rmse_v_blk_imp=float(np.sqrt(np.mean((iv - tv)[ok & gap] ** 2))) if (ok & gap).any() else None)
        print("pass2", pn, {k: round(v["rmse_v_nb_imp"], 3) for k, v in per[pn].items()}, flush=True)
    fam = p["families"]; groups = [[i for i, x in enumerate(PANELS) if fam[x] == f] for f in sorted(set(fam.values()))]
    dates = sorted({r["date"] for r in rows[("C11", "all")]}); even = [d for i, d in enumerate(dates) if i % 2 == 0]
    res = dict(arms=arms, pooled_weights={k: v[1] for k, v in W.items()}, per_panel=per, contrasts={})
    ref = arms[0]["name"]
    for a in arms:
        for base in ("C11", ref):
            if base == a["name"]:
                continue
            for part in ("nonblock", "all"):
                x = own_m.paired_state([r for r in rows[(base, part)] if r["date"] in even], [r for r in rows[(a["name"], part)] if r["date"] in even],
                                       PANELS, even, groups)
                k = f"{a['name']}_vs_{base}_{part}"
                res["contrasts"][k] = dict(delta=x["delta_S_state"], ci=x["ci95"], verdict=x["verdict"])
                print(k, round(x["delta_S_state"], 5), [round(c, 5) for c in x["ci95"]], x["verdict"], flush=True)
    (OUT / f"own_{name}.json").write_text(json.dumps(res, indent=1, default=str))


def opt(name, arms, nboot=1000):
    import flow_screen as FS
    dates = P.P["OPT_held_dates"]; D = len(dates); par = np.arange(D) % 2
    jun = np.array([d.startswith("2030-06") for d in dates])
    comps = {}; regs = {}; stats = {a["name"]: {0: {}, 1: {}} for a in arms}; cache = {}
    for pn in PANELS:
        held, tv, tq, pv, pq = P.held_panel(pn)
        r = np.load(EV.OUT / f"ref_opt_{pn}.npz"); cv, cq = r["c11_v"].astype(float), r["c11_q"].astype(float)
        T = np.asarray(held.target, bool)
        la = np.broadcast_to(np.maximum(held.stations.sort_values("idx").lanes.to_numpy(float), 1.), T.shape)
        ok = T & np.isfinite(tv) & np.isfinite(tq); day = np.broadcast_to(par[:, None, None], T.shape)
        imps = {}
        for a in arms:
            x = load_imp(a, "opt_jun", pn); y = load_imp(a, "opt_jul", pn)
            iv = np.zeros_like(cv); iq = np.zeros_like(cq); iv[jun], iq[jun] = x; iv[~jun], iq[~jun] = y
            imps[a["name"]] = (iv, iq)
            for f in (0, 1):
                m = ok & (day == f)
                stats[a["name"]][f][pn] = dict(v_nb=EV.sse_grid(cv, iv, tv, m), v_blk=np.zeros(len(EV.GRID)),
                                               q_nb=EV.sse_grid(cq, iq, tq, m, la), q_blk=np.zeros(len(EV.GRID)))
        cache[pn] = imps
        print("pass1", pn, flush=True)
    W = {a["name"]: {f: EV.fit_weights(stats[a["name"]][1 - f])[0] for f in (0, 1)} for a in arms}
    rows = []
    for pn in PANELS:
        held, tv, tq, pv, pq = P.held_panel(pn)
        r = np.load(EV.OUT / f"ref_opt_{pn}.npz"); cv, cq = r["c11_v"].astype(float), r["c11_q"].astype(float)
        T = np.asarray(held.target, bool); gap = np.zeros_like(T)
        comps[pn] = {}
        comps[pn]["C11"], regs[pn] = FS.day_components(held, tv, tq, cv, cq)
        for a in arms:
            iv, iq = cache[pn][a["name"]]; bv, bq = cv.copy(), cq.copy()
            for f in (0, 1):
                x = EV.blend(cv, cq, iv, iq, gap, W[a["name"]][f][pn]); bv[par == f] = x[0][par == f]; bq[par == f] = x[1][par == f]
            bv = np.where(T, bv, cv); bq = np.where(T, bq, cq)
            pv_, pq_, _ = PJ.project(bv, bq, T, held.stations, ST.R, ST.A, ST.G)
            comps[pn][a["name"]], _ = FS.day_components(held, tv, tq, pv_, pq_)
        one = {n: FS.aggregate(c, regs[pn], np.ones(D)) for n, c in comps[pn].items()}
        rows.append(dict(panel=pn, **{f"dJ_{n}": one[n]["J"] - one["C11"]["J"] for n in one if n != "C11"}))
        print("pass2", pn, {k: round(v, 5) for k, v in rows[-1].items() if k != "panel"}, flush=True)
        del cache[pn]
    M = ("J", "S_state", "S_phys", "S_LWR", "S_FD")
    res = dict(arms=arms, per_panel=rows, contrasts={})
    ref = arms[0]["name"]
    for a in arms:
        for base in ("C11", ref):
            if base == a["name"]:
                continue
            agg = lambda n, p, w: FS.aggregate(comps[p][n], regs[p], w)
            per = {p: {m: agg(a["name"], p, np.ones(D))[m] - agg(base, p, np.ones(D))[m] for m in M} for p in PANELS}
            rng = np.random.default_rng(0); boots = []
            for _ in range(nboot):
                w = np.bincount(rng.integers(0, D, D), minlength=D).astype(float)
                boots.append({m: np.mean([agg(a["name"], p, w)[m] - agg(base, p, w)[m] for p in PANELS]) for m in M})
            Bt = pd.DataFrame(boots); k = f"{a['name']}_vs_{base}"
            res["contrasts"][k] = {m: dict(mean=float(np.mean([per[p][m] for p in PANELS])), ci=[float(Bt[m].quantile(.025)), float(Bt[m].quantile(.975))],
                                           n_pos=int(sum(per[p][m] > 0 for p in PANELS))) for m in M}
            print(k, {m: (round(v["mean"], 5), [round(c, 5) for c in v["ci"]], v["n_pos"]) for m, v in res["contrasts"][k].items()}, flush=True)
    (OUT / f"opt_{name}.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"own": own, "opt": opt}[sys.argv[1]](sys.argv[2], [parse(a) for a in sys.argv[3:]])
