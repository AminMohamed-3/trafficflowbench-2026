"""Claude helper 2026-10-04 (ledger 0403, B6 v2 idea G): uncertainty-gated blend. The imputer's seed spread (std over seeds
of the per-cell prediction) says where the imputer is unsure; trust C11 more there. Base blend = b6full per panel shrunk
weights (fit odd days). Gate: pooled additive weight offsets per spread bin (quartiles of the pooled odd-day spread,
separately for speed and flow and for non-block / block cells), w = clip(w_panel + delta_bin, 0, 1), deltas on a .05 grid
fit on odd days (coordinate descent, 2 sweeps). Then phys3 (BLP). Scored on even days with own_instrument.paired_state vs
the ungated BLP of the same imputer arm. OPT: same, cross-fit by day parity (gap = none).
Usage: python docs/reference_code/b6v2_gate.py own|opt <out_name> <arm with >= 2 seeds, b6v2_eval syntax>"""
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
import b6v2_eval as V2

PANELS = P.P["panels"]
DG = np.round(np.arange(-.5, .51, .05), 2)
NB = 4
MODE2 = os.environ.get("B6V2_GATE", "1") == "2"   # 2: bins = spread quartile x (C11 speed < v_cut)
MODE3 = os.environ.get("B6V2_GATE", "1") == "3"   # 3: 'spread' := |IMP - C11| disagreement (works with one seed)
TAGG = {"1": "", "2": "2", "3": "3"}[os.environ.get("B6V2_GATE", "1")]
FITPAR = int(os.environ.get("B6V2_FITPAR", "1"))   # OWN: fit on day index parity FITPAR (1 = odd days, default), score the other


def load_sd(arm, mode, panel):
    zs = [np.load(V2.find(arm, f"imp_{arm['pre']}{mode}_{panel}_s{k}.npz")) for k in arm["seeds"]]
    v = np.stack([z["speed" + arm["suf"]] for z in zs]).astype(float); q = np.stack([z["flow" + arm["suf"]] for z in zs]).astype(float)
    return v.mean(0), q.mean(0), v.std(0), q.std(0)


def disagree(cv, cq, iv, iq):
    return np.abs(iv - cv), np.abs(iq - cq)


def cells(cv, cq, iv, iq, sv, sq, tv, tq, la, gap, m, w, cg):
    """Flattened per-cell arrays for the fit: for key k: (C11, IMP, truth, base weight, spread)."""
    out = {}
    for k, (c, i, t, s, sc) in dict(v=(cv, iv, tv, sv, None), q=(cq, iq, tq, sq, la)).items():
        for part, mm in (("nb", m & ~gap), ("blk", m & gap)):
            if not mm.any():
                continue
            d = 1. if sc is None else sc[mm]
            sp = s[mm] if sc is None else s[mm] / sc[mm]
            out[f"{k}_{part}"] = (c[mm] / d, i[mm] / d, t[mm] / d, np.full(mm.sum(), w[f"{k}_{part}"]), sp, cg[mm])
    return out


def fit_gate(pool):
    """pool[key] = list of per-panel tuples; returns bin edges and deltas per key."""
    res = {}
    for k, L in pool.items():
        c, i, t, w, s, cg = (np.concatenate([x[j] for x in L]) for j in range(6))
        edges = np.quantile(s, np.linspace(0, 1, NB + 1)[1:-1])
        b = np.searchsorted(edges, s) + (NB * cg if MODE2 else 0)
        nb = 2 * NB if MODE2 else NB
        dl = np.zeros(nb)
        for _ in range(2):
            for j in range(nb):
                if not (b == j).any():
                    continue
                mm = b == j
                err = [np.sum((np.clip(w[mm] + g, 0, 1) * c[mm] + (1 - np.clip(w[mm] + g, 0, 1)) * i[mm] - t[mm]) ** 2) for g in DG]
                dl[j] = DG[int(np.argmin(err))]
        res[k] = dict(edges=edges.tolist(), delta=dl.tolist())
    return res


def gated(cv, cq, iv, iq, sv, sq, la, gap, w, G, cg):
    g = np.asarray(gap, bool); out = []
    for k, c, i, s in (("v", cv, iv, sv), ("q", cq, iq, sq / la)):
        wt = np.where(g, w[f"{k}_blk"], w[f"{k}_nb"]).astype(float)
        for part, mm in (("nb", ~g), ("blk", g)):
            key = f"{k}_{part}"
            if key in G:
                b = np.searchsorted(np.asarray(G[key]["edges"]), s) + (NB * cg if MODE2 else 0)
                wt = np.where(mm, np.clip(wt + np.asarray(G[key]["delta"])[b], 0, 1), wt)
        out.append(wt * c + (1 - wt) * i)
    return out


def own(name, arm):
    import flow_own as FO
    own_m, binding = FO.own, FO.binding
    p = P.P
    items = {i["panel"]: i for i in json.loads(Path(p["OWN_metadata_binding"]["manifest"]).read_text())["items"] if i["period"] == "validation"}
    data = {}; stats = {}; stats_all = {}
    for pn in PANELS:
        view, tv, tq, st_, hidden = own_m.load_item(P.ROOT, binding.bound_own_item(p, items[pn]))
        st_ = np.asarray(st_, bool); gap = np.asarray(view.gap, bool); tv = np.asarray(tv, float); tq = np.asarray(tq, float)
        r = np.load(EV.OUT / f"ref_own_{pn}.npz"); cv, cq = r["c11_v"].astype(float), r["c11_q"].astype(float)
        iv, iq, sv, sq = load_sd(arm, "own", pn)
        if MODE3:
            sv, sq = disagree(cv, cq, iv, iq)
        la = np.broadcast_to(np.maximum(view.stations.sort_values("idx").lanes.to_numpy(float), 1.), tv.shape)
        day = np.broadcast_to(np.arange(tv.shape[0])[:, None, None], tv.shape)
        m = st_ & np.isfinite(tv) & np.isfinite(tq) & (day % 2 == FITPAR)
        ma = st_ & np.isfinite(tv) & np.isfinite(tq)
        stats_all[pn] = dict(v_nb=EV.sse_grid(cv, iv, tv, ma & ~gap), v_blk=EV.sse_grid(cv, iv, tv, ma & gap),
                             q_nb=EV.sse_grid(cq, iq, tq, ma & ~gap, la), q_blk=EV.sse_grid(cq, iq, tq, ma & gap, la))
        stats[pn] = dict(v_nb=EV.sse_grid(cv, iv, tv, m & ~gap), v_blk=EV.sse_grid(cv, iv, tv, m & gap),
                         q_nb=EV.sse_grid(cq, iq, tq, m & ~gap, la), q_blk=EV.sse_grid(cq, iq, tq, m & gap, la))
        cg = (cv < view.stations.sort_values("idx").v_cut.to_numpy(float)).astype(int)
        data[pn] = (view, tv, tq, st_, gap, cv, cq, iv, iq, sv, sq, la, m, cg)
    W, _ = EV.fit_weights(stats)
    pool = {}
    for pn, (view, tv, tq, st_, gap, cv, cq, iv, iq, sv, sq, la, m, cg) in data.items():
        for k, x in cells(cv, cq, iv, iq, sv, sq, tv, tq, la, gap, m, W[pn], cg).items():
            pool.setdefault(k, []).append(x)
    G = fit_gate(pool); print("gate", json.dumps(G), flush=True)
    WA, _ = EV.fit_weights(stats_all); pool = {}   # deployment: weights and gate fit on all OWN days
    for pn, (view, tv, tq, st_, gap, cv, cq, iv, iq, sv, sq, la, m, cg) in data.items():
        ma = st_ & np.isfinite(tv) & np.isfinite(tq)
        for k, x in cells(cv, cq, iv, iq, sv, sq, tv, tq, la, gap, ma, WA[pn], cg).items():
            pool.setdefault(k, []).append(x)
    GA = fit_gate(pool); print("gate all days", json.dumps(GA), flush=True)
    rows = {}
    for pn, (view, tv, tq, st_, gap, cv, cq, iv, iq, sv, sq, la, m, cg) in data.items():
        for n, (bv, bq) in (("BLP", EV.blend(cv, cq, iv, iq, gap, W[pn])), ("GATE", gated(cv, cq, iv, iq, sv, sq, la, gap, W[pn], G, cg))):
            pv_, pq_, _ = PJ.project(bv, bq, st_, view.stations, ST.R, ST.A, ST.G, gap=gap)
            for part, mm in (("all", st_), ("nonblock", st_ & ~gap)):
                rows.setdefault((n, part), []).extend(own_m.state_rows(view, tv, tq, pv_, pq_, mm)[0])
        print(pn, flush=True)
    fam = p["families"]; groups = [[i for i, x in enumerate(PANELS) if fam[x] == f] for f in sorted(set(fam.values()))]
    dates = sorted({r["date"] for r in rows[("BLP", "all")]}); even = [d for i, d in enumerate(dates) if i % 2 != FITPAR]
    res = dict(arm=arm, gate=G, gate_all_days=GA, weights_all_days=WA, contrasts={})
    for part in ("nonblock", "all"):
        x = own_m.paired_state([r for r in rows[("BLP", part)] if r["date"] in even], [r for r in rows[("GATE", part)] if r["date"] in even], PANELS, even, groups)
        res["contrasts"][f"GATE_vs_BLP_{part}"] = dict(delta=x["delta_S_state"], ci=x["ci95"], verdict=x["verdict"])
        print(f"GATE_vs_BLP_{part}", round(x["delta_S_state"], 5), [round(c, 5) for c in x["ci95"]], x["verdict"], flush=True)
    (V2.OUT / f"gate{TAGG}_own_{name}{'' if FITPAR == 1 else '_fiteven'}.json").write_text(json.dumps(res, indent=1, default=str))


def opt(name, arm, nboot=1000):
    import flow_screen as FS
    MON = os.environ.get("B6V2_OPTMON")  # optional: restrict the OPT test to one held month (jun or jul)
    dates = P.P["OPT_held_dates"]
    sel = [i for i, d in enumerate(dates) if MON is None or d.startswith({"jun": "2030-06", "jul": "2030-07"}[MON])]
    D = len(sel); par = np.arange(D) % 2
    jun = np.array([dates[i].startswith("2030-06") for i in sel])
    data = {}; stats = {0: {}, 1: {}}
    for pn in PANELS:
        held, tv, tq, pv, pq = P.held_panel(pn)
        r = np.load(EV.OUT / f"ref_opt_{pn}.npz"); cv, cq = r["c11_v"].astype(float), r["c11_q"].astype(float)
        if MON:
            cache = P.data.load_cache(pn, "train", root=P.ROOT)
            held = P.base.subset_cache(cache, [cache.dates.index(dates[i]) for i in sel]); tv, tq, cv, cq = tv[sel], tq[sel], cv[sel], cq[sel]
        T = np.asarray(held.target, bool); gap = np.zeros_like(T)
        a = load_sd(arm, "opt_jun", pn) if jun.any() else None; b = load_sd(arm, "opt_jul", pn) if (~jun).any() else None
        iv = np.zeros_like(cv); iq = np.zeros_like(cv); sv = np.zeros_like(cv); sq = np.zeros_like(cv)
        for arr, j in ((iv, 0), (iq, 1), (sv, 2), (sq, 3)):
            if a is not None:
                arr[jun] = a[j]
            if b is not None:
                arr[~jun] = b[j]
        if MODE3:
            sv, sq = disagree(cv, cq, iv, iq)
        la = np.broadcast_to(np.maximum(held.stations.sort_values("idx").lanes.to_numpy(float), 1.), T.shape)
        ok = T & np.isfinite(tv) & np.isfinite(tq); day = np.broadcast_to(par[:, None, None], T.shape)
        for f in (0, 1):
            m = ok & (day == f)
            stats[f][pn] = dict(v_nb=EV.sse_grid(cv, iv, tv, m), v_blk=np.zeros(len(EV.GRID)), q_nb=EV.sse_grid(cq, iq, tq, m, la), q_blk=np.zeros(len(EV.GRID)))
        cg = (cv < held.stations.sort_values("idx").v_cut.to_numpy(float)).astype(int)
        data[pn] = (held, tv, tq, T, gap, cv, cq, iv, iq, sv, sq, la, ok, day, cg)
        print("load", pn, flush=True)
    W = {f: EV.fit_weights(stats[1 - f])[0] for f in (0, 1)}
    G = {}
    for f in (0, 1):  # gate for days of parity f fitted on the other parity
        pool = {}
        for pn, (held, tv, tq, T, gap, cv, cq, iv, iq, sv, sq, la, ok, day, cg) in data.items():
            for k, x in cells(cv, cq, iv, iq, sv, sq, tv, tq, la, gap, ok & (day == 1 - f), W[f][pn], cg).items():
                pool.setdefault(k, []).append(x)
        G[f] = fit_gate(pool)
    print("gate", json.dumps(G), flush=True)
    comps = {}; regs = {}
    for pn, (held, tv, tq, T, gap, cv, cq, iv, iq, sv, sq, la, ok, day, cg) in data.items():
        comps[pn] = {}
        for n in ("BLP", "GATE"):
            bv, bq = cv.copy(), cq.copy()
            for f in (0, 1):
                x = EV.blend(cv, cq, iv, iq, gap, W[f][pn]) if n == "BLP" else gated(cv, cq, iv, iq, sv, sq, la, gap, W[f][pn], G[f], cg)
                bv[par == f] = x[0][par == f]; bq[par == f] = x[1][par == f]
            bv = np.where(T, bv, cv); bq = np.where(T, bq, cq)
            pv_, pq_, _ = PJ.project(bv, bq, T, held.stations, ST.R, ST.A, ST.G)
            comps[pn][n], regs[pn] = FS.day_components(held, tv, tq, pv_, pq_)
        print(pn, flush=True)
    M = ("J", "S_state", "S_phys", "S_LWR", "S_FD")
    agg = lambda n, p, w: FS.aggregate(comps[p][n], regs[p], w)
    per = {p: {m: agg("GATE", p, np.ones(D))[m] - agg("BLP", p, np.ones(D))[m] for m in M} for p in PANELS}
    rng = np.random.default_rng(0); boots = []
    for _ in range(nboot):
        w = np.bincount(rng.integers(0, D, D), minlength=D).astype(float)
        boots.append({m: np.mean([agg("GATE", p, w)[m] - agg("BLP", p, w)[m] for p in PANELS]) for m in M})
    Bt = pd.DataFrame(boots)
    res = dict(arm=arm, gate=G, GATE_vs_BLP={m: dict(mean=float(np.mean([per[p][m] for p in PANELS])), ci=[float(Bt[m].quantile(.025)), float(Bt[m].quantile(.975))],
                                                   n_pos=int(sum(per[p][m] > 0 for p in PANELS))) for m in M})
    print("GATE_vs_BLP", {m: (round(v["mean"], 5), [round(c, 5) for c in v["ci"]], v["n_pos"]) for m, v in res["GATE_vs_BLP"].items()}, flush=True)
    (V2.OUT / f"gate{TAGG}_opt_{name}{'_' + os.environ['B6V2_OPTMON'] if os.environ.get('B6V2_OPTMON') else ''}.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"own": own, "opt": opt}[sys.argv[1]](sys.argv[2], V2.parse("a=" + sys.argv[3]))
