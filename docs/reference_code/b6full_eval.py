"""Claude helper 2026-10-04 (ledger 0399, B6 full): blend the per-split imputer (b6full_imp, seeds averaged) with the C11 stack
(b6full_ref) and score.
Blend: v = w_v C11 + (1 - w_v) IMP, q likewise with its own weight; separate weights for non-block and block (gap) target cells.
Weights: grid .05; per panel w = .5 w_panel + .5 w_pooled (pooled = argmin of the summed squared error over all panels),
speed on km/h error, flow on per-lane error. Variants: IMP (imputer alone), BL (blend), BLP (blend then phys3 projection
r=3 a=.75 g=1, gap cells skipped).
OWN (March, I-S-OWN): weights fitted on odd days, scored on even days (primary); also two-way cross-fit over all days.
own_instrument.paired_state (date bootstrap, family groups) for non-block and all score targets, pooled and per panel.
OPT (train held dates, imputer per held month): weights cross-fitted by day parity (fit on one parity, applied to the other),
scored with flow_screen.day_components (official mode A physics incl. S_LWR) and a date bootstrap over 10 panels.
Usage: python docs/reference_code/b6full_eval.py own | opt  [seeds e.g. 0,1,2]"""
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

OUT = P.ROOT / "artifacts/claude_ref/b6full"
PANELS = P.P["panels"]
GRID = np.round(np.linspace(0, 1, 21), 2)
LAM = .5


def imp(mode, panel, seeds):
    zs = [np.load(OUT / f"imp_{mode}_{panel}_s{s}.npz") for s in seeds if (OUT / f"imp_{mode}_{panel}_s{s}.npz").exists()]
    assert zs, (mode, panel)
    return np.mean([z["speed"] for z in zs], 0).astype(float), np.mean([z["flow"] for z in zs], 0).astype(float), len(zs)


def sse_grid(a, b, t, m, scale=None):
    """Squared error of w a + (1-w) b vs t over mask m for each grid w (scale = per-cell divisor, e.g. lanes)."""
    if not m.any():
        return np.zeros(len(GRID))
    a, b, t = a[m], b[m], t[m]
    if scale is not None:
        s = scale[m]; a, b, t = a / s, b / s, t / s
    return np.array([np.sum((w * a + (1 - w) * b - t) ** 2) for w in GRID])


def fit_weights(stats):
    """stats[panel][key] = sse grid; returns per panel shrunk weights and pooled weights per key."""
    keys = next(iter(stats.values())).keys()
    pool = {k: float(GRID[np.argmin(sum(stats[p][k] for p in stats))]) for k in keys}
    out = {p: {k: LAM * float(GRID[np.argmin(stats[p][k])]) + (1 - LAM) * pool[k] if stats[p][k].any() else pool[k] for k in keys} for p in stats}
    return out, pool


def blend(cv, cq, iv, iq, gap, w):
    g = np.asarray(gap, bool)
    wv = np.where(g, w["v_blk"], w["v_nb"]); wq = np.where(g, w["q_blk"], w["q_nb"])
    return wv * cv + (1 - wv) * iv, wq * cq + (1 - wq) * iq


def own(seeds="0,1,2"):
    import flow_own as FO
    own_m, binding = FO.own, FO.binding
    seeds = [int(s) for s in seeds.split(",")]; p = P.P
    items = [i for i in json.loads(Path(p["OWN_metadata_binding"]["manifest"]).read_text())["items"] if i["period"] == "validation"]
    cache = {}; stats = {0: {}, 1: {}}; nseed = {}
    for item in items:
        pn = item["panel"]
        view, tv, tq, st_, hidden = own_m.load_item(P.ROOT, binding.bound_own_item(p, item))
        st_ = np.asarray(st_, bool); gap = np.asarray(view.gap, bool)
        r = np.load(OUT / f"ref_own_{pn}.npz"); cv, cq = r["c11_v"].astype(float), r["c11_q"].astype(float)
        assert (r["target"] == st_).all()
        iv, iq, nseed[pn] = imp("own", pn, seeds)
        la = np.broadcast_to(np.maximum(view.stations.sort_values("idx").lanes.to_numpy(float), 1.), tv.shape)
        D = tv.shape[0]; day = np.broadcast_to(np.arange(D)[:, None, None], tv.shape)
        ok = st_ & np.isfinite(tv) & np.isfinite(tq)
        for par in (0, 1):  # stats from days of this parity
            m = ok & (day % 2 == par)
            stats[par][pn] = dict(v_nb=sse_grid(cv, iv, tv, m & ~gap), v_blk=sse_grid(cv, iv, tv, m & gap),
                                  q_nb=sse_grid(cq, iq, tq, m & ~gap, la), q_blk=sse_grid(cq, iq, tq, m & gap, la))
        cache[pn] = (view, tv, tq, st_, gap, cv, cq, iv, iq, day)
        print(pn, "seeds", nseed[pn], "rmse_v C11/IMP", round(float(np.sqrt(np.mean((cv - tv)[ok] ** 2))), 3),
              round(float(np.sqrt(np.mean((iv - tv)[ok] ** 2))), 3), flush=True)
    W1, pool1 = fit_weights(stats[1]); W0, pool0 = fit_weights(stats[0])
    allw, _ = fit_weights({pn: {k: stats[0][pn][k] + stats[1][pn][k] for k in stats[0][pn]} for pn in stats[0]})
    rows = {}  # (scheme, variant, part) -> list
    per = {}
    for pn, (view, tv, tq, st_, gap, cv, cq, iv, iq, day) in cache.items():
        odd = day % 2 == 1
        b_even = blend(cv, cq, iv, iq, gap, W1[pn])          # fit odd -> used on even days
        b_odd = blend(cv, cq, iv, iq, gap, W0[pn])           # fit even -> used on odd days
        bv = np.where(odd, b_odd[0], b_even[0]); bq = np.where(odd, b_odd[1], b_even[1])
        pv_, pq_, _ = PJ.project(bv, bq, st_, view.stations, ST.R, ST.A, ST.G, gap=gap)
        preds = dict(C11=(cv, cq), IMP=(iv, iq), BL=(bv, bq), BLP=(pv_, pq_))
        for n, (v, q) in preds.items():
            for part, m in (("all", st_), ("nonblock", st_ & ~gap)):
                rows.setdefault((n, part), []).extend(own_m.state_rows(view, tv, tq, v, q, m)[0])
        ok = st_ & np.isfinite(tv) & np.isfinite(tq); la = np.maximum(view.stations.sort_values("idx").lanes.to_numpy(float), 1.)
        per[pn] = dict(weights_fit_odd=W1[pn], weights_all_days=allw[pn], n_seeds=nseed[pn],
                       **{f"rmse_v_nb_{n}": float(np.sqrt(np.mean((v - tv)[ok & ~gap] ** 2))) for n, (v, q) in preds.items()},
                       **{f"rmse_ql_nb_{n}": float(np.sqrt(np.mean(((q - tq) / la)[ok & ~gap] ** 2))) for n, (v, q) in preds.items()},
                       **{f"rmse_v_blk_{n}": float(np.sqrt(np.mean((v - tv)[ok & gap] ** 2))) for n, (v, q) in preds.items()})
        print(pn, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in per[pn].items() if k.startswith("rmse")}, flush=True)
    panels = p["panels"]; fam = p["families"]
    groups = [[i for i, x in enumerate(panels) if fam[x] == f] for f in sorted(set(fam.values()))]
    dates = sorted({r["date"] for r in rows[("C11", "all")]})
    even = [d for i, d in enumerate(dates) if i % 2 == 0]
    assert all(int(d[-2:]) % 2 == 1 for d in even)  # day index 0 = March 1; even index = odd calendar day
    res = dict(seeds=seeds, lam=LAM, pooled_fit_odd=pool1, pooled_fit_even=pool0, per_panel=per, pooled={}, by_panel={})
    for n in ("IMP", "BL", "BLP"):
        for part in ("nonblock", "all"):
            for scheme, ds in (("even", even), ("crossfit", dates)):
                ref = [r for r in rows[("C11", part)] if r["date"] in ds]; new = [r for r in rows[(n, part)] if r["date"] in ds]
                x = own_m.paired_state(ref, new, panels, ds, groups)
                key = f"{n}_{part}_{scheme}"
                res["pooled"][key] = dict(delta=x["delta_S_state"], ci=x["ci95"], verdict=x["verdict"], ref=x["incumbent_S_state"])
                bp = {}
                for i, pn in enumerate(panels):
                    y = own_m.paired_state([r for r in ref if r["panel"] == pn], [r for r in new if r["panel"] == pn], [pn], ds, [[0]])
                    bp[pn] = (round(y["delta_S_state"], 5), [round(c, 5) for c in y["ci95"]])
                res["by_panel"][key] = bp
                print(key, round(x["delta_S_state"], 5), [round(c, 5) for c in x["ci95"]], x["verdict"], flush=True)
    print("weights fit odd (pooled)", pool1)
    print("per panel even nonblock BL", res["by_panel"]["BL_nonblock_even"])
    (OUT / f"own_eval_s{''.join(map(str, seeds))}.json").write_text(json.dumps(res, indent=1))


def opt(seeds="0,1,2", nboot=1000):
    import flow_screen as FS
    seeds = [int(s) for s in seeds.split(",")]
    dates = P.P["OPT_held_dates"]; D = len(dates); par = np.arange(D) % 2
    jun = np.array([d.startswith("2030-06") for d in dates])
    cache = {}; stats = {0: {}, 1: {}}
    for pn in PANELS:
        held, tv, tq, pv, pq = P.held_panel(pn)
        r = np.load(OUT / f"ref_opt_{pn}.npz"); cv, cq = r["c11_v"].astype(float), r["c11_q"].astype(float)
        a = imp("opt_jun", pn, seeds); b = imp("opt_jul", pn, seeds)
        iv = np.zeros_like(cv); iq = np.zeros_like(cq)
        iv[jun], iq[jun] = a[0], a[1]; iv[~jun], iq[~jun] = b[0], b[1]
        T = np.asarray(held.target, bool); gap = np.zeros_like(T)
        la = np.broadcast_to(np.maximum(held.stations.sort_values("idx").lanes.to_numpy(float), 1.), T.shape)
        ok = T & np.isfinite(tv) & np.isfinite(tq); day = np.broadcast_to(par[:, None, None], T.shape)
        for f in (0, 1):
            m = ok & (day == f)
            stats[f][pn] = dict(v_nb=sse_grid(cv, iv, tv, m), v_blk=np.zeros(len(GRID)), q_nb=sse_grid(cq, iq, tq, m, la), q_blk=np.zeros(len(GRID)))
        cache[pn] = (held, tv, tq, T, gap, cv, cq, iv, iq)
        print(pn, "nseeds", a[2], b[2], "rmse_v C11/IMP", round(float(np.sqrt(np.mean((cv - tv)[ok] ** 2))), 3),
              round(float(np.sqrt(np.mean((iv - tv)[ok] ** 2))), 3), flush=True)
    W = {f: fit_weights(stats[1 - f])[0] for f in (0, 1)}  # weights for days of parity f fitted on the other parity
    comps = {}; regs = {}; rows = []
    for pn, (held, tv, tq, T, gap, cv, cq, iv, iq) in cache.items():
        bv, bq = cv.copy(), cq.copy()
        for f in (0, 1):
            x = blend(cv, cq, iv, iq, gap, W[f][pn]); bv[par == f] = x[0][par == f]; bq[par == f] = x[1][par == f]
        bv = np.where(T, bv, cv); bq = np.where(T, bq, cq)
        pv_, pq_, _ = PJ.project(bv, bq, T, held.stations, ST.R, ST.A, ST.G)
        preds = dict(C11=(cv, cq), IMP=(np.where(T, iv, cv), np.where(T, iq, cq)), BL=(bv, bq), BLP=(pv_, pq_))
        comps[pn] = {}
        for n, (v, q) in preds.items():
            comps[pn][n], regs[pn] = FS.day_components(held, tv, tq, v, q)
        one = {n: FS.aggregate(comps[pn][n], regs[pn], np.ones(D)) for n in preds}
        rows.append(dict(panel=pn, w0=W[0][pn], w1=W[1][pn], **{f"dJ_{n}": one[n]["J"] - one["C11"]["J"] for n in preds if n != "C11"},
                         **{f"dS_state_{n}": one[n]["S_state"] - one["C11"]["S_state"] for n in preds if n != "C11"},
                         **{f"dS_LWR_{n}": one[n]["S_LWR"] - one["C11"]["S_LWR"] for n in preds if n != "C11"}))
        print(pn, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in rows[-1].items() if k.startswith("d")}, flush=True)
    M = ("J", "S_state", "S_phys", "S_LWR", "S_FD")
    res = dict(seeds=seeds, per_panel=rows, contrasts={})
    for a in ("IMP", "BL", "BLP"):
        per = {p: {m: FS.aggregate(comps[p][a], regs[p], np.ones(D))[m] - FS.aggregate(comps[p]["C11"], regs[p], np.ones(D))[m] for m in M} for p in PANELS}
        rng = np.random.default_rng(0); boots = []
        for _ in range(nboot):
            w = np.bincount(rng.integers(0, D, D), minlength=D).astype(float)
            boots.append({m: np.mean([FS.aggregate(comps[p][a], regs[p], w)[m] - FS.aggregate(comps[p]["C11"], regs[p], w)[m] for p in PANELS]) for m in M})
        B = pd.DataFrame(boots)
        res["contrasts"][f"{a}_vs_C11"] = dict(summary={m: dict(mean=float(np.mean([per[p][m] for p in PANELS])), ci=[float(B[m].quantile(.025)), float(B[m].quantile(.975))],
                                                                n_pos=int(sum(per[p][m] > 0 for p in PANELS))) for m in M},
                                               per_panel={p: {m: round(v, 6) for m, v in d.items()} for p, d in per.items()})
        print(f"{a} vs C11", {m: (round(v["mean"], 5), [round(c, 5) for c in v["ci"]], v["n_pos"]) for m, v in res["contrasts"][f"{a}_vs_C11"]["summary"].items()}, flush=True)
    (OUT / f"opt_eval_s{''.join(map(str, seeds))}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"own": own, "opt": opt}[sys.argv[1]](*sys.argv[2:3])
