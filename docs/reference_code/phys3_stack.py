"""Claude helper 2026-10-04 (ledger 0372): does the 0368 state hedge stack with phys3 (both move v in congestion)?
ref = F4 + 0360 + 0358.  H = 0360 -> 0368 -> 0358 (0368 alone on ref).  S1 = ref + phys3.  S2 = 0360 -> 0368 -> 0358 -> phys3.
phys3 fixed r=3, a=.75, g=1 (gap cells excluded where a gap mask exists). 0368 = state3_hedge alpha=1, EPS=0 (raw mixture on
candidates); its own reference is F4 + 0360 with F4 flow, which is exactly its input position in S2, so no refit is needed.
OPT: 0368 cross-fit predictions from artifacts/claude_ref/state3/opt_pred_<panel>.npz (LOPO x parity), 0360/0358 fold fits as phys3_screen.
OWN: 0368 models fitted on all OPT candidates (saved to artifacts/claude_ref/phys3/m368_*.txt for the probe builder).
Usage: python docs/reference_code/phys3_stack.py opt | own"""
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
import state2_cm as CM
import phys3_proj as PJ
import phys3_screen as S3

OUT = P.ROOT / "artifacts/claude_ref/phys3"
PANELS = P.P["panels"]
R, A, G = 3., .75, 1.
NAMES = ("ref", "H", "S1", "S2")
M = ("J", "S_state", "S_phys", "S_LWR")


def blend_folds(panel, held, v, pq, D):
    stats = {p: dict(np.load(FS.OUT / f"{p}_stats.npz")) for p in PANELS}
    N = np.load(FS.OUT / f"{panel}_N.npy").astype(float); den = np.load(FS.OUT / f"{panel}_den.npy").astype(float)
    T = np.asarray(held.target, bool); par = np.arange(D) % 2; nq = pq.copy()
    for f in (0, 1):
        fq = FK.blend(v, pq, N, T, held.stations, FS.fit_weights(stats, panel, f, D), den=den)
        nq[par == f] = fq[par == f]
    return nq


def opt(nboot=1000):
    D = len(P.P["OPT_held_dates"]); comps = {}; regs = {}
    for panel in PANELS:
        held, tv, tq, pv, pq = P.held_panel(panel)
        T = np.asarray(held.target, bool)
        rv, rq = S3.reference(panel, held, pv, pq, D)
        z = np.load(P.ROOT / "artifacts/claude_ref/state3" / f"opt_pred_{panel}.npz"); ix = np.load(P.ROOT / "artifacts/claude_ref/state3" / f"opt_{panel}.npz")["ix"]
        hv = rv.copy(); hv[ix[:, 0], ix[:, 1], ix[:, 2]] = np.where(z["cand"], z["v"].astype(float), rv[ix[:, 0], ix[:, 1], ix[:, 2]])
        hq = blend_folds(panel, held, hv, pq, D)
        preds = dict(ref=(rv, rq), H=(hv, hq), S1=PJ.project(rv, rq, T, held.stations, R, A, G)[:2], S2=PJ.project(hv, hq, T, held.stations, R, A, G)[:2])
        comps[panel] = {}
        for n, (v, q) in preds.items():
            comps[panel][n], regs[panel] = FS.day_components(held, tv, tq, v, q)
        # overlap: share of S1-changed cells also changed by 0368
        m1 = PJ.project(rv, rq, T, held.stations, R, A, G)[2]; mh = np.abs(hv - rv) > 1e-9
        ov = dict(phys3_cells=int(m1.sum()), h_cells=int(mh.sum()), both=int((m1 & mh).sum()))
        np.savez(OUT / f"stack_{panel}.npz", regimes=np.array(regs[panel]), **comps[panel])
        one = {n: FS.aggregate(comps[panel][n], regs[panel], np.ones(D)) for n in NAMES}
        print(panel, {n: round(one[n]["J"] - one["ref"]["J"], 5) for n in NAMES[1:]}, ov, flush=True)
    res = {}
    pairs = [("H", "ref"), ("S1", "ref"), ("S2", "ref"), ("S2", "S1"), ("S2", "H")]
    for a, b in pairs:
        per = {p: {m: FS.aggregate(comps[p][a], regs[p], np.ones(D))[m] - FS.aggregate(comps[p][b], regs[p], np.ones(D))[m] for m in M} for p in PANELS}
        rng = np.random.default_rng(0); boots = []
        for _ in range(nboot):
            w = np.bincount(rng.integers(0, D, D), minlength=D).astype(float)
            boots.append({m: np.mean([FS.aggregate(comps[p][a], regs[p], w)[m] - FS.aggregate(comps[p][b], regs[p], w)[m] for p in PANELS]) for m in M})
        B = pd.DataFrame(boots)
        res[f"{a}_vs_{b}"] = dict(summary={m: dict(mean=float(np.mean([per[p][m] for p in PANELS])), ci=[float(B[m].quantile(.025)), float(B[m].quantile(.975))],
                                                   n_pos=int(sum(per[p][m] > 0 for p in PANELS))) for m in M},
                                  per_panel={p: {m: round(v, 6) for m, v in d.items()} for p, d in per.items()})
        print(f"{a} vs {b}", {m: (round(v["mean"], 5), [round(c, 5) for c in v["ci"]], v["n_pos"]) for m, v in res[f"{a}_vs_{b}"]["summary"].items()}, flush=True)
    j = {k: res[k]["summary"]["J"]["mean"] for k in res}
    res["additivity_J"] = dict(H=j["H_vs_ref"], S1=j["S1_vs_ref"], sum=j["H_vs_ref"] + j["S1_vs_ref"], S2=j["S2_vs_ref"])
    print("additivity", res["additivity_J"])
    (OUT / "stack_opt.json").write_text(json.dumps(res, indent=1))


def fit_368():
    import state3_hedge as SH
    data = [SH.load(p) for p in PANELS]; cs = [SH.candidate(d[0], d[3]) for d in data]
    models = SH.fit(np.concatenate([d[0][c] for d, c in zip(data, cs)]), np.concatenate([d[2][c] for d, c in zip(data, cs)]),
                    np.concatenate([d[3][c] for d, c in zip(data, cs)]))
    for n, m in zip(("clf", "gc", "gf"), models):
        m.save_model(str(OUT / f"m368_{n}.txt"))
    return models


def load_368():
    import lightgbm as lgb
    return tuple(lgb.Booster(model_file=str(OUT / f"m368_{n}.txt")) for n in ("clf", "gc", "gf"))


def apply_368(models, speed, observed, stations, vr, pq, mask):
    """0368 hedge (alpha 1, EPS 0) on mask cells that are candidates; returns new speed field."""
    import state3_hedge as SH, state3_feat as F
    X, ix = F.build(speed, observed, stations, vr, pq, mask)
    X = X.astype(float); cut = stations.sort_values("idx").v_cut.to_numpy(float)[ix[:, 2]]
    c = SH.candidate(X, cut); out = vr.copy()
    if c.any():
        clf, gc, gf = models; Xc = X[c]
        p = clf.predict(Xc); v = p * (Xc[:, SH.FI["vr"]] + gc.predict(Xc)) + (1 - p) * (Xc[:, SH.FI["vr"]] + gf.predict(Xc))
        out[ix[c, 0], ix[c, 1], ix[c, 2]] = v
    return out, int(c.sum())


def own():
    import flow_own as FO
    own_m, binding = FO.own, FO.binding
    models = fit_368() if not (OUT / "m368_clf.txt").exists() else load_368()
    print("0368 models ready", flush=True)
    p = P.P
    w = np.asarray(json.loads((P.ROOT / "artifacts/claude_ref/flow/opt/opt_summary.json").read_text())["pooled_weights"])
    cj = json.loads((CM.OUT / "opt_coef.json").read_text()); cf = (cj["b_free"], cj["b_cong"])
    items = [i for i in json.loads(Path(p["OWN_metadata_binding"]["manifest"]).read_text())["items"] if i["period"] == "validation"]
    rows = {n: [] for n in NAMES}; nb = {n: [] for n in NAMES}; per = {}
    for item in items:
        view, tv, tq, st_, hidden = own_m.load_item(P.ROOT, binding.bound_own_item(p, item))
        st_ = np.asarray(st_, bool); gap = np.asarray(view.gap, bool)
        pv, pq = (np.load(P.D / "F4_predictions/OWN" / f"{item['panel']}_validation" / f"{n}.npy").astype(float) for n in ("speed", "flow"))
        speed = np.asarray(view.speed, float)
        observed = np.asarray(view.elig, bool) & ~np.asarray(view.target, bool) & ~gap & np.isfinite(speed)
        gr, _ = CM.common_factor(speed, observed, view.stations)
        vcut = view.stations.sort_values("idx").v_cut.to_numpy(float)
        rv = np.where(st_, CM.apply(pv, gr, vcut, cf), pv)
        N, den = FK.krige_log_flow(np.asarray(view.flow, float), FK.positions(view.stations))
        rq = FK.blend(rv, pq, N, st_, view.stations, w, den=den)
        hv, nc = apply_368(models, speed, observed, view.stations, rv, pq, st_)
        hq = FK.blend(hv, pq, N, st_, view.stations, w, den=den)
        preds = dict(ref=(rv, rq), H=(hv, hq), S1=PJ.project(rv, rq, st_, view.stations, R, A, G, gap=gap)[:2],
                     S2=PJ.project(hv, hq, st_, view.stations, R, A, G, gap=gap)[:2])
        for n, (v, q) in preds.items():
            rows[n] += own_m.state_rows(view, tv, tq, v, q, st_)[0]
            nb[n] += own_m.state_rows(view, tv, tq, v, q, st_ & ~gap)[0]
        mm = st_ & np.isfinite(tv)
        per[item["panel"]] = dict(cand=nc, **{f"rmse_v_{n}": float(np.sqrt(np.mean((v - tv)[mm] ** 2))) for n, (v, q) in preds.items()})
        print(item["panel"], {k: round(x, 4) for k, x in per[item["panel"]].items()}, flush=True)
    panels = p["panels"]; fam = p["families"]
    groups = [[i for i, x in enumerate(panels) if fam[x] == f] for f in sorted(set(fam.values()))]
    dates = sorted({r["date"] for r in rows["ref"]})
    res = {}
    for a, b in (("H", "ref"), ("S1", "ref"), ("S2", "ref"), ("S2", "S1")):
        for tag, src in (("all", rows), ("nonblock", nb)):
            x = own_m.paired_state(src[b], src[a], panels, dates, groups)
            res[f"{a}_vs_{b}_{tag}"] = x
            print(f"{a} vs {b} {tag}", round(x["delta_S_state"], 5), [round(c, 5) for c in x["ci95"]], x["verdict"], flush=True)
    (OUT / "stack_own.json").write_text(json.dumps(dict(results=res, per_panel=per), indent=1))


if __name__ == "__main__":
    {"opt": opt, "own": own}[sys.argv[1]]()
