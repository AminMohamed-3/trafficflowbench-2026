"""Claude helper 2026-10-04 (ledger 0389, idea S2): learned stage 2 over the post-processor signals (s2stage_feat.FEATS).
Base = shipped stack S = F4 -> 0360 -> 0368 -> 0358 -> phys3 (OPT predictions cached by s7_audit.py, identical to phys3_stack S2;
OPT has no gap cells). One LightGBM residual model per output on target cells: y_v = v_true - v_S, y_q = (q_true - q_S) / lanes.
OPT: cross-fit leave-one-panel-out x other day parity (as phys3_screen); outputs V (speed only), Q (flow only), VQ (both);
choice by OPT J_S (official physics mode A for every panel, S7). Paired date bootstrap vs S over 10 panels.
OWN: models fitted on all OPT target cells, stack rebuilt on March items as phys3_stack.own (gap = blocks), I-S-OWN paired state.
Usage: python docs/reference_code/s2stage.py feat | opt [variant_tag] | own [variant_tag]"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "3")
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phys_lwr_diag as P
import flow_screen as FS
import flow_krige as FK
import state2_cm as CM
import phys3_proj as PJ
import phys3_stack as ST
import s2stage_feat as SF

OUT = P.ROOT / "artifacts/claude_ref/s2stage"
S7 = P.ROOT / "artifacts/claude_ref/s7"
PANELS = P.P["panels"]
FI = {f: i for i, f in enumerate(SF.FEATS)}
PRM = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=2000, feature_fraction=0.8,
           bagging_fraction=0.5, bagging_freq=1, lambda_l2=10.0, num_threads=3, verbose=-1, seed=0)
ROUNDS = 600   # 2026-10-09 ledger 0768 (C budget): 63 leaves x 600 rounds replaces 15 x 300 (C30 board +.00005)
M = ("J", "S_state", "S_phys", "S_LWR", "S_FD")


def lanes_of(stations):
    return np.maximum(stations.sort_values("idx").lanes.to_numpy(float), 1.)


def feat():
    OUT.mkdir(parents=True, exist_ok=True)
    for panel in PANELS:
        held, tv, tq, pv, pq = P.held_panel(panel)
        T = np.asarray(held.target, bool); z = np.load(S7 / f"pred_{panel}.npz")
        cm = np.load(CM.OUT / f"opt_{panel}_cm.npz")
        N = np.load(FS.OUT / f"{panel}_N.npy").astype(float); den = np.load(FS.OUT / f"{panel}_den.npy").astype(float)
        h = np.load(P.ROOT / "artifacts/claude_ref/state3" / f"opt_pred_{panel}.npz")
        ix3 = np.load(P.ROOT / "artifacts/claude_ref/state3" / f"opt_{panel}.npz")["ix"]
        hp = np.full(T.shape, np.nan); hp[ix3[:, 0], ix3[:, 1], ix3[:, 2]] = np.where(h["cand"], h["p"], np.nan)
        f = {k: z[k].astype(float) for k in z.files}
        m = T & np.isfinite(tv) & np.isfinite(tq)
        X, ix = SF.build(np.asarray(held.speed, float), np.asarray(held.flow, float), held.elig, T, held.gap, held.stations,
                         pv, pq, f["ref_v"], cm["gr"].astype(float), cm["n"], N, den, hp, f["H_v"], f["H_q"], f["S2_v"], f["S2_q"], m)
        la = lanes_of(held.stations)
        yv = (tv - f["S2_v"])[m]; yq = ((tq - f["S2_q"]) / la)[m]
        np.savez(OUT / f"opt_X_{panel}.npz", X=X, ix=ix, yv=yv.astype(np.float32), yq=yq.astype(np.float32))
        print(panel, X.shape, "rmse v", round(float(np.sqrt(np.mean(yv ** 2))), 3), "rmse q/lane", round(float(np.sqrt(np.mean(yq ** 2))), 2), flush=True)


MAXROWS = 3_000_000   # 2026-10-10 ledger 0768/0779 (C budget): all OPT rows (C34 board +.00005); was 1_500_000


def fit(X, y):
    if len(y) > MAXROWS:  # fixed-seed row subsample (speed; 3-thread laptop budget)
        s = np.random.default_rng(1).choice(len(y), MAXROWS, replace=False); X, y = X[s], y[s]
    return lgb.train(PRM, lgb.Dataset(X, y, free_raw_data=True), ROUNDS)


def opt(tag="strong", nboot=1000):
    D = len(P.P["OPT_held_dates"]); par = np.arange(D) % 2
    data = {p: dict(np.load(OUT / f"opt_X_{p}.npz")) for p in PANELS}
    gv = {p: np.zeros(len(data[p]["yv"])) for p in PANELS}; gq = {p: np.zeros(len(data[p]["yq"])) for p in PANELS}
    imp = {"v": np.zeros(len(SF.FEATS)), "q": np.zeros(len(SF.FEATS))}
    for panel in PANELS:
        for f in (0, 1):
            tr = [(d["X"][par[d["ix"][:, 0]] != f], d["yv"][par[d["ix"][:, 0]] != f], d["yq"][par[d["ix"][:, 0]] != f]) for q, d in data.items() if q != panel]
            Xt = np.concatenate([a[0] for a in tr]); sel = par[data[panel]["ix"][:, 0]] == f
            for out, k in ((gv, 1), (gq, 2)):
                mdl = fit(Xt, np.concatenate([a[k] for a in tr]))
                out[panel][sel] = mdl.predict(data[panel]["X"][sel])
                imp["v" if k == 1 else "q"] += mdl.feature_importance("gain")
            del Xt
        print(panel, "cross-fit done", flush=True)
    comps = {}; regs = {}; rows = []
    for panel in PANELS:
        held, tv, tq, pv, pq = P.held_panel(panel); z = np.load(S7 / f"pred_{panel}.npz")
        sv, sq = z["S2_v"].astype(float), z["S2_q"].astype(float); ix = data[panel]["ix"]; I = (ix[:, 0], ix[:, 1], ix[:, 2])
        st = held.stations.sort_values("idx"); la = lanes_of(held.stations); cap = st.capacity_vph.to_numpy(float)
        nv = sv.copy(); nv[I] = np.maximum(sv[I] + gv[panel], 3.)
        nq = sq.copy(); nq[I] = np.clip(sq[I] + la[ix[:, 2]] * gq[panel], 50., 1.3 * cap[ix[:, 2]])
        T = np.asarray(held.target, bool)
        preds = dict(S=(sv, sq), V=(nv, sq), Q=(sv, nq), VQ=(nv, nq), VQP=PJ.project(nv, nq, T, held.stations, ST.R, ST.A, ST.G)[:2])
        comps[panel] = {}
        for n, (v, q) in preds.items():
            comps[panel][n], regs[panel] = FS.day_components(held, tv, tq, v, q)
        one = {n: FS.aggregate(comps[panel][n], regs[panel], np.ones(D)) for n in preds}
        m = T & np.isfinite(tv) & np.isfinite(tq)
        rows.append(dict(panel=panel, rmse_v_S=float(np.sqrt(np.mean((sv - tv)[m] ** 2))), rmse_v_new=float(np.sqrt(np.mean((nv - tv)[m] ** 2))),
                         rmse_ql_S=float(np.sqrt(np.mean(((sq - tq) / la)[m] ** 2))), rmse_ql_new=float(np.sqrt(np.mean(((nq - tq) / la)[m] ** 2))),
                         **{f"dJ_{n}": one[n]["J"] - one["S"]["J"] for n in preds if n != "S"}))
        print(panel, {k: round(v, 5) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)
    res = dict(tag=tag, params=PRM, rounds=ROUNDS, feats=SF.FEATS, per_panel=rows,
               importance={k: dict(zip(SF.FEATS, (v / v.sum()).round(4).tolist())) for k, v in imp.items()}, contrasts={})
    for a in ("V", "Q", "VQ", "VQP"):
        per = {p: {m: FS.aggregate(comps[p][a], regs[p], np.ones(D))[m] - FS.aggregate(comps[p]["S"], regs[p], np.ones(D))[m] for m in M} for p in PANELS}
        rng = np.random.default_rng(0); boots = []
        for _ in range(nboot):
            w = np.bincount(rng.integers(0, D, D), minlength=D).astype(float)
            boots.append({m: np.mean([FS.aggregate(comps[p][a], regs[p], w)[m] - FS.aggregate(comps[p]["S"], regs[p], w)[m] for p in PANELS]) for m in M})
        B = pd.DataFrame(boots)
        res["contrasts"][f"{a}_vs_S"] = dict(summary={m: dict(mean=float(np.mean([per[p][m] for p in PANELS])), ci=[float(B[m].quantile(.025)), float(B[m].quantile(.975))],
                                                              n_pos=int(sum(per[p][m] > 0 for p in PANELS))) for m in M},
                                             per_panel={p: {m: round(v, 6) for m, v in d.items()} for p, d in per.items()})
        print(f"{a} vs S", {m: (round(v["mean"], 5), [round(c, 5) for c in v["ci"]], v["n_pos"]) for m, v in res["contrasts"][f"{a}_vs_S"]["summary"].items()}, flush=True)
    print("importance", json.dumps(res["importance"]), flush=True)
    (OUT / f"opt_{tag}.json").write_text(json.dumps(res, indent=1))


def fit_all():
    data = [dict(np.load(OUT / f"opt_X_{p}.npz")) for p in PANELS]
    X = np.concatenate([d["X"] for d in data])
    mv = fit(X, np.concatenate([d["yv"] for d in data])); mq = fit(X, np.concatenate([d["yq"] for d in data]))
    mv.save_model(str(OUT / "m_v.txt")); mq.save_model(str(OUT / "m_q.txt"))
    return mv, mq


def load_all():
    return tuple(lgb.Booster(model_file=str(OUT / f"m_{k}.txt")) for k in ("v", "q"))


def hedge_368(models, speed, observed, stations, vr, pq, mask):
    """phys3_stack.apply_368 that also returns p (nan off candidates)."""
    import state3_hedge as SH, state3_feat as F3
    X, ix = F3.build(speed, observed, stations, vr, pq, mask)
    X = X.astype(float); cut = stations.sort_values("idx").v_cut.to_numpy(float)[ix[:, 2]]
    c = SH.candidate(X, cut); out = vr.copy(); hp = np.full(vr.shape, np.nan)
    if c.any():
        clf, gc, gf = models; Xc = X[c]
        p = clf.predict(Xc); v = p * (Xc[:, SH.FI["vr"]] + gc.predict(Xc)) + (1 - p) * (Xc[:, SH.FI["vr"]] + gf.predict(Xc))
        I = (ix[c, 0], ix[c, 1], ix[c, 2]); out[I] = v; hp[I] = p
    return out, hp


def apply_stage2(models, view_arrays, stations, pv, pq, rv, gr, ncm, N, den, hp, hv, hq, sv, sq, mask, variant):
    speed, flow, elig, target, gap = view_arrays
    X, ix = SF.build(speed, flow, elig, target, gap, stations, pv, pq, rv, gr, ncm, N, den, hp, hv, hq, sv, sq, mask)
    I = (ix[:, 0], ix[:, 1], ix[:, 2]); st = stations.sort_values("idx")
    la = lanes_of(stations); cap = st.capacity_vph.to_numpy(float)
    nv, nq = sv.copy(), sq.copy()
    if "V" in variant:
        nv[I] = np.maximum(sv[I] + models[0].predict(X), 3.)
    if "Q" in variant:
        nq[I] = np.clip(sq[I] + la[ix[:, 2]] * models[1].predict(X), 50., 1.3 * cap[ix[:, 2]])
    if variant.endswith("P"):
        nv, nq, _ = PJ.project(nv, nq, mask, stations, ST.R, ST.A, ST.G, gap=gap)
    return nv, nq


def own(variant="VQ"):
    import flow_own as FO
    own_m, binding = FO.own, FO.binding
    models = load_all() if (OUT / "m_v.txt").exists() else fit_all()
    m368 = ST.load_368(); p = P.P
    w = np.asarray(json.loads((P.ROOT / "artifacts/claude_ref/flow/opt/opt_summary.json").read_text())["pooled_weights"])
    cj = json.loads((CM.OUT / "opt_coef.json").read_text()); cf = (cj["b_free"], cj["b_cong"])
    items = [i for i in json.loads(Path(p["OWN_metadata_binding"]["manifest"]).read_text())["items"] if i["period"] == "validation"]
    vs = variant.split(","); rows = {n: [] for n in ["S"] + vs}; nb = {n: [] for n in ["S"] + vs}; per = {}
    for item in items:
        view, tv, tq, st_, hidden = own_m.load_item(P.ROOT, binding.bound_own_item(p, item))
        st_ = np.asarray(st_, bool); gap = np.asarray(view.gap, bool)
        pv, pq = (np.load(P.D / "F4_predictions/OWN" / f"{item['panel']}_validation" / f"{n}.npy").astype(float) for n in ("speed", "flow"))
        speed = np.asarray(view.speed, float); flow = np.asarray(view.flow, float)
        observed = np.asarray(view.elig, bool) & ~np.asarray(view.target, bool) & ~gap & np.isfinite(speed)
        gr, ncm = CM.common_factor(speed, observed, view.stations)
        vcut = view.stations.sort_values("idx").v_cut.to_numpy(float)
        rv = np.where(st_, CM.apply(pv, gr, vcut, cf), pv)
        N, den = FK.krige_log_flow(flow, FK.positions(view.stations))
        hv, hp = hedge_368(m368, speed, observed, view.stations, rv, pq, st_)
        hq = FK.blend(hv, pq, N, st_, view.stations, w, den=den)
        sv, sq, _ = PJ.project(hv, hq, st_, view.stations, ST.R, ST.A, ST.G, gap=gap)
        out = {"S": (sv, sq)}
        for vn in vs:
            out[vn] = apply_stage2(models, (speed, flow, np.asarray(view.elig, bool), np.asarray(view.target, bool) | st_, gap), view.stations,
                                   pv, pq, rv, gr, ncm, N, den, hp, hv, hq, sv, sq, st_, vn)
        mm = st_ & np.isfinite(tv) & np.isfinite(tq); la = lanes_of(view.stations); per[item["panel"]] = {}
        for n, (v, q) in out.items():
            rows[n] += own_m.state_rows(view, tv, tq, v, q, st_)[0]
            nb[n] += own_m.state_rows(view, tv, tq, v, q, st_ & ~gap)[0]
            per[item["panel"]].update({f"rmse_v_{n}": float(np.sqrt(np.mean((v - tv)[mm] ** 2))), f"rmse_ql_{n}": float(np.sqrt(np.mean(((q - tq) / la)[mm] ** 2))),
                                       f"rmse_v_nb_{n}": float(np.sqrt(np.mean((v - tv)[mm & ~gap] ** 2)))})
        print(item["panel"], {k: round(x, 4) for k, x in per[item["panel"]].items()}, flush=True)
    panels = p["panels"]; fam = p["families"]
    groups = [[i for i, x in enumerate(panels) if fam[x] == f] for f in sorted(set(fam.values()))]
    dates = sorted({r["date"] for r in rows["S"]})
    res = {}
    for vn in vs:
        for tag, src in (("all", rows), ("nonblock", nb)):
            x = own_m.paired_state(src["S"], src[vn], panels, dates, groups)
            res[f"{vn}_{tag}"] = x
            print(f"{vn} vs S {tag}", round(x["delta_S_state"], 5), [round(c, 5) for c in x["ci95"]], x["verdict"], flush=True)
    (OUT / f"own_{variant.replace(',', '_')}.json").write_text(json.dumps(dict(variant=variant, results=res, per_panel=per), indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "feat":
        feat()
    elif cmd == "opt":
        opt(*sys.argv[2:3])
    else:
        own(*sys.argv[2:3])
