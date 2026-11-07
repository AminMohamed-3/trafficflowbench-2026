"""Claude helper phys4 (2026-10-04, ledger 0405): stage 3 = LightGBM huber residual models of log q and log v on top of
C14 (= C13 + LSA clipped nudge), features phys4_feat (+ LSA diffs/support), target cells, non-gap cells only.
OPT cross-fit result: phys4_dens.py eval --tag hub. This file: all-days fit (fit), I-S-OWN March check (own), probe (probe).
Usage: python docs/reference_code/phys4_s3.py fit | own | probe <out.csv>"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, hashlib
from pathlib import Path
import numpy as np
import phys4_common as C
import phys4_feat as PF
import phys4_lsa_eval as LE
import phys4_dens as DN
import phys_lwr_diag as P
import polish_eval as PE
from tfb import metric

OUT = C.OUT
LSA_PRM = {"wq": 0.5, "cq": 0.015, "wv": 0.3, "cv": 0.02, "G": 0.9, "xgap": 1}
TAG = os.environ.get("PHYS4_S3", "")  # "" = C15 models (150 rounds, lr .1, 31 leaves, 1.5M rows); "L" = 600, .05, 127, 3M
if TAG == "L":
    DN.ROUNDS2 = 600; DN.PRM2 = dict(DN.PRM2, learning_rate=.05, num_leaves=127); DN.NSUB = 3_000_000


def fit():
    import lightgbm as lgb
    Z = DN.load(); rng = np.random.default_rng(1)
    tot = sum(len(Z[pn]["yq"]) for pn in Z); keep = min(1., DN.NSUB / tot)
    sel = {pn: rng.random(len(Z[pn]["yq"])) < keep for pn in Z}
    X = np.concatenate([Z[pn]["X"][sel[pn]] for pn in Z])
    for lab in ("yq", "yv"):
        y = np.concatenate([Z[pn][lab][sel[pn]] for pn in Z])
        m = lgb.train(dict(DN.PRM2, alpha=DN.ALPHA[lab]), lgb.Dataset(X, y), DN.ROUNDS2)
        m.save_model(str(OUT / f"s3_{lab}{TAG}.txt")); print("saved", lab, len(y), flush=True)
    (OUT / "s3_names.json").write_text(json.dumps([str(x) for x in Z[C.PANELS[0]]["names"]]))


def models():
    import lightgbm as lgb
    return lgb.Booster(model_file=str(OUT / f"s3_yq{TAG}.txt")), lgb.Booster(model_file=str(OUT / f"s3_yv{TAG}.txt"))


def stage3(mq, mv, v, q, T, gap, speed_obs, flow_obs, p, stations, length, vf, scale=(1., 1.)):
    """v, q: C13 arrays. Returns (C14 v, q), (C15 v, q)."""
    F = LE.lsa_fields(speed_obs, flow_obs, vf)
    v14, q14 = LE.apply(v, q, T, vf, F, LSA_PRM, gap)
    m = T & ~gap & np.isfinite(v14) & np.isfinite(q14) & (q14 > 0) & (v14 > 0)
    X, names, ix = PF.build(speed_obs, flow_obs, T, v14, q14, p, stations, length, m)
    d, t, s = ix.T.astype(int)
    lq = np.log(np.maximum(q14, 1.)); lv = np.log(np.maximum(v14, 1.))
    X = np.concatenate([X, np.stack([(F[0] - lq)[d, t, s], F[1][d, t, s], (F[2] - lv)[d, t, s], F[3][d, t, s]], 1).astype(np.float32)], 1)
    assert names + ["lsa_dq", "lsa_nq", "lsa_dv", "lsa_nv"] == json.loads((OUT / "s3_names.json").read_text())
    v15, q15 = v14.copy(), q14.copy()
    q15[d, t, s] = q14[d, t, s] * np.exp(scale[0] * mq.predict(X, num_threads=3))
    v15[d, t, s] = v14[d, t, s] * np.exp(scale[1] * mv.predict(X, num_threads=3))
    return (v14, q14), (v15, q15)


def own():
    import flow_own as FO
    own_m, binding = FO.own, FO.binding
    p = P.P; mq, mv = models()
    items = [i for i in json.loads(Path(p["OWN_metadata_binding"]["manifest"]).read_text())["items"] if i["period"] == "validation"]
    rows = {}; info = {}
    for item in items:
        pn = item["panel"]
        view, tv, tq, st_, hidden = own_m.load_item(P.ROOT, binding.bound_own_item(p, item))
        st_ = np.asarray(st_, bool); gap = np.asarray(view.gap, bool)
        z = np.load(PE.OUT / f"c12_own_{pn}.npz"); assert (z["target"] == st_).all()
        speed = np.asarray(view.speed, float); flow = np.asarray(view.flow, float)
        observed = np.asarray(view.elig, bool) & ~np.asarray(view.target, bool) & ~gap & np.isfinite(speed) & np.isfinite(flow)
        ctx = PE.Ctx(view.stations, st_, np.where(observed, speed, np.nan), z["p"].astype(float), z["bl_v"].astype(float), z["bl_q"].astype(float), gap)
        v13, q13 = PE.apply_chain(C.CHAIN["_all"], ctx, z["v"].astype(float), z["q"].astype(float))
        prm = metric._parameters(view, P.ROOT)[0]
        (v14, q14), (v15, q15) = stage3(mq, mv, v13, q13, st_, gap, np.where(observed, speed, np.nan), np.where(observed, flow, np.nan),
                                        z["p"].astype(float), view.stations, prm.length_km.to_numpy(float), prm.free_speed_kmh.to_numpy(float))
        for name, (v, q) in (("c13", (v13, q13)), ("c14", (v14, q14)), ("c15", (v15, q15))):
            for part, m in (("all", st_), ("nonblock", st_ & ~gap)):
                rows.setdefault((name, part), []).extend(own_m.state_rows(view, tv, tq, v, q, m)[0])
        ok = st_ & ~gap & np.isfinite(tv) & np.isfinite(tq)
        la = np.maximum(view.stations.sort_values("idx").lanes.to_numpy(float), 1.)
        info[pn] = {f"{k}_{n}": float(np.sqrt(np.mean(x[ok] ** 2))) for n, (a, b) in (("c13", (v13, q13)), ("c14", (v14, q14)), ("c15", (v15, q15)))
                    for k, x in (("rv", a - tv), ("rql", (b - tq) / la))}
        print(pn, {k: round(x, 4) for k, x in info[pn].items()}, flush=True)
    panels = p["panels"]; fam = p["families"]
    groups = [[i for i, x in enumerate(panels) if fam[x] == f] for f in sorted(set(fam.values()))]
    dates = sorted({r["date"] for r in rows[("c13", "all")]})
    res = dict(per_panel=info)
    for ref, new in (("c14", "c15"), ("c13", "c15")):
        for part in ("nonblock", "all"):
            x = own_m.paired_state(rows[(ref, part)], rows[(new, part)], panels, dates, groups)
            bp = {pn: round(own_m.paired_state([r for r in rows[(ref, part)] if r["panel"] == pn], [r for r in rows[(new, part)] if r["panel"] == pn], [pn], dates, [[0]])["delta_S_state"], 6) for pn in panels}
            res[f"{new}_vs_{ref}_{part}"] = dict(delta=x["delta_S_state"], ci=x["ci95"], verdict=x["verdict"], by_panel=bp, n_pos=int(sum(v > 0 for v in bp.values())))
            print(new, "vs", ref, part, round(x["delta_S_state"], 6), [round(c, 6) for c in x["ci95"]], x["verdict"], "panels up", res[f"{new}_vs_{ref}_{part}"]["n_pos"], flush=True)
    (OUT / f"eval_s3{TAG}_own.json").write_text(json.dumps(res, indent=1, default=float))


class _HashW:
    def __init__(self):
        self.h = hashlib.sha256()

    def write(self, x):
        self.h.update(x.encode("utf-8"))


def probe(out):
    import pandas as pd
    sys.path.insert(0, str(P.ROOT / "src"))
    from tfb.data import load_cache
    import polish_probe as PP
    mq, mv = models()
    base = OUT / "c14_lsa_on_c13.csv"; c13f = PE.OUT / "c13_polish_on_c03.csv"
    sub = pd.read_csv(base, float_precision="round_trip", dtype={"task": "category"})
    c13 = pd.read_csv(c13f, float_precision="round_trip", usecols=["speed_kmh", "flow_vph"])
    par = pd.read_csv(PP.PARENT, float_precision="round_trip", dtype={"task": "category"})
    kf = P.ROOT / "data/raw/kaggle_public/submission_key.csv"
    key = pd.read_csv(kf, usecols=["submission_id"], dtype=np.int64)
    for col in ("task", "split", "panel", "timestamp", "link_id"):
        key[col] = pd.read_csv(kf, usecols=[col], dtype=str)[col].astype("category")
    assert (sub.submission_id.to_numpy() == key.submission_id.to_numpy()).all()
    st_rows = (key.task == "state").to_numpy(); other = ~st_rows
    for col in ("speed_kmh", "flow_vph", "queue_pred", "path_flow"):
        assert np.array_equal(sub[col].to_numpy()[other], par[col].to_numpy()[other], equal_nan=True)
    speed, flow = sub.speed_kmh.to_numpy(float).copy(), sub.flow_vph.to_numpy(float).copy()
    s13, f13 = c13.speed_kmh.to_numpy(float), c13.flow_vph.to_numpy(float); log = {}
    for (split, panel), g in key[st_rows].groupby(["split", "panel"], observed=True):
        c = load_cache(panel, split); st = c.stations.sort_values("idx")
        ts = pd.to_datetime(g.timestamp.astype(str))
        d = np.array([c.dates.index(x) for x in ts.dt.strftime("%Y-%m-%d")]); t = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
        s = g.link_id.astype(str).map(dict(zip(st.link_id.astype(str), st.idx))).to_numpy().astype(int)
        tgt = np.asarray(c.target, bool); gap = np.asarray(c.gap, bool)
        assert tgt[d, t, s].all() and tgt.sum() == len(g)
        idx = g.index.to_numpy()
        cv = np.full(tgt.shape, np.nan); cq = cv.copy(); cv[d, t, s] = s13[idx]; cq[d, t, s] = f13[idx]
        sp = np.asarray(c.speed, float); fl = np.asarray(c.flow, float)
        obs = np.asarray(c.elig, bool) & ~tgt & ~gap & np.isfinite(sp) & np.isfinite(fl)
        f4v = np.full(tgt.shape, np.nan); f4q = f4v.copy()
        f4v[d, t, s] = par.speed_kmh.to_numpy(float)[idx]; f4q[d, t, s] = par.flow_vph.to_numpy(float)[idx]
        pp = PP.official_p(c, tgt, gap, f4v, f4q)
        prm = metric._parameters(c, P.ROOT)[0]
        (v14, q14), (v15, q15) = stage3(mq, mv, cv, cq, tgt, gap, np.where(obs, sp, np.nan), np.where(obs, fl, np.nan), pp, c.stations,
                                        prm.length_km.to_numpy(float), prm.free_speed_kmh.to_numpy(float))
        assert np.allclose(v14[d, t, s], speed[idx], atol=1e-9) and np.allclose(q14[d, t, s], flow[idx], atol=1e-9), "C14 rebuild mismatch"
        nv, nq = v15[d, t, s], q15[d, t, s]
        assert np.isfinite(nv).all() and np.isfinite(nq).all()
        log[f"{split}/{panel}"] = dict(rows=len(g), changed_share=float(np.mean((np.abs(nv - speed[idx]) > 1e-9) | (np.abs(nq - flow[idx]) > 1e-9))),
                                       mean_abs_dv=float(np.abs(nv - speed[idx]).mean()), mean_abs_dq=float(np.abs(nq - flow[idx]).mean()),
                                       mean_dq=float((nq - flow[idx]).mean()), mean_dv=float((nv - speed[idx]).mean()))
        speed[idx] = nv; flow[idx] = nq
        print(split, panel, {k: round(x, 5) for k, x in log[f"{split}/{panel}"].items()}, flush=True)
    b_v, b_q = sub.speed_kmh.to_numpy(float).copy(), sub.flow_vph.to_numpy(float).copy()
    sub["speed_kmh"] = speed; sub["flow_vph"] = flow
    for col in ("speed_kmh", "flow_vph", "queue_pred", "path_flow"):
        assert np.array_equal(sub[col].to_numpy()[other], par[col].to_numpy()[other], equal_nan=True)
    assert len(sub) == 6980503 and np.isfinite(speed[st_rows]).all() and np.isfinite(flow[st_rows]).all()
    if str(out).endswith(".delta.csv.gz"):  # disk rule 2026-10-04: changed state rows only, sha of the would-be full CSV
        ch = st_rows & ((speed != b_v) | (flow != b_q))
        sub["speed_kmh"] = b_v; sub["flow_vph"] = b_q
        hb = _HashW(); sub.to_csv(hb, index=False); h_base = hb.h.hexdigest()
        sub["speed_kmh"] = speed; sub["flow_vph"] = flow
        hw = _HashW(); sub.to_csv(hw, index=False); h = hw.h.hexdigest()
        sub.loc[ch, ["submission_id", "speed_kmh", "flow_vph"]].to_csv(out, index=False)
        extra = dict(delta_rows=int(ch.sum()), base_full_sha256=h_base, full_sha256=h, delta_sha256=hashlib.sha256(Path(out).read_bytes()).hexdigest())
        print("base sha", h_base, "full sha", h, extra, flush=True)
    else:
        sub.to_csv(out, index=False)
        h = hashlib.sha256(Path(out).read_bytes()).hexdigest(); extra = {}
    by_split = {sp: {k: float(np.average([v[k] for kk, v in log.items() if kk.startswith(sp)], weights=[v["rows"] for kk, v in log.items() if kk.startswith(sp)]))
                     for k in ("changed_share", "mean_abs_dv", "mean_abs_dq", "mean_dq", "mean_dv")} for sp in ("validation", "private")}
    print("wrote", out, "sha256", h); print("by split", by_split)
    Path(str(out).replace(".delta.csv.gz", ".csv")).with_suffix(".json").write_text(json.dumps(dict(lsa=LSA_PRM, base=base.name, sha256=h, by_split=by_split, per_panel=log, **extra), indent=1))


if __name__ == "__main__":
    {"fit": fit, "own": own, "probe": lambda: probe(sys.argv[2])}[sys.argv[1]]()
