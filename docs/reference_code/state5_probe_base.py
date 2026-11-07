"""Claude helper 2026-10-04 (state5, ledger 0407): probe = the C13 Task 1 stack rebuilt with S08 as its input instead of F4
(replacement base). NOT uploaded.
Per split/panel (official view): the C11 stack exactly as s2stage_probe (VQP: 0360 -> 0368 on time-filled arrays -> 0358 -> phys3
-> stage 2 -> phys3) fed with S08 official rows (S's 0400 delivery) instead of the C03 state rows (= F4, checked); the same code
fed with C03 rows is asserted to reproduce the C11 file. Then the B6 imputer blend (imputer files and seeds as polish_probe;
weights per panel = b6full_eval weights_all_days refitted on I-S-OWN March for the S08-based C11, state5/c11_mar_s08_*) and the
C13 polish chain (s3 soft phys3 -> TV -> S4, all-days params). Every non-state row equals C03.
Usage: python docs/reference_code/state5_probe_base.py <out.csv> [s08 | s08r] [s3l]"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
from tfb.data import load_cache
import flow_krige as FK
import state2_cm as CM
import phys3_proj as PJ
import phys3_stack as ST
import phys3_build_probe as PB
import s2stage as S2
import b6full_eval as EV
import polish_eval as E
import polish_probe as PP
import state5_lib as L

C13 = ROOT / "artifacts/claude_ref/polish/c13_polish_on_c03.csv"


def c11_official(c, tgt, gap, pv, pq, m368, models, w, cf):
    st = c.stations.sort_values("idx")
    vobs = np.asarray(c.speed, float); fobs = np.asarray(c.flow, float)
    observed = np.asarray(c.elig, bool) & ~tgt & ~gap & np.isfinite(vobs)
    vcut = st.v_cut.to_numpy(float)
    gr, ncm = CM.common_factor(vobs, observed, c.stations)
    rv = np.where(tgt, CM.apply(pv, gr, vcut, cf), np.nan)
    full = PB.fill_time(np.where(tgt, rv, np.where(observed, vobs, np.nan)))
    pqf = PB.fill_time(np.where(tgt, pq, np.where(np.isfinite(fobs), fobs, np.nan)))
    hvf, hp = S2.hedge_368(m368, vobs, observed, c.stations, full, pqf, tgt)
    hv = np.where(tgt, hvf, np.nan)
    N, den = FK.krige_log_flow(fobs, FK.positions(c.stations))
    hq = FK.blend(np.where(tgt, hv, 0.), pq, N, tgt, c.stations, w, den=den)
    sv, sq, _ = PJ.project(hv, hq, tgt, c.stations, ST.R, ST.A, ST.G, gap=gap)
    return S2.apply_stage2(models, (vobs, fobs, np.asarray(c.elig, bool), tgt, gap), c.stations,
                           pv, pq, rv, gr, ncm, N, den, hp, hv, hq, sv, sq, tgt, "VQP")


def s08_weights(variant="s08"):
    """weights_all_days (as b6full_eval.own) for the S08-based C11 on I-S-OWN March, imputer seeds 0,1,2."""
    stats = {}
    for pn in L.PANELS:
        r = dict(np.load(L.OUT / f"c11_mar_{variant}_{pn}.npz"))
        tv, tq, st_, gap = r["tv"].astype(float), r["tq"].astype(float), r["target"], r["gap"]
        cv, cq = r["c11_v"].astype(float), r["c11_q"].astype(float)
        iv, iq, _ = EV.imp("own", pn, [0, 1, 2])
        la = np.broadcast_to(L.lanes_of(pn)[0], tv.shape)
        ok = st_ & np.isfinite(tv) & np.isfinite(tq)
        stats[pn] = dict(v_nb=EV.sse_grid(cv, iv, tv, ok & ~gap), v_blk=EV.sse_grid(cv, iv, tv, ok & gap),
                         q_nb=EV.sse_grid(cq, iq, tq, ok & ~gap, la), q_blk=EV.sse_grid(cq, iq, tq, ok & gap, la))
    return EV.fit_weights(stats)[0]


def fit_s3l(split):
    """Split-local stage 3 (state5_s3loc features/params) fitted on ALL I-S-OWN non-block cells of that split (S08-base stack)."""
    import lightgbm as lgb, state5_s3loc as S3L
    tag = {"validation": "mar", "private": "apr"}[split]
    Xs, yv, yq = [], [], []
    for pn in L.PANELS:
        z = dict(np.load(L.OUT / f"own_{tag}_s08_{pn}.npz"))
        la, vc, cap, _ = L.lanes_of(pn, split)
        v, q = z["v13"].astype(float), z["q13"].astype(float)
        cols = S3L.feats(z, v, q, la, vc, cap, {"f4v": z["f4v"].astype(float) - v, "f4q": (z["f4q"].astype(float) - q) / la})
        tv, tq = z["tv"].astype(float), z["tq"].astype(float)
        m = z["target"] & ~z["gap"] & np.isfinite(tv) & np.isfinite(tq)
        Xs.append(np.stack([np.asarray(a, float)[m] for a in cols.values()], 1).astype(np.float32)); yv.append((tv - v)[m]); yq.append(((tq - q) / la)[m])
    X = np.concatenate(Xs)
    return lgb.train(S3L.PRM, lgb.Dataset(X, np.concatenate(yv)), S3L.ROUNDS), lgb.train(S3L.PRM, lgb.Dataset(X, np.concatenate(yq)), S3L.ROUNDS), list(cols)


def apply_s3l(mods, c, tgt, gap, obs, v, q, f4v, f4q):
    """Stage 3 on official non-gap targets, then the C13 final soft phys3 + TV (as state5_s3loc variant 1P)."""
    import state5_s3loc as S3L, research2_tv as TVM
    st = c.stations.sort_values("idx"); la = np.maximum(st.lanes.to_numpy(float), 1.); vc = st.v_cut.to_numpy(float); cap = st.capacity_vph.to_numpy(float)
    sp = np.asarray(c.speed, float); fl = np.asarray(c.flow, float)
    z = dict(vobs=np.where(obs, sp, np.nan), qobs=np.where(obs, fl, np.nan), dates=np.asarray(c.dates))
    cols = S3L.feats(z, v, q, la, vc, cap, {"f4v": f4v - v, "f4q": (f4q - q) / la})
    assert list(cols) == mods[2]
    m = tgt & ~gap; ix = np.argwhere(m); I = (ix[:, 0], ix[:, 1], ix[:, 2])
    X = np.stack([np.asarray(a, float)[m] for a in cols.values()], 1).astype(np.float32)
    nv = v.copy(); nq = q.copy()
    nv[I] = np.maximum(v[I] + mods[0].predict(X), 3.); nq[I] = np.clip(q[I] + la[ix[:, 2]] * mods[1].predict(X), 50., 1.3 * cap[ix[:, 2]])
    nv, nq = E.project_soft(E.Ctx(c.stations, tgt, z["vobs"], None, nv, nq, gap), dict(r=3., a=.9, g=1., h=.1))
    return TVM.smooth(nv, nq, tgt & ~gap, c.stations, .004, 0.)


def main(out, variant="s08", stage3=""):
    """variant s08: shipped 0368/stage-2 models (fit on F4-based OPT); s08r: models refit on S08-based OPT (state5_refit)."""
    import state5_c13apr as CA
    chain = json.loads(C13.with_suffix(".json").read_text())["chain"]
    W = s08_weights(variant); print("S08-base blend weights", {p: {k: round(x, 3) for k, x in w.items()} for p, w in W.items()}, flush=True)
    models, m368 = CA.load_models(variant)
    f4models, f4m368 = S2.load_all(), ST.load_368()
    w = np.asarray(json.loads((ROOT / "artifacts/claude_ref/flow/opt/opt_summary.json").read_text())["pooled_weights"])
    cf = json.loads((ROOT / "artifacts/claude_ref/state2/opt_coef.json").read_text()); cf = (cf["b_free"], cf["b_cong"])
    sub = pd.read_csv(C13, float_precision="round_trip", dtype={"task": "category"})
    par = pd.read_csv(PP.PARENT, float_precision="round_trip", dtype={"task": "category"})
    kf = ROOT / "data/raw/kaggle_public/submission_key.csv"
    key = pd.read_csv(kf, usecols=["submission_id"], dtype=np.int64)
    for col in ("task", "split", "panel", "timestamp", "link_id"):
        key[col] = pd.read_csv(kf, usecols=[col], dtype=str)[col].astype("category")
    assert (sub.submission_id.to_numpy() == key.submission_id.to_numpy()).all() and (par.submission_id.to_numpy() == key.submission_id.to_numpy()).all()
    st_rows = (key.task == "state").to_numpy(); other = ~st_rows

    def same_other(a, b):
        ok = (a.task.astype(str).to_numpy()[other] == b.task.astype(str).to_numpy()[other]).all()
        for col in ("speed_kmh", "flow_vph", "queue_pred", "path_flow"):
            ok &= np.array_equal(a[col].to_numpy()[other], b[col].to_numpy()[other], equal_nan=True)
        return ok
    assert same_other(sub, par)
    speed, flow = sub.speed_kmh.to_numpy(float).copy(), sub.flow_vph.to_numpy(float).copy()
    f4s, f4f = par.speed_kmh.to_numpy(float), par.flow_vph.to_numpy(float)
    c11 = pd.read_csv(PP.C11, float_precision="round_trip", usecols=["speed_kmh", "flow_vph"])
    c11v, c11q = c11.speed_kmh.to_numpy(float), c11.flow_vph.to_numpy(float); del c11
    log = {}
    for split in ("validation", "private"):
        s3m = fit_s3l(split) if stage3 == "s3l" else None
        s08 = pd.read_parquet(L.S08 / f"official_{split}.parquet", columns=["panel", "timestamp", "link_id", "speed_kmh", "flow_vph"])
        for panel, g in key[st_rows & (key.split == split).to_numpy()].groupby("panel", observed=True):
            c = load_cache(panel, split); st = c.stations.sort_values("idx")
            ts = pd.to_datetime(g.timestamp.astype(str))
            d = np.array([c.dates.index(x) for x in ts.dt.strftime("%Y-%m-%d")]); t = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
            s = g.link_id.astype(str).map(dict(zip(st.link_id.astype(str), st.idx))).to_numpy().astype(int)
            tgt = np.asarray(c.target, bool); gap = np.asarray(c.gap, bool)
            assert tgt[d, t, s].all() and tgt.sum() == len(g)
            idx = g.index.to_numpy()
            m = g[["timestamp", "link_id"]].astype(str).merge(s08[s08.panel == panel].astype({"timestamp": str, "link_id": str}),
                                                              on=["timestamp", "link_id"], how="left")
            assert len(m) == len(g) and m.speed_kmh.notna().all()

            def grid(vals):
                a = np.full(tgt.shape, np.nan); a[d, t, s] = vals; return a
            # parity: C03 (F4) rows through the same code reproduce the C11 file
            fv, fq = c11_official(c, tgt, gap, grid(f4s[idx]), grid(f4f[idx]), f4m368, f4models, w, cf)
            assert np.allclose(fv[d, t, s], c11v[idx], atol=1e-6) and np.allclose(fq[d, t, s], c11q[idx], atol=1e-6), "C11 rebuild mismatch"
            nv11, nq11 = c11_official(c, tgt, gap, grid(m.speed_kmh.to_numpy(float)), grid(m.flow_vph.to_numpy(float)), m368, models, w, cf)
            mode = {"validation": "valown", "private": "priv"}[split]
            zs = [np.load(EV.OUT / f"imp_{mode}_{panel}_s{k}.npz") for k in PP.SEEDS[split]]
            iv = np.mean([z["speed"] for z in zs], 0).astype(float); iq = np.mean([z["flow"] for z in zs], 0).astype(float)
            bl_v, bl_q = EV.blend(nv11, nq11, iv, iq, gap, W[panel])
            sp = np.asarray(c.speed, float); obs = np.asarray(c.elig, bool) & ~tgt & ~gap & np.isfinite(sp)
            nv3, nq3 = E.apply_chain(chain, E.Ctx(c.stations, tgt, np.where(obs, sp, np.nan), None, bl_v, bl_q, gap), None, None)
            if s3m is not None:
                nv3, nq3 = apply_s3l(s3m, c, tgt, gap, obs, nv3, nq3, grid(f4s[idx]), grid(f4f[idx]))
            nv, nq = nv3[d, t, s], nq3[d, t, s]
            assert np.isfinite(nv).all() and np.isfinite(nq).all()
            # generator speed cap last (ledger 0409): speed <= 1.15 x topology free_flow_speed_kmh
            topo = pd.read_csv(ROOT / f"data/raw/kaggle_public/corridors/{panel}/network/lwr_mainline_topology.csv")
            cl = 1.15 * g.link_id.astype(str).map(dict(zip(topo.link_id.astype(str), topo.free_flow_speed_kmh.astype(float)))).to_numpy(float)
            assert np.isfinite(cl).all()
            n_clip = int((nv > cl + 1e-9).sum()); nv = np.minimum(nv, cl)
            gp = gap[d, t, s]
            log[f"{split}/{panel}"] = dict(rows=len(g), mean_abs_dv=float(np.abs(nv - speed[idx]).mean()), mean_abs_dq=float(np.abs(nq - flow[idx]).mean()),
                                           mean_dv=float((nv - speed[idx]).mean()), mean_dq=float((nq - flow[idx]).mean()),
                                           mean_abs_dv_gap=float(np.abs(nv - speed[idx])[gp].mean()) if gp.any() else 0.,
                                           rms_s08_f4_v=float(np.sqrt(np.mean((m.speed_kmh.to_numpy(float) - f4s[idx]) ** 2))), n_speed_cap=n_clip)
            speed[idx] = nv; flow[idx] = nq
            print(split, panel, {k: round(x, 4) for k, x in log[f"{split}/{panel}"].items()}, flush=True)
    new = sub; new["speed_kmh"] = speed; new["flow_vph"] = flow
    assert same_other(new, par)
    assert len(new) == 6980503 and not new.isna().any().any()
    new.to_csv(out, index=False)
    h = hashlib.sha256(Path(out).read_bytes()).hexdigest()
    keys = ("mean_abs_dv", "mean_abs_dq", "mean_dv", "mean_dq", "rms_s08_f4_v")
    caps = {sp: int(sum(v["n_speed_cap"] for kk, v in log.items() if kk.startswith(sp))) for sp in ("validation", "private")}
    by_split = {sp: {k: float(np.average([v[k] for kk, v in log.items() if kk.startswith(sp)], weights=[v["rows"] for kk, v in log.items() if kk.startswith(sp)]))
                     for k in keys} for sp in ("validation", "private")}
    print("wrote", out, "rows", len(new), "sha256", h); print("by split", by_split)
    Path(out).with_suffix(".json").write_text(json.dumps(dict(variant="S08 replacement base " + variant + (" + split-local stage 3 (1P)" if stage3 else ""), blend_weights=W, chain=chain, base=C13.name,
                                                              parent=PP.PARENT.name, sha256=h, by_split=by_split, speed_cap_rows=caps, per_panel=log), indent=1, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:4])
