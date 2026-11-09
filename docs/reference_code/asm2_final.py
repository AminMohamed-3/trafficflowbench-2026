"""asm2 (ledger 0475) DRAFT single entry script for the final candidate (award rule, forum 744187: organizers rerun the code and
the result must match the selected submission within 1% of its private score).

Final candidate F1 (2026-10-05) = C17 board chain + phys5 stage 4 + onset Q0436:
  Task 1/3 state rows, per split and panel, in memory, every stage from the split's released official view:
    1. asm chain K5 (asm2_build.Stages): S08 input -> 0360 -> 0368 -> 0358 -> phys3 -> stage 2 -> B6 imputer blend -> polish
       -> phys4 LSA -> phys4 stage 3 (L) -> gap specialist (skip with --no-gap = isolation X1) -> speed clip
    2. S lane S0424 + S0426 + S0447 (s0447_probe.predict: occupancy residuals and joint demand-block flow factor)
    3. phys5 stage 4 (tag AL+FL, phys5_stage4.apply) + speed clip   (skip with --no-phys5 = C17 state)
  Task 2 queue rows:
    onset   = Q03 rows (--onset-rows; default the C03 rows) + Q0436 delta (--onset-delta; '' to skip)
    ongoing = C03 CNN adapted ens2 eiou rows (--ongoing-rows; default C03 rows) + selcond decode (--selcond rows csv; '' = X2)
  Task 4 ODME rows = NNLS .05 rows (--odme-rows; default the C03 rows)
Rows given as files: parquet/csv with submission_id + queue_pred / path_flow (asm2_qo.py output), or for --ongoing-rows /
--selcond the (window_id, timestamp, link_id, queue_pred[_new]) layout of qcnn_avg_probs.py / asm2_selcond.py.

Frozen fitted inputs are the files listed in the manifests (asm K5 manifest, lane_s 0447_probe manifest, phys5 s4 model);
see docs/report/ASM2_DEPENDENCIES.md for how each is regenerated and what cannot be.
Usage:
  python docs/reference_code/asm2_final.py --out F.csv [--no-gap] [--no-selcond] [--no-phys5] [--panels P1,P2]
        [--queue-base artifacts/claude_ref/asm2/c03_parent.parquet | none] [--onset-rows X] [--onset-delta Y] [--ongoing-rows Z]
        [--selcond S] [--odme-rows O] [--t1-from F.state.parquet] [--check SPEC]
  --check SPEC compares the result with a reference (asm2_diff spec 'full.csv+delta+delta').
Writes F.csv (LF line ends; the manifest also records the CRLF sha that Windows probe files carry) and F.csv.manifest.json.
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, time, argparse, platform, hashlib
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
import asm2_phys_lwr_diag as _P
sys.modules["phys_lwr_diag"] = _P
import asm2_build as AB

CR = ROOT / "artifacts/claude_ref"
DEF = dict(queue_base="artifacts/claude_ref/asm2/c03_parent.parquet",
           onset_delta="artifacts/claude_ref/onset8/c19_onset8_q0436_on_c17.delta.csv",
           selcond="artifacts/claude_ref/gapres/selcond_official_rows.csv",
           hedge_rows="artifacts/claude_ref/onset9/hedge_B_rows.csv")
T1 = ["s08in", "c10", "s2", "b6", "polish", "lsa", "s3p4", "gap", "clip"]
PHYS5_TAG = "AL+FL"


def rd(p, cols=None):
    p = str(p)
    return pd.read_parquet(p, columns=cols) if p.endswith(".parquet") else pd.read_csv(p, float_precision="round_trip", usecols=cols)


def slane_setup():
    """S lane modules (S lane 0424/0426/0447, docs/reference_code/s04*). Import the asm feature modules first so they keep
    the full phys_lwr_diag shim; s0416_density.modules swaps in a metadata-only stub for the S lane's own imports."""
    import phys4_feat, phys4_lsa, flow_screen, phys4_lsa_eval, polish_eval  # noqa: F401
    import lightgbm as lgb
    import s0447_probe as S47
    P424 = S47.Q.P
    meta = json.loads((P424.B.INP / "metadata.json").read_text()); _, pf, _ = P424.B.modules(meta)
    sys.modules["phys_lwr_diag"] = _P
    for split in ("validation", "private"):
        S47.F.gate(split)
    files = {k: [P424.C.O.BASE_OUT / f"occupancy_{k}.txt"] + [P424.C.O.BASE_OUT / f"seed{s}/occupancy_{k}.txt" for s in (4702, 4703)] for k in ("v", "q")}
    models = {k: [lgb.Booster(model_file=str(p)) for p in pp] for k, pp in files.items()}
    used = [p for pp in files.values() for p in pp]
    used += [S47.Q.LOCAL / sp / f"seed{s}" / f"occupancy_{k}.txt" for sp in ("validation", "private") for s in (4701, 4702, 4703) for k in ("v", "q")]
    used += [S47.F.OUT / sp / f"seed{s}/joint.txt" for sp in ("validation", "private") for s in (4701, 4702, 4703)]
    return S47, pf, models, used


class _Bag:
    """Mean prediction of several LightGBM boosters (stage-4 bag, tune)."""
    def __init__(s, ms):
        s.ms = ms

    def predict(s, X, num_threads=3):
        return np.mean([m.predict(X, num_threads=num_threads) for m in s.ms], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--panels", default=None)
    ap.add_argument("--no-gap", action="store_true"); ap.add_argument("--no-selcond", action="store_true"); ap.add_argument("--no-phys5", action="store_true")
    ap.add_argument("--queue-base", default=DEF["queue_base"]); ap.add_argument("--onset-rows", default=None)
    ap.add_argument("--onset-delta", default=DEF["onset_delta"]); ap.add_argument("--ongoing-rows", default=None)
    ap.add_argument("--selcond", default=DEF["selcond"]); ap.add_argument("--odme-rows", default=None)
    ap.add_argument("--t1-from", default=None, help="reuse state rows (parquet submission_id, speed_kmh, flow_vph) of an earlier run")
    ap.add_argument("--check", default=None)
    ap.add_argument("--s0631-query", default=None, help="write the S 0631 TabPFN query rows (official target nongap switch cells, "
                    "parent = final state) to this npz (asm2_s0631.py predict)")
    ap.add_argument("--state-only", action="store_true", help="stop after the Task 1 state parquet (and --s0631-query)")
    ap.add_argument("--state-rows", default=None, help="speed overlay (npz/parquet submission_id, speed_kmh) on state rows, "
                    "e.g. the S 0631 rows from asm2_s0631.py; flow unchanged")
    ap.add_argument("--state-atol", type=float, default=1e-8, help="maximum absolute state float difference accepted by --check")
    ap.add_argument("--hedge", choices=["A", "B"], default="A", help="final-pair onset file: A = Q0436, B = onset9 hedge")
    ap.add_argument("--hedge-rows", default=DEF["hedge_rows"])
    ap.add_argument("--hedge-rows-A", default=None, help="C_pair 2026-10-12 (exp/C_pair/PAIR.md): rows file for file A too (the split pair: A' "
                    "keeps one site of a private two-site onset decode, B the other; c_hedge_b.py build --rule split). Same layout and "
                    "asserts as --hedge-rows. Default none = file A unchanged")
    ap.add_argument("--noT", action="store_true", help="replace all Task 2 rows with T's qualified strict retrained delivery")
    ap.add_argument("--noT-rows", default="artifacts/T_noT/delivery/task2_noT.parquet")
    ap.add_argument("--noT-receipt", default="artifacts/T_noT/delivery/rows_receipt.json")
    ap.add_argument("--tune", default=None, help="tune knob config JSON (ledger 0479-0481): per-split B6 arms/weights, polish chain, LSA, "
                    "stage 3 bag/scale, stage 4 bag/scale; omitted = shipped F1 chain")
    a = ap.parse_args(); t0 = time.time()
    if a.noT and a.hedge != "A":
        ap.error("--noT requires hedge A until a no-T hedge is qualified")
    if a.hedge_rows_A and (a.noT or a.hedge != "A"):
        ap.error("--hedge-rows-A applies to --hedge A without --noT (file B takes --hedge-rows)")
    inputs = {}

    def reg(p):
        p = Path(p); p = p if p.is_absolute() else ROOT / p
        try:
            name = str(p.relative_to(ROOT))
        except ValueError:
            name = str(p)
        inputs[name.replace("\\", "/")] = AB.sha(p); return p

    # ---- key and the queue/odme base
    kf = reg(AB.KEY)
    key = pd.read_csv(kf, usecols=["submission_id"], dtype=np.int64)
    for col in ["task", "split", "panel", "timestamp", "link_id", "window_id"]:
        key[col] = pd.read_csv(kf, usecols=[col], dtype=str)[col].astype("category")
    N = len(key); assert N == AB.NROWS
    ids = key.submission_id.to_numpy()
    speed = np.zeros(N); flow = np.zeros(N)
    if a.queue_base.lower() == "none":   # no prior submission file: every queue/odme row must come from --*-rows (asserted below)
        qpred = np.full(N, np.nan); pflow = np.full(N, np.nan)
    else:
        base = rd(reg(a.queue_base), ["submission_id", "queue_pred", "path_flow"])
        assert (base.submission_id.to_numpy() == ids).all()
        qpred = base.queue_pred.to_numpy(float).copy(); pflow = base.path_flow.to_numpy(float).copy(); del base
    st_rows = (key.task == "state").to_numpy(); q_rows = (key.task == "queue").to_numpy(); o_rows = (key.task == "odme").to_numpy()
    log = dict(stages={})
    noT_rows = None
    if a.noT:
        import asm2_noT_rows as NT
        noT_rows, log["noT"] = NT.validate(ROOT, a.noT_rows, a.noT_receipt, key.loc[q_rows])
        inputs.update(log["noT"]["input_sha256"])

    # ---- Task 1 state rows
    comps = [x for x in T1 if not (a.no_gap and x == "gap")]
    if a.t1_from:
        tf = rd(reg(a.t1_from)); r = np.searchsorted(ids, tf.submission_id.to_numpy()); assert (ids[r] == tf.submission_id.to_numpy()).all()
        speed[r] = tf.speed_kmh.to_numpy(float); flow[r] = tf.flow_vph.to_numpy(float)
        assert r.size == st_rows.sum() and st_rows[r].all()
    else:
        tune = json.loads(reg(a.tune).read_text()) if a.tune else None
        S = AB.Stages(comps, None, "L", tune)
        S47, pf, smodels, sused = slane_setup()
        for p in sused:
            reg(p)
        import phys5_stage4 as S4, asm_inter as AI
        from tfb.data import load_cache
        from tfb import metric
        if not a.no_phys5:
            m4 = S4.models(PHYS5_TAG); reg(S4.OUT / f"s4_{PHYS5_TAG}_yq.txt"); reg(S4.OUT / f"s4_{PHYS5_TAG}_names.json")
            s4scale = 1.
            if tune and tune.get("s4"):   # bag of stage-4 yq models (mean of log residuals) times scale
                import lightgbm as lgb
                m4["yq"] = _Bag([lgb.Booster(model_file=str(reg(f))) for f in tune["s4"]["models"]]); s4scale = float(tune["s4"].get("scale", 1.))
        s08 = {}; panels = set(a.panels.split(",")) if a.panels else None; s631 = []
        for (split, panel), g in key[st_rows].groupby(["split", "panel"], observed=True):
            if panels and panel not in panels:
                continue
            tp = time.time()
            if split not in s08:
                s08[split] = pd.read_parquet(S._in(AB.S08_DIR / f"official_{split}.parquet"), columns=["panel", "timestamp", "link_id", "speed_kmh", "flow_vph"])
            m8 = g[["timestamp", "link_id"]].astype(str).merge(s08[split][s08[split].panel == panel].astype({"timestamp": str, "link_id": str}),
                                                               on=["timestamp", "link_id"], how="left", validate="one_to_one")
            assert len(m8) == len(g) and m8.speed_kmh.notna().all()
            vin, qin = m8.speed_kmh.to_numpy(float), m8.flow_vph.to_numpy(float); del m8
            c = load_cache(panel, split); st = c.stations.sort_values("idx")
            ts = pd.to_datetime(g.timestamp.astype(str))
            d = np.array([c.dates.index(x) for x in ts.dt.strftime("%Y-%m-%d")]); t = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
            s = g.link_id.astype(str).map(dict(zip(st.link_id.astype(str), st.idx))).to_numpy().astype(int)
            tgt = np.asarray(c.target, bool); gap = np.asarray(c.gap, bool)
            assert tgt[d, t, s].all() and tgt.sum() == len(g)
            idx = g.index.to_numpy()
            res = S.run_panel(split, panel, c, d, t, s, np.full(len(g), np.nan), np.full(len(g), np.nan), {}, vin, qin)
            v, q = res[comps[-1]]
            # S lane (in memory, parent = the asm rows; its predict returns changed rows only)
            parent = pd.DataFrame({"submission_id": ids[idx], "split": split, "panel": panel, "timestamp": g.timestamp.astype(str).to_numpy(),
                                   "link_id": g.link_id.astype(str).to_numpy(), "speed_kmh": v, "flow_vph": q})
            dl, sst = S47.predict(c, parent, pf, smodels)
            pos = pd.Index(parent.submission_id).get_indexer(dl.submission_id); assert (pos >= 0).all()
            v2, q2 = v.copy(), q.copy(); v2[pos] = dl.speed_kmh.to_numpy(float); q2[pos] = dl.flow_vph.to_numpy(float)
            if not a.no_phys5:
                cv = np.full(tgt.shape, np.nan); cq = cv.copy(); cv[d, t, s] = v2; cq[d, t, s] = q2
                sp = np.asarray(c.speed, float); fl = np.asarray(c.flow, float)
                obs = np.asarray(c.elig, bool) & ~tgt & ~gap & np.isfinite(sp) & np.isfinite(fl)
                prm = metric._parameters(c, ROOT)[0]
                nv, nq, n4 = S4.apply(m4, cv, cq, tgt, gap, np.where(obs, sp, np.nan), np.where(obs, fl, np.nan), c.stations,
                                      prm.length_km.to_numpy(float), prm.free_speed_kmh.to_numpy(float), s4scale)
                nv = AI.clipv(nv, tgt, AI.clipvec(panel, c.stations))
                v3, q3 = nv[d, t, s], nq[d, t, s]
                assert np.array_equal(v3[gap[d, t, s]], v2[gap[d, t, s]]) and np.array_equal(q3[gap[d, t, s]], q2[gap[d, t, s]])
            else:
                v3, q3 = v2, q2
            assert np.isfinite(v3).all() and np.isfinite(q3).all()
            speed[idx], flow[idx] = v3, q3
            if a.s0631_query:   # S 0631 query rows on this panel: S's fields() on the official view, parent = final state v3
                import types, asm2_s0631 as S631
                sp0 = np.asarray(c.speed, float)
                hq_ = types.SimpleNamespace(speed=sp0, visible=np.isfinite(sp0) & np.isfinite(np.asarray(c.flow, float)),
                                            elig=np.asarray(c.elig, bool), target=tgt, gap=gap, stations=c.stations)
                par = np.full(tgt.shape, np.nan); par[d, t, s] = v3
                x6, ix6, sc6 = S631.fields(hq_, par, S.hp)
                rp = np.full(tgt.shape, -1, np.int64); rp[d, t, s] = np.arange(len(g)); pos6 = rp[tuple(ix6.T)]
                assert (pos6 >= 0).all() and not gap[tuple(ix6.T)].any()
                topo = metric._parameters(c, ROOT)[1]; st6 = c.stations.sort_values("idx")
                cap6 = 1.15 * topo.free_flow_speed_kmh.reindex(st6.link_id.astype(str)).to_numpy(float)
                s631.append(dict(X=x6, scale=sc6, submission_id=ids[idx[pos6]], parent_v=v3[pos6], cap=cap6[ix6[:, 2]],
                                 split=np.full(len(ix6), split), panel=np.full(len(ix6), panel), ix=ix6))
            log["stages"][f"{split}/{panel}"] = dict(rows=len(g), slane_changed=int(len(dl)), phys5_mean_abs_dq=float(np.abs(q3 - q2).mean()),
                                                     asm_vs_s08_mean_abs_dv=float(np.abs(v - vin).mean()), s=round(time.time() - tp))
            print(split, panel, log["stages"][f"{split}/{panel}"], flush=True)
        inputs.update(S.inputs)
        if a.s0631_query:
            qz = {k: np.concatenate([x[k] for x in s631]) for k in s631[0]}
            np.savez_compressed(a.s0631_query, **qz); log["s0631_query"] = dict(file=a.s0631_query, rows=int(len(qz["submission_id"])))
            print("S0631 query rows", len(qz["submission_id"]), flush=True)
        if panels:
            print("debug panels: no file written"); return
    if a.state_rows:   # speed-only overlay on state rows (S 0631)
        sr = np.load(reg(a.state_rows)) if a.state_rows.endswith(".npz") else rd(reg(a.state_rows))
        sid_, sv_ = np.asarray(sr["submission_id"], np.int64), np.asarray(sr["speed_kmh"], float)
        r = np.searchsorted(ids, sid_); assert (r < len(ids)).all() and (ids[r] == sid_).all() and st_rows[r].all() and len(np.unique(r)) == len(r)
        assert np.isfinite(sv_).all() and (sv_ > 0).all()
        log["state_rows"] = dict(file=a.state_rows, rows=int(len(r)), changed=int((speed[r] != sv_).sum()),
                                 mean_abs_dv=float(np.abs(speed[r] - sv_).mean()) if len(r) else 0.); speed[r] = sv_
    pq_ = pd.DataFrame({"submission_id": ids[st_rows], "speed_kmh": speed[st_rows], "flow_vph": flow[st_rows]})
    pq_.to_parquet(str(a.out) + ".state.parquet", index=False, compression="zstd")
    if a.state_only:
        Path(str(a.out) + ".state.json").write_text(json.dumps(dict(log=log, inputs=inputs, runtime_s=round(time.time() - t0)), indent=1, default=str))
        print("state only: wrote", str(a.out) + ".state.parquet", flush=True); return

    # ---- queue rows
    wi = pd.concat([pd.read_csv(p, dtype=str) for p in (ROOT / "data/raw/kaggle_public/task2").glob("*/*/window_index.csv")])
    cond = key.window_id.astype(str).map(dict(zip(wi.window_id, wi.condition))).to_numpy()
    on_rows = q_rows & (cond == "queue_onset"); og_rows = q_rows & (cond == "queue_ongoing")
    kq = key.loc[q_rows, ["window_id", "timestamp", "link_id"]].astype(str).reset_index()

    def put_rows(f, col, allowed):
        df = rd(reg(f))
        if "submission_id" in df:
            assert df.submission_id.is_unique, f"{f}: duplicate submission ids"
            r = np.searchsorted(ids, df.submission_id.to_numpy())
            assert (r < len(ids)).all() and (ids[r] == df.submission_id.to_numpy()).all(), f"{f}: ids outside the key"
            vals = df[col].to_numpy(float)
        else:
            vc = "queue_pred_new" if "queue_pred_new" in df else "queue_pred"
            m = kq.merge(df[["window_id", "timestamp", "link_id", vc]].astype({"window_id": str, "timestamp": str, "link_id": str}),
                         on=["window_id", "timestamp", "link_id"], how="inner", validate="one_to_one")
            assert len(m) == len(df), f"{f}: rows not on the key"
            r = m["index"].to_numpy(); vals = m[vc].to_numpy(float)
        assert allowed[r].all(), f"{f}: rows outside the allowed condition"
        return r, vals

    for name, f, col, allowed in (("onset_rows", a.onset_rows, "queue_pred", on_rows), ("ongoing_rows", a.ongoing_rows, "queue_pred", og_rows),
                                  ("odme_rows", a.odme_rows, "path_flow", o_rows)):
        if f:
            r, vals = put_rows(f, col, allowed); tgt_arr = pflow if col == "path_flow" else qpred
            log[name] = dict(file=f, rows=int(len(r)), changed=int((tgt_arr[r] != vals).sum())); tgt_arr[r] = vals
            if name == "onset_rows":
                assert len(r) == on_rows.sum()
    if a.selcond and not a.no_selcond:
        r, vals = put_rows(a.selcond, "queue_pred", og_rows); log["selcond"] = dict(file=a.selcond, changed=int((qpred[r] != vals).sum())); qpred[r] = vals
    if a.onset_delta:
        r, vals = put_rows(a.onset_delta, "queue_pred", on_rows); log["onset_delta"] = dict(file=a.onset_delta, changed=int((qpred[r] != vals).sum())); qpred[r] = vals
    hedge_file = a.hedge_rows if a.hedge == "B" else a.hedge_rows_A   # file A takes rows only when --hedge-rows-A is given (split pair)
    if hedge_file:
        h = rd(reg(hedge_file))
        r, vals = put_rows(hedge_file, "queue_pred", on_rows)
        assert (key.iloc[r].split == "private").all(), "hedge rows must be private onset rows"
        assert (key.iloc[r].window_id.astype(str).to_numpy() == h.window_id.astype(str).to_numpy()).all()
        assert (qpred[r] == h.queue_pred_A.to_numpy(float)).all(), "onset differs from hedge A: rerun onset9_hedge.py"
        log["hedge"] = dict(file=hedge_file, windows=sorted(h.window_id.astype(str).unique()), changed=int((qpred[r] != vals).sum()))
        qpred[r] = vals

    if noT_rows is not None:
        r = pd.Index(ids).get_indexer(noT_rows.submission_id)
        assert (r >= 0).all() and q_rows[r].all() and len(r) == int(q_rows.sum())
        log["noT"]["changed_vs_default"] = int((qpred[r] != noT_rows.queue_pred.to_numpy()).sum())
        qpred[r] = noT_rows.queue_pred.to_numpy(float)

    # ---- assemble + validate
    if a.queue_base.lower() == "none":
        assert not np.isnan(qpred[q_rows]).any(), "--queue-base none: queue rows not covered by --onset-rows/--ongoing-rows"
        assert not np.isnan(pflow[o_rows]).any(), "--queue-base none: odme rows not covered by --odme-rows"
        qpred[~q_rows] = 0.; pflow[~o_rows] = 0.
    sub = pd.DataFrame({"submission_id": ids, "task": key.task.astype(str).to_numpy(), "speed_kmh": np.where(st_rows, speed, 0.),
                        "flow_vph": np.where(st_rows, flow, 0.), "queue_pred": np.where(q_rows, qpred, 0.), "path_flow": np.where(o_rows, pflow, 0.)})
    assert not sub.isna().any().any() and np.isfinite(sub[["speed_kmh", "flow_vph", "path_flow"]].to_numpy()).all()
    assert np.isin(sub.queue_pred[q_rows], [0., 1.]).all() and (sub.path_flow[o_rows] >= 0).all()
    assert (sub.speed_kmh[st_rows] > 0).all() and (sub.flow_vph[st_rows] >= 0).all()
    sub.to_csv(a.out, index=False, lineterminator="\n")
    man = dict(output=str(a.out), sha256_lf=AB.sha(a.out), sha256_crlf=AB.csv_sha(sub), rows=N, tune=a.tune, components=dict(t1=comps, slane="S0424+S0426+S0447",
               phys5=None if a.no_phys5 else PHYS5_TAG, state_rows=a.state_rows, selcond=not a.no_selcond and bool(a.selcond), onset_delta=a.onset_delta, hedge=a.hedge, hedge_rows=hedge_file, noT=a.noT),
               log=log, inputs=inputs, runtime_s=round(time.time() - t0),
               env=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, platform=platform.platform()))
    import lightgbm
    man["env"]["lightgbm"] = lightgbm.__version__
    if a.check:
        import asm2_diff as DF
        _, R = DF.load_spec(a.check)
        man["check"] = dict(ref=a.check, **{c: dict(max_abs=float(np.abs(sub[c].to_numpy(float) - R[c]).max()), n_diff=int((sub[c].to_numpy(float) != R[c]).sum()))
                                            for c in ("speed_kmh", "flow_vph", "queue_pred", "path_flow")})
        print("check", json.dumps(man["check"]), flush=True)
        man["check"]["state_atol"] = a.state_atol
        man["check"]["passed"] = all(man["check"][c]["max_abs"] <= a.state_atol for c in ("speed_kmh", "flow_vph")) and all(
            man["check"][c]["n_diff"] == 0 for c in ("queue_pred", "path_flow"))
    Path(str(a.out) + ".manifest.json").write_text(json.dumps(man, indent=1, default=str))
    if a.check and not man["check"]["passed"]:
        raise ValueError("reference parity failed; diagnostics saved in output manifest")
    print("wrote", a.out, "crlf sha", man["sha256_crlf"], "runtime", man["runtime_s"], "s", flush=True)


if __name__ == "__main__":
    main()
