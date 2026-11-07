"""Claude helper 2026-10-04 (ledger 0402, polish): probe = C12 file with the accepted polish chain applied to its Task 1 state rows.
NOT uploaded. Base: artifacts/claude_ref/probes/c12_state_c11_b6blp_on_c03.csv (sha 6489028851d3...). Chain parameters = OPT all-days
fits (polish_eval json). Per split/panel the chain context comes only from the released official view: target/gap masks,
observed speed (elig, not target, not gap), station constants. If the chain has s3, the pre-projection blend is rebuilt
exactly as b6full_probe (C11 file + imputer, weights_all_days of own_eval_s012.json, val seeds 1,2 / priv seed 0) and the final
phys3 is re-run with the retuned parameters. s5 needs the 0368 p on the official view (s2stage.hedge_368 on F4 + 0360).
Every non-state row equals C03. Usage: python docs/reference_code/polish_probe.py '<chain json>' <out.csv>"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
from tfb.data import load_cache
import polish_eval as E

C12 = ROOT / "artifacts/claude_ref/probes/c12_state_c11_b6blp_on_c03.csv"
C11 = ROOT / "artifacts/claude_ref/probes/c11_state_stack_s2stage_on_c03.csv"
PARENT = ROOT / "submissions/c03_queue_cnn_adapt_ongoing.csv"
B6W = ROOT / "artifacts/claude_ref/b6full/own_eval_s012.json"
SEEDS = {"validation": [1, 2], "private": [0]}


def official_p(c, tgt, gap, pv, pq):
    """0368 congestion probability on the official view, built as s2stage_probe (F4 = C03 state rows -> 0360 -> 0368)."""
    import state2_cm as CM, phys3_stack as ST, phys3_build_probe as PB, s2stage as S2
    cf = json.loads((ROOT / "artifacts/claude_ref/state2/opt_coef.json").read_text()); cf = (cf["b_free"], cf["b_cong"])
    st = c.stations.sort_values("idx"); vcut = st.v_cut.to_numpy(float)
    vobs = np.asarray(c.speed, float); fobs = np.asarray(c.flow, float)
    observed = np.asarray(c.elig, bool) & ~tgt & ~gap & np.isfinite(vobs)
    gr, ncm = CM.common_factor(vobs, observed, c.stations)
    rv = np.where(tgt, CM.apply(pv, gr, vcut, cf), np.nan)
    full = PB.fill_time(np.where(tgt, rv, np.where(observed, vobs, np.nan)))
    pqf = PB.fill_time(np.where(tgt, pq, np.where(np.isfinite(fobs), fobs, np.nan)))
    _, hp = S2.hedge_368(ST.load_368(), vobs, observed, c.stations, full, pqf, tgt)
    return hp


def main(chain_json, out):
    chain = json.loads(chain_json)
    sub = pd.read_csv(C12, float_precision="round_trip", dtype={"task": "category"})
    par = pd.read_csv(PARENT, float_precision="round_trip", dtype={"task": "category"})
    kf = ROOT / "data/raw/kaggle_public/submission_key.csv"
    key = pd.read_csv(kf, usecols=["submission_id"], dtype=np.int64)
    for col in ("task", "split", "panel", "timestamp", "link_id"):  # one column at a time (laptop RAM)
        key[col] = pd.read_csv(kf, usecols=[col], dtype=str)[col].astype("category")
    assert (sub.submission_id.to_numpy() == key.submission_id.to_numpy()).all() and (par.submission_id.to_numpy() == key.submission_id.to_numpy()).all()
    st_rows = (key.task == "state").to_numpy(); other = ~st_rows
    def same_other(a, b):
        ok = (a.task.astype(str).to_numpy()[other] == b.task.astype(str).to_numpy()[other]).all()
        for col in ("speed_kmh", "flow_vph", "queue_pred", "path_flow"):
            ok &= np.array_equal(a[col].to_numpy()[other], b[col].to_numpy()[other], equal_nan=True)
        return ok
    assert same_other(sub, par)
    speed, flow = sub.speed_kmh.to_numpy(float).copy(), sub.flow_vph.to_numpy(float).copy(); log = {}
    if "s3" in chain:
        import b6full_eval as EV
        W = json.loads(B6W.read_text())["per_panel"]
        c11 = pd.read_csv(C11, float_precision="round_trip", usecols=["speed_kmh", "flow_vph"])
        c11v, c11q = c11.speed_kmh.to_numpy(float), c11.flow_vph.to_numpy(float); del c11
    for (split, panel), g in key[st_rows].groupby(["split", "panel"], observed=True):
        c = load_cache(panel, split); st = c.stations.sort_values("idx")
        ts = pd.to_datetime(g.timestamp.astype(str))
        d = np.array([c.dates.index(x) for x in ts.dt.strftime("%Y-%m-%d")]); t = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
        s = g.link_id.astype(str).map(dict(zip(st.link_id.astype(str), st.idx))).to_numpy().astype(int)
        tgt = np.asarray(c.target, bool); gap = np.asarray(c.gap, bool)
        assert tgt[d, t, s].all() and tgt.sum() == len(g)
        idx = g.index.to_numpy()
        cv = np.full(tgt.shape, np.nan); cq = cv.copy(); cv[d, t, s] = speed[idx]; cq[d, t, s] = flow[idx]
        sp = np.asarray(c.speed, float)
        obs = np.asarray(c.elig, bool) & ~tgt & ~gap & np.isfinite(sp)
        bl_v = bl_q = None
        if "s3" in chain:
            v11 = np.full(tgt.shape, np.nan); q11 = v11.copy(); v11[d, t, s] = c11v[idx]; q11[d, t, s] = c11q[idx]
            mode = {"validation": "valown", "private": "priv"}[split]
            zs = [np.load(EV.OUT / f"imp_{mode}_{panel}_s{k}.npz") for k in SEEDS[split]]
            assert all(list(z["dates"]) == list(c.dates) for z in zs)
            iv = np.mean([z["speed"] for z in zs], 0).astype(float); iq = np.mean([z["flow"] for z in zs], 0).astype(float)
            bl_v, bl_q = EV.blend(v11, q11, iv, iq, gap, W[panel]["weights_all_days"])
            chk = E.Ctx(c.stations, tgt, np.where(obs, sp, np.nan), None, bl_v, bl_q, gap)
            rv, rq = E.project_soft(chk, E.REF_S3)
            assert np.allclose(rv[d, t, s], cv[d, t, s], atol=1e-6) and np.allclose(rq[d, t, s], cq[d, t, s], atol=1e-6), "C12 rebuild mismatch"
        if "s5" in chain:
            f4v = np.full(tgt.shape, np.nan); f4q = f4v.copy()
            f4v[d, t, s] = par.speed_kmh.to_numpy(float)[idx]; f4q[d, t, s] = par.flow_vph.to_numpy(float)[idx]
            p = official_p(c, tgt, gap, f4v, f4q)
        else:
            p = np.full(tgt.shape, np.nan)
        ctx = E.Ctx(c.stations, tgt, np.where(obs, sp, np.nan), p, bl_v, bl_q, gap)
        nv3, nq3 = E.apply_chain(chain, ctx, cv, cq)
        nv, nq = nv3[d, t, s], nq3[d, t, s]
        assert np.isfinite(nv).all() and np.isfinite(nq).all()
        log[f"{split}/{panel}"] = dict(rows=len(g), changed_share=float(np.mean((np.abs(nv - speed[idx]) > 1e-9) | (np.abs(nq - flow[idx]) > 1e-9))),
                                       mean_abs_dv=float(np.abs(nv - speed[idx]).mean()), mean_abs_dq=float(np.abs(nq - flow[idx]).mean()),
                                       mean_dq=float((nq - flow[idx]).mean()))
        speed[idx] = nv; flow[idx] = nq
        print(split, panel, {k: round(x, 5) for k, x in log[f"{split}/{panel}"].items()}, flush=True)
    new = sub; new["speed_kmh"] = speed; new["flow_vph"] = flow
    assert same_other(new, par)
    assert len(new) == 6980503 and not new.isna().any().any()
    new.to_csv(out, index=False)
    h = hashlib.sha256(Path(out).read_bytes()).hexdigest()
    by_split = {sp: {k: float(np.average([v[k] for kk, v in log.items() if kk.startswith(sp)], weights=[v["rows"] for kk, v in log.items() if kk.startswith(sp)]))
                     for k in ("changed_share", "mean_abs_dv", "mean_abs_dq", "mean_dq")} for sp in ("validation", "private")}
    print("wrote", out, "rows", len(new), "sha256", h); print("by split", by_split)
    Path(out).with_suffix(".json").write_text(json.dumps(dict(chain=chain, base=C12.name, parent=PARENT.name, sha256=h, by_split=by_split, per_panel=log), indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
