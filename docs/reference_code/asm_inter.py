"""Claude helper asm (2026-10-04, ledger 0411): interaction tests of the Task 1 pieces at their final position.

opt: OPT J_S (10 panels, official mode A, day bootstrap 1000 of the mean panel delta, panels up). Variants per panel:
  C13 (polish chain cross-fit by day parity, phys4_common.c13_opt), C14 = C13 + phys4 LSA (all-days prm), C15 = C14 + phys4
  stage 3 (cross-fit predictions pred_dens_hub.npz, fit on the other day parity), clip (gen, v <= 1.15 vf_topology) on C13 / C15.
  Contrasts: K1 = C15+clip vs C13 (combined), LSA vs C13, S3 vs C14, clip|C13, clip|C15 (interaction: clip alone vs on top).
own: I-S-OWN March (all-days chain params, as phys4_s3.own), non-block and all targets, same variants and contrasts; plus
  the B6 v2 seed-spread gate (b6v2 gate_own_v1s012.json: gate and C12 blend weights fit on odd days) carried through the
  whole downstream chain (polish -> LSA -> S3 -> clip) and scored on even days: gain of the gate on top of the chain vs alone.
Usage: python docs/reference_code/asm_inter.py opt | own"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phys4_common as C
import phys4_lsa_eval as LE
import phys_lwr_diag as P
import polish_eval as PE
import flow_screen as FS

OUT = P.ROOT / "artifacts/claude_ref/asm"
LSA_PRM = {"wq": 0.5, "cq": 0.015, "wv": 0.3, "cv": 0.02, "G": 0.9, "xgap": 1}


def clipvec(pn, stations):
    topo = pd.read_csv(P.ROOT / f"data/raw/kaggle_public/corridors/{pn}/network/lwr_mainline_topology.csv")
    s = stations.sort_values("idx")
    c = 1.15 * topo.set_index(topo.link_id.astype(str)).free_flow_speed_kmh.reindex(s.link_id.astype(str)).to_numpy(float)
    assert np.isfinite(c).all()
    return c


def clipv(v, T, c):
    return np.where(T & (v > c[None, None, :] + 1e-9), c[None, None, :], v)


def opt():
    D = C.D_OPT; pred = np.load(C.OUT / "pred_dens_hub.npz")
    V = ("C13", "C13c", "C14", "C15", "C15c"); comps = {k: {} for k in V}; t0 = time.time()
    for pn in C.PANELS:
        held, tv, tq, ctx, v13, q13 = C.c13_opt(pn)
        T = np.asarray(held.target, bool); el = np.asarray(held.elig, bool); vf = LE.vf_of(held)
        F = LE.lsa_fields(np.where(el & ~T, tv, np.nan), np.where(el & ~T, tq, np.nan), vf)
        v14, q14 = LE.apply(v13, q13, T, vf, F, LSA_PRM)
        d, t, s = np.load(C.OUT / f"feat_opt_{pn}.npz")["ix"].T.astype(int)
        v15, q15 = v14.copy(), q14.copy()
        q15[d, t, s] = q14[d, t, s] * np.exp(pred[f"q_{pn}"]); v15[d, t, s] = v14[d, t, s] * np.exp(pred[f"v_{pn}"])
        cl = clipvec(pn, held.stations)
        arr = dict(C13=(v13, q13), C13c=(clipv(v13, T, cl), q13), C14=(v14, q14), C15=(v15, q15), C15c=(clipv(v15, T, cl), q15))
        for k, (v, q) in arr.items():
            comps[k][pn] = FS.day_components(held, tv, tq, v, q)
        print(pn, "clip cells C13/C15", int((T & (v13 > cl + 1e-9)).sum()), int((T & (v15 > cl + 1e-9)).sum()), round(time.time() - t0), "s", flush=True)
    res = {}
    for name, (new, ref) in dict(K1_vs_C13=("C15c", "C13"), LSA_vs_C13=("C14", "C13"), S3_vs_C14=("C15", "C14"), phys4_vs_C13=("C15", "C13"),
                                 clip_on_C13=("C13c", "C13"), clip_on_C15=("C15c", "C15")).items():
        summ, per = PE.boot(comps[new], comps[ref], D)
        res[name] = dict(summary=summ, per_panel=per)
        print("OPT", name, {m: (round(x["mean"], 7), [round(c, 7) for c in x["ci"]], x["n_pos"]) for m, x in summ.items()}, flush=True)
    (OUT / "inter_opt.json").write_text(json.dumps(res, indent=1))


def own():
    import flow_own as FO
    import phys4_s3 as S3
    import phys3_proj as PJ, phys3_stack as ST
    import b6full_eval as EV, b6v2_gate as GT
    from tfb import metric
    own_m, binding = FO.own, FO.binding
    p = P.P; mq, mv = S3.models(); panels = p["panels"]
    gj = json.loads((P.ROOT / "artifacts/claude_ref/b6v2/gate_own_v1s012.json").read_text()); G = gj["gate"]
    arm = dict(dir="b6full", pre="", seeds=[0, 1, 2], suf="")
    items = {i["panel"]: i for i in json.loads(Path(p["OWN_metadata_binding"]["manifest"]).read_text())["items"] if i["period"] == "validation"}
    stats = {}
    for pn in panels:  # pass 1: odd-day blend weights (as b6v2_gate.own, FITPAR 1)
        view, tv, tq, st_, hidden = own_m.load_item(P.ROOT, binding.bound_own_item(p, items[pn]))
        st_ = np.asarray(st_, bool); gap = np.asarray(view.gap, bool); tv = np.asarray(tv, float); tq = np.asarray(tq, float)
        r = np.load(EV.OUT / f"ref_own_{pn}.npz"); cv, cq = r["c11_v"].astype(float), r["c11_q"].astype(float)
        iv, iq, _, _ = GT.load_sd(arm, "own", pn)
        la = np.broadcast_to(np.maximum(view.stations.sort_values("idx").lanes.to_numpy(float), 1.), tv.shape)
        day = np.broadcast_to(np.arange(tv.shape[0])[:, None, None], tv.shape)
        m = st_ & np.isfinite(tv) & np.isfinite(tq) & (day % 2 == 1)
        stats[pn] = dict(v_nb=EV.sse_grid(cv, iv, tv, m & ~gap), v_blk=EV.sse_grid(cv, iv, tv, m & gap),
                         q_nb=EV.sse_grid(cq, iq, tq, m & ~gap, la), q_blk=EV.sse_grid(cq, iq, tq, m & gap, la))
        print(pn, "stats", flush=True)
    W, _ = EV.fit_weights(stats)
    rows = {}; info = {}
    for pn in panels:
        view, tv, tq, st_, hidden = own_m.load_item(P.ROOT, binding.bound_own_item(p, items[pn]))
        st_ = np.asarray(st_, bool); gap = np.asarray(view.gap, bool); tv = np.asarray(tv, float); tq = np.asarray(tq, float)
        z = np.load(PE.OUT / f"c12_own_{pn}.npz"); assert (z["target"] == st_).all()
        speed = np.asarray(view.speed, float); flow = np.asarray(view.flow, float)
        observed = np.asarray(view.elig, bool) & ~np.asarray(view.target, bool) & ~gap & np.isfinite(speed) & np.isfinite(flow)
        so, fo = np.where(observed, speed, np.nan), np.where(observed, flow, np.nan)
        prm = metric._parameters(view, P.ROOT)[0]; length = prm.length_km.to_numpy(float); vf = prm.free_speed_kmh.to_numpy(float)
        cl = clipvec(pn, view.stations); hp = z["p"].astype(float)

        def chain(bl_v, bl_q, v12, q12):
            ctx = PE.Ctx(view.stations, st_, so, hp, bl_v, bl_q, gap)
            v13, q13 = PE.apply_chain(C.CHAIN["_all"], ctx, v12, q12)
            (v14, q14), (v15, q15) = S3.stage3(mq, mv, v13, q13, st_, gap, so, fo, hp, view.stations, length, vf)
            return dict(C13=(v13, q13), C13c=(clipv(v13, st_, cl), q13), C14=(v14, q14), C15=(v15, q15), C15c=(clipv(v15, st_, cl), q15))

        arr = chain(z["bl_v"].astype(float), z["bl_q"].astype(float), z["v"].astype(float), z["q"].astype(float))
        arr["C12"] = (z["v"].astype(float), z["q"].astype(float))
        # B6 v2 gate (even days scored): BLP and GATE blends with odd-fit weights / gate, then the full chain
        r = np.load(EV.OUT / f"ref_own_{pn}.npz"); cv, cq = r["c11_v"].astype(float), r["c11_q"].astype(float)
        iv, iq, sv, sq = GT.load_sd(arm, "own", pn)
        la = np.broadcast_to(np.maximum(view.stations.sort_values("idx").lanes.to_numpy(float), 1.), tv.shape)
        bb = EV.blend(cv, cq, iv, iq, gap, W[pn]); bg = GT.gated(cv, cq, iv, iq, sv, sq, la, gap, W[pn], G, np.zeros(tv.shape, int))
        even = np.broadcast_to((np.arange(tv.shape[0]) % 2 == 0)[:, None, None], tv.shape)
        chk = float(np.nanmax(np.abs(np.where(even & st_, bb[0] - z["bl_v"].astype(float), 0.))))
        gv12, gq12, _ = PJ.project(bg[0], bg[1], st_, view.stations, ST.R, ST.A, ST.G, gap=gap)
        bv12, bq12, _ = PJ.project(bb[0], bb[1], st_, view.stations, ST.R, ST.A, ST.G, gap=gap)
        ga = chain(bg[0], bg[1], gv12, gq12); ba = chain(bb[0], bb[1], bv12, bq12)
        arr.update({"G12": (gv12, gq12), "B12": (bv12, bq12), "G15c": ga["C15c"], "B15c": ba["C15c"], "G13": ga["C13"], "B13": ba["C13"]})
        for name, (v, q) in arr.items():
            for part, mm in (("all", st_), ("nonblock", st_ & ~gap)):
                rows.setdefault((name, part), []).extend(own_m.state_rows(view, tv, tq, v, q, mm)[0])
        info[pn] = dict(blend_check_even_maxabs=chk, clip_cells_C13=int((st_ & (arr["C13"][0] > cl + 1e-9)).sum()),
                        clip_cells_C15=int((st_ & (arr["C15"][0] > cl + 1e-9)).sum()))
        print(pn, info[pn], flush=True)
    fam = p["families"]; groups = [[i for i, x in enumerate(panels) if fam[x] == f] for f in sorted(set(fam.values()))]
    dates = sorted({r["date"] for r in rows[("C13", "all")]}); even = [d for i, d in enumerate(dates) if i % 2 == 0]
    res = dict(per_panel=info)
    con = dict(K1_vs_C13=("C15c", "C13", dates), LSA_vs_C13=("C14", "C13", dates), S3_vs_C14=("C15", "C14", dates),
               phys4_vs_C13=("C15", "C13", dates), clip_on_C13=("C13c", "C13", dates), clip_on_C15=("C15c", "C15", dates),
               K1_vs_C12=("C15c", "C12", dates), gate_alone_even=("G12", "B12", even), gate_after_polish_even=("G13", "B13", even),
               gate_full_chain_even=("G15c", "B15c", even))
    for name, (new, ref, dd) in con.items():
        for part in ("nonblock", "all"):
            a_ = [x for x in rows[(ref, part)] if x["date"] in dd]; b_ = [x for x in rows[(new, part)] if x["date"] in dd]
            x = own_m.paired_state(a_, b_, panels, dd, groups)
            bp = {pn: round(own_m.paired_state([r_ for r_ in a_ if r_["panel"] == pn], [r_ for r_ in b_ if r_["panel"] == pn], [pn], dd, [[0]])["delta_S_state"], 7) for pn in panels}
            res[f"{name}_{part}"] = dict(delta=x["delta_S_state"], ci=x["ci95"], verdict=x["verdict"], n_pos=int(sum(v > 0 for v in bp.values())), by_panel=bp)
            print("OWN", name, part, round(x["delta_S_state"], 7), [round(c, 7) for c in x["ci95"]], x["verdict"], "up", res[f"{name}_{part}"]["n_pos"], flush=True)
    (OUT / "inter_own.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    {"opt": opt, "own": own}[sys.argv[1]]()
