"""C knobs (helper PK, 2026-10-11): re-tune of the Task 1 post-processing knobs on J (I-S-OWN) with the current upstream.
Pre-registration, grids and the decision rule: exp/C_knobs/KNOBS.md section 0.

Cached path of c_budget_s2chain.py arm A (form = C_REF = stage 2 3M rows, S08 arm l127_r4800): the c11 arrays written by
the baseline run (c_budget/s2/chain/c11_<stag>_<TAGF>_<pn>.npz) are reused; every stage after c11 is re-run here with the
same calls as the chain script, sharing each stage between knob variants whose upstream settings agree:
  blend (ship-arm cross-parity weights) -> phys3 -> polish chain [D1, D2, E] -> LSA [C1..C4] -> stage 3 L [B1, B2]
  -> clip [F] -> phys5 stage 4 AL+FL [A] -> clip [F] -> final bounds [G1..G3].
Per variant and panel: state5_lib.stats and flow_screen.day_components (non-gap OWN target view) are stored.

Usage (repo root, PYTHONPATH=src):
  python c_knobs.py run <validation|private> [names,comma | all] [panels,comma]   -> artifacts/claude_ref/c_knobs/<stag>/<name>_<pn>.npz
  python c_knobs.py parity <validation|private>   cached path (arm base vs arm ship) vs the baseline chain json, must be exact
  python c_knobs.py score [append]                J(mix) gain of every variant vs base, joint date bootstrap; table + json
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "5")
os.environ["PHYS4_S3"] = "L"
import sys, json, time, types, gc, copy
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import state5_c13apr as C5
import state5_lib as L
import b6full_eval as EV
import phys4_s3 as S3
import phys5_stage4 as S4
import asm_inter as AI
import asm_inter2 as AI2
import flow_screen as FS

P = C5.P
S2DIR = P.ROOT / "artifacts/claude_ref/c_budget/s2"; CH = S2DIR / "chain"
OUT = P.ROOT / "artifacts/claude_ref/c_knobs"
F3M = "all_l63_m2000_n3000000_r600"; ARM = "l127_r4800"
S08ALT = f"artifacts/claude_ref/c_budget/s08/{ARM}"
TAGF = f"{F3M}_vs_{F3M}_s08{ARM}"
NT = int(os.environ.get("KNOB_THREADS", "4"))
STAG = {"validation": ("mar", "own_validation"), "private": ("apr", "own_private")}
SHIP = dict(chain=C5.CHAIN_ALL, lsa=dict(S3.LSA_PRM), s3scale=[1., 1.], s4scale=1., clipf=1.15, vfloor=None, qcap=None, qfloor=None)
EXP = Path(P.ROOT) / "exp/C_knobs"


def grid():
    """name -> (knob id, value, overrides). 'base' = shipped values."""
    V = {"base": ("base", None, {})}
    ch = lambda **kw: copy.deepcopy({**SHIP["chain"], **{k: {**SHIP["chain"][k], **v} for k, v in kw.items()}})
    for x in (.8, .9, 1.1, 1.2): V[f"A_s4_{x}"] = ("A", x, dict(s4scale=x))
    for x in (.9, 1.1): V[f"B1_s3q_{x}"] = ("B1", x, dict(s3scale=[x, 1.]))
    for x in (.8, .9, 1.1): V[f"B2_s3v_{x}"] = ("B2", x, dict(s3scale=[1., x]))
    for x in (.4, .6): V[f"C1_wq_{x}"] = ("C1", x, dict(lsa=dict(SHIP["lsa"], wq=x)))
    for x in (.2, .4): V[f"C2_wv_{x}"] = ("C2", x, dict(lsa=dict(SHIP["lsa"], wv=x)))
    for x in (.01, .02): V[f"C3_cq_{x}"] = ("C3", x, dict(lsa=dict(SHIP["lsa"], cq=x)))
    for x in (.015, .03): V[f"C4_cv_{x}"] = ("C4", x, dict(lsa=dict(SHIP["lsa"], cv=x)))
    for x in (.7, .8, 1.0): V[f"D1_a_{x}"] = ("D1", x, dict(chain=ch(s3=dict(a=x))))
    for x in (1.5, 6.): V[f"D2_r_{x}"] = ("D2", x, dict(chain=ch(s3=dict(r=x))))
    for x in (.002, .006): V[f"E_tau_{x}"] = ("E", x, dict(chain=ch(tv=dict(tau=x))))
    for x in (1.13, 1.14, 1.16, None): V[f"F_clip_{x if x else 'off'}"] = ("F", x if x else "off", dict(clipf=x))
    for x in (5., 8.): V[f"G1_vfloor_{x}"] = ("G1", x, dict(vfloor=x))
    for x in (1.0, 1.1, 1.2): V[f"G2_qcap_{x}"] = ("G2", x, dict(qcap=x))
    for x in (50., 100.): V[f"G3_qfloor_{x}"] = ("G3", x, dict(qfloor=x))
    extra = EXP / "grid_extra.json"   # edge rule / joint configs: {name: [knob id, value, overrides]}
    if extra.exists():
        for n, (kid, val, ov) in json.loads(extra.read_text()).items():
            V[n] = (kid, val, ov)
    return V


def _imp(period):
    return (lambda pn: EV.imp("own", pn, [0, 1, 2])) if period == "validation" else C5.imp_apr


def ship_weights(period):
    """Blend weights of the chain's ship arm (cross parity), exactly as pass 1 of c_budget_s2chain.py."""
    from tfb.data import load_cache
    stag, _ = STAG[period]; f = OUT / f"W_{stag}.json"
    if f.exists():
        j = json.loads(f.read_text()); return {int(k): v for k, v in j.items()}
    imp = _imp(period); stats = {0: {}, 1: {}}
    for item in C5.items(period):
        pn = item["panel"]
        r = dict(np.load(C5.OUT / f"own_{stag}_s08_{pn}.npz")); z = np.load(CH / f"c11_{stag}_{TAGF}_{pn}.npz")
        cv, cq = z["sv"].astype(float), z["sq"].astype(float)
        tv, tq, st_, gap = r["tv"].astype(float), r["tq"].astype(float), r["target"], r["gap"]
        iv, iq, ns = imp(pn)
        stn = load_cache(pn, period).stations
        la = np.broadcast_to(np.maximum(stn.sort_values("idx").lanes.to_numpy(float), 1.), tv.shape)
        day = np.broadcast_to(np.arange(tv.shape[0])[:, None, None], tv.shape)
        ok = st_ & np.isfinite(tv) & np.isfinite(tq)
        for par in (0, 1):
            m = ok & (day % 2 == par)
            stats[par][pn] = dict(v_nb=EV.sse_grid(cv, iv, tv, m & ~gap), v_blk=EV.sse_grid(cv, iv, tv, m & gap),
                                  q_nb=EV.sse_grid(cq, iq, tq, m & ~gap, la), q_blk=EV.sse_grid(cq, iq, tq, m & gap, la))
        print(period, pn, "weights stats", flush=True)
    W = {par: EV.fit_weights(stats[par])[0] for par in (0, 1)}
    OUT.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(W, indent=1))
    return W


def _k(*x):
    return json.dumps(x, sort_keys=True)


def run(period, names="all", panels=None):
    import phys3_proj as PJ, phys3_stack as ST, phys4_lsa_eval as LE, phys4_feat as PF
    import s0445_density_loss as DL
    from tfb import metric
    from tfb.data import load_cache
    stag, grp = STAG[period]; imp = _imp(period)
    _LSA0 = LE.lsa_fields
    V = grid(); V["ship"] = ("ship", None, {})
    names = list(V) if names == "all" else names.split(",")
    dest = OUT / stag; dest.mkdir(parents=True, exist_ok=True)
    W = ship_weights(period)
    _, m368 = C5.load_models(None)
    cj = json.loads((C5.CM.OUT / "opt_coef.json").read_text()); cf = (cj["b_free"], cj["b_cong"])
    mq, mv = S3.models(); m4 = S4.models("AL+FL"); assert m4["yv"] is None
    s3names = json.loads((S3.OUT / "s3_names.json").read_text())
    cfgs = {n: {**SHIP, **V[n][2]} for n in names}
    # order: ship first, then by upstream keys so every stage is computed once per distinct upstream setting
    order = sorted(names, key=lambda n: (n != "ship", _k(cfgs[n]["chain"]) != _k(SHIP["chain"]), _k(cfgs[n]["chain"]),
                                         _k(cfgs[n]["lsa"]) != _k(SHIP["lsa"]), _k(cfgs[n]["lsa"]),
                                         _k(cfgs[n]["s3scale"], cfgs[n]["clipf"]) != _k(SHIP["s3scale"], SHIP["clipf"]), _k(cfgs[n]["s3scale"], cfgs[n]["clipf"])))
    t0 = time.time()
    for pn in (panels.split(",") if panels else L.PANELS):
        todo = [n for n in order if not (dest / f"{n}_{pn}.npz").exists()]
        if not todo:
            continue
        r = dict(np.load(C5.OUT / f"own_{stag}_s08_{pn}.npz")); z = np.load(CH / f"c11_{stag}_{TAGF}_{pn}.npz")
        tv, tq, st_, gap = r["tv"].astype(float), r["tq"].astype(float), r["target"].astype(bool), r["gap"].astype(bool)
        iv, iq, _ = imp(pn)
        cache = load_cache(pn, period); stn = cache.stations
        la, vc, cap, stn2 = L.lanes_of(pn, period)
        odd = (np.arange(tv.shape[0]) % 2 == 1)[:, None, None]
        vo, qo = r["vobs"].astype(float), r["qobs"].astype(float); okk = np.isfinite(vo) & np.isfinite(qo)
        so = np.where(okk, vo, np.nan); fo = np.where(okk, qo, np.nan)
        prm = metric._parameters(cache, L.ROOT)[0]; length = prm.length_km.to_numpy(float); vf = prm.free_speed_kmh.to_numpy(float)
        cl = AI.clipvec(pn, stn2)
        s08 = L.load_s08(grp, pn)
        pS = AI2.p_of(vo, st_, gap, stn2, s08["speed"].astype(float), s08["flow"].astype(float), m368, cf)
        s2 = np.load(P.ROOT / S08ALT / f"own_{period}" / f"{pn}.npz"); assert (s2["target"] == st_).all()
        pN = AI2.p_of(vo, st_, gap, stn2, s2["speed"].astype(float), s2["flow"].astype(float), m368, cf)
        h = types.SimpleNamespace(path=cache.path, meta=cache.meta, stations=cache.stations, ramps=cache.ramps, **cache.arrays); h.target = st_; h.gap = gap; h.regimes = [str(x) for x in r["regimes"]]; h.dates = [str(x) for x in r["dates"]]
        assert list(cache.dates) == h.dates
        hv = DL.proxy_view(h)
        F = _LSA0(so, fo, vf)
        # phys5_stage4.features recomputes lsa_fields(so, fo, vf) on every call (8 s of single-thread convolution, same
        # inputs): serve the panel's F for exactly these objects (read-only use), anything else goes to the original
        LE.lsa_fields = lambda a_, b_, c_, mp=None, _F=F, _o=(so, fo, vf): _F if (a_ is _o[0] and b_ is _o[1] and c_ is _o[2] and mp is None) else _LSA0(a_, b_, c_, mp)
        c13, c14, c15 = [None, None], [None, None], [None, None]
        for n in todo:
            cfg = cfgs[n]; arm = "ship" if n == "ship" else "new"
            p3 = pS if arm == "ship" else pN
            k13 = _k(arm, cfg["chain"])
            if c13[0] != k13:
                cv, cq = (z["sv"].astype(float), z["sq"].astype(float)) if arm == "ship" else (z["nv"].astype(float), z["nq"].astype(float))
                b_even = EV.blend(cv, cq, iv, iq, gap, W[1][pn]); b_odd = EV.blend(cv, cq, iv, iq, gap, W[0][pn])
                bv = np.where(odd, b_odd[0], b_even[0]); bq = np.where(odd, b_odd[1], b_even[1])
                v, q, _ = PJ.project(bv, bq, st_, stn, ST.R, ST.A, ST.G, gap=gap)
                ctx = C5.E.Ctx(stn, st_, vo, (z["p"] if arm == "ship" else z["pnew"]).astype(float), bv, bq, gap)
                v13, q13 = C5.E.apply_chain(cfg["chain"], ctx, v, q)
                c13 = [k13, (v13.astype(np.float32).astype(float), q13.astype(np.float32).astype(float))]; c14 = [None, None]; c15 = [None, None]
            v13, q13 = c13[1]
            k14 = _k(k13, cfg["lsa"])
            if c14[0] != k14:   # as phys4_s3.stage3: LSA nudge, stage 3 features and log-residual predictions
                v14, q14 = LE.apply(v13, q13, st_, vf, F, cfg["lsa"], gap)
                m = st_ & ~gap & np.isfinite(v14) & np.isfinite(q14) & (q14 > 0) & (v14 > 0)
                X, nm3, ix = PF.build(so, fo, st_, v14, q14, p3, stn2, length, m)
                d, t, s = ix.T.astype(int)
                lq = np.log(np.maximum(q14, 1.)); lv = np.log(np.maximum(v14, 1.))
                X = np.concatenate([X, np.stack([(F[0] - lq)[d, t, s], F[1][d, t, s], (F[2] - lv)[d, t, s], F[3][d, t, s]], 1).astype(np.float32)], 1)
                assert nm3 + ["lsa_dq", "lsa_nq", "lsa_dv", "lsa_nv"] == s3names
                c14 = [k14, (v14, q14, (d, t, s), mq.predict(X, num_threads=NT), mv.predict(X, num_threads=NT))]; c15 = [None, None]; del X
            v14, q14, (d, t, s), pq3, pv3 = c14[1]
            clv = None if cfg["clipf"] is None else (cl if cfg["clipf"] == 1.15 else cl / 1.15 * cfg["clipf"])
            k15 = _k(k14, cfg["s3scale"], cfg["clipf"])
            if c15[0] != k15:
                v15, q15 = v14.copy(), q14.copy()
                q15[d, t, s] = q14[d, t, s] * np.exp(cfg["s3scale"][0] * pq3)
                v15[d, t, s] = v14[d, t, s] * np.exp(cfg["s3scale"][1] * pv3)
                if clv is not None:
                    v15 = AI.clipv(v15, st_, clv)
                X4, nm4, ix4 = S4.features(v15, q15, st_, gap, so, fo, stn2, length, vf)
                X4 = X4[:, [nm4.index(x) for x in m4["names"]]]
                c15 = [k15, (v15, q15, tuple(ix4.T.astype(int)), m4["yq"].predict(X4, num_threads=NT))]; del X4
            v15, q15, (d4, t4, s4), p4 = c15[1]
            v16, q16 = v15.copy(), q15.copy()
            q16[d4, t4, s4] = q15[d4, t4, s4] * np.exp(cfg["s4scale"] * p4)
            if clv is not None:
                v16 = AI.clipv(v16, st_, clv)
            if cfg["vfloor"] is not None:
                v16 = np.where(st_ & (v16 < cfg["vfloor"]), cfg["vfloor"], v16)
            if cfg["qcap"] is not None:
                qc = np.broadcast_to(cfg["qcap"] * cap[None, None, :], q16.shape); q16 = np.where(st_ & (q16 > qc), qc, q16)
            if cfg["qfloor"] is not None:
                q16 = np.where(st_ & (q16 < cfg["qfloor"]), cfg["qfloor"], q16)
            comp, reg = FS.day_components(hv, tv, tq, v16, q16)
            tmp = dest / f"tmp_{os.getpid()}_{pn}.npz"   # atomic: a killed worker never leaves a partial result file
            np.savez(tmp, st=L.stats(pn, r, v16, q16, la), comps=comp, regs=np.asarray(reg),
                     moved=np.array([float(np.nanmean(np.abs(v16 - c15[1][0])[st_])), float(np.nanmean(np.abs(q16 - c15[1][1])[st_]))]))
            os.replace(tmp, dest / f"{n}_{pn}.npz")
        print(period, pn, len(todo), "variants", round((time.time() - t0) / 60, 1), "min", flush=True)
        LE.lsa_fields = _LSA0
        del r, z, iv, iq, cache, F, c13, c14, c15; gc.collect()
    print("RUN DONE", period, flush=True)


def _load(stag, name):
    st, comps, regs = {}, {}, {}
    for pn in L.PANELS:
        zz = np.load(OUT / stag / f"{name}_{pn}.npz"); st[pn] = zz["st"]; comps[pn] = zz["comps"]; regs[pn] = [str(x) for x in zz["regs"]]
    return st, comps, regs


def _phys(ca, cr, regs, wt):
    fam = P.P["families"]; fams = sorted(set(fam.values()))
    d = {m: [] for m in ("S_phys", "S_LWR", "S_FD")}
    ag = {p: (FS.aggregate(ca[p], regs[p], wt), FS.aggregate(cr[p], regs[p], wt)) for p in L.PANELS}   # same values as the chain script, one call per panel and arm
    for f_ in fams:
        pp = [p for p in L.PANELS if fam[p] == f_]
        for m in d:
            d[m].append(np.mean([ag[p][0][m] - ag[p][1][m] for p in pp]))
    return {m: float(np.mean(v)) for m, v in d.items()}


def parity(period):
    """Cached path: base (arm A, shipped knobs) vs ship arm, against the baseline chain json (must be exact)."""
    stag, _ = STAG[period]; sh = L.share_array(period)
    j = json.loads((S2DIR / f"chain_{stag}_{TAGF}.json").read_text())["A_vs_ship"]
    sa, ca, regs = _load(stag, "base"); ss, cs, _ = _load(stag, "ship")
    st = L.compare(ss, sa, sh); D = len(next(iter(ca.values())))
    ph = _phys(ca, cs, regs, np.ones(D))
    rng = np.random.default_rng(0); boots = [_phys(ca, cs, regs, np.bincount(rng.choice(D, D), minlength=D).astype(float)) for _ in range(500)]
    ci = [float(np.quantile([b["S_phys"] for b in boots], q)) for q in (.025, .975)]
    Jm = .35 * st["mix"]["delta"] + .15 * ph["S_phys"]; Jn = .35 * st["nb"]["delta"] + .15 * ph["S_phys"]
    diffs = dict(J_mix=abs(Jm - j["J_mix"]), J_nb=abs(Jn - j["J_nb"]),
                 state=max(max(abs(st[p]["delta"] - j["state_all_days"][p]["delta"]), abs(st[p]["ci"][0] - j["state_all_days"][p]["ci"][0]),
                               abs(st[p]["ci"][1] - j["state_all_days"][p]["ci"][1]), abs(st[p]["ref"] - j["state_all_days"][p]["ref"])) for p in ("nb", "mix", "all")),
                 phys=max(abs(ph[m] - j["phys_all"][m]) for m in ("S_phys", "S_LWR", "S_FD")),
                 phys_ci=max(abs(ci[i] - j["phys_all"]["ci"]["S_phys"][i]) for i in (0, 1)))
    ok = all(v == 0 for v in diffs.values())
    line = f"- parity {period}: cached path A vs ship J(mix) {Jm:+.6f} J(nb) {Jn:+.6f} (chain json {j['J_mix']:+.6f} / {j['J_nb']:+.6f}); max |diff| {diffs} -> {'EXACT' if ok else 'MISMATCH'}"
    print(line, flush=True)
    return ok, line


def score(append=False):
    V = grid(); res = {}; W = None
    for period in ("validation", "private"):
        stag, _ = STAG[period]; sh = L.share_array(period)
        if not all((OUT / stag / f"base_{pn}.npz").exists() for pn in L.PANELS):
            continue
        sb, cb, regs = _load(stag, "base"); D = len(next(iter(cb.values())))
        A = np.stack([sb[p] for p in L.PANELS])
        rng = np.random.default_rng(4701); W = [np.bincount(rng.integers(0, D, D), minlength=D).astype(float) for _ in range(1000)]
        one = np.ones(D)
        for n, (kid, val, ov) in V.items():
            if n == "base" or not all((OUT / stag / f"{n}_{pn}.npz").exists() for pn in L.PANELS):
                continue
            sn, cn, _ = _load(stag, n); B = np.stack([sn[p] for p in L.PANELS])
            st = L.compare(sb, sn, sh, pops=("nb", "mix"), nboot=1)
            dmix = L.reduce(B, one, "mix", sh) - L.reduce(A, one, "mix", sh); ph = _phys(cn, cb, regs, one)
            J = .35 * dmix + .15 * ph["S_phys"]
            bs = np.array([[L.reduce(B, w, "mix", sh) - L.reduce(A, w, "mix", sh), _phys(cn, cb, regs, w)["S_phys"]] for w in W])
            Jb = .35 * bs[:, 0] + .15 * bs[:, 1]
            q = lambda x: [float(np.quantile(x, .025)), float(np.quantile(x, .975))]
            hal = []
            for par in (0, 1):
                w = (np.arange(D) % 2 == par).astype(float)
                hal.append(.35 * (L.reduce(B, w, "mix", sh) - L.reduce(A, w, "mix", sh)) + .15 * _phys(cn, cb, regs, w)["S_phys"])
            res.setdefault(n, dict(knob=kid, value=val))[stag] = dict(J=float(J), J_ci=q(Jb), state_mix=float(dmix), state_mix_ci=q(bs[:, 0]), state_nb=st["nb"]["delta"],
                                                                     phys=ph["S_phys"], phys_ci=q(bs[:, 1]), LWR=ph["S_LWR"], FD=ph["S_FD"], halves=[float(x) for x in hal])
            print(stag, n, "J %+.7f [%+.7f, %+.7f] state %+.7f phys %+.7f" % (J, *q(Jb), dmix, ph["S_phys"]), flush=True)
    (OUT / "score.json").write_text(json.dumps(res, indent=1))
    f6 = lambda x: "%+.6f" % x
    rows = ["| knob | value | March J(mix) [CI] | March state mix / phys | April J(mix) [CI] | April state mix / phys | March CI>0 | April CI>0 |", "|---|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        c = [r["knob"], str(r["value"])]
        for stag in ("mar", "apr"):
            x = r.get(stag)
            c += [f"{f6(x['J'])} [{f6(x['J_ci'][0])}, {f6(x['J_ci'][1])}]", f"{f6(x['state_mix'])} / {f6(x['phys'])}"] if x else ["-", "-"]
        c += [("yes" if r[s]["J"] > 0 and r[s]["J_ci"][0] > 0 else "no") if s in r else "-" for s in ("mar", "apr")]
        rows.append("| " + " | ".join(c) + " |")
    txt = "\n".join(rows); print(txt)
    if append:
        with open(EXP / "KNOBS.md", "a", encoding="utf-8", newline="\n") as fh:
            fh.write(f"\n### Score table ({time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())}; J(mix) gain vs the shipped value, joint date bootstrap 1000)\n\n" + txt + "\n")
    return res


def probe(state_parquet, name):
    """Laptop probe on C43 (pattern of c_s08_ship.probe): PARENT state rows must equal the shipped-knob control build
    (official/l127_r4800/t1.csv.state.parquet, |d| <= 1e-6); full = PARENT with the new state rows; tools/probe.py make + diff."""
    import pandas as pd, subprocess
    control = P.ROOT / "artifacts/claude_ref/c_budget/s08/official/l127_r4800/t1.csv.state.parquet"
    parent = P.ROOT / "artifacts/claude_ref/probes/c36_s08l127r4800_on_c36.csv"
    new = pd.read_parquet(state_parquet); ctl = pd.read_parquet(control)
    assert new.submission_id.equals(ctl.submission_id)
    par = pd.read_csv(parent, float_precision="round_trip")
    pos = pd.Index(par.submission_id).get_indexer(ctl.submission_id); assert (pos >= 0).all() and (par.task.iloc[pos] == "state").all()
    assert (par.task == "state").sum() == len(ctl)
    d = max(float(np.max(np.abs(par.speed_kmh.to_numpy()[pos] - ctl.speed_kmh.to_numpy()))), float(np.max(np.abs(par.flow_vph.to_numpy()[pos] - ctl.flow_vph.to_numpy()))))
    print("control vs parent (C43) state rows max|d|", d, flush=True); assert d <= 1e-6, "parent state rows are not the shipped-knob control build"
    dv = np.abs(new.speed_kmh.to_numpy() - ctl.speed_kmh.to_numpy()); dq = np.abs(new.flow_vph.to_numpy() - ctl.flow_vph.to_numpy())
    print("new vs control: rows", len(new), "speed changed", int((dv > 1e-9).sum()), "flow changed", int((dq > 1e-9).sum()), "mean|dv|", float(dv.mean()), "max|dv|", float(dv.max()),
          "mean|dq|", float(dq.mean()), "max|dq|", float(dq.max()), "mean dq", float((new.flow_vph.to_numpy() - ctl.flow_vph.to_numpy()).mean()), flush=True)
    before = {c: par[c].to_numpy().copy() for c in ("queue_pred", "path_flow")}; other = (par.task != "state").to_numpy()
    par.loc[pos, "speed_kmh"] = new.speed_kmh.to_numpy(); par.loc[pos, "flow_vph"] = new.flow_vph.to_numpy()
    for c in before:
        assert np.array_equal(par[c].to_numpy(), before[c], equal_nan=True)
    assert np.isfinite(par.speed_kmh.to_numpy()[pos]).all() and np.isfinite(par.flow_vph.to_numpy()[pos]).all() and other.sum() + len(pos) == len(par)
    probes = P.ROOT / "artifacts/claude_ref/probes"
    eol = "\r\n" if open(parent, "rb").readline().endswith(b"\r\n") else "\n"
    full = probes / f"{name}.csv"; assert not full.exists(), full
    par.to_csv(full, index=False, lineterminator=eol)
    subprocess.run([sys.executable, str(P.ROOT / "tools/probe.py"), "make", str(parent), str(full), str(probes / f"{name}.delta.csv.gz")], check=True)
    subprocess.run([sys.executable, str(P.ROOT / "tools/probe.py"), "diff", str(parent), str(full)], check=True)
    print("PROBE DONE", full, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "probe":
        probe(a[1], a[2]); sys.exit(0)
    if a[0] == "run":
        run(a[1], *(a[2:4]))
    elif a[0] == "parity":
        ok, line = parity(a[1])
        if len(a) > 2:
            with open(EXP / "KNOBS.md", "a", encoding="utf-8", newline="\n") as fh:
                fh.write(line + "\n")
        sys.exit(0 if ok else 2)
    elif a[0] == "score":
        score(len(a) > 1)
