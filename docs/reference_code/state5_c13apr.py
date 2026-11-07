"""Claude helper 2026-10-04 (state5): rebuild the C13 Task 1 stack on I-S-OWN April (private), the same way the March OWN
reference was built (b6full_ref.own -> polish_ref.own -> polish chain), so state ideas get an April check.
Stage c11: F4 (state5_f4own private, parity-checked parent arm) -> 0360 -> 0368 -> 0358 -> phys3 -> 0389 stage 2 VQP (models fit
on all OPT, unchanged). Stage c13: blend with the April OWN imputer (state5_imp_apr, seeds 0,1,2; b6full weights fit on the
other day parity, as polish_ref.own) -> phys3 -> C13 polish chain (all-days params). Writes state5/own_apr_<panel>.npz with the
same keys as state5_cache own (tv, tq, target, gap, v13, q13, f4v, f4q, vobs, qobs, dates, regimes) plus c11 arrays.
Usage: python docs/reference_code/state5_c13apr.py c11 | c13 [f4 | s08] [apr | mar]   (s08: S08 replaces F4 as the stack input)"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phys_lwr_diag as P
import flow_krige as FK
import state2_cm as CM
import phys3_proj as PJ
import phys3_stack as ST
import s2stage as S2
import b6full_eval as EV
import polish_eval as E

OUT = P.ROOT / "artifacts/claude_ref/state5"
S2.PRM["num_threads"] = 3
CHAIN_ALL = json.loads((E.OUT / "eval_combo.json").read_text())["base"]["_all"]


def items(period):
    import flow_own as FO
    return [i for i in json.loads(Path(P.P["OWN_metadata_binding"]["manifest"]).read_text())["items"] if i["period"] == period]


def c11_arrays(view, st_, gap, pv, pq, models, m368, w, cf):
    speed = np.asarray(view.speed, float); flow = np.asarray(view.flow, float)
    observed = np.asarray(view.elig, bool) & ~np.asarray(view.target, bool) & ~gap & np.isfinite(speed)
    gr, ncm = CM.common_factor(speed, observed, view.stations)
    vcut = view.stations.sort_values("idx").v_cut.to_numpy(float)
    rv = np.where(st_, CM.apply(pv, gr, vcut, cf), pv)
    N, den = FK.krige_log_flow(flow, FK.positions(view.stations))
    hv, hp = S2.hedge_368(m368, speed, observed, view.stations, rv, pq, st_)
    hq = FK.blend(hv, pq, N, st_, view.stations, w, den=den)
    sv, sq, _ = PJ.project(hv, hq, st_, view.stations, ST.R, ST.A, ST.G, gap=gap)
    cv, cq = S2.apply_stage2(models, (speed, flow, np.asarray(view.elig, bool), np.asarray(view.target, bool) | st_, gap), view.stations,
                             pv, pq, rv, gr, ncm, N, den, hp, hv, hq, sv, sq, st_, "VQP")
    return cv, cq, hp, observed


def load_models(base):
    """Stage-2 and 0368 models: shipped (fit on F4-based OPT) or, for base 's08r', refit on S08-based OPT (state5_refit)."""
    if base == "s08r":
        import lightgbm as lgb
        return (tuple(lgb.Booster(model_file=str(OUT / f"m2s08_{k}.txt")) for k in ("v", "q")),
                tuple(lgb.Booster(model_file=str(OUT / f"m368s08_{n}.txt")) for n in ("clf", "gc", "gf")))
    return S2.load_all(), ST.load_368()


def c11(period="private", tag="apr", base=None):
    import flow_own as FO
    own_m, binding = FO.own, FO.binding
    models, m368 = load_models(base)
    w = np.asarray(json.loads((P.ROOT / "artifacts/claude_ref/flow/opt/opt_summary.json").read_text())["pooled_weights"])
    cj = json.loads((CM.OUT / "opt_coef.json").read_text()); cf = (cj["b_free"], cj["b_cong"])
    for item in items(period):
        pn = item["panel"]
        f = OUT / (f"c11_{tag}_{pn}.npz" if base is None else f"c11_{tag}_{base}_{pn}.npz")
        if f.exists():
            print("skip", f.name, flush=True); continue
        view, tv, tq, st_, hidden = own_m.load_item(P.ROOT, binding.bound_own_item(P.P, item))
        st_ = np.asarray(st_, bool); gap = np.asarray(view.gap, bool)
        a = np.load(OUT / f"f4own_{period}_{pn}.npz"); assert (a["target"] == st_).all()
        f4v, f4q = a["speed"].astype(float), a["flow"].astype(float)
        if base is None:
            pv, pq = f4v, f4q
        else:
            s = np.load(P.ROOT / "artifacts/lane_s/0398_s08" / f"own_{period}" / f"{pn}.npz"); assert (s["target"] == st_).all()
            pv = np.where(st_, s["speed"].astype(float), f4v); pq = np.where(st_, s["flow"].astype(float), f4q)
        cv, cq, hp, observed = c11_arrays(view, st_, gap, pv, pq, models, m368, w, cf)
        speed = np.asarray(view.speed, float); flow = np.asarray(view.flow, float)
        np.savez_compressed(f, c11_v=cv.astype(np.float32), c11_q=cq.astype(np.float32), p=hp.astype(np.float32),
                            tv=np.asarray(tv, np.float32), tq=np.asarray(tq, np.float32), target=st_, gap=gap,
                            f4v=f4v.astype(np.float32), f4q=f4q.astype(np.float32),
                            vobs=np.where(observed, speed, np.nan).astype(np.float32), qobs=np.where(observed, flow, np.nan).astype(np.float32),
                            dates=np.asarray(view.dates), regimes=np.asarray(view.regimes))
        ok = st_ & ~gap & np.isfinite(tv)
        print(pn, "rmse_v nb F4/C11", round(float(np.sqrt(np.mean((f4v - tv)[ok] ** 2))), 4),
              round(float(np.sqrt(np.mean((cv - tv)[ok] ** 2))), 4), flush=True)


def imp_apr(pn, seeds=(0, 1, 2)):
    zs = [np.load(OUT / f"imp_ownapr_{pn}_s{s}.npz") for s in seeds if (OUT / f"imp_ownapr_{pn}_s{s}.npz").exists()]
    assert zs, pn
    return np.mean([z["speed"] for z in zs], 0).astype(float), np.mean([z["flow"] for z in zs], 0).astype(float), len(zs)


def c13(tag="apr", base=None, imp=None, period="private"):
    """imp: function pn -> (iv, iq, n) (default April OWN imputer); writes own_<tag>[_<base>]_<pn>.npz."""
    imp = imp or imp_apr
    import sys as _s
    _s.path.insert(0, str(P.ROOT / "src"))
    from tfb.data import load_cache
    name = lambda pn: OUT / (f"c11_{tag}_{pn}.npz" if base is None else f"c11_{tag}_{base}_{pn}.npz")
    stats = {0: {}, 1: {}}; cache = {}
    for pn in P.P["panels"]:
        r = dict(np.load(name(pn)))
        tv, tq, st_, gap = r["tv"].astype(float), r["tq"].astype(float), r["target"], r["gap"]
        cv, cq = r["c11_v"].astype(float), r["c11_q"].astype(float)
        iv, iq, ns = imp(pn)
        stn = load_cache(pn, period).stations
        la = np.broadcast_to(np.maximum(stn.sort_values("idx").lanes.to_numpy(float), 1.), tv.shape)
        day = np.broadcast_to(np.arange(tv.shape[0])[:, None, None], tv.shape)
        ok = st_ & np.isfinite(tv) & np.isfinite(tq)
        for par in (0, 1):
            m = ok & (day % 2 == par)
            stats[par][pn] = dict(v_nb=EV.sse_grid(cv, iv, tv, m & ~gap), v_blk=EV.sse_grid(cv, iv, tv, m & gap),
                                  q_nb=EV.sse_grid(cq, iq, tq, m & ~gap, la), q_blk=EV.sse_grid(cq, iq, tq, m & gap, la))
        print(pn, "stats; imputer seeds", ns, flush=True)
    W1, _ = EV.fit_weights(stats[1]); W0, _ = EV.fit_weights(stats[0])
    for pn in P.P["panels"]:
        r = dict(np.load(name(pn)))
        tv, st_, gap = r["tv"].astype(float), r["target"], r["gap"]
        cv, cq = r["c11_v"].astype(float), r["c11_q"].astype(float)
        iv, iq, _ = imp(pn)
        stn = load_cache(pn, period).stations
        odd = (np.arange(tv.shape[0]) % 2 == 1)[:, None, None]
        b_even = EV.blend(cv, cq, iv, iq, gap, W1[pn]); b_odd = EV.blend(cv, cq, iv, iq, gap, W0[pn])
        bv = np.where(odd, b_odd[0], b_even[0]); bq = np.where(odd, b_odd[1], b_even[1])
        v, q, _ = PJ.project(bv, bq, st_, stn, ST.R, ST.A, ST.G, gap=gap)
        ctx = E.Ctx(stn, st_, r["vobs"].astype(float), r["p"].astype(float), bv, bq, gap)
        v13, q13 = E.apply_chain(CHAIN_ALL, ctx, v, q)
        out = {k: r[k] for k in ("tv", "tq", "target", "gap", "f4v", "f4q", "vobs", "qobs", "dates", "regimes")}
        out.update(v13=v13.astype(np.float32), q13=q13.astype(np.float32), c11_v=r["c11_v"], c11_q=r["c11_q"],
                   v12=v.astype(np.float32), q12=q.astype(np.float32), bl_v=bv.astype(np.float32), bl_q=bq.astype(np.float32))
        np.savez_compressed(OUT / (f"own_{tag}_{pn}.npz" if base is None else f"own_{tag}_{base}_{pn}.npz"), **out)
        ok = st_ & ~gap & np.isfinite(tv)
        print(pn, "rmse_v nb F4/C11/C13", *[round(float(np.sqrt(np.mean((a - tv)[ok] ** 2))), 4) for a in (r["f4v"], cv, v13)],
              "w", {k: round(x, 2) for k, x in W1[pn].items()}, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    base = a[1] if len(a) > 1 else None
    mar = len(a) > 2 and a[2] == "mar"
    kw = dict(period="validation", tag="mar") if mar else {}
    if a[0] == "c11":
        c11(base=None if base == "f4" else base, **kw)
    else:
        imp = (lambda pn: EV.imp("own", pn, [0, 1, 2])) if mar else None
        c13(base=None if base == "f4" else base, imp=imp, **kw)
