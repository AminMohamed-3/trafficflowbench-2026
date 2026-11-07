"""Claude helper phys4 (2026-10-04): density-targeted stage 3 on top of C13 (OPT feasibility + full test).
Label: log k_true - log k_C13 on target cells; LightGBM L1 (median) with weight L * k_C13 (= vehicle count, the S_LWR
boundary cost of a singleton target is 2 L |k_hat - k|). Pooled over 10 panels; cross-fit by day parity (fit on the other
parity). Apply: dk = s * pred; log q += beta dk, log v -= (1 - beta) dk (beta by class free/cong of C13 v vs v_cut).
Usage: python docs/reference_code/phys4_dens.py build | eval [--tag T]"""
import sys, json, time, argparse
import numpy as np
import phys4_common as C
import phys4_feat as PF
import phys_lwr_diag as P
import polish_eval as PE
import flow_screen as FS

OUT = C.OUT
PRM = dict(objective="l1", learning_rate=.05, num_leaves=31, min_data_in_leaf=500, feature_fraction=.8, bagging_fraction=.8,
           bagging_freq=1, lambda_l2=1., num_threads=3, verbose=-1, seed=0)
ROUNDS = 400


BASE_PRM = None  # LSA nudge applied to C13 before the stage (json via --base)


def base_opt(pn):
    import phys4_lsa_eval as LE
    held, tv, tq, ctx, v, q = C.c13_opt(pn)
    T = np.asarray(held.target, bool); el = np.asarray(held.elig, bool); vf = LE.vf_of(held)
    F = LE.lsa_fields(np.where(el & ~T, tv, np.nan), np.where(el & ~T, tq, np.nan), vf)
    if BASE_PRM:
        v, q = LE.apply(v, q, T, vf, F, BASE_PRM)
    return held, tv, tq, ctx, v, q, F


def build():
    for pn in C.PANELS:
        held, tv, tq, ctx, v, q, F = base_opt(pn)
        T = np.asarray(held.target, bool); c = C.consts(held)
        el = np.asarray(held.elig, bool)
        ov = np.where(el & ~T & np.isfinite(tv), tv, np.nan); oq = np.where(el & ~T & np.isfinite(tq), tq, np.nan)
        m = T & np.isfinite(tv) & np.isfinite(tq) & (tq > 0) & (tv > 0)
        X, names, ix = PF.build(ov, oq, T, v, q, ctx.p, held.stations, c["length"], m)
        d, t, s = ix.T
        lq = np.log(np.maximum(q, 1.)); lv = np.log(np.maximum(v, 1.))
        X = np.concatenate([X, np.stack([(F[0] - lq)[d, t, s], F[1][d, t, s], (F[2] - lv)[d, t, s], F[3][d, t, s]], 1).astype(np.float32)], 1)
        names = names + ["lsa_dq", "lsa_nq", "lsa_dv", "lsa_nv"]
        yk = (np.log(tq / tv) - np.log(q / np.maximum(v, 1.)))[d, t, s]
        yq = (np.log(tq) - np.log(q))[d, t, s]; yv = (np.log(tv) - np.log(np.maximum(v, 1.)))[d, t, s]
        w = (c["length"][None, None, :] * q / np.maximum(v, 1.))[d, t, s]
        np.savez_compressed(OUT / f"feat_opt_{pn}.npz", X=X, ix=ix, yk=yk.astype(np.float32), yq=yq.astype(np.float32),
                            yv=yv.astype(np.float32), w=w.astype(np.float32), names=np.array(names))
        print(pn, X.shape, flush=True)


def load():
    return {pn: dict(np.load(OUT / f"feat_opt_{pn}.npz")) for pn in C.PANELS}


PRM2 = dict(objective="huber", learning_rate=.1, num_leaves=31, min_data_in_leaf=1000, feature_fraction=.8, bagging_fraction=.7,
            bagging_freq=1, lambda_l2=1., num_threads=3, verbose=-1, seed=0)
ROUNDS2 = 150
ALPHA = dict(yq=.03, yv=.01, yk=.03)
NSUB = 1_500_000


def crossfit(Z, label):
    import lightgbm as lgb
    pred = {pn: np.zeros(len(Z[pn][label]), np.float32) for pn in Z}
    rng = np.random.default_rng(0)
    for f in (0, 1):
        sel = {pn: np.flatnonzero(C.PAR[Z[pn]["ix"][:, 0]] != f) for pn in Z}
        tot = sum(len(x) for x in sel.values()); keep = min(1., NSUB / tot)
        sel = {pn: x[rng.random(len(x)) < keep] for pn, x in sel.items()}
        Xs = np.concatenate([Z[pn]["X"][sel[pn]] for pn in Z]); ys = np.concatenate([Z[pn][label][sel[pn]] for pn in Z])
        t0 = time.time()
        m = lgb.train(dict(PRM2, alpha=ALPHA[label]), lgb.Dataset(Xs, ys, free_raw_data=True), ROUNDS2)
        del Xs, ys
        print(label, "fold", f, "fit", round(time.time() - t0), "s", flush=True)
        for pn in Z:
            o = C.PAR[Z[pn]["ix"][:, 0]] == f
            pred[pn][o] = m.predict(Z[pn]["X"][o], num_threads=3)
    return pred


def evaluate(Z, pq_, pv_, variants, tag):
    D = C.D_OPT; cref = {}; cnew = {k: {} for k in variants}
    for pn in C.PANELS:
        held, tv, tq, ctx, v, q, _ = base_opt(pn)
        cref[pn] = FS.day_components(held, tv, tq, v, q)
        d, t, s = Z[pn]["ix"].T.astype(int)
        for k, (sq, sv) in variants.items():
            nv, nq = v.copy(), q.copy()
            nq[d, t, s] = q[d, t, s] * np.exp(sq * pq_[pn]); nv[d, t, s] = v[d, t, s] * np.exp(sv * pv_[pn])
            cnew[k][pn] = FS.day_components(held, tv, tq, nv, nq)
        print(pn, "eval", flush=True)
    res = {}
    for k in variants:
        summ, per = PE.boot(cnew[k], cref, D)
        res[k] = dict(summary=summ, per_panel=per)
        print(tag, k, {m: (round(x["mean"], 6), [round(c, 6) for c in x["ci"]], x["n_pos"]) for m, x in summ.items()}, flush=True)
    (OUT / f"eval_dens_{tag}.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    global BASE_PRM
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--tag", default="base"); ap.add_argument("--base", default=None)
    a = ap.parse_args()
    BASE_PRM = json.loads(a.base) if a.base else None
    if a.cmd == "build":
        build(); return
    Z = load()
    pq_ = crossfit(Z, "yq"); pv_ = crossfit(Z, "yv")
    np.savez_compressed(OUT / f"pred_dens_{a.tag}.npz", **{f"q_{pn}": pq_[pn] for pn in pq_}, **{f"v_{pn}": pv_[pn] for pn in pv_})
    for pn in Z:
        y = Z[pn]; print(pn, "rmse yq", round(float(np.sqrt(np.mean(y["yq"] ** 2))), 5), "->", round(float(np.sqrt(np.mean((y["yq"] - pq_[pn]) ** 2))), 5),
                         "yv", round(float(np.sqrt(np.mean(y["yv"] ** 2))), 5), "->", round(float(np.sqrt(np.mean((y["yv"] - pv_[pn]) ** 2))), 5), flush=True)
    variants = {"q1v1": (1., 1.), "q.5v.5": (.5, .5), "q1v0": (1., 0.), "q0v1": (0., 1.)}
    evaluate(Z, pq_, pv_, variants, a.tag)


if __name__ == "__main__":
    main()
