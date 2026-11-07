"""phys5 (C helper, 2026-10-05): stage-3 arms on C14 OPT (cross-fit by day parity, pooled 10 panels), to test NEW feature
groups against the phys4 feature set. Arm A = phys4 features (+LSA diffs) = phys4 stage 3; arm B = A + demand-step block
features (travel-time aligned phase in the half-hour block, own-station observed block means of log q for blocks b-1, b,
b+1 relative to C14, corridor-wide block steps). Same LightGBM params per arm (size S: 150 rounds lr .1 31 leaves
1.5M rows; L: 600 rounds lr .05 127 leaves 3M rows = phys4 S3L).
Usage: python phys5_s3.py build [panels] | fit ARM [S|L] | eval TAGS... [--ref TAG]"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import phys4_common as C
import phys4_feat as PF
import phys4_lsa_eval as LE
import phys5_diag as DG
import polish_eval as PE
import flow_screen as FS
import gen_blk as B

OUT = DG.OUT
PRM = dict(objective="huber", learning_rate=.1, num_leaves=31, min_data_in_leaf=1000, feature_fraction=.8, bagging_fraction=.7,
           bagging_freq=1, lambda_l2=1., num_threads=3, verbose=-1, seed=0)
ALPHA = dict(yq=float(os.environ.get("PHYS5_AQ", ".03")), yv=float(os.environ.get("PHYS5_AV", ".01")))
ASFX = "" if (ALPHA["yq"], ALPHA["yv"]) == (.03, .01) else f"a{ALPHA['yq']}_{ALPHA['yv']}"
SIZE = dict(S=dict(rounds=150, lr=.1, leaves=31, nsub=1_500_000), L=dict(rounds=600, lr=.05, leaves=127, nsub=3_000_000))
WEIGHT = float(os.environ.get("PHYS5_W", "0"))
LABS = os.environ.get("PHYS5_LABS", "yq,yv").split(",")
STEP = ["bm_prev", "bm_prev_n", "bm_cur", "bm_cur_n", "bm_next", "bm_next_n", "cstep_next", "cstep_prev", "phase"]


def step_feats(T, el, tv, tq, mp, vf, lq):
    """Demand-step block features, dict of arrays broadcastable to [D, T, S]."""
    D, Tn, S = T.shape
    obs = el & ~T & np.isfinite(tv) & np.isfinite(tq) & (tq > 0)
    x = np.where(obs & (tv > .9 * vf), np.log(np.maximum(np.nan_to_num(tq), 1)), np.nan)
    blk, pos = B.block_index(Tn, mp, vf.mean(), .5)
    b0 = blk.min(); nb = blk.max() - b0 + 1; bi = blk - b0 + 1           # padded: blocks 1..nb, 0 and nb+1 empty
    clean = (pos > .5) & (pos < 5.5)
    sums = np.zeros((D, nb + 2, S)); cnts = np.zeros((D, nb + 2, S))
    for t in range(Tn):
        ok = np.isfinite(x[:, t]) & clean[t][None]
        sums[:, bi[t], np.arange(S)] += np.where(ok, x[:, t], 0); cnts[:, bi[t], np.arange(S)] += ok
    M = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    with np.errstate(all="ignore"):
        cstep = np.nanmedian(M[:, 1:] - M[:, :-1], axis=2)              # [D, nb+1]; j = step from block j to j+1 (padded idx)
    out = {}
    Sx = np.arange(S)[None, :]
    for k, nm in ((-1, "bm_prev"), (0, "bm_cur"), (1, "bm_next")):
        j = np.clip(bi + k, 0, nb + 1)                                   # [T, S]
        out[nm] = M[:, j, Sx] - lq; out[nm + "_n"] = cnts[:, j, Sx]
    out["cstep_next"] = cstep[:, np.clip(bi, 0, nb)]                     # [D, T, S]
    out["cstep_prev"] = cstep[:, np.clip(bi - 1, 0, nb)]
    out["phase"] = np.broadcast_to(pos[None], T.shape)
    return out


BASE = os.environ.get("PHYS5_BASE", "")   # stage-4 mode: features/labels relative to C14 x exp(pred_BASE), p dropped
PFX = f"feat_{BASE}_" if BASE else "feat_"


def build(panels):
    for pn in panels:
        f = OUT / f"{PFX}{pn}.npz"
        if f.exists():
            continue
        t0 = time.time()
        held, tv, tq, v, q, p = DG.load(pn)
        if BASE:
            v, q = apply(pn, BASE, v, q, np.load(OUT / f"feat_{pn}.npz")["ix"]); p = np.full_like(p, np.nan)
        T = np.asarray(held.target, bool); el = np.asarray(held.elig, bool); c = C.consts(held); vf = c["vf"]
        st = held.stations.sort_values("idx"); mp = st.milepost.to_numpy(float)
        ov = np.where(el & ~T & np.isfinite(tv), tv, np.nan); oq = np.where(el & ~T & np.isfinite(tq), tq, np.nan)
        F = LE.lsa_fields(ov, oq, vf)
        m = T & np.isfinite(tv) & np.isfinite(tq) & (tq > 0) & (tv > 0)
        X, names, ix = PF.build(ov, oq, T, v, q, p, held.stations, c["length"], m)
        d, t, s = ix.T
        lq = np.log(np.maximum(q, 1.)); lv = np.log(np.maximum(v, 1.))
        X = np.concatenate([X, np.stack([(F[0] - lq)[d, t, s], F[1][d, t, s], (F[2] - lv)[d, t, s], F[3][d, t, s]], 1).astype(np.float32)], 1)
        names = names + ["lsa_dq", "lsa_nq", "lsa_dv", "lsa_nv"]
        del F
        SF = step_feats(T, el, tv, tq, mp, vf, lq)
        cols = [np.asarray(np.broadcast_to(SF[k], T.shape))[d, t, s].astype(np.float32) for k in STEP]
        X = np.concatenate([X, np.stack(cols, 1)], 1); names = names + STEP
        yq = (np.log(tq) - lq)[d, t, s]; yv = (np.log(tv) - lv)[d, t, s]
        np.savez(f, X=X, ix=ix, yq=yq.astype(np.float32), yv=yv.astype(np.float32), names=np.array(names))
        print(pn, X.shape, round(time.time() - t0), "s", flush=True)


def cols_of(names, arm):
    """names = feat names + feat2 names (if loaded). A base, B +step, C +profile, D +step +profile."""
    import phys5_feat2 as F2
    names = [str(n) for n in names]
    import phys5_feat3 as F3
    N4 = ["phase_adj", "delta_s", "ramp_ex"]
    drop = {"A": STEP + F2.NAMES2 + F3.NAMES3, "B": F2.NAMES2 + F3.NAMES3, "C": STEP + F3.NAMES3, "D": F3.NAMES3, "E": F2.NAMES2,
            "F": F2.NAMES2 + F3.NAMES3}[arm]
    if arm != "F":
        drop = drop + N4
    return [i for i, n in enumerate(names) if n not in drop]


def load_Z(arm):
    Z = {}
    for pn in C.PANELS:
        z = dict(np.load(OUT / f"{PFX}{pn}.npz")); names = [str(n) for n in z["names"]]
        if arm in ("C", "D"):
            z2 = np.load(OUT / f"feat2_{pn}.npz"); z["X"] = np.concatenate([z["X"], z2["X"]], 1); names += [str(n) for n in z2["names"]]
        if arm in ("E",):
            z3 = np.load(OUT / f"feat3_{pn}.npz"); z["X"] = np.concatenate([z["X"], z3["X"]], 1); names += [str(n) for n in z3["names"]]
        if arm in ("F",):
            z4 = np.load(OUT / (f"feat4_{BASE}_{pn}.npz" if BASE else f"feat4_{pn}.npz")); z["X"] = np.concatenate([z["X"], z4["X"]], 1); names += [str(n) for n in z4["names"]]
        z["names"] = names; Z[pn] = z
    return Z


def fit(arm, size):
    import lightgbm as lgb
    Z = load_Z(arm)
    names = Z[C.PANELS[0]]["names"]; ci = cols_of(names, arm); sz = SIZE[size]
    prm = dict(PRM, learning_rate=sz["lr"], num_leaves=sz["leaves"])
    rng = np.random.default_rng(0)
    ixs = {pn: Z[pn]["ix"] for pn in C.PANELS}
    preds = {}
    for lab in LABS:
        pr = {pn: np.zeros(len(ixs[pn]), np.float32) for pn in C.PANELS}
        for f in (0, 1):
            sel = {pn: np.flatnonzero(C.PAR[ixs[pn][:, 0]] != f) for pn in C.PANELS}
            tot = sum(len(x) for x in sel.values()); keep = min(1., sz["nsub"] / tot)
            sel = {pn: x[rng.random(len(x)) < keep] for pn, x in sel.items()}
            Xs = np.concatenate([Z[pn]["X"][sel[pn]][:, ci] for pn in C.PANELS]); ys = np.concatenate([Z[pn][lab][sel[pn]] for pn in C.PANELS])
            ws = None
            if WEIGHT:   # J-relevance weight: flow ~ q/cap (state error is absolute vph), speed ~ v/v_cut
                wc = names.index("q_cap" if lab == "yq" else "v_cut")
                ws = np.concatenate([Z[pn]["X"][sel[pn], wc] for pn in C.PANELS]).astype(float); ws = np.clip(ws, .05, None) ** WEIGHT; ws /= ws.mean()
            t0 = time.time()
            mdl = lgb.train(dict(prm, alpha=ALPHA[lab]), lgb.Dataset(Xs, ys, weight=ws, free_raw_data=True), sz["rounds"])
            del Xs, ys
            for pn in C.PANELS:
                o = C.PAR[ixs[pn][:, 0]] == f
                pr[pn][o] = mdl.predict(Z[pn]["X"][o][:, ci], num_threads=3)
            if f == 0:
                imp = sorted(zip(mdl.feature_importance("gain"), [names[i] for i in ci]), reverse=True)[:12]
                print(lab, "top gain", [(n, int(g)) for g, n in imp], flush=True)
            print(arm, size, lab, "fold", f, round(time.time() - t0), "s", flush=True)
        preds[lab] = pr
        for pn in C.PANELS:
            y = Z[pn][lab]; print("  ", pn, lab, "rel rmse %.4f" % (np.sqrt(np.mean((y - pr[pn]) ** 2)) / np.sqrt(np.mean(y ** 2))), flush=True)
    np.savez_compressed(OUT / f"pred_{BASE + '+' if BASE else ''}{arm}{size}{'w' + str(WEIGHT) if WEIGHT else ''}{ASFX}.npz", **{f"{lab}_{pn}": preds[lab][pn] for lab in preds for pn in C.PANELS})


def fitall(arm, size):
    """All-OPT-days models for application (OWN, official). Saved as s4_<BASE>+<arm><size>_{yq,yv}.txt + names json."""
    import lightgbm as lgb
    Z = load_Z(arm)
    names = Z[C.PANELS[0]]["names"]; ci = cols_of(names, arm); sz = SIZE[size]
    prm = dict(PRM, learning_rate=sz["lr"], num_leaves=sz["leaves"])
    rng = np.random.default_rng(1)
    tot = sum(len(Z[pn]["yq"]) for pn in C.PANELS); keep = min(1., sz["nsub"] / tot)
    sel = {pn: rng.random(len(Z[pn]["yq"])) < keep for pn in C.PANELS}
    X = np.concatenate([Z[pn]["X"][sel[pn]][:, ci] for pn in C.PANELS])
    tag = f"{BASE + '+' if BASE else ''}{arm}{size}"
    for lab in LABS:
        y = np.concatenate([Z[pn][lab][sel[pn]] for pn in C.PANELS])
        m = lgb.train(dict(prm, alpha=ALPHA[lab]), lgb.Dataset(X, y), sz["rounds"])
        m.save_model(str(OUT / f"s4_{tag}_{lab}.txt")); print("saved", tag, lab, len(y), flush=True)
    (OUT / f"s4_{tag}_names.json").write_text(json.dumps([names[i] for i in ci]))


def apply(pn, tag, v, q, ix):
    """tag 'X' = pred_X on C14; 'X+Y' = pred_X then pred_X+Y (stage 4 on base X)."""
    d, t, s = ix.T.astype(int); nv, nq = v.copy(), q.copy()
    parts = tag.split("+")
    for i in range(len(parts)):
        z = np.load(OUT / f"pred_{'+'.join(parts[:i + 1])}.npz")
        if f"yq_{pn}" in z: nq[d, t, s] = nq[d, t, s] * np.exp(z[f"yq_{pn}"])
        if f"yv_{pn}" in z: nv[d, t, s] = nv[d, t, s] * np.exp(z[f"yv_{pn}"])
    return nv, nq


def evaluate(tags, ref=None):
    """J deltas of each tag vs ref (None = C14) with day bootstrap."""
    D = C.D_OPT; comps = {t: {} for t in tags}; cref = {}
    for pn in C.PANELS:
        held, tv, tq, v, q, p = DG.load(pn); ix = np.load(OUT / f"feat_{pn}.npz")["ix"]
        base = apply(pn, ref, v, q, ix) if ref else (v, q)
        cref[pn] = FS.day_components(held, tv, tq, *base)
        for tg in tags:
            comps[tg][pn] = FS.day_components(held, tv, tq, *apply(pn, tg, v, q, ix))
        print(pn, "eval", flush=True)
    res = {}
    for tg in tags:
        summ, per = PE.boot(comps[tg], cref, D)
        res[tg] = dict(summary=summ, per_panel=per)
        print(tg, "vs", ref or "C14", {m: (round(x["mean"], 6), [round(c_, 6) for c_ in x["ci"]], x["n_pos"]) for m, x in summ.items()}, flush=True)
    (OUT / ("eval_" + "_".join(tags) + "_vs_" + (ref or "C14") + ".json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2:] or C.PANELS)
    elif cmd == "fit":
        fit(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "S")
    elif cmd == "fitall":
        fitall(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "S")
    elif cmd == "eval":
        a = sys.argv[2:]; ref = None
        if "--ref" in a:
            i = a.index("--ref"); ref = a[i + 1]; a = a[:i] + a[i + 2:]
        evaluate(a, ref)
