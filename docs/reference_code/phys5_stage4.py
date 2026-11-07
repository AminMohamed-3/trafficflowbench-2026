"""phys5 (C helper, 2026-10-05): apply a phys5 stage-4 model (phys5_s3 fitall, features = phys4 stage-3 set with p dropped
+ LSA diffs + demand-step block features) to any base (v, q) given the released observed fields. Same feature code as the
OPT build (phys4_feat.build, phys4_lsa_eval.lsa_fields, phys5_s3.step_feats)."""
import json
import numpy as np
import phys4_feat as PF
import phys4_lsa_eval as LE
import phys5_s3 as P5

OUT = P5.OUT


def models(tag):
    import lightgbm as lgb
    names = json.loads((OUT / f"s4_{tag}_names.json").read_text())
    m = dict(names=names)
    for lab in ("yq", "yv"):
        f = OUT / f"s4_{tag}_{lab}.txt"
        m[lab] = lgb.Booster(model_file=str(f)) if f.exists() else None
    return m


def features(v, q, T, gap, so, fo, stations, length, vf):
    """so/fo: released observed speed/flow (nan elsewhere, never target/gap). Returns X (all names), names, ix."""
    ok = np.isfinite(so) & np.isfinite(fo)
    m = T & ~gap & np.isfinite(v) & np.isfinite(q) & (q > 0) & (v > 0)
    X, names, ix = PF.build(so, fo, T, v, q, np.full(T.shape, np.nan), stations, length, m)
    d, t, s = ix.T
    lq = np.log(np.maximum(q, 1.)); lv = np.log(np.maximum(v, 1.))
    F = LE.lsa_fields(so, fo, vf)
    X = np.concatenate([X, np.stack([(F[0] - lq)[d, t, s], F[1][d, t, s], (F[2] - lv)[d, t, s], F[3][d, t, s]], 1).astype(np.float32)], 1)
    names = names + ["lsa_dq", "lsa_nq", "lsa_dv", "lsa_nv"]
    del F
    mp = stations.sort_values("idx").milepost.to_numpy(float)
    SF = P5.step_feats(~ok, ok, so, fo, mp, vf, lq)
    X = np.concatenate([X, np.stack([np.asarray(np.broadcast_to(SF[k], T.shape))[d, t, s].astype(np.float32) for k in P5.STEP], 1)], 1)
    import phys5_feat4 as F4
    X4, _ = F4.feats_from(so, fo, T, vf, mp, lq[d, t, s], ix)
    X = np.concatenate([X, X4], 1)
    return X, names + P5.STEP + F4.NAMES4, ix


def apply(m4, v, q, T, gap, so, fo, stations, length, vf, scale=1.):
    X, names, ix = features(v, q, T, gap, so, fo, stations, length, vf)
    ci = [names.index(n) for n in m4["names"]]
    Xs = X[:, ci]; d, t, s = ix.T.astype(int)
    nv, nq = v.copy(), q.copy()
    if m4["yq"] is not None:
        nq[d, t, s] = q[d, t, s] * np.exp(scale * m4["yq"].predict(Xs, num_threads=3))
    if m4["yv"] is not None:
        nv[d, t, s] = v[d, t, s] * np.exp(scale * m4["yv"].predict(Xs, num_threads=3))
    return nv, nq, len(ix)
