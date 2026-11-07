"""Claude onset6 2026-10-04 idea F evaluation: first-of-day site share switch (onset6_first.py), alone and stacked on S1.
lrF = log((F + .5) / (N + 1)); null = gap-emulated train windows of other months, same panel, site, history days kd,
origin within +-3 h (onset5_eval2.null_quantiles logic). Fire at uF < a or > 1 - a; odds x exp(bF * (lrF - medF)).
Rule declared before reading: a = .02, bF = 1 (as S1). Methods: q03, s1 (current best), f (F on Q03), s1f (S1 then F).
Usage: python docs/reference_code/onset6_evalF.py [a] [bF]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append(str(Path(__file__).resolve().parent))
import onset6_common as C
import onset5_eval2 as V


def prepF():
    files = {k: C.O6 / f"uF_{k}.parquet" for k in ("train", "val", "pri", "off")}
    if all(f.exists() for f in files.values()):
        return {k: pd.read_parquet(f) for k, f in files.items()}
    pops = {}
    for k in files:
        df = pd.read_parquet(C.O6 / f"first_{k}.parquet")
        df["lr"] = np.log((df.F + .5) / (df.N + 1)); df["mon"] = df.date.str[:7]
        pops[k] = df
    null = pops["train"]
    for k, df in pops.items():
        if k == "train":
            df = df[df.date >= "2031-01-01"].copy()
        u, med, nn = V.null_quantiles(null, df)
        pops[k] = df.assign(u=u, med=med, nn=nn); pops[k].to_parquet(files[k])
    return pops


def adjF(g, a, b):
    fire = (g.uF < a) | (g.uF > 1 - a)
    p = np.clip(g.pq.to_numpy(), 1e-6, 1 - 1e-6); o = p / (1 - p) * np.where(fire, np.exp(b * (g.lrF - g.medF)), 1.)
    return o / (1 + o)


def extra_frames():
    P = prepF()
    ren = {"lr": "lrF", "u": "uF", "med": "medF"}
    return {"tol": P["train"], "tolm": P["train"], "val": P["val"], "pri": P["pri"]}, ren


def main(a=.02, b=1.):
    a, b = float(a), float(b)
    X, ren = extra_frames()
    extra = {k: v[["wkey", "site", "j", "lr", "u", "med"]].rename(columns=ren) for k, v in X.items()}
    M = {"q03": lambda g: g.pq.to_numpy(), "s1": C.s1,
         "f": lambda g: adjF(g, a, b),
         "s1f": lambda g: adjF(g.assign(pq=C.s1(g)), a, b)}
    res = C.score(M, extra=extra)
    pd.to_pickle(res, C.O6 / f"evalF_a{a:g}_b{b:g}.pkl")
    for x, y in (("s1", "q03"), ("f", "q03"), ("s1f", "s1")):
        C.report(res, x, y, f"F a={a:g} b={b:g}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
