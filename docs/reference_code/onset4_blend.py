"""Claude onset4 2026-10-04: logit blend of Q03 site probabilities with the onset4 physics site model.
p = sigmoid(w * logit(p_q03) + (1 - w) * logit(p_phys)); w grid; decoded by Q03 choose_cells.
Q03 probabilities: TOL from onset2 tol_windows p_q03; March/April pseudo from onset q03_pseudo_{split} k=0.
Usage: python docs/reference_code/onset4_blend.py [variant]   (variant of onset4_eval.VARS, default C)"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append(str(Path(__file__).resolve().parent))
import onset4_eval as E
ROOT = E.ROOT; OUT = E.OUT


def q03_probs():
    out = {}
    T = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / "artifacts/claude_ref/onset2/tol_windows_*.parquet")))])
    for r in T.itertuples():
        for k, p in zip(r.site_keys, r.p_q03):
            out[(f"{r.panel}|{r.date}|{r.t}", f"{r.panel}/{k}")] = p
    for split in ("validation", "private"):
        q = pd.read_parquet(ROOT / f"artifacts/claude_ref/onset/q03_pseudo_{split}.parquet"); q = q[q.k == "0"]
        for r in q.itertuples():
            for k, p in zip(r.site_keys.split(";"), r.p):
                out[(f"{r.panel}|{r.date}|{r.t}", f"{r.panel}/{k}")] = p
    return out


def lg(p):
    p = np.clip(p, 1e-5, 1 - 1e-5); return np.log(p / (1 - p))


def run(variant="C", ws=(0, .25, .5, .6, .75, .9, 1.), extra=None, fitpops=("fit",)):
    fitd = pd.concat([pd.read_parquet(OUT / f"feat_{k}.parquet") for k in fitpops], ignore_index=True)
    cols = E.VARS[variant]
    m = E.fit(fitd, cols)
    qp = q03_probs(); ref = E.q03_refs()
    res = {}
    for k in ("tol", "val", "pri"):
        df = pd.read_parquet(OUT / f"feat_{k}.parquet")
        pp = m.predict(E.X(df, cols))
        pq = np.array([qp[(w, s)] for w, s in zip(df.wkey, df.site_uid)])
        df.assign(p_phys=pp, p_q03=pq)[["wkey", "site_uid", "panel", "date", "t", "y", "p_phys", "p_q03"]].to_parquet(OUT / f"probs_{variant}_{k}.parquet")
        for w in ws:
            p = 1 / (1 + np.exp(-(w * lg(pq) + (1 - w) * lg(pp))))
            cells = E.decode(df, p)
            sc = E.score(df, cells, ref, k)
            res[(k, w)] = sc
    print(f"variant {variant} fit {fitpops}")
    for w in ws:
        line = f"  w={w:.2f}"
        for k in ("tol", "val", "pri"):
            sc = res[(k, w)]
            if k == "tol":
                for nm, sub in (("Jan", sc[sc.date < "2031-02-01"]), ("Feb", sc[sc.date >= "2031-02-01"])):
                    d, lo, hi = E.boot(sub, "new", "q03")
                    line += f" | {nm} {d:+.4f} [{lo:+.4f},{hi:+.4f}]"
            else:
                d, lo, hi = E.boot(sc, "new", "q03")
                d12 = sc[sc.date.str[-2:].astype(int) <= 12]
                line += f" | {k} {d:+.4f} [{lo:+.4f},{hi:+.4f}] d<=12 {E.macro(d12,'new')-E.macro(d12,'q03'):+.4f}(n{len(d12)})"
        print(line, flush=True)
    return res


if __name__ == "__main__":
    run(*(sys.argv[1:2] or ["C"]))
