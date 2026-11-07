"""ong3 2026-10-05: per selector-type analog diagnostics on I-Q-SPLIT hard pseudo windows (days <= 12).
Types from the pseudo gate (masked layer, coverage >= .5, some link with >= 2 queued steps):
  A: t-1 gated, its horizon held a queue and pers(t-1) > .9  (official type A analog)
  C: t-1 failed the gate (queue newly visible or coverage)   (official type C analog)
  M: t-1 gated with pers(t-1) <= .9 (official selector would have picked t-1; only reachable as type B)
Per window: eiou IoU, count-oracle IoU (top n_true of the same ranking), k (eiou size), n_true, psum.
Input: an export dir with hard_<split>.parquet (ong2_sccnn_export.py), column p (p_truth or p_unk).
Out: <export>/types_<col>.parquet.  Usage: python ong3_types_diag.py <export_dir> [p_truth|p_unk]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
from tfb.data import load_cache
from qcnn_data import cube
from gapres_selcond_rule import iou, eiou_order


def last_state(obs, q, lo, hi):
    last = np.zeros(q.shape[1], bool)
    for s in range(lo, hi):
        last = np.where(obs[s], q[s], last)
    return last


def pers_at(obs, q, t):
    last = last_state(obs, q, t - 12, t); val = obs[t + 1:t + 7]; fut = q[t + 1:t + 7] & val
    p = np.broadcast_to(last, fut.shape) & val; u = (p | fut).sum()
    return ((p & fut).sum() / u if u else 1.), bool(fut.any())


def gate(obs, q, t, cov=.5):
    return obs[t - 12:t].mean() >= cov and bool(((q[t - 12:t] & obs[t - 12:t]).sum(0) >= 2).any())


def wtype(obs, q, t):
    if not gate(obs, q, t - 1):
        return "C"
    p1, f1 = pers_at(obs, q, t - 1)
    if not f1:
        return "Z"
    return "A" if p1 > .9 else "M"


def main(exp, col="p_truth"):
    rows = []
    for split in ("validation", "private"):
        a = pd.read_parquet(Path(exp) / f"hard_{split}.parquet")
        for pn, gp in a.groupby("panel"):
            cc = load_cache(pn, split); st = cc.stations.sort_values("idx"); links = st.link_id.astype(str).tolist()
            c3 = cube(*[np.asarray(cc[x], np.float32) for x in ("speed", "flow", "occ")], st)
            OBS, Q = c3[:, :, 3] > .5, c3[:, :, 4] > .5
            for (date, t), g in gp.groupby(["date", "t"], sort=False):
                d = cc.day_index(str(date)); L = len(g) // 6
                idx = np.array([links.index(l) for l in g.link_id.to_numpy()[:L]])
                pm = np.broadcast_to(last_state(OBS[d], Q[d], t - 12, t)[idx], (6, L)).ravel()
                V = g.valid.to_numpy(); Y = g.label.to_numpy() & V; f = g[col].to_numpy(np.float64)
                k, o = eiou_order(f); m = np.zeros(len(f), bool); m[o[:k]] = True
                nt = int(Y.sum()); ov = o[V[o]]; mk = np.zeros(len(f), bool); mk[ov[:max(nt, 1)]] = True
                rows.append(dict(split=split, panel=pn, date=str(date), t=int(t), typ=wtype(OBS[d], Q[d], t),
                                 cross=bool(g.cross.iloc[0]), iou=iou(m & V, Y), iou_k=iou(mk & V, Y), k=int((m & V).sum()), n_true=nt,
                                 psum=f.sum(), npers=int(pm.sum()), pers_pred=iou(m, pm), pers_true=iou(pm & V, Y)))
    df = pd.DataFrame(rows); df.to_parquet(Path(exp) / f"types_{col}.parquet")
    df["day"] = pd.to_datetime(df.date).dt.day
    print(df.groupby(["split", "typ"]).agg(n=("iou", "size"), iou=("iou", "mean"), iou_k=("iou_k", "mean"),
                                            pers_true=("pers_true", "mean"), k_over_true=("k", "sum"),
                                            nt=("n_true", "sum")).assign(k_over_true=lambda x: x.k_over_true / x.nt).round(3))
    print("cross flag vs typ A:", pd.crosstab(df.cross, df.typ))
    return df


if __name__ == "__main__":
    main(*sys.argv[1:])
