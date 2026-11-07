"""ong3 2026-10-05: score ensembles of per-model I-Q-SPLIT exports (ong3_export.py) and compare them.
  score <name> <dir[,dir...]> : average p_truth / p_unk over the model dirs, decode every window, per-window IoU on
                                valid cells -> artifacts/claude_ref/ong3/eval_<name>.parquet
     decodes: e   eiou;  c17 eiou + selcond .92 with persistence t-12..t (as C17 ships);
              v2  eiou + selcond .92 with persistence t-12..t-1, + band raise to .8 on type A (ong2 0469 chain);
              u_* the same on p_unk (sccnn with t-1 status unknown = official type B input).
  cmp <x_name> <x_col> <y_name> <y_col> : date-bootstrap deltas by month x type (A, C, M) x day half.
Official mix (val 23/8/9 A/B/C): MIX = .55 A + .25 M(B input) + .20 C.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
from tfb.data import load_cache
from qcnn_data import cube
from gapres_selcond_rule import iou, eiou_order, constrained
from ong2_band_rule import raise_to
from ong3_types_diag import last_state

OUT = ROOT / "artifacts/claude_ref/ong3"


def chain(f, pm, pmT, cross, kind, lo=.8):
    k, o = eiou_order(f)
    if kind == "e":
        m = np.zeros(len(f), bool); m[o[:k]] = True; return m
    if kind == "c17":
        return constrained(f, o, k, pmT, .92)[0]
    m, _ = constrained(f, o, k, pm, .92)
    if cross and iou(m, pm) < lo:
        m = raise_to(m, f, pm, lo)
    return m


def load_ens(dirs, split):
    base = None
    for i, d in enumerate(dirs):
        a = pd.read_parquet(Path(d) / f"hard_{split}.parquet")
        if base is None:
            base = a; base["p_truth"] = base.p_truth.astype(np.float64); base["p_unk"] = base.p_unk.astype(np.float64)
        else:
            assert len(a) == len(base) and (a.t.to_numpy() == base.t.to_numpy()).all() and (a.link_id.to_numpy() == base.link_id.to_numpy()).all()
            base["p_truth"] += a.p_truth.to_numpy(); base["p_unk"] += a.p_unk.to_numpy()
    base["p_truth"] /= len(dirs); base["p_unk"] /= len(dirs)
    return base


def score(name, dirs, kinds=("e", "c17", "v2"), unk=True):
    rows = []
    for split in ("validation", "private"):
        a = load_ens(dirs.split(","), split)
        for pn, gp in a.groupby("panel"):
            cc = load_cache(pn, split); st = cc.stations.sort_values("idx"); links = st.link_id.astype(str).tolist()
            c3 = cube(*[np.asarray(cc[x], np.float32) for x in ("speed", "flow", "occ")], st)
            OBS, Q = c3[:, :, 3] > .5, c3[:, :, 4] > .5
            for (date, t), g in gp.groupby(["date", "t"], sort=False):
                L = len(g) // 6; d = cc.day_index(str(date))
                idx = np.array([links.index(l) for l in g.link_id.to_numpy()[:L]])
                pm = np.broadcast_to(last_state(OBS[d], Q[d], t - 12, t)[idx], (6, L)).ravel()
                pmT = np.broadcast_to(last_state(OBS[d], Q[d], t - 12, t + 1)[idx], (6, L)).ravel()
                V = g.valid.to_numpy(); Y = g.label.to_numpy() & V; typ = g.typ.iloc[0]; cr = typ == "A"
                r = dict(split=split, panel=pn, date=str(date), t=int(t), typ=typ, grid=bool(g.grid.iloc[0]),
                         n_true=int(Y.sum()), pers_true=iou(pm & V, Y))
                for nm, col in (("", "p_truth"), ("u_", "p_unk")) if unk else (("", "p_truth"),):
                    f = g[col].to_numpy(np.float64)
                    for kind in kinds:
                        r[nm + kind] = iou(chain(f, pm, pmT, cr, kind) & V, Y)
                rows.append(r)
        print(name, split, "done", flush=True)
    df = pd.DataFrame(rows); df.to_parquet(OUT / f"eval_{name}.parquet")
    return df


def boot(x, dates, B=2000, seed=0):
    rng = np.random.default_rng(seed); g = pd.DataFrame(dict(d=dates, x=x)).groupby("d").x.agg(["sum", "size"])
    s, n = g["sum"].to_numpy(), g["size"].to_numpy()
    bs = [s[i].sum() / n[i].sum() for i in (rng.integers(0, len(s), len(s)) for _ in range(B))]
    return s.sum() / n.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def cmp(xn, xc, yn, yc, out=True):
    x = pd.read_parquet(OUT / f"eval_{xn}.parquet"); y = pd.read_parquet(OUT / f"eval_{yn}.parquet")
    k = ["split", "panel", "date", "t"]
    assert (x[k].to_numpy() == y[k].to_numpy()).all()
    df = x[k + ["typ", "grid"]].copy(); df["dx"] = x[xc].to_numpy() - y[yc].to_numpy(); df["y"] = y[yc].to_numpy()
    df["day"] = pd.to_datetime(df.date).dt.day; df["half"] = np.where(df.day % 2 == 1, "odd", "even")
    res = []
    for split, gs in df.groupby("split"):
        for typ in ("A", "C", "M"):
            gt = gs[gs.typ == typ]
            for half, g in list(gt.groupby("half")) + [("all", gt)]:
                e, lo, hi = boot(g.dx.to_numpy(), g.date.tolist())
                res.append(dict(split=split[:3], typ=typ, half=half, n=len(g), base=g.y.mean(), d=e, lo=lo, hi=hi,
                                nchg=int((g.dx != 0).sum())))
    r = pd.DataFrame(res)
    if out:
        print(f"== {xn}:{xc} - {yn}:{yc}")
        print(r.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    return r


if __name__ == "__main__":
    if sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3])
    else:
        cmp(*sys.argv[2:6])
