"""Claude onset4 helper 2026-10-04: site model on physics features (onset4_data.py), decoded with Q03's choose_cells.

Variants (LightGBM binary per (window, site) row, fit on train Jun-Dec 2030 selector windows):
  A  = site_uid + all features
  B  = site-agnostic physics (no site_uid, no time-of-day/weekday)
  C  = B + tod6, m30 (demand-step timing)
Scored on I-Q-TOL (Jan dev / Feb confirm, truth/valid from onset2) vs frozen Q03 (iou_q03), and on March/April
pseudo onset windows (onset_score populations; masked history) vs Q03 k=0. Macro panel -> family -> mean,
date bootstrap B=2000 seed 0.
Usage: python docs/reference_code/onset4_eval.py [variant ...]"""
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/claude_ref/onset4"
LANEQ = ROOT / "src"
sys.path.insert(0, str(LANEQ))
from tfb.lane_q.onset_model import choose_cells
sys.path.pop(0)
META = json.loads((ROOT / "artifacts/lane_q/0267/Q03_seed0_model/metadata.json").read_text())
PHYS = ["other", "n_cells", "cap_c", "fc_T", "fc_last", "fc_h", "fc_slope", "prof_T", "prof_ratio", "fc_anom", "fc_proj",
        "brk", "head", "head_proj", "day_anom", "fR_max", "fR_anom_max", "vR_T", "vR_T1", "vR_h", "vR_slope", "vP_T",
        "oR_max", "obsP_T", "other_minv_T", "q_today"]
VARS = {"A": ["site_uid"] + PHYS + ["tod6", "m30", "weekend", "weekday"], "B": PHYS, "C": PHYS + ["tod6", "m30"]}
PARAMS = {"objective": "binary", "num_leaves": 15, "learning_rate": .05, "min_data_in_leaf": 30, "deterministic": True,
          "force_col_wise": True, "num_threads": 3, "seed": 0, "verbosity": -1, "feature_fraction": .8,
          "bagging_fraction": .8, "bagging_freq": 1}
CATS = sorted({f"{p}/{s['key']}" for p, ss in META["sites"].items() for s in ss})


def X(df, cols):
    x = df[[c for c in cols if c != "site_uid"]].astype(np.float32).copy()
    if "site_uid" in cols:
        x.insert(0, "site_uid", pd.Categorical(df.site_uid, categories=CATS).codes.astype(np.float32))
    return x


def fit(df, cols, rounds=300):
    x = X(df, cols)
    ds = lgb.Dataset(x, label=df.y.values, categorical_feature=["site_uid"] if "site_uid" in cols else [])
    return lgb.train(PARAMS, ds, num_boost_round=rounds)


def decode(df, p):
    """df rows of one population (all sites per window, in META order); returns {wkey: set(cells)}."""
    out = {}
    df = df.assign(p=p)
    for w, g in df.groupby("wkey", sort=False):
        sites = META["sites"][g.panel.iloc[0]]
        probs = g.set_index("site_uid").p.reindex([f"{g.panel.iloc[0]}/{s['key']}" for s in sites]).to_numpy()
        cells = choose_cells(sites, probs, close_ratio=META.get("close_ratio", .8))
        pos = {l: i for s in sites for l, i in zip(s["links"], s["positions"])}
        out[w] = {pos[l] for l in cells}
    return out


def iou(pred, truth, valid):
    pv = set(pred) & set(valid); t = set(truth); u = pv | t
    return 1. if not u else len(pv & t) / len(u)


def macro(d, c):
    pn = d.groupby("panel")[c].mean()
    return float(pn.groupby(pn.index.str.rsplit("_", n=1).str[0]).mean().mean())


def boot(d, a, b, B=2000, seed=0):
    """Date bootstrap of macro(a) - macro(b); vectorised via per-(date, panel) sums and counts."""
    rng = np.random.default_rng(seed); ds = np.array(sorted(d.date.unique())); pans = sorted(d.panel.unique())
    di = {x: i for i, x in enumerate(ds)}; pi = {x: i for i, x in enumerate(pans)}
    S = np.zeros((len(ds), len(pans))); N = np.zeros_like(S)
    np.add.at(S, (d.date.map(di).values, d.panel.map(pi).values), (d[a] - d[b]).values)
    np.add.at(N, (d.date.map(di).values, d.panel.map(pi).values), 1)
    fam = np.array([p.rsplit("_", 1)[0] for p in pans]); fams = sorted(set(fam))
    W = rng.multinomial(len(ds), np.ones(len(ds)) / len(ds), size=B)  # B x dates counts
    s = W @ S; n = W @ N
    with np.errstate(all="ignore"):
        pm = s / n
    out = []
    for row in pm:
        fm = [np.nanmean(row[fam == f]) for f in fams]
        out.append(np.nanmean(fm))
    return macro(d, a) - macro(d, b), *np.percentile(out, [2.5, 97.5])


def windows(df):
    w = df.groupby("wkey", sort=False).first()[["panel", "date", "t", "truth", "valid"]].reset_index()
    w["truth_l"] = w.truth.map(lambda s: [int(x) for x in s.split(";") if x])
    w["valid_l"] = w.valid.map(lambda s: [int(x) for x in s.split(";") if x])
    return w


def q03_refs():
    T = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / "artifacts/claude_ref/onset2/tol_windows_*.parquet")))])
    ref = {f"{r.panel}|{r.date}|{r.t}": set(map(int, r.q03)) for r in T.itertuples()}
    for split in ("validation", "private"):
        q = pd.read_parquet(ROOT / f"artifacts/claude_ref/onset/q03_pseudo_{split}.parquet"); q = q[q.k == "0"]
        for p, g in q.groupby("panel"):
            links = [l for s in META["sites"][p] for l in s["links"]]
            pos = {l: i for s in META["sites"][p] for l, i in zip(s["links"], s["positions"])}
            for r in g.itertuples():
                ref[f"{p}|{r.date}|{r.t}"] = {pos[l] for l in r.cells.split(";") if l}
    return ref


def score(pop_df, cells, ref, name):
    w = windows(pop_df)
    w["new"] = [iou(cells[k], t, v) for k, t, v in zip(w.wkey, w.truth_l, w.valid_l)]
    w["q03"] = [iou(ref[k], t, v) for k, t, v in zip(w.wkey, w.truth_l, w.valid_l)]
    w["chg"] = [cells[k] != ref[k] for k in w.wkey]
    return w


def report(w, label):
    d, lo, hi = boot(w, "new", "q03")
    print(f"  {label:10s} n={len(w):3d} chg={int(w.chg.sum()):3d} new {macro(w,'new'):.4f} q03 {macro(w,'q03'):.4f} "
          f"delta {d:+.4f} CI [{lo:+.4f},{hi:+.4f}]", flush=True)


def main(variants):
    fitd = pd.read_parquet(OUT / "feat_fit.parquet")
    pops = {k: pd.read_parquet(OUT / f"feat_{k}.parquet") for k in ("tol", "val", "pri")}
    ref = q03_refs()
    for v in variants:
        cols = VARS[v]
        m = fit(fitd, cols)
        print(f"variant {v}", flush=True)
        for k, df in pops.items():
            p = m.predict(X(df, cols))
            cells = decode(df, p)
            w = score(df, cells, ref, k)
            w.to_parquet(OUT / f"scores_{v}_{k}.parquet")
            if k == "tol":
                report(w[w.date < "2031-02-01"], "tol-Jan"); report(w[w.date >= "2031-02-01"], "tol-Feb"); report(w, "tol-all")
            else:
                report(w, k)
        imp = pd.Series(m.feature_importance("gain"), index=m.feature_name()).sort_values(ascending=False)
        print("  top gain:", (imp / imp.sum()).head(10).round(3).to_dict(), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["A", "B", "C"])
