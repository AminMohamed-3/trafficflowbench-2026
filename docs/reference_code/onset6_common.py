"""Claude onset6 2026-10-04: shared scorer for onset6 ideas on top of onset5 S1 (copy-not-edit of onset5_eval2 logic).

Populations (same as onset5_eval2): 'tol' I-Q-TOL raw-history Q03 (Jan dev / Feb confirm), 'tolm' masked-history Q03,
'val'/'pri' March/April pseudo onset windows (Q03 k=0, gap-emulated S1 statistic for history days j=0..8).
A method is a function m(g) -> site probabilities, where g is the per-(window, site) frame for one history day j with
columns wkey, site, site_uid, panel, pq (Q03), lr/u/med (S1 statistic) and any extra columns merged in by the caller.
score(methods, pops) -> {pop: DataFrame per window with one column per method (IoU averaged over valid j), plus
'unobs' = all predicted cells of any method unobserved at T+30 (pseudo-instrument artefact flag)}.
Usage: imported."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append(str(Path(__file__).resolve().parent))
import onset4_eval as E
import onset5_eval2 as V
ROOT = E.ROOT
O6 = ROOT / "artifacts/claude_ref/onset6"; O6.mkdir(parents=True, exist_ok=True)


_DC = {}
SITES = {pn: [f"{pn}/{s['key']}" for s in ss] for pn, ss in E.META["sites"].items()}
POS = {pn: {l: i for s in ss for l, i in zip(s["links"], s["positions"])} for pn, ss in E.META["sites"].items()}


def decode(df, p):
    """Memoised equivalent of onset4_eval.decode (same choose_cells call), keyed by panel and rounded probabilities."""
    out = {}
    df = df.assign(p=p)
    for w, g in df.groupby("wkey", sort=False):
        pn = g.panel.iloc[0]
        probs = g.set_index("site_uid").p.reindex(SITES[pn]).to_numpy()
        key = (pn, tuple(np.round(probs, 10)))
        if key not in _DC:
            cells = E.choose_cells(E.META["sites"][pn], probs, close_ratio=E.META.get("close_ratio", .8))
            _DC[key] = frozenset(POS[pn][l] for l in cells)
        out[w] = set(_DC[key])
    return out


def s1(g, alpha=.02, beta=1.):
    return V.adjust(g.pq.to_numpy(), g.lr.to_numpy(), g.u.to_numpy(), g.med.to_numpy(), alpha, beta)[0]


def frames(stat="all"):
    pops = V.prep(stat); F = V.base_frames()
    return {"tol": (F["tol"], pops["train"]), "tolm": (F["tolm"], pops["train"]), "val": (F["val"], pops["val"]),
            "pri": (F["pri"], pops["pri"])}


def score(methods, pops=("tol", "tolm", "val", "pri"), jmax=8, extra=None, stat="all"):
    """methods: {name: fn(g) -> p}. extra: {pop: DataFrame keyed by (wkey, site) [, j]} merged into g."""
    FR = frames(stat); out = {}
    for pop in pops:
        f, st = FR[pop]
        if extra is not None and pop in extra and "j" not in extra[pop].columns:
            f = f.merge(extra[pop], on=["wkey", "site"], how="left")
        w = E.windows(f.drop_duplicates("wkey")).set_index("wkey")
        rows = []
        for j in range(jmax + 1):
            g = f.merge(st[st.j == j][["wkey", "site", "lr", "u", "med"]], on=["wkey", "site"], how="inner")
            if extra is not None and pop in extra and "j" in extra[pop].columns:
                x = extra[pop]; g = g.merge(x[x.j == j].drop(columns="j"), on=["wkey", "site"], how="left")
            if not len(g):
                continue
            dec = {nm: decode(g, fn(g)) for nm, fn in methods.items()}
            for k in dec[next(iter(dec))]:
                t, v = w.loc[k, "truth_l"], w.loc[k, "valid_l"]; vs = set(v)
                r = dict(wkey=k, j=j, unobs=float(any(not (set(d[k]) & vs) for d in dec.values())))
                for nm, d in dec.items():
                    r[nm] = E.iou(d[k], t, v); r["c_" + nm] = tuple(sorted(d[k]))
                rows.append(r)
        r = pd.DataFrame(rows)
        num = [c for c in r.columns if c not in ("wkey", "j") and not c.startswith("c_")]
        a = r.groupby("wkey")[num].mean()
        names = list(methods)
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                a[f"chg_{n1}_{n2}"] = (r[f"c_{n1}"] != r[f"c_{n2}"]).groupby(r.wkey).mean()
        out[pop] = w[["panel", "date", "t"]].join(a, how="inner").reset_index()
    return out


def report(res, a, b, label=""):
    line = f"{label} {a} vs {b}"
    for pop, w in res.items():
        subs = [("Jan", w[w.date < "2031-02-01"]), ("Feb", w[w.date >= "2031-02-01"])] if pop == "tol" else \
            [("TOLm", w)] if pop == "tolm" else [({"val": "Mar", "pri": "Apr"}[pop], w)]
        for nm, s in subs:
            d, lo, hi = E.boot(s, a, b)
            ch = s[f"chg_{a}_{b}"].sum() if f"chg_{a}_{b}" in s else s[f"chg_{b}_{a}"].sum()
            line += f" | {nm} {d:+.4f} [{lo:+.4f},{hi:+.4f}] chg {ch:.1f}"
        if pop in ("val", "pri"):
            s = w[w.unobs == 0]; d, lo, hi = E.boot(s, a, b)
            line += f" (obs-only {d:+.4f} [{lo:+.4f},{hi:+.4f}])"
    print(line, flush=True)
    return line
