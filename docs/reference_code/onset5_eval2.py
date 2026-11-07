"""Claude onset5 2026-10-04: calibrated bottleneck-shift switch on Q03, OFFICIAL-GAP-EMULATED instrument (stat2).

Statistic (onset5_stat2.py): lr_s = log((A_s + C) / (E_s + C)), site queue activity in the window's own online history
(days s..d-1 of the same month plus day d before t; emulated official gaps hidden) vs the train expectation.
Null: the same statistic on train windows of OTHER months, same panel, site, history days j, origin within +-NULL_T.
u_s = null CDF, med_s = null median.
Rule (declared before any read): site s fires when u_s < ALPHA or u_s > 1 - ALPHA; then odds_s *= exp(BETA*(lr_s - med_s)).
Non-firing sites keep Q03's odds. Decode with Q03 choose_cells. Baseline = the same Q03 probabilities decoded unadjusted.
Populations: I-Q-TOL Jan (dev) / Feb (confirm) with raw-history Q03 ('tol') or masked-history Q03 ('tolm', fidelity);
March/April pseudo windows (masked labels, Q03 k=0 probabilities). A window's score = mean over its valid j.
Macro panel -> family -> mean, paired date bootstrap (onset4_eval.boot).
Usage: python docs/reference_code/onset5_eval2.py [stat=all|loc] [alpha] [beta] [jmax]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append(str(Path(__file__).resolve().parent))
import onset4_eval as E
import onset4_blend as BL
ROOT = E.ROOT
O5 = ROOT / "artifacts/claude_ref/onset5"
C = 5.
NULL_T = 36


def load(stat):
    pops = {}
    for k in ("train", "val", "pri", "off"):
        df = pd.read_parquet(O5 / f"stat2_{k}.parquet")
        df["lr"] = np.log((df[f"A_{stat}"] + C) / (df[f"E_{stat}"] + C)); df["mon"] = df.date.str[:7]
        pops[k] = df
    return pops


def null_quantiles(null, df):
    u = np.full(len(df), .5); med = np.zeros(len(df)); nn = np.zeros(len(df), int)
    null = null.assign(kd=null.kdays.clip(upper=8))
    groups = {key: (g.t.to_numpy(), g.mon.to_numpy(), g.lr.to_numpy()) for key, g in null.groupby(["panel", "site", "kd"])}
    for i, (p, s, kd, t, mon, lr) in enumerate(zip(df.panel, df.site, df.kdays.clip(upper=8), df.t, df.mon, df.lr)):
        g = groups.get((p, s, kd))
        if g is None:
            continue
        m = (np.abs(g[0] - t) <= NULL_T) & (g[1] != mon); x = g[2][m]
        if len(x) < 20:
            continue
        u[i] = ((x < lr).sum() + .5 * (x == lr).sum()) / len(x); med[i] = np.median(x); nn[i] = len(x)
    return u, med, nn


def prep(stat):
    files = {k: O5 / f"u2_{stat}_{k}.parquet" for k in ("train", "val", "pri", "off")}
    if all(f.exists() for f in files.values()):
        return {k: pd.read_parquet(f) for k, f in files.items()}
    pops = load(stat); null = pops["train"]
    for k in pops:
        df = pops[k]
        if k == "train":
            df = df[df.date >= "2031-01-01"].copy()
        u, med, nn = null_quantiles(null, df)
        pops[k] = df.assign(u=u, med=med, nn=nn); pops[k].to_parquet(files[k])
    return pops


def adjust(pq, lr, u, med, alpha, beta):
    fire = (u < alpha) | (u > 1 - alpha)
    pq = np.clip(pq, 1e-6, 1 - 1e-6); o = pq / (1 - pq) * np.where(fire, np.exp(beta * (lr - med)), 1.)
    return o / (1 + o), fire


_CACHE = {}


def base_frames():
    if "f" in _CACHE:
        return _CACHE["f"]
    qp = BL.q03_probs()
    qm = pd.read_parquet(E.OUT / "q03m_tol.parquet").set_index(["wkey", "site_uid"]).p_q03m.to_dict()
    out = {}
    for k, fn in (("tol", "feat_tol"), ("val", "feat_val"), ("pri", "feat_pri")):
        f = pd.read_parquet(E.OUT / f"{fn}.parquet")[["wkey", "site_uid", "panel", "date", "t", "truth", "valid"]].copy()
        f["site"] = f.site_uid.str.split("/").str[1]
        f["pq"] = [qp[(w, s)] for w, s in zip(f.wkey, f.site_uid)]
        if k == "tol":
            g = f.copy(); g["pq"] = [qm[(w, s)] for w, s in zip(g.wkey, g.site_uid)]
            out["tolm"] = g
        out[k] = f
    _CACHE["f"] = out
    return out


def score_pop(f, st, alpha, beta, jmax):
    w = E.windows(f.drop_duplicates("wkey").assign()).set_index("wkey")
    rows = []
    for j in range(jmax + 1):
        g = f.merge(st[st.j == j][["wkey", "site", "lr", "u", "med"]], on=["wkey", "site"], how="inner")
        if not len(g):
            continue
        p, fire = adjust(g.pq.to_numpy(), g.lr.to_numpy(), g.u.to_numpy(), g.med.to_numpy(), alpha, beta)
        new = E.decode(g, p); old = E.decode(g, g.pq.to_numpy())
        fw = g.assign(fire=fire).groupby("wkey").fire.any()
        for k in new:
            t, v = w.loc[k, "truth_l"], w.loc[k, "valid_l"]
            rows.append(dict(wkey=k, j=j, new=E.iou(new[k], t, v), q03=E.iou(old[k], t, v), chg=new[k] != old[k], fire=fw[k],
                             unobs=float(not (set(new[k]) & set(v)) or not (set(old[k]) & set(v)))))
    r = pd.DataFrame(rows)
    a = r.groupby("wkey")[["new", "q03", "chg", "fire", "unobs"]].mean()
    return w[["panel", "date", "t"]].join(a, how="inner").reset_index()


def run(stat="all", alpha=.02, beta=1., jmax=8, verbose=True):
    alpha, beta, jmax = float(alpha), float(beta), int(jmax)
    pops = prep(stat); F = base_frames()
    res = {"tol": score_pop(F["tol"], pops["train"], alpha, beta, jmax), "tolm": score_pop(F["tolm"], pops["train"], alpha, beta, jmax),
           "val": score_pop(F["val"], pops["val"], alpha, beta, jmax), "pri": score_pop(F["pri"], pops["pri"], alpha, beta, jmax)}
    out = {}; line = f"stat={stat} a={alpha:g} b={beta:g} j<={jmax}"
    for nm, w in (("Jan", res["tol"][res["tol"].date < "2031-02-01"]), ("Feb", res["tol"][res["tol"].date >= "2031-02-01"]),
                  ("TOLm", res["tolm"]), ("Mar", res["val"]), ("Apr", res["pri"])):
        d, lo, hi = E.boot(w, "new", "q03")
        out[nm] = (d, lo, hi, w.chg.sum(), w.fire.mean(), len(w))
        line += f" | {nm} {d:+.4f} [{lo:+.4f},{hi:+.4f}] chg {w.chg.sum():.1f} n {len(w)}"
    if verbose:
        print(line, flush=True)
    return res, out


if __name__ == "__main__":
    run(*sys.argv[1:5])
