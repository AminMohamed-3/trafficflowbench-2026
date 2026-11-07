"""Claude onset9 2026-10-07: final-pair onset hedge (research_round3_big.md Idea 1) on the FINAL onset method.

File A = final onset rows = Q0436 (served probabilities p_candidate, artifacts/lane_q/0436_demand_agreement).
File B = A except, in chosen PRIVATE onset windows, the decode with the top-two site probabilities swapped
(single-site decode -> the second site's own expected-IoU cell set), or for a union decode one site only.
Validation rows and every other task identical, so B's public score equals A's.

Calibration of P(top true), P(second true), P(both) per window (window level, j <= 4, S1F probabilities, onset8 resid):
  pooled  : March + April pseudo by r bin (as tested in big_hedge_sim.py)
  march   : March pseudo only by r bin
  pairM   : march r-bin rate shrunk toward the March pseudo rate of the same (panel, s1, s2) ordered pair (m = 4 windows)
            -> SELECTION calibration (D-0062: April may use released March data; no April labels used)
  pairA   : the same with April pseudo windows: CONFIRM ONLY (April labels after the official days are post-T for the
            private windows, so they must not choose the window set)
IoU model as big_hedge_sim: right single site a = .93; single vs two-site truth a/2; union vs one-site truth a/2.
Window weight .30/80 S_total. Gain = E[max(A, B)] - E[A] (Kaggle ranks the better of the two selected files).
Usage: python docs/reference_code/onset9_hedge.py   (writes artifacts/claude_ref/onset9/hedge_*.csv/.txt)"""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tfb.lane_q.onset_model import choose_cells
O9 = ROOT / "artifacts/claude_ref/onset9"; O9.mkdir(parents=True, exist_ok=True)
META = json.loads((ROOT / "artifacts/lane_q/0267/Q03_seed0_model/metadata.json").read_text())
A_IOU, W, N = .93, .30 / 80, 40000
BINS = [0, .1, .25, .5, .75, 1.01]


PROBS = ROOT / "artifacts/lane_q/0436_demand_agreement/official_probabilities.parquet"
STAT = ROOT / "artifacts/claude_ref/onset5/stat2_off.parquet"


def served(path=None):
    """Per official onset window: served site probabilities (Q0436 p_candidate), top two sites, r, A and swap-B cells."""
    p = pd.read_parquet(path or PROBS)
    out = []
    for wk, g in p.groupby("wkey"):
        pn = g.panel.iloc[0]; sites = META["sites"][pn]
        keys = [s["key"] for s in sites]
        pr = g.set_index("site").p_candidate.reindex(keys).to_numpy()
        named = g[g.site != "other"].sort_values("p_candidate", ascending=False)
        s1, p1 = named.site.iloc[0], named.p_candidate.iloc[0]
        s2, p2 = (named.site.iloc[1], named.p_candidate.iloc[1]) if len(named) > 1 else ("none", 0.)
        a = choose_cells(sites, pr, close_ratio=META["close_ratio"])
        i1, i2 = keys.index(s1), (keys.index(s2) if s2 in keys else None)
        sw = pr.copy()
        if i2 is not None:
            sw[i1], sw[i2] = pr[i2], pr[i1]
        b_swap = choose_cells(sites, sw, close_ratio=META["close_ratio"])
        cells = {k: set(s["links"]) for k, s in zip(keys, sites)}
        only = lambda k: tuple(sorted(cells[k] & set(a))) if k in cells else ()
        union = len(set(a) & cells[s1]) > 0 and i2 is not None and len(set(a) & cells[s2]) > 0
        out.append(dict(window_id=wk, split=g.split.iloc[0], panel=pn, s1=s1, p1=p1, s2=s2, p2=p2, r=p2 / p1,
                        union=union, a_cells=";".join(a), b_swap=";".join(b_swap),
                        b_s1only=";".join(only(s1)), b_s2only=";".join(only(s2)) if i2 is not None else "",
                        b_swap_site=",".join(k for k in keys if set(b_swap) & cells[k])))
    return pd.DataFrame(out)


def calib():
    d = pd.read_parquet(ROOT / "artifacts/claude_ref/big/hedge_pseudo.parquet")
    d["panel"] = d.wkey.str.split("|").str[0]
    d["both"] = d.top_true & d.sec_true
    w = d.groupby(["pop", "panel", "wkey"]).agg(r=("r", "mean"), s1=("s1", "first"), s2=("s2", "first"),
                                                 top=("top_true", "mean"), sec=("sec_true", "mean"),
                                                 both=("both", "mean")).reset_index()
    w["b"] = np.digitize(w.r, BINS) - 1
    return w


def window_probs(o, w, mode, m=4.):
    """(top_only, sec_only, both) per official window under a calibration mode."""
    pops = {"pooled": ["val", "pri"], "march": ["val"], "pairM": ["val"], "pairA": ["pri"]}[mode]
    ww = w[w["pop"].isin(pops)]
    base = ww.groupby("b")[["top", "sec", "both"]].mean()
    res = []
    for x in o.itertuples():
        b = np.digitize([x.r], BINS)[0] - 1
        t, s, bo = base.loc[b, ["top", "sec", "both"]]
        if mode in ("pairM", "pairA"):
            q = ww[(ww.panel == x.panel) & (ww.r > .25) & (((ww.s1 == x.s1) & (ww.s2 == x.s2)) | ((ww.s1 == x.s2) & (ww.s2 == x.s1)))]
            # pair data in the orientation of this window: rate that site s1 (resp. s2) is true
            n = len(q)
            if n:
                k1 = np.where(q.s1 == x.s1, q.top, q.sec).sum(); k2 = np.where(q.s1 == x.s1, q.sec, q.top).sum()
                kb = q.both.sum()
                t = (k1 + m * t) / (n + m); s = (k2 + m * s) / (n + m); bo = (kb + m * bo) / (n + m)
        res.append((max(t - bo, 0), max(s - bo, 0), bo, len(q) if mode.startswith("pair") else 0))
    return res


def simulate(o, probs, sel, union_opt="s2", seed=1):
    """sel: indices into o. Returns E[B-A], sd, E[max(A,B)]-E[A], P(max-A >= .003)."""
    rng = np.random.default_rng(seed)
    dB = np.zeros(N)
    for i in sel:
        x = o.iloc[i]; pt, ps, pb, _ = probs[i]
        u = rng.random(N)
        cat = np.where(u < pt, 0, np.where(u < pt + ps, 1, np.where(u < pt + ps + pb, 2, 3)))  # top, sec, both, other
        if x.union:
            a = np.select([cat == 0, cat == 1, cat == 2], [A_IOU / 2, A_IOU / 2, A_IOU], 0.)
            if union_opt == "s2":
                b = np.select([cat == 1, cat == 2], [A_IOU, A_IOU / 2], 0.)
            else:
                b = np.select([cat == 0, cat == 2], [A_IOU, A_IOU / 2], 0.)
        else:
            a = np.select([cat == 0, cat == 2], [A_IOU, A_IOU / 2], 0.)
            b = np.select([cat == 1, cat == 2], [A_IOU, A_IOU / 2], 0.)
        dB += b - a
    dB *= W
    g = np.maximum(dB, 0)
    return dB.mean(), dB.std(), g.mean(), (g >= .003).mean()


def final_set(pri, th=.4, stat=None):
    """FINAL RULE on a frame of windows with a 0..n-1 index: r > th on served probabilities, minus windows whose second
    site has zero online activity in the month's history while some site has activity (research kill rule; online S1
    statistic stat2_off: days 0..d of the split, day d only steps before T). Returns (keep positions, killed ids, act)."""
    st = pd.read_parquet(stat or STAT)
    act = st.groupby(["wkey", "site"]).A_all.sum()
    tot = st[st.site != "other"].groupby("wkey").A_all.sum()
    keep = [i for i in range(len(pri)) if pri.r[i] > th
            and not (act.get((pri.window_id[i], pri.s2[i]), 0) == 0 and tot.get(pri.window_id[i], 0) > 0)]
    killed = [pri.window_id[i] for i in range(len(pri)) if pri.r[i] > th and i not in keep]
    return keep, killed, act


def hedge_frame(pri, keep, act):
    """Rows of the hedged windows with the B cell set (union decode -> second site only, single -> swapped decode)."""
    fin = pri.loc[keep].copy()
    fin["b_cells"] = [x.b_s2only if x.union else x.b_swap for x in fin.itertuples()]
    fin["b_site"] = [x.s2 for x in fin.itertuples()]
    fin["A_s2_online"] = [act.get((x.window_id, x.s2), 0) for x in fin.itertuples()]
    return fin


def main():
    o = served()
    o.to_csv(O9 / "hedge_official_q0436.csv", index=False)
    w = calib()
    lines = []
    pri = o[(o.split == "private") & (o.r > .25)].sort_values("r", ascending=False).reset_index(drop=True)
    P = {mode: window_probs(pri, w, mode) for mode in ("pooled", "march", "pairM", "pairA")}
    tab = pri[["window_id", "s1", "p1", "s2", "p2", "r", "union", "b_swap_site"]].copy()
    for mode, pr in P.items():
        tab[f"sec_{mode}"] = [x[1] + x[2] for x in pr]; tab[f"top_{mode}"] = [x[0] + x[2] for x in pr]
    tab["n_pairM"] = [x[3] for x in P["pairM"]]; tab["n_pairA"] = [x[3] for x in P["pairA"]]
    lines.append("private windows r > .25 (served Q0436 probabilities); sec_/top_ = P(second / top site true)")
    lines.append(tab.round(3).to_string(index=False))
    # threshold rules (as tested) under every calibration
    lines.append("\nrule: r > th, union -> s2 only (and s1 only); E[B-A], sd, gain = E[max]-E[A], P(gain >= .003)")
    for th in (.25, .4, .5):
        sel = [i for i in range(len(pri)) if pri.r[i] > th]
        for uo in ("s2", "s1"):
            row = f"th {th} n {len(sel)} union {uo}: " + "  ".join(
                f"{mode} {simulate(pri, P[mode], sel, uo)[0]:+.5f}/{simulate(pri, P[mode], sel, uo)[1]:.5f}/"
                f"{simulate(pri, P[mode], sel, uo)[2]:+.5f}/{simulate(pri, P[mode], sel, uo)[3]:.2f}" for mode in P)
            lines.append(row)
    # subset choice on the SELECTION calibration (pairM), confirm on pairA and pooled
    best = []
    idx = list(range(len(pri)))
    for k in range(1, len(idx) + 1):
        for sel in itertools.combinations(idx, k):
            for uo in ("s2", "s1"):
                if uo == "s1" and not any(pri.union[i] for i in sel):
                    continue
                best.append((simulate(pri, P["pairM"], sel, uo)[2], sel, uo))
    best.sort(key=lambda z: -z[0])
    lines.append("\ntop subsets by pairM gain (selection), with march / pooled / pairA(confirm) gains:")
    for g, sel, uo in best[:8]:
        lines.append(f"{g:+.5f} {[pri.window_id[i] for i in sel]} union {uo} | march {simulate(pri, P['march'], sel, uo)[2]:+.5f} "
                     f"pooled {simulate(pri, P['pooled'], sel, uo)[2]:+.5f} pairA {simulate(pri, P['pairA'], sel, uo)[2]:+.5f}")
    # FINAL RULE (as tested): r > .4 on served probabilities, union -> second site only, minus windows whose second
    # site has zero online activity in the month's history while some site has activity (research kill rule).
    keep, killed, act = final_set(pri, .4)
    lines.append(f"\nFINAL set r > .4 minus zero-online-share second sites (killed {killed}): "
                 f"{[pri.window_id[i] for i in keep]}")
    for mode in P:
        e, sd, g, p3 = simulate(pri, P[mode], keep, "s2")
        lines.append(f"  {mode}: E[B-A] {e:+.5f} sd {sd:.5f} gain E[max]-E[A] {g:+.5f} P(gain>=.003) {p3:.2f}")
    fin = hedge_frame(pri, keep, act)
    fin.to_csv(O9 / "hedge_set.csv", index=False)
    txt = "\n".join(lines); print(txt)
    (O9 / "hedge_select.txt").write_text(txt)
    tab.to_csv(O9 / "hedge_private_calib.csv", index=False)


if __name__ == "__main__":
    main()
