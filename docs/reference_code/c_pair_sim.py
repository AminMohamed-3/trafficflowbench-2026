"""C pair 2026-10-12 (helper PB): what should file B contain? Monte Carlo of E[max(A, B)] - E[A] on private.

D = B - A on private S_total = onset bets (window level, discrete) + ongoing pieces (normal). Gain = E[D+].

ONSET (per private window with a hedge): truth category in {top only, second only, both, neither}.
  IoU model 'cells' (default): a = .93 for the right site, two-site truth or union decode pay by cell counts
  (n1, n2 = cells of the top / second site in the served decode); 'half' = the onset9 model (a/2), for parity only.
  Calibrations of the category probabilities (window level, pseudo windows artifacts/claude_ref/big/hedge_pseudo.parquet):
    bin    : March pseudo by r bin (HEDGE.md "march")                                   train + March
    dom    : March pseudo by r bin x orientation (is the top site the panel's train-dominant site, META n_windows),
             shrunk to the r-bin rate with m = 4 windows                               train + March
    pairM  : onset9 pair-specific March rate (m = 4)                                    train + March (cross-month persistence)
    binA / domA : the same with APRIL pseudo windows                                    APRIL-DERIVED: CONFIRM ONLY
  scale k: second-only probability x k (mass moved from / to top-only).
  boot: parameter uncertainty = bootstrap of the March pseudo windows, one replicate per draw (shared by all windows).
ONGOING pieces: X = mu + kappa * S_ind_private * Z (c_pair_data.py exposures; kappa = rms of board delta / S_ind over
  the 13 board-read ongoing pieces = spread of a real board read in units of independent fair-coin cells).
Scenarios: OPT (instruments at face value), CEN (board transfer), PES (March board read as mean, spread x 1.5).
Usage: PYTHONPATH=src python docs/reference_code/c_pair_sim.py [--n 400000]
Writes artifacts/claude_ref/c_pair/sim_*.csv and prints RESULT lines.
"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/reference_code")); sys.path.insert(0, str(ROOT / "src"))
import onset9_hedge as H
OUT = ROOT / "artifacts/claude_ref/c_pair"
A_IOU, W = H.A_IOU, H.W
META = json.loads((ROOT / "artifacts/lane_q/0267/Q03_seed0_model/metadata.json").read_text())
NW = {(pn, s["key"]): s.get("n_windows", 0) for pn, ss in META["sites"].items() for s in ss if not s.get("other")}
DOM = {}
for (pn, k), n in NW.items():
    if pn not in DOM or n > NW[(pn, DOM[pn])]:
        DOM[pn] = k
RB = [0, .1, .25, .5, 1.01]          # r bins of the orientation table (the two top onset9 bins merged: 8 + 10 windows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 40)


# ------------------------------------------------------------------ calibration tables
def pseudo_windows():
    w = H.calib()
    w["s1dom"] = [DOM.get(p) == s for p, s in zip(w.panel, w.s1)]
    w["to"] = w.top - w.both; w["so"] = w.sec - w.both; w["ne"] = (1 - w.top - w.sec + w.both).clip(lower=0)
    w["b2"] = np.digitize(w.r, RB) - 1
    return w


def norm4(t, s, b, n):
    v = np.clip(np.array([t, s, b, n], float), 0, None)
    return v / v.sum()


def probs(o, w, mode, k=1.):
    """[n windows, 4] = P(top only, second only, both, neither)."""
    pop = "pri" if mode.endswith("A") and mode != "pairA" else "val"
    out = []
    if mode in ("bin", "binA"):
        ww = w[w["pop"] == pop]; base = ww.groupby("b")[["to", "so", "both", "ne"]].mean()
        for x in o.itertuples():
            b = np.digitize([x.r], H.BINS)[0] - 1
            out.append(norm4(*base.loc[b]))
    elif mode in ("dom", "domA"):
        ww = w[w["pop"] == pop]; m = 4.
        base = ww.groupby("b2")[["to", "so", "both", "ne"]].mean()
        cell = ww.groupby(["b2", "s1dom"])[["to", "so", "both", "ne"]].agg(["mean", "size"])
        for x in o.itertuples():
            b = np.digitize([x.r], RB)[0] - 1; v = base.loc[b].to_numpy(float)
            d = DOM.get(x.panel) == x.s1
            if b >= 2 and (b, d) in cell.index:
                c = cell.loc[(b, d)]; n = c[("to", "size")]
                cm = np.array([c[(q, "mean")] for q in ("to", "so", "both", "ne")])
                v = (n * cm + m * v) / (n + m)
            out.append(norm4(*v))
    elif mode in ("pairM", "pairA"):
        for t, s, b, _ in H.window_probs(o, w, mode):
            out.append(norm4(t, s, b, max(1 - t - s - b, 0)))
    else:
        raise ValueError(mode)
    p = np.array(out)
    if k != 1.:
        ns = np.minimum(p[:, 1] * k, p[:, 0] + p[:, 1]); p[:, 0] += p[:, 1] - ns; p[:, 1] = ns
    return p


def boot_probs(o, w, mode, k, R, seed=11):
    """R bootstrap replicates of the March pseudo windows -> [R, n, 4]; a replicate with an empty bin falls back."""
    rng = np.random.default_rng(seed); base = probs(o, w, mode, k)
    wm = w[w["pop"] == "val"].reset_index(drop=True); rest = w[w["pop"] != "val"]
    P = np.empty((R,) + base.shape)
    for i in range(R):
        wb = pd.concat([wm.iloc[rng.integers(0, len(wm), len(wm))], rest])
        try:
            P[i] = probs(o, wb, mode, k)
        except (KeyError, IndexError):
            P[i] = base
    return P


# ------------------------------------------------------------------ payoffs
def payoff(x, opt, iou="cells"):
    """D of one hedged window by truth category [top only, second only, both, neither], in S_total."""
    n1 = len(str(x.b_s1only).split(";")) if isinstance(x.b_s1only, str) and x.b_s1only else len(x.a_cells.split(";"))
    if x.union:
        n2 = len(x.b_s2only.split(";"))
    else:
        n2 = len(x.b_swap.split(";"))
    f1, f2 = (n1 / (n1 + n2), n2 / (n1 + n2)) if iou == "cells" else (.5, .5)
    a = A_IOU
    if x.union:
        A = np.array([a * f1, a * f2, a, 0.])
        B = np.array([0., a, a * f2, 0.]) if opt == "s2" else np.array([a, 0., a * f1, 0.])
    else:
        A = np.array([a, 0., a * f1 if iou == "cells" else a / 2, 0.])
        B = np.array([0., a, a * f2 if iou == "cells" else a / 2, 0.])
    return W * (B - A), W * A


def draw_cats(P, n, seed):
    """P [R, nw, 4] (or [nw, 4]) -> categories [n, nw]; one replicate per draw."""
    rng = np.random.default_rng(seed)
    if P.ndim == 2:
        P = P[None]
    rep = rng.integers(0, P.shape[0], n)
    c = np.cumsum(P, axis=2)[rep]                       # [n, nw, 4]
    u = rng.random((n, P.shape[1], 1))
    return (u > c[:, :, :3]).sum(axis=2)


def stats(D):
    g = np.maximum(D, 0)
    return dict(gain=g.mean(), EBA=D.mean(), sd=D.std(), PBgtA=(D > 1e-9).mean(), p05=np.quantile(D, .05), p95=np.quantile(D, .95))


# ------------------------------------------------------------------ sets and rules
def cand(o, split):
    return o[(o.split == split)].sort_values("r", ascending=False).reset_index(drop=True)


def rule_sets(c):
    """name -> list of (position, union option). TH = onset9 family (r > th, kill rule, union -> second site only).
    DOM = bet on the panel's train-dominant site: union -> dominant site only; single-site decode with a non-dominant top
    and the dominant site second -> swap. Per window, uses only that window's served probabilities at T and META."""
    s = {}
    for th in (.25, .4, .5, .75):
        keep, _, _ = H.final_set(c, th)
        s[f"TH{th}"] = [(i, "s2") for i in keep]
        s[f"TH{th}_nounion"] = [(i, "s2") for i in keep if not c.union[i]]
        kd, _, _ = H.final_set(c, th); dm = []
        for i in kd:
            d = DOM.get(c.panel[i])
            if c.union[i] and d in (c.s1[i], c.s2[i]):
                dm.append((i, "s2" if c.s2[i] == d else "s1"))
            elif not c.union[i] and c.s2[i] == d:
                dm.append((i, "s2"))
        s[f"DOM{th}"] = dm
    return s


def gain_of(c, cats, sel, iou="cells"):
    D = np.zeros(len(cats))
    for i, opt in sel:
        d, _ = payoff(c.iloc[i], opt, iou); D += d[cats[:, i]]
    return D


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=400000); ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args(); N = a.n
    o = H.served(); w = pseudo_windows()
    pri, val = cand(o, "private"), cand(o, "validation")
    lines = []

    def say(s=""):
        print(s); lines.append(s)

    # ---- calibration tables (March = evidence; April = confirm only)
    for pop, tag in (("val", "MARCH pseudo (evidence)"), ("pri", "APRIL pseudo (CONFIRM ONLY)")):
        ww = w[w["pop"] == pop]
        say(f"calibration {tag}: r bin (RB .25/.5) x top site is the panel's train-dominant site")
        say(ww[ww.r > .25].groupby(["b2", "s1dom"]).agg(n=("r", "size"), top_only=("to", "mean"), sec_only=("so", "mean"), both=("both", "mean"),
                                                        neither=("ne", "mean")).round(3).to_string())
    # ---- per-window probabilities of the private candidates
    c8 = pri[pri.r > .25].reset_index(drop=True)
    tab = c8[["window_id", "s1", "s2", "r", "union"]].copy(); tab["s1dom"] = [DOM.get(p) == s for p, s in zip(c8.panel, c8.s1)]
    for mode in ("bin", "dom", "pairM", "binA", "domA", "pairA"):
        p = probs(c8, w, mode); tab[f"sec_{mode}"] = p[:, 1]; tab[f"top_{mode}"] = p[:, 0]
    say("\nprivate candidates r > .25: P(second only) / P(top only) by calibration (binA, domA, pairA = April, confirm only)")
    say(tab.round(2).to_string(index=False)); tab.to_csv(OUT / "sim_window_probs.csv", index=False, lineterminator="\n")

    # ---- parity with HEDGE.md (half model, bin calibration, exact numbers +.00146 / +.00149)
    sets = rule_sets(pri)
    for name in ("TH0.4", "TH0.75"):
        cats = draw_cats(probs(pri, w, "bin"), N, 1)
        g = stats(gain_of(pri, cats, sets[name], "half"))
        say(f"\nparity {name} (half model, bin): gain {g['gain']:+.5f} E[B-A] {g['EBA']:+.5f}  (HEDGE.md: TH.4 +.00146 / -.0022, TH.75 +.00149 / -.0005)")

    # ---- rule choice on MARCH official windows (train + March only), then the same rule on April windows
    rows = []
    vs, ps = rule_sets(val), rule_sets(pri)
    for mode in ("bin", "dom", "pairM"):
        cv = draw_cats(probs(val, w, mode), N, 2); cp = draw_cats(probs(pri, w, mode), N, 3)
        for name in vs:
            gv = stats(gain_of(val, cv, vs[name])); gp = stats(gain_of(pri, cp, ps[name]))
            rows.append(dict(calib=mode, rule=name, n_mar=len(vs[name]), gain_on_march=gv["gain"], EBA_march=gv["EBA"],
                             n_apr=len(ps[name]), gain_on_april_r=gp["gain"], EBA_april=gp["EBA"],
                             apr_windows=";".join(pri.window_id[i].replace("Q2_", "").replace("_private_", ":") + "/" + op for i, op in ps[name])))
    rt = pd.DataFrame(rows); rt.to_csv(OUT / "sim_rule_table.csv", index=False, lineterminator="\n")
    say("\nrule table: gain on the 40 MARCH official windows (selection evidence) and on the April windows by their own r (March calibrations)")
    say(rt.round(5).to_string(index=False))

    # ---- named B onset sets on private
    ix = {wid: i for i, wid in enumerate(pri.window_id)}
    U3, S1, S3, W5 = (ix["Q2_D12_I5_S_private_003"], ix["Q2_D7_I405_S_private_001"], ix["Q2_D7_I405_S_private_003"], ix["Q2_D7_I210_W_private_005"])
    named = {"none": [], "asbuilt4 (today)": [(U3, "s2"), (S1, "s2"), (S3, "s2"), (W5, "s2")], "march2 (TH.75)": [(U3, "s2"), (S1, "s2")],
             "D12_I5_S:003 only (DOM)": [(U3, "s2")], "D7_I405_S:001 only": [(S1, "s2")],
             "TH.25 (6)": ps["TH0.25"], "D12_I5_S:003 -> site_9": [(U3, "s1")]}
    # ---- stress: scale of the second-only probability, both March calibrations
    rows = []
    for mode in ("bin", "dom", "pairM"):
        for k in (.5, .75, 1., 1.25, 1.5):
            cats = draw_cats(probs(pri, w, mode, k), N, 5)
            for name, sel in named.items():
                if sel:
                    g = stats(gain_of(pri, cats, sel)); rows.append(dict(calib=mode, k=k, B=name, **g))
    st = pd.DataFrame(rows); st.to_csv(OUT / "sim_onset_stress.csv", index=False, lineterminator="\n")
    say("\nonset sets, gain E[max(A,B)] - E[A] by calibration and scale k of the second-only probability (cells model)")
    say(st.pivot_table(index=["calib", "B"], columns="k", values="gain", sort=False).round(5).to_string())
    say("\nonset sets at k = 1: E[B-A], P(B > A), 5th percentile of B - A")
    say(st[st.k == 1.].drop(columns=["k", "sd", "p95"]).round(5).to_string(index=False))
    # April confirm (never used to choose)
    rows = []
    for mode in ("binA", "domA", "pairA"):
        cats = draw_cats(probs(pri, w, mode), N, 6)
        for name, sel in named.items():
            if sel:
                rows.append(dict(calib=mode, B=name, **stats(gain_of(pri, cats, sel))))
    ca = pd.DataFrame(rows); ca.to_csv(OUT / "sim_onset_april_confirm.csv", index=False, lineterminator="\n")
    say("\nAPRIL-DERIVED calibrations (confirm only, not used for any choice): gain")
    say(ca.pivot_table(index="B", columns="calib", values="gain", sort=False).round(5).to_string())

    # ---- pair split of the union window: A' = site_42 only, B' = site_9 only (changes A on one private window)
    x = pri.iloc[U3]; n1 = len(x.b_s1only.split(";")); n2 = len(x.b_s2only.split(";")); f1, f2 = n1 / (n1 + n2), n2 / (n1 + n2)
    for mode in ("bin", "dom", "pairM"):
        p = probs(pri, w, mode)[U3]
        un = np.array([f1, f2, 1, 0]); s1o = np.array([1, 0, f1, 0]); s2o = np.array([0, 1, f2, 0])
        e = lambda v: float((p * v).sum()) * A_IOU * W
        say(f"\nunion window D12_I5_S:003 ({mode}: top-only {p[0]:.2f} second-only {p[1]:.2f} both {p[2]:.2f}): E[A=union] {e(un):.5f}; "
            f"pair (union, site_42) +{e(np.maximum(un, s2o)) - e(un):.5f}; pair (union, site_9) +{e(np.maximum(un, s1o)) - e(un):.5f}; "
            f"pair (site_42, site_9) +{e(np.maximum(s1o, s2o)) - e(un):.5f} with E[A'=site_42] - E[A] {e(s2o) - e(un):+.5f}")

    # ---- ongoing pieces
    ex = pd.read_csv(OUT / "expo_pieces.csv")
    pub = ex[ex.split == "validation"].set_index("piece"); prv = ex[ex.split == "private"].set_index("piece")
    z = (pub.board / pub.S_ind).dropna(); kappa = float(np.sqrt((z ** 2).mean()))
    chi = np.random.default_rng(0).chisquare(len(z), 20000); klo, khi = np.quantile(np.sqrt((z ** 2).sum() / chi), [.05, .95])
    say(f"\nongoing noise model: {len(z)} board-read ongoing pieces, z = board delta / S_ind public: mean {z.mean():+.2f}, rms kappa {kappa:.2f} "
        f"(90% interval {klo:.2f} to {khi:.2f}); pieces: " + ", ".join(f"{k} {v:+.2f}" for k, v in z.items()))
    # piece: (private S_ind, optimistic mu, central mu, pessimistic mu, family, source)
    ratio = lambda p: prv.M[p] / pub.M[p]
    PIECES = {
        "xsd5s":   (prv.S_ind["c_xsd5s"], +.00043, .5 * -.00022 * ratio("c_xsd5s"), -.00022 * ratio("c_xsd5s"), "cold"),
        "w35":     (prv.S_ind["c39_w35"], +.00005, .5 * -.00085 * ratio("c39_w35"), -.00085 * ratio("c39_w35"), "member"),
        "tinye24": (prv.S_ind["c40_tinye24"], +.00003, .5 * -.00055 * ratio("c40_tinye24"), -.00055 * ratio("c40_tinye24"), "member"),
        "k2bc":    (prv.S_ind["c24_k2bc"], +.00079, .5 * -.00140 * ratio("c24_k2bc"), -.00140 * ratio("c24_k2bc"), "k2"),
        "marpool": (prv.S_ind["c36_xsd"], +.00047, -.15 * .00047, -.5 * .00047, "cold"),       # exposure GUESSED = C36 XSD private
        "r13":     (prv.S_ind["c35_mh5"], +.00009, 0., -.5 * .00009, "r13"),                    # exposure GUESSED = mh5 private
    }
    pt = pd.DataFrame([dict(piece=k, S_ind_private=v[0], sd_cen=kappa * v[0], mu_opt=v[1], mu_cen=v[2], mu_pes=v[3], family=v[4]) for k, v in PIECES.items()])
    for sc, mu, sdm in (("opt", "mu_opt", 1.), ("cen", "mu_cen", 1.), ("pes", "mu_pes", 1.5)):
        s = pt.sd_cen * sdm; m = pt[mu]; zz = m / s
        from math import erf, sqrt, pi, exp
        pt[f"alone_{sc}"] = [si * exp(-.5 * zi * zi) / sqrt(2 * pi) + mi * .5 * (1 + erf(zi / sqrt(2))) for si, mi, zi in zip(s, m, zz)]
    say("\nongoing pieces on PRIVATE rows: spread (kappa x S_ind), mean by scenario, gain if the piece were ALONE in B")
    say(pt.round(5).to_string(index=False)); pt.to_csv(OUT / "sim_ongoing_pieces.csv", index=False, lineterminator="\n")

    def ongoing_draws(names, sc, seed, rho=.5):
        rng = np.random.default_rng(seed); X = np.zeros(N)
        fam = {}
        for nme in names:
            s0, mo, mc, mp, f = PIECES[nme]
            mu, sd = {"opt": (mo, kappa * s0), "cen": (mc, kappa * s0), "pes": (mp, 1.5 * kappa * s0)}[sc]
            if f not in fam:
                fam[f] = rng.standard_normal(N)
            X += mu + sd * (np.sqrt(rho) * fam[f] + np.sqrt(1 - rho) * rng.standard_normal(N))
        return X

    # ---- two-file compositions. A file option per window: dec (served decode), s1 (top site only), s2 (second site only)
    def iou_vec(x, opt):
        n1 = len(x.b_s1only.split(";")) if x.union else len(x.a_cells.split(";"))
        n2 = len(x.b_s2only.split(";")) if x.union else len(x.b_swap.split(";"))
        f1, f2 = n1 / (n1 + n2), n2 / (n1 + n2); q = A_IOU * W
        if opt == "dec":
            opt = "un" if x.union else "s1"
        return q * np.array({"s1": [1, 0, f1, 0], "s2": [0, 1, f2, 0], "un": [f1, f2, 1, 0]}[opt])

    def comp_D(cats, comp):
        """comp: list of (window position, option in file A', option in file B'). Returns A' - A and B' - A per draw."""
        DA = np.zeros(len(cats)); DB = np.zeros(len(cats))
        for i, oa, ob in comp:
            x = pri.iloc[i]; base = iou_vec(x, "dec")
            DA += (iou_vec(x, oa) - base)[cats[:, i]]; DB += (iou_vec(x, ob) - base)[cats[:, i]]
        return DA, DB

    def pstats(DA, DB):
        m = np.maximum(DA, DB); d = DB - DA
        return dict(gain=m.mean(), EA=DA.mean(), EBA=d.mean(), PBgtA=(d > 1e-9).mean(), p05=np.quantile(d, .05), p05max=np.quantile(m, .05))

    COMPS = {
        "asbuilt4 (today)":            [(U3, "dec", "s2"), (S1, "dec", "s2"), (S3, "dec", "s2"), (W5, "dec", "s2")],
        "march2 (TH.75)":              [(U3, "dec", "s2"), (S1, "dec", "s2")],
        "003 -> site_42 in B":         [(U3, "dec", "s2")],
        "I405S:001 swap in B":         [(S1, "dec", "s2")],
        "SPLIT 003 (A site_42, B site_9)":            [(U3, "s2", "s1")],
        "SPLIT 003 + I405S:001 swap in B":            [(U3, "s2", "s1"), (S1, "dec", "s2")],
        "SPLIT 003 + I405S:001 swap in A":            [(U3, "s2", "s1"), (S1, "s2", "dec")],
    }
    # stress calibrations for the pessimistic column (all train + March)
    def stress_P(kind):
        if kind == "dom_k.5":
            return boot_probs(pri, w, "dom", .5, a.boot)
        if kind == "pairM":
            return probs(pri, w, "pairM")
        P = boot_probs(pri, w, "dom", 1., a.boot).copy()               # both + .20, taken pro rata from the single-site cells
        take = np.minimum(.20, P[:, :, 0] + P[:, :, 1]); tot = np.maximum(P[:, :, 0] + P[:, :, 1], 1e-9)
        P[:, :, 2] += take; P[:, :, 0] -= take * P[:, :, 0] / tot; P[:, :, 1] -= take * P[:, :, 1] / tot
        return P
    CAL = {"opt": [("bin", probs(pri, w, "bin"))], "cen": [("dom boot", boot_probs(pri, w, "dom", 1., a.boot))],
           "pes": [(kd, stress_P(kd)) for kd in ("dom_k.5", "pairM", "dom_both+.2")]}
    combos = [("none", ())] + [(p_, (p_,)) for p_ in PIECES] + [("xsd5s+marpool", ("xsd5s", "marpool")), ("w35+tinye24", ("w35", "tinye24")),
                                                                 ("all 6", tuple(PIECES))]
    rows = []
    for sc, cals in CAL.items():
        for cname_, P in cals:
            cats = draw_cats(P, N, 21)
            for bname, comp in COMPS.items():
                DA, DB = comp_D(cats, comp)
                for oname, cp in combos:
                    X = ongoing_draws(cp, sc, 31) if cp else 0.
                    rows.append(dict(scenario=sc, calib=cname_, onset=bname, ongoing=oname, **pstats(DA, DB + X)))
            for oname, cp in combos[1:]:
                X = ongoing_draws(cp, sc, 31); rows.append(dict(scenario=sc, calib=cname_, onset="no onset bet", ongoing=oname, **pstats(np.zeros(N), X)))
    r = pd.DataFrame(rows); r.to_csv(OUT / "sim_scenarios_all.csv", index=False, lineterminator="\n")
    say("\npessimistic column, by stress calibration (onset only): gain")
    say(r[(r.scenario == "pes") & (r.ongoing == "none")].pivot_table(index="onset", columns="calib", values="gain", sort=False).round(5).to_string())
    # pessimistic = the worst stress calibration for that composition
    worst = r.loc[r.groupby(["scenario", "onset", "ongoing"], sort=False).gain.idxmin()].reset_index(drop=True)
    worst.to_csv(OUT / "sim_scenarios.csv", index=False, lineterminator="\n")
    say("\nSCENARIOS. gain = E[max(A', B')] - E[A today]. opt: onset r-bin March calibration, ongoing instruments at face value; cen: onset "
        "orientation calibration with bootstrap, ongoing mean = half the March read (exposure scaled); pes: worst of 3 stress calibrations, "
        "ongoing mean = March read, spread x 1.5")
    t = worst[worst.ongoing == "none"].pivot_table(index="onset", columns="scenario", values=["gain", "EA", "EBA", "PBgtA", "p05"], sort=False)
    say(t.round(5).to_string())
    say("\ngain by onset composition (rows) and ongoing addition to B (columns), per scenario")
    for sc in CAL:
        say(f"-- {sc}")
        say(worst[worst.scenario == sc].pivot_table(index="onset", columns="ongoing", values="gain", sort=False).round(5).to_string())
    for bname in COMPS:
        g = worst[(worst.onset == bname) & (worst.ongoing == "none")].set_index("scenario")
        say(f"RESULT pair [{bname}]: E[max]-E[A] opt {g.gain['opt']:+.5f} cen {g.gain['cen']:+.5f} pes {g.gain['pes']:+.5f}; "
            f"E[A'-A] {g.EA['opt']:+.5f}/{g.EA['cen']:+.5f}/{g.EA['pes']:+.5f}; P(B>A') {g.PBgtA['opt']:.2f}/{g.PBgtA['cen']:.2f}/{g.PBgtA['pes']:.2f}; "
            f"E[B-A'] {g.EBA['opt']:+.5f}/{g.EBA['cen']:+.5f}/{g.EBA['pes']:+.5f}; p05(B-A') {g.p05['opt']:+.5f}/{g.p05['cen']:+.5f}/{g.p05['pes']:+.5f}")

    # ---- exhaustive two-file search on the 4 as-built windows (exact, independent windows, point calibrations)
    wins = [U3, S1, S3, W5]; opts = [["dec", "s1", "s2"] if pri.union[i] else ["s1", "s2"] for i in wins]
    files = list(itertools.product(*opts)); states = list(itertools.product(range(4), repeat=len(wins)))
    V = np.array([[sum(iou_vec(pri.iloc[i], o)[st[j]] for j, (i, o) in enumerate(zip(wins, f))) for st in states] for f in files])
    base = files.index(tuple("dec" if pri.union[i] else "s1" for i in wins))
    say("\nexhaustive search, two files over the 4 as-built windows (options per window: decode / top site / second site), exact:")
    for mode in ("bin", "dom", "pairM", "domA"):
        P = probs(pri, w, mode)[wins]; ps_ = np.array([np.prod([P[j, st[j]] for j in range(len(wins))]) for st in states])
        best = []
        for ia in range(len(files)):
            for ib in range(ia, len(files)):
                best.append((float((np.maximum(V[ia], V[ib]) * ps_).sum() - (V[base] * ps_).sum()), ia, ib))
        best.sort(reverse=True)
        lab = lambda f: ",".join(f"{pri.window_id[i][3:].replace('_private_', ':')}={o}" for i, o in zip(wins, f) if o != ("dec" if pri.union[i] else "s1")) or "A today"
        say(f"  {mode}{' (APRIL, confirm only)' if mode.endswith('A') else ''}: " + " | ".join(f"{g:+.5f} [{lab(files[ia])}] + [{lab(files[ib])}]" for g, ia, ib in best[:4]))
    (OUT / "sim_out.txt").write_text("\n".join(lines), newline="\n")


if __name__ == "__main__":
    main()
