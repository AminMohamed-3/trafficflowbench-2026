"""K6 helper 2026-10-11 (ledger 0801, exp/C_b6long/B6LONG.md section 6): I-S-OWN read of a B6 long FULL configuration.

Candidate set = OWN-view imputer files imp_<own|ownapr>_<panel>_s{0,1,2}.npz of a configuration (it12000_w32, it24000_w32)
found under artifacts/claude_ref/c_b6long/kaggle_full/<kernel>/b6long/<cfg>/ or artifacts/claude_ref/c_b6long/full/<cfg>/.
Reference = the shipped OWN members (3 seeds) = c_knobs' base result (artifacts/claude_ref/c_knobs/<stag>/base_<pn>.npz).
Chain = c_knobs.run, knob set 'base', with c_knobs._imp and c_knobs.ship_weights replaced in this process (as
c_b6long_score.py). Weights: A = shipped cross-parity weights; R = refit with the candidate imputers, same month,
other-day parity; M (April) = refit on March OWN all days with the candidate imputers, applied to April.
J(mix) = .35 S_state(mix) + .15 S_phys, all 10 panels, family macro, joint date bootstrap 1000 (seed 4701).
A panel counts for a configuration when all 3 seeds exist in both months; panels not there yet stay at the shipped
members (partial lines = information only). Hybrid h24_12 = it24000_w32 where the panel is there, else it12000_w32.
Usage (repo root):
  python c_b6long_fullscore.py auto        score whatever changed since the last call; RESULT lines -> B6LONG.md
  python c_b6long_fullscore.py status      which panels are there per configuration
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "KNOB_THREADS"):
    os.environ.setdefault(k, "2")
import sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_knobs as CK
import state5_lib as L
import b6full_eval as EV
import flow_screen as FS

C5 = CK.C5; ROOT = Path(L.ROOT); CR = ROOT / "artifacts/claude_ref"
BASE = CR / "c_b6long"; OUT = BASE / "own_full"; KN = CR / "c_knobs"; EXP = ROOT / "exp/C_b6long"
if os.environ.get("B6LONG_TEST"):       # dry run on stand-in files: own dirs, never the real tables or B6LONG.md
    BASE = Path(os.environ["B6LONG_TEST"]); OUT = BASE / "own_full"; EXP = BASE / "exp"; EXP.mkdir(parents=True, exist_ok=True)
STAG = {"validation": "mar", "private": "apr"}; PERIOD = {v: k for k, v in STAG.items()}
TAG = {"validation": "own", "private": "ownapr"}
CFGS = ("it12000_w32", "it24000_w32")
SCREEN = ("it24000_w48", "it24000_w32", "D12_I5_S", (0, 1))   # extra laptop arm: new cfg, reference cfg, panel, seeds
_IMP0 = CK._imp
f6 = lambda x: "%+.6f" % x
now = lambda: time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())


def dirs(cfg):
    return sorted((BASE / "kaggle_full").glob(f"*/b6long/{cfg}")) + [BASE / "full" / cfg]


def files(cfg, period, pn, seeds=(0, 1, 2)):
    out = []
    for s in seeds:
        hit = [d / f"imp_{TAG[period]}_{pn}_s{s}.npz" for d in dirs(cfg) if (d / f"imp_{TAG[period]}_{pn}_s{s}.npz").exists() and (d / f"run_{TAG[period]}_{pn}_s{s}.json").exists()]
        if not hit:
            return None
        out.append(hit[0])
    return out


def available(cfg, seeds=(0, 1, 2)):
    return [pn for pn in L.PANELS if all(files(cfg, p, pn, seeds) for p in STAG)]


def imputer(cfg, period, pn, seeds=(0, 1, 2)):
    zs = [np.load(f, allow_pickle=False) for f in files(cfg, period, pn, seeds)]
    return np.mean([z["speed"] for z in zs], 0).astype(float), np.mean([z["flow"] for z in zs], 0).astype(float), len(zs)


def w_fixed(period):
    return {int(k): v for k, v in json.loads((KN / f"W_{STAG[period]}.json").read_text()).items()}


def panel_stats(period, pn, imp, arm="s"):
    """SSE grids of the blend per day parity (as c_knobs.ship_weights / pass 1 of c_budget_s2chain.py).
    arm 's' = the chain's ship-arm c11 (sv, sq: stage 2 3M on the shipped S08 0398; what c_knobs fits its weights on),
    'n' = the new-arm c11 (nv, nq: the same on S08 l127_r4800 = the arrays the chain actually blends)."""
    from tfb.data import load_cache
    stag = STAG[period]
    r = dict(np.load(C5.OUT / f"own_{stag}_s08_{pn}.npz")); z = np.load(CK.CH / f"c11_{stag}_{CK.TAGF}_{pn}.npz")
    cv, cq = z[arm + "v"].astype(float), z[arm + "q"].astype(float)
    tv, tq, st_, gap = r["tv"].astype(float), r["tq"].astype(float), r["target"], r["gap"]
    iv, iq, _ = imp
    stn = load_cache(pn, period).stations
    la = np.broadcast_to(np.maximum(stn.sort_values("idx").lanes.to_numpy(float), 1.), tv.shape)
    day = np.broadcast_to(np.arange(tv.shape[0])[:, None, None], tv.shape)
    ok = st_ & np.isfinite(tv) & np.isfinite(tq); out = {}
    for par in (0, 1):
        m = ok & (day % 2 == par)
        out[par] = dict(v_nb=EV.sse_grid(cv, iv, tv, m & ~gap), v_blk=EV.sse_grid(cv, iv, tv, m & gap),
                        q_nb=EV.sse_grid(cq, iq, tq, m & ~gap, la), q_blk=EV.sse_grid(cq, iq, tq, m & gap, la))
    rm = lambda a, b, m, s=1.: float(np.sqrt(np.mean(((a - b) / s)[m] ** 2)))
    return out, dict(rmse_v_nb=rm(iv, tv, ok & ~gap), rmse_v_blk=rm(iv, tv, ok & gap), rmse_ql_nb=rm(iq, tq, ok & ~gap, la))


def _stats_io(f, compute):
    if f.exists():
        j = json.loads(f.read_text())
        return {int(par): {pn: {k: np.asarray(v) for k, v in d.items()} for pn, d in pp.items()} for par, pp in j.items()}
    st = compute(); f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({par: {pn: {k: v.tolist() for k, v in d.items()} for pn, d in pp.items()} for par, pp in st.items()}))
    return st


def ship_stats(period):
    """All 10 panels with the shipped 3-seed imputers; the fit must reproduce c_knobs' W exactly."""
    def compute():
        imp = _IMP0(period); st = {0: {}, 1: {}}
        for pn in L.PANELS:
            s, _ = panel_stats(period, pn, imp(pn))
            for par in (0, 1):
                st[par][pn] = s[par]
        W = {par: EV.fit_weights(st[par])[0] for par in (0, 1)}
        assert json.dumps(W, sort_keys=True) == json.dumps(w_fixed(period), sort_keys=True), "shipped stats do not reproduce c_knobs W"
        return st
    return _stats_io(OUT / f"stats_ship_{STAG[period]}.json", compute)


def cand_stats(cfg, period):
    """Shipped stats with every panel of the (complete) configuration replaced by the candidate imputers."""
    def compute():
        st = {par: dict(d) for par, d in ship_stats(period).items()}
        for pn in available(cfg):
            s, _ = panel_stats(period, pn, imputer(cfg, period, pn))
            for par in (0, 1):
                st[par][pn] = s[par]
        return st
    return _stats_io(OUT / f"stats_{cfg}_{STAG[period]}.json", compute)


def weights(mode, cfg, period):
    if mode == "A":
        return w_fixed(period)
    if mode == "R":
        st = cand_stats(cfg, period)
        return {par: EV.fit_weights(st[par])[0] for par in (0, 1)}
    st = cand_stats(cfg, "validation")      # M: March all days, used unchanged on the scored month (April)
    allw = EV.fit_weights({pn: {k: st[0][pn][k] + st[1][pn][k] for k in st[0][pn]} for pn in st[0]})[0]
    return {0: allw, 1: allw}


def run_chain(dest, period, panels, imp_of, W):
    """c_knobs.run for the panels still missing under dest/<stag>/."""
    stag = STAG[period]; todo = [pn for pn in panels if not (dest / stag / f"base_{pn}.npz").exists()]
    if not todo:
        return
    orig = (CK.OUT, CK._imp, CK.ship_weights)
    try:
        (dest / stag).mkdir(parents=True, exist_ok=True)
        CK.OUT = dest; CK._imp = lambda p: imp_of; CK.ship_weights = lambda p: W
        CK.run(period, "base", ",".join(todo))
    finally:
        CK.OUT, CK._imp, CK.ship_weights = orig
    (dest / stag / "weights.json").write_text(json.dumps({par: {pn: W[par][pn] for pn in panels} for par in W}, indent=1))


def run_cfg(cfg, mode, panels):
    for period in STAG:
        if mode == "M" and period == "validation":
            continue
        run_chain(OUT / mode / cfg, period, panels, lambda pn, _p=period: imputer(cfg, _p, pn), weights(mode, cfg, period))


def _load(d, stag, pn):
    z = np.load(d / stag / f"base_{pn}.npz")
    return z["st"], z["comps"], [str(x) for x in z["regs"]]


def compare(src, stag, base=None):
    """src: panel -> result dir of the candidate (panels not listed = shipped); base: the same for the reference
    (default: shipped everywhere). 10-panel J(mix) gain of src over base + per panel."""
    period = PERIOD[stag]; sh = L.share_array(period); base = base or {}
    ship = {pn: _load(KN, stag, pn) for pn in L.PANELS}
    ref = {pn: (_load(base[pn], stag, pn) if pn in base else ship[pn]) for pn in L.PANELS}
    new = {pn: (_load(src[pn], stag, pn) if pn in src else ship[pn]) for pn in L.PANELS}
    A = np.stack([ref[p][0] for p in L.PANELS]); B = np.stack([new[p][0] for p in L.PANELS]); D = A.shape[2]
    cb = {p: ref[p][1] for p in L.PANELS}; cn = {p: new[p][1] for p in L.PANELS}; regs = {p: ref[p][2] for p in L.PANELS}
    rng = np.random.default_rng(4701); W = [np.ones(D)] + [np.bincount(rng.integers(0, D, D), minlength=D).astype(float) for _ in range(1000)]
    x = np.array([[L.reduce(B, w, "mix", sh) - L.reduce(A, w, "mix", sh), CK._phys(cn, cb, regs, w)["S_phys"]] for w in W])
    J = .35 * x[:, 0] + .15 * x[:, 1]
    res = dict(J=float(J[0]), ci=[float(np.quantile(J[1:], .025)), float(np.quantile(J[1:], .975))], state_mix=float(x[0, 0]), phys=float(x[0, 1]), per_panel={})
    g0 = L.GROUPS; L.GROUPS = [[0]]
    try:
        for i, pn in enumerate(L.PANELS):
            if pn in src:
                one = np.ones(D)
                dj = (.35 * (L.reduce(B[i:i + 1], one, "mix", sh[i:i + 1]) - L.reduce(A[i:i + 1], one, "mix", sh[i:i + 1]))
                      + .15 * (FS.aggregate(cn[pn], regs[pn], one)["S_phys"] - FS.aggregate(cb[pn], regs[pn], one)["S_phys"]))
                res["per_panel"][pn] = float(dj)
    finally:
        L.GROUPS = g0
    res["n_pos"] = int(sum(v > 0 for v in res["per_panel"].values()))
    return res


def fmt(r):
    return {st: f"{f6(r[st]['J'])} [{f6(r[st]['ci'][0])}, {f6(r[st]['ci'][1])}]" for st in r}


def append(line):
    print(line, flush=True)
    with open(EXP / "B6LONG.md", "a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")


def table(title, r):
    rows = [f"### {title} ({now()})", "", "| panel | March dJ (panel alone) | April dJ (panel alone) |", "|---|---|---|"]
    for pn in L.PANELS:
        if pn in r["mar"]["per_panel"]:
            rows.append(f"| {pn} | {f6(r['mar']['per_panel'][pn])} | {f6(r['apr']['per_panel'][pn])} |")
    rows.append(f"| ALL 10 (missing = shipped) | {fmt(r)['mar']} state {f6(r['mar']['state_mix'])} phys {f6(r['mar']['phys'])} | {fmt(r)['apr']} state {f6(r['apr']['state_mix'])} phys {f6(r['apr']['phys'])} |")
    with open(EXP / "full_table.md", "a", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n\n")


def score_cfg(cfg, panels):
    run_cfg(cfg, "A", panels)
    src = {pn: OUT / "A" / cfg for pn in panels}
    rA = {st: compare(src, st) for st in ("mar", "apr")}
    n = len(panels); full = n == len(L.PANELS); pos = lambda r: r["J"] > 0 and r["ci"][0] > 0
    line = (f"RESULT 0801 {now()} B6 long FULL {cfg} vs shipped B6, {n}/10 panels{'' if full else ' (PARTIAL, information only; rest = shipped)'}: "
            f"weights A (shipped, fixed) J(mix) March {fmt(rA)['mar']} April {fmt(rA)['apr']}, panels up {rA['mar']['n_pos']}/{n} and {rA['apr']['n_pos']}/{n}")
    out = dict(cfg=cfg, panels=panels, A=rA)
    if full:
        run_cfg(cfg, "R", panels); run_cfg(cfg, "M", panels)
        rR = {st: compare({pn: OUT / "R" / cfg for pn in panels}, st) for st in ("mar", "apr")}
        rM = {"apr": compare({pn: OUT / "M" / cfg for pn in panels}, "apr")}
        adopt = pos(rA["mar"]) and pos(rA["apr"])
        refit = rR["mar"]["J"] >= rA["mar"]["J"] and rM["apr"]["J"] >= rA["apr"]["J"]
        line += (f"; weights R (refit, same month cross-parity) March {fmt(rR)['mar']} April {fmt(rR)['apr']}; weights M (refit on March, applied to April) April {fmt(rM)['apr']}"
                 f"; ADOPT rule (A, CI > 0 both months): {'PASS' if adopt else 'FAIL'}; build weights by the rule: {'refit (March R >= A and April M >= A)' if refit else 'keep shipped'}")
        out.update(R=rR, M=rM, adopt=adopt, refit=refit)
    line += ". Per panel: exp/C_b6long/full_table.md"
    table(f"{cfg} vs shipped, weights A, {n}/10 panels", rA)
    append(line)
    return out


def score_hybrid(p24, p12):
    src = {pn: OUT / "A" / CFGS[0] for pn in p12}; src.update({pn: OUT / "A" / CFGS[1] for pn in p24})
    r = {st: compare(src, st) for st in ("mar", "apr")}
    d = {st: compare(src, st, base={pn: OUT / "A" / CFGS[0] for pn in p12}) for st in ("mar", "apr")}   # 24k vs 12k, same panels
    table(f"hybrid h24_12 (24k on {len(p24)} panels, 12k on {len(set(p12) - set(p24))}) vs shipped, weights A", r)
    append(f"RESULT 0801 {now()} B6 long FULL hybrid h24_12 vs shipped B6 (24k on {len(p24)} panels: {','.join(p24)}; 12k on {len(set(p12) - set(p24))}; weights A): "
           f"J(mix) March {fmt(r)['mar']} April {fmt(r)['apr']}; 24k over 12k on those panels: March {fmt(d)['mar']} April {fmt(d)['apr']}, panels up {d['mar']['n_pos']}/{len(p24)} and {d['apr']['n_pos']}/{len(p24)}"
           f" (panel alone, March/April: {', '.join(pn + ' ' + f6(d['mar']['per_panel'][pn]) + '/' + f6(d['apr']['per_panel'][pn]) for pn in p24)})"
           f"{' (PARTIAL, information only)' if len(p24) < len(L.PANELS) else '; 24k replaces 12k by the rule only if both are > 0'}. Per panel: exp/C_b6long/full_table.md")
    return r


W48, W32 = "it12000_w48", "it12000_w32"


def score_w48(panels):
    """Task 3 (B6LONG.md section 8): 12k width 48 vs 12k width 32 (the adopted set). Panels not at width 48 yet = width 32 in
    both arms. Fixed form = shipped weights A in both arms; build form (complete only) = each arm with its own refit weights."""
    run_cfg(W48, "A", panels)
    allp = list(L.PANELS); n = len(panels); full = n == len(allp)
    mix = lambda mode: {pn: OUT / mode / (W48 if pn in panels else W32) for pn in allp}
    b32 = lambda mode: {pn: OUT / mode / W32 for pn in allp}
    dA = {st: compare(mix("A"), st, base=b32("A")) for st in ("mar", "apr")}
    pos = lambda r: r["J"] > 0 and r["ci"][0] > 0
    line = (f"RESULT 0801 {now()} B6 12k WIDTH 48 vs 12k width 32, {n}/10 panels{'' if full else ' (PARTIAL, information only; rest = width 32 in both arms)'}: "
            f"fixed form (shipped weights A, both arms) J(mix) March {fmt(dA)['mar']} April {fmt(dA)['apr']}, panels up {dA['mar']['n_pos']}/{n} and {dA['apr']['n_pos']}/{n}")
    out = dict(panels=panels, A=dA)
    if full:
        run_cfg(W48, "R", panels); run_cfg(W48, "M", panels)
        dB = {"mar": compare(mix("R"), "mar", base=b32("R")), "apr": compare(mix("M"), "apr", base=b32("M"))}
        sB = {"mar": compare(mix("R"), "mar"), "apr": compare(mix("M"), "apr")}
        adopt = pos(dB["mar"]) and pos(dB["apr"]); wonly = adopt and min(dA["mar"]["J"], dA["apr"]["J"]) <= 0
        line += (f"; build form (own refit weights, March R, April M) March {fmt(dB)['mar']} April {fmt(dB)['apr']}, panels up {dB['mar']['n_pos']}/{n} and {dB['apr']['n_pos']}/{n}"
                 f"; width 48 vs SHIPPED B6 in build form March {fmt(sB)['mar']} April {fmt(sB)['apr']}"
                 f"; ADOPT rule (build form, CI > 0 both months): {'PASS' if adopt else 'FAIL'}{' BUT weights only (fixed form <= 0 in a month): not adopted without C' if wonly else ''}")
        out.update(build=dB, vs_shipped_build=sB, adopt=adopt, weights_only=wonly)
        table("12k width 48 vs 12k width 32, build form (own refit weights), 10/10 panels", dB)
    table(f"12k width 48 vs 12k width 32, fixed form (weights A), {n}/10 panels", dA)
    append(line + ". Per panel: exp/C_b6long/full_table.md")
    (OUT / "w48_score.json").write_text(json.dumps(out, indent=1))
    return out


def score_screen():
    """Extra laptop arm: 24k width 48 vs 24k width 32, one panel, 2 seeds, weights A, J of that panel (as the first screen)."""
    new, refc, pn, seeds = SCREEN
    res = {}
    for cfg in (new, refc):
        for period in STAG:
            run_chain(OUT / "A" / f"{cfg}_s{''.join(map(str, seeds))}", period, [pn], lambda p_, _c=cfg, _p=period: imputer(_c, _p, p_, seeds), w_fixed(period))
    for st in ("mar", "apr"):
        period = PERIOD[st]; i = L.PANELS.index(pn); sh = L.share_array(period)[i:i + 1]
        a = _load(OUT / "A" / f"{refc}_s01", st, pn); b = _load(OUT / "A" / f"{new}_s01", st, pn); D = a[0].shape[1]
        rng = np.random.default_rng(4701); W = [np.ones(D)] + [np.bincount(rng.integers(0, D, D), minlength=D).astype(float) for _ in range(1000)]
        g0 = L.GROUPS; L.GROUPS = [[0]]
        try:
            J = np.array([.35 * (L.reduce(b[0][None], w, "mix", sh) - L.reduce(a[0][None], w, "mix", sh))
                          + .15 * (FS.aggregate(b[1], b[2], w)["S_phys"] - FS.aggregate(a[1], a[2], w)["S_phys"]) for w in W])
        finally:
            L.GROUPS = g0
        res[st] = dict(J=float(J[0]), ci=[float(np.quantile(J[1:], .025)), float(np.quantile(J[1:], .975))])
    append(f"RESULT 0801 {now()} B6 screen arm 24k width 48 vs 24k width 32 ({pn} alone, seeds 0,1, weights A; shipped 2-seed noise .000056 / .000080): "
           f"J(mix) March {fmt(res)['mar']} April {fmt(res)['apr']}")
    return res


def auto():
    sf = OUT / "state.json"; OUT.mkdir(parents=True, exist_ok=True)
    state = json.loads(sf.read_text()) if sf.exists() else {}
    av = {cfg: available(cfg) for cfg in CFGS}; changed = False
    for cfg in CFGS:
        if av[cfg] and av[cfg] != state.get(cfg):
            score_cfg(cfg, av[cfg]); state[cfg] = av[cfg]; changed = True
            sf.write_text(json.dumps(state))
    if av[CFGS[1]] and av[CFGS[0]] and [av[CFGS[1]], av[CFGS[0]]] != state.get("hybrid"):
        score_hybrid(av[CFGS[1]], av[CFGS[0]]); state["hybrid"] = [av[CFGS[1]], av[CFGS[0]]]; changed = True
        sf.write_text(json.dumps(state))
    a48 = available(W48)
    if a48 and len(av[W32]) == len(L.PANELS) and a48 != state.get("w48"):
        score_w48(a48); state["w48"] = a48; changed = True
        sf.write_text(json.dumps(state))
    new, refc, pn, seeds = SCREEN
    if not state.get("screen") and all(files(c, p, pn, seeds) for c in (new, refc) for p in STAG):
        score_screen(); state["screen"] = True; changed = True
        sf.write_text(json.dumps(state))
    print("AUTO", "changed" if changed else "nothing new", {c: len(v) for c, v in av.items()}, {W48: len(a48)}, flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "auto":
        auto()
    elif a[0] == "status":
        print({cfg: available(cfg) for cfg in CFGS}, "screen arm files:", bool(all(files(c, p, SCREEN[2], SCREEN[3]) for c in SCREEN[:2] for p in STAG)))
