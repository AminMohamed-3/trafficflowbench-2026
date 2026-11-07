"""F14 data: the joint date-bootstrap draws of the last state pieces on their own instruments (worker F2, 2026-10-21).

The instruments kept only the 95% interval in their JSON outputs, so the draws are regenerated here with the SAME code path, the SAME
stored prediction arms and the SAME seed (4701, 500 draws, joint over dates): every part asserts that its point value and its 2.5% /
97.5% quantiles equal the stored JSON (parity CSV). Read-only on artifacts/; writes only into figures/out/.

Parts (one subprocess each, so the gap bench RSS guard of 1.9 GB and the I-S-INEL scorer never share a process):
  isi  l_isinel           s2b->fv (FV-FT), fv->fv2 (FV2), fv2->s7c_x (S7c-INEL)     I-S-INEL  (exp/L_isinel/isi_score.py)
  isi  l_inelfold/isi     fv2->fold (FV-INEL W1.0, held-out folds)                  INEL-FOLD (exp/L_inelfold/ifo_chain.py score)
  isi  l_isinel/s7cf/isi  fold->s7c_fv2x (shipped S7c recipe on the INEL-FOLD chain) S7C-FOLD  (exp/L_isinel/s7cf.py score)
  gap  l_gap3             S -> B0 (gap edge taper + unsmooth, ADOPTED row 29)          gap bench state part (exp/L_gap3/gap3_bench.py)
Value per draw: I-S-INEL family = fc = .88 (.35 dState + .45 dPhys); gap bench = .308 dState (its board rule). Both are S_total units.

Usage (repo root): .venv_tfb/Scripts/python.exe final_solution/report/figures/f14_draws.py all
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "3")
import sys, json, subprocess, importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = Path(os.environ["F14_OUT"]) if os.environ.get("F14_OUT") else HERE / "out"   # F14_OUT: determinism checks only
PARTS = OUT / "f14_parts"
CR = ROOT / "artifacts/claude_ref"
SPLITS = {"validation": "March", "private": "April"}
NB = 500

ISI_PARTS = {   # name -> (root, pairs, stored JSON per split)
    "isinel": (CR / "l_isinel", [("s2b", "fv"), ("fv", "fv2"), ("fv2", "s7c_x")],
               lambda sp: [CR / f"l_isinel/score_inst_{sp}.json", CR / f"l_isinel/score_s7c_{sp}.json"]),
    "inelfold": (CR / "l_inelfold/isi", [("fv2", "fold")], lambda sp: [CR / f"l_inelfold/score_fold_{sp}.json"]),
    "s7cf": (CR / "l_isinel/s7cf/isi", [("fold", "s7c_fv2x")], lambda sp: [CR / f"l_isinel/s7cf/score_s7cf_{sp}.json"]),
}
PIECE = {  # pair -> (piece label, ADOPTED row, instrument)
    "s2b->fv": ("FV-FT", "28-29", "I-S-INEL"),
    "fv->fv2": ("FV2", "30", "I-S-INEL"),
    "fv2->fold": ("FV-INEL W1.0 (held-out folds)", "31+34", "INEL-FOLD"),
    "fv2->s7c_x": ("S7c-INEL", "35", "I-S-INEL"),
    "fold->s7c_fv2x": ("S7c-INEL on the INEL-FOLD chain", "35", "S7C-FOLD"),
    "S->B0": ("gap edge taper + unsmooth", "29", "gap bench"),
}


def _q(x):
    return float(np.quantile(x, .025)), float(np.quantile(x, .975))


# ------------------------------------------------------------------------------------------ I-S-INEL family
def isi_part(name, split):
    """exp/L_isinel/isi_score.main, unchanged arithmetic, returning the draws instead of only the quantiles."""
    root, pairs, stored = ISI_PARTS[name]
    sys.path.insert(0, str(ROOT / "exp/L_isinel"))
    import types
    import isi_lib as IL
    IL.OUT = root   # the chain arms of this instrument (as ifo_chain._isi / s7cf._isi set it)
    import isi_score as SC
    assert SC.IL is IL
    L, FS, DL = SC.L, SC.FS, SC.DL
    from tfb.data import load_cache
    arms = sorted({a for p in pairs for a in p})
    ST = {a: {} for a in arms}; CP = {a: {} for a in arms}; regs = {}; E = {a: {} for a in arms}
    for pn in IL.PANELS:
        c2, inel, tv, tq = IL.inel_cache(pn, split)
        c0 = load_cache(pn, split)
        la, _, _, _ = L.lanes_of(pn, split)
        gap = np.asarray(c2.gap, bool)
        z = dict(tv=tv, tq=tq, target=inel, gap=np.zeros_like(inel), regimes=np.asarray(c2.regimes))
        h = types.SimpleNamespace(path=c2.path, meta=c2.meta, stations=c2.stations, ramps=c2.ramps, **c2.arrays)
        h.target = np.asarray(c2.target, bool); h.gap = gap; h.regimes = [str(x) for x in c2.regimes]; h.dates = list(c2.dates)
        hv = DL.proxy_view(h)
        tvp = np.asarray(c0.speed, float); tqp = np.asarray(c0.flow, float)
        lan = la[None, None, :]
        for a in arms:
            v, q = SC.grids(split, a, pn, inel.shape)
            assert np.isfinite(v[inel]).all() and np.isfinite(q[inel]).all()
            ST[a][pn] = L.stats(pn, z, v, q, la)
            CP[a][pn], regs[pn] = FS.day_components(hv, tvp, tqp, v, q)
            e = .54 * ((v - tv) / 25) ** 2 + .46 * ((q - tq) / lan / 600) ** 2
            E[a][pn] = np.array([np.nansum(np.where(inel[d], e[d], np.nan)) for d in range(inel.shape[0])]), inel.sum((1, 2))
            del v, q, e
        del c2, c0, hv, h, tv, tq, tvp, tqp
        print("F14 scored", name, split, pn, flush=True)
    sh = L.share_array(split); fam = L.P.P["families"]
    PU = [p for p in L.PANELS if p in IL.PANELS]
    assert len(PU) == len(L.PANELS)
    fams = sorted({fam[p] for p in PU})
    D = len(next(iter(regs.values())))

    def phys(a, b, wt):
        return float(np.mean([np.mean([FS.aggregate(CP[b][p], regs[p], wt)["S_phys"] - FS.aggregate(CP[a][p], regs[p], wt)["S_phys"]
                                        for p in IL.PANELS if fam[p] == f]) for f in fams]))

    def state(a, b, wt):
        A = np.stack([ST[a][p] for p in PU]); B = np.stack([ST[b][p] for p in PU])
        return L.reduce(B, wt, "nb", sh) - L.reduce(A, wt, "nb", sh)

    rng = np.random.default_rng(4701); W = [np.bincount(rng.integers(0, D, D), minlength=D).astype(float) for _ in range(NB)]
    st = {}
    for f in stored(split):
        st.update(json.loads(f.read_text())["pairs"])
    rows, par = [], []
    for a, b in pairs:
        key = f"{a}->{b}"
        one = np.ones(D); s0, p0 = state(a, b, one), phys(a, b, one); fc0 = .88 * (.35 * s0 + .45 * p0)
        bs = np.array([(state(a, b, w), phys(a, b, w)) for w in W])
        fcb = .88 * (.35 * bs[:, 0] + .45 * bs[:, 1])
        ref = st[key]; lo, hi = _q(fcb)
        ok = bool(np.isclose(fc0, ref["fc"], rtol=1e-9, atol=1e-13) and np.isclose(lo, ref["ci_fc"][0], rtol=1e-9, atol=1e-13)
                  and np.isclose(hi, ref["ci_fc"][1], rtol=1e-9, atol=1e-13))
        par.append(dict(part=name, split=split, pair=key, point=fc0, lo=lo, hi=hi, stored_point=ref["fc"], stored_lo=ref["ci_fc"][0],
                        stored_hi=ref["ci_fc"][1], parity="EXACT" if ok else "FAIL"))
        print("F14_PARITY", name, split, key, "EXACT" if ok else "FAIL", fc0, ref["fc"], flush=True)
        rows += [dict(pair=key, split=split, draw=i, value=float(x), dState=float(s)) for i, (x, s) in enumerate(zip(fcb, bs[:, 0]))]
        rows.append(dict(pair=key, split=split, draw=-1, value=float(fc0), dState=float(s0)))   # draw -1 = point (all dates once)
    return rows, par


# ------------------------------------------------------------------------------------------ gap bench
def gap_part(split):
    spec = importlib.util.spec_from_file_location("g3b", ROOT / "exp/L_gap3/gap3_bench.py")
    G3 = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "exp/L_gap3"))
    spec.loader.exec_module(G3)    # same patched GL2 (chain = bundle18 S2b bench chain) as the stored read
    GL2 = G3.GL2
    ev, nb, nb_old, dates = GL2.load(split, G3.MEM)
    del nb_old
    zs = np.load(G3.G1D / f"arrays_stack_own8k_{split}.npz")      # S = the C69 bench stack (gap3 parity reference)
    zb = np.load(G3.OUT / f"arrays_B0_{split}.npz")               # B0 = edge taper on the unsmoothed members (row 29)
    for pn, e in ev.items():
        for k in [k for k in e if k.startswith("mem_") or k in ("X", "c12", "bl_v3", "vf", "vc")]:
            del e[k]
        e["S"] = (zs[f"{pn}|0"], zs[f"{pn}|1"]); e["B0"] = (zb[f"{pn}|0"], zb[f"{pn}|1"])
    hm = GL2.half_mask(dates, "all"); W = GL2.boot_weights(int(hm.sum()), NB)
    A = GL2.state_arrays(ev, nb, dates, "S"); B = GL2.state_arrays(ev, nb, dates, "B0")
    xs, _ = GL2.state_delta(A, B, hm, W)
    ref = json.loads((G3.OUT / ("bench_validation.json" if split == "validation" else "bench_private_paired.json")).read_text())["B0_vs_S"]
    lo, hi = _q(xs[1:])
    ok = bool(np.isclose(xs[0], ref["state"], rtol=1e-9, atol=1e-13) and np.isclose(lo, ref["state_ci"][0], rtol=1e-9, atol=1e-13)
              and np.isclose(hi, ref["state_ci"][1], rtol=1e-9, atol=1e-13))
    print("F14_PARITY gap", split, "EXACT" if ok else "FAIL", xs[0], ref["state"], flush=True)
    par = [dict(part="gap3", split=split, pair="S->B0", point=.308 * xs[0], lo=.308 * lo, hi=.308 * hi, stored_point=ref["board"],
                stored_lo=ref["board_ci"][0], stored_hi=ref["board_ci"][1], parity="EXACT" if ok else "FAIL")]
    rows = [dict(pair="S->B0", split=split, draw=i, value=float(.308 * x), dState=float(x)) for i, x in enumerate(xs[1:])]
    rows.append(dict(pair="S->B0", split=split, draw=-1, value=float(.308 * xs[0]), dState=float(xs[0])))
    return rows, par


# ------------------------------------------------------------------------------------------ driver
def run_part(kind, name, split):
    rows, par = gap_part(split) if kind == "gap" else isi_part(name, split)
    PARTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PARTS / f"{name}_{split}_draws.csv", index=False, lineterminator="\n", float_format="%.12e")
    pd.DataFrame(par).to_csv(PARTS / f"{name}_{split}_parity.csv", index=False, lineterminator="\n", float_format="%.12e")


def run_all():
    py = sys.executable
    jobs = [("isi", n, sp) for n in ISI_PARTS for sp in SPLITS] + [("gap", "gap3", sp) for sp in SPLITS]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    for kind, name, sp in jobs:
        if (PARTS / f"{name}_{sp}_draws.csv").exists():
            continue
        r = subprocess.run([py, str(Path(__file__)), "part", kind, name, sp], cwd=str(ROOT), env=env)
        if r.returncode:
            print("F14 part FAILED", kind, name, sp, r.returncode, flush=True)
    merge()


def merge():
    d, p = [], []
    for n in list(ISI_PARTS) + ["gap3"]:
        for sp in SPLITS:
            f = PARTS / f"{n}_{sp}_draws.csv"
            if f.exists():
                d.append(pd.read_csv(f)); p.append(pd.read_csv(PARTS / f"{n}_{sp}_parity.csv"))
    D = pd.concat(d, ignore_index=True); P = pd.concat(p, ignore_index=True)
    for df in (D, P):
        df["piece"] = df["pair"].map(lambda k: PIECE[k][0]); df["adopted_row"] = df["pair"].map(lambda k: PIECE[k][1])
        df["instrument"] = df["pair"].map(lambda k: PIECE[k][2]); df["month"] = df["split"].map(SPLITS)
    D.to_csv(OUT / "F14_bootstrap_draws.csv", index=False, lineterminator="\n", float_format="%.12e")
    P.to_csv(OUT / "F14_bootstrap_parity.csv", index=False, lineterminator="\n", float_format="%.12e")
    print("F14 merged", len(D), "draw rows;", (P.parity == "EXACT").sum(), "of", len(P), "parity EXACT", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "part":
        sys.path.insert(0, str(ROOT / "src"))
        run_part(*a[1:4])
    elif a[0] == "merge":
        merge()
    else:
        run_all()
