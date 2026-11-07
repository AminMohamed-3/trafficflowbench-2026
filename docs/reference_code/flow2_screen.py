"""Claude helper 2026-10-04 (ledger 0362): congested target-cell flow on top of F4 + 0358 (flow_krige), OPT screen.
Candidate: shrink F4 flow toward the congested-branch FD flow implied by F4 speed on predicted-congested target cells:
  q_FD = v * w_c * k_jam / (v + w_c), w_c = cap / (k_jam - k_crit)  (triangular FD, totals not per lane)
  q = (1-a) q_ref + a q_FD where F4 speed < g * v_cut.
Why: in congestion F4 speed error (log sd ~.07) dominates density error; d log q_FD / d log v = w_c/(v+w_c), so the shrink
makes flow error co-move with speed error and cuts the density (q/v) error that S_LWR scores. Same-time and backward-wave
kriging were screened first (in-sample blend sd cut <= 5%, scratch) and dropped.
(a, g) chosen per panel leave-one-panel-out and on the other even/odd date fold by J_S from per-day components.
Reference = F4 + 0358 with its own LOPO/fold band weights (identical to flow_screen.stage2).
Usage: python docs/reference_code/flow2_screen.py"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import phys_lwr_diag as P
import flow_krige as FK
import flow_screen as FS

OUT = P.ROOT / "artifacts/claude_ref/flow/congested"
A_GRID = (0., .4, .8, 1.)  # repair 1: v0 grid (0-.8, g .8/1) peaked at its edge; extended
G_GRID = (.8, 1., 1.15)


def fd_flow(pv, stations):
    s = stations.sort_values("idx")
    cap = s.capacity_vph.to_numpy(float); kj = s.k_jam.to_numpy(float); kc = s.critical_density.to_numpy(float)
    wc = cap / (kj - kc)
    return pv * wc * kj / (pv + wc)


def shrink(pv, qref, target, stations, a, g, gap=None):
    """gap: optional [D,T,S] bool of 90-min gap cells (official/OWN blocks); those keep qref (OWN Mar: FD flow on block rows is far worse)."""
    vcut = stations.sort_values("idx").v_cut.to_numpy(float)
    cap = stations.sort_values("idx").capacity_vph.to_numpy(float)
    m = target & (pv < g * vcut)
    if gap is not None:
        m &= ~np.asarray(gap, bool)
    q = np.clip((1 - a) * qref + a * fd_flow(pv, stations), 50., 1.3 * cap)
    return np.where(m, q, qref)


def reference_flow(panel, held, pv, pq, stats, D):
    N = np.load(FS.OUT / f"{panel}_N.npy").astype(float); den = np.load(FS.OUT / f"{panel}_den.npy").astype(float)
    T = np.asarray(held.target, bool); nq = pq.copy()
    for fold in (0, 1):
        fq = FK.blend(pv, pq, N, T, held.stations, FS.fit_weights(stats, panel, fold, D), den=den)
        sel = np.arange(D) % 2 == fold; nq[sel] = fq[sel]
    return nq


def main(nboot=1000):
    OUT.mkdir(parents=True, exist_ok=True)
    panels = P.P["panels"]; D = len(P.P["OPT_held_dates"])
    stats = {p: dict(np.load(FS.OUT / f"{p}_stats.npz")) for p in panels}
    comps = {}; regs = {}; extra = {}
    for panel in panels:
        held, tv, tq, pv, pq = P.held_panel(panel)
        qref = reference_flow(panel, held, pv, pq, stats, D)
        T = np.asarray(held.target, bool)
        comps[panel] = {}
        for g in G_GRID:
            for a in A_GRID:
                if a == 0 and g != G_GRID[0]:
                    comps[panel][(a, g)] = comps[panel][(0., G_GRID[0])]; continue
                q = shrink(pv, qref, T, held.stations, a, g)
                comps[panel][(a, g)], regs[panel] = FS.day_components(held, tv, tq, pv, q)
        la = held.stations.sort_values("idx").lanes.to_numpy(float); vcut = held.stations.sort_values("idx").v_cut.to_numpy(float)
        extra[panel] = dict(share_cong_target=float((T & (pv < vcut)).sum() / T.sum()))
        print(panel, {f"a{a}_g{g}": round(FS.aggregate(c, regs[panel], np.ones(D))["J"], 5) for (a, g), c in comps[panel].items()}, flush=True)

    def chosen(panel, fold):
        days = (np.arange(D) % 2 != fold).astype(float)
        best = max(comps[panel], key=lambda k: np.mean([FS.aggregate(comps[p][k], regs[p], days)["J"] for p in panels if p != panel]))
        return best

    sel = {p: {f: chosen(p, f) for f in (0, 1)} for p in panels}
    print("chosen (a,g) per panel/fold", {p: [sel[p][0], sel[p][1]] for p in panels}, flush=True)
    newc = {}
    for p in panels:
        C = comps[p][(0., G_GRID[0])].copy()
        for f in (0, 1):
            rows = np.arange(D) % 2 == f; C[rows] = comps[p][sel[p][f]][rows]
        newc[p] = C
    rows = []
    for p in panels:
        a = FS.aggregate(comps[p][(0., G_GRID[0])], regs[p], np.ones(D)); b = FS.aggregate(newc[p], regs[p], np.ones(D))
        rows.append(dict(panel=p, **{f"ref_{k}": v for k, v in a.items()}, **{f"new_{k}": v for k, v in b.items()}, **extra[p],
                         chosen=str([sel[p][0], sel[p][1]])))
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(0); boots = []
    for _ in range(nboot):
        w = np.bincount(rng.integers(0, D, D), minlength=D).astype(float)
        dl = {k: [] for k in ("J", "S_state", "S_phys", "S_LWR")}
        for p in panels:
            a = FS.aggregate(comps[p][(0., G_GRID[0])], regs[p], w); b = FS.aggregate(newc[p], regs[p], w)
            for k in dl: dl[k].append(b[k] - a[k])
        boots.append({k: np.mean(v) for k, v in dl.items()})
    B = pd.DataFrame(boots)
    summary = {k: dict(mean_delta=float((df[f"new_{k}"] - df[f"ref_{k}"]).mean()), ci=[float(B[k].quantile(.025)), float(B[k].quantile(.975))])
               for k in ("J", "S_state", "S_phys", "S_LWR")}
    grid = {f"a{a}_g{g}": float(np.mean([FS.aggregate(comps[p][(a, g)], regs[p], np.ones(D))["J"] - FS.aggregate(comps[p][(0., G_GRID[0])], regs[p], np.ones(D))["J"] for p in panels]))
            for (a, g) in comps[panels[0]]}
    print(df.drop(columns=["ref_S_FD", "new_S_FD"]).round(5).to_string()); print(json.dumps(summary, indent=1)); print("in-sample grid mean dJ", grid)
    df.to_csv(OUT / "opt_screen.csv", index=False)
    (OUT / "opt_summary.json").write_text(json.dumps(dict(summary=summary, grid_mean_dJ=grid, chosen={p: [list(sel[p][0]), list(sel[p][1])] for p in panels}), indent=1))


if __name__ == "__main__":
    main()
