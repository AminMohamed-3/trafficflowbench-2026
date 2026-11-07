"""Claude onset5 2026-10-04: bottleneck activity statistic under OFFICIAL-DAY GAP EMULATION (fixes a selection bias).

Why: on official days the masked layer hides T+1..T+18 on all links after each of the 10 official windows (5 onset,
5 ongoing). Those gaps sit exactly on queue episodes (an onset window's gap hides the first 65 min of its own queue), so
site queue activity measured on official days is biased low, and train (no gaps) is the wrong null.
Emulation (per window (d, t) and start day s = d - j, j = 0..8, same month): replay the official selection from day s
(onset_select replica: first 5 onset windows from s, the window itself must be one of them; 5 ongoing windows drawn at
random (seeded) among ongoing candidates on days s..last onset day), hide T'+1..T'+18 for every emulated window T'
before (d, t), then A_s / E_s as in onset5_stat.py over days s..d-1 plus day d before t.
Selection layer: train raw (replica 39/40 on released train windows), validation/private masked. Statistic layer:
masked everywhere. Test months keep their real gaps (pseudo windows on early days get both; noted).
Official windows: real masked layer from day 1, no emulation (k = d - 1).
Output: artifacts/claude_ref/onset5/stat2_{train,val,pri,off}.parquet (rows: wkey, panel, split, date, t, j, rank,
site, A_all, E_all, A_loc, E_loc, kdays).
Usage: python docs/reference_code/onset5_stat2.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append(str(Path(__file__).resolve().parent))
import onset5_stat as S
import onset5_select as SEL
ROOT = S.ROOT; OUT = S.OUT
J = 9
GAP = 18


def emu_gaps(cands, dates, s, d, t, seed):
    """Return (rank, gaps) for window (d, t) under official selection from day s; gaps = {day: [t0, ...]} of emulated
    windows before (d, t). rank None when (d, t) is not among the first 5 onset windows from s."""
    on, _ = SEL.official_from(cands, dates, s)
    if (d, t) not in on:
        return None, None
    last = on[-1][0]
    og = [(dd, tt) for dd in range(s, last + 1) for tt in cands[dd][1]]
    rng = np.random.default_rng(seed)
    if len(og) > 5:
        og = [og[i] for i in sorted(rng.choice(len(og), 5, replace=False))]
    gaps = {}
    for dd, tt in on + og:
        if (dd, tt) < (d, t):
            gaps.setdefault(dd, []).append(tt)
    return on.index((d, t)), gaps


def stat(keys, arr, s, d, t, gaps):
    dates, Qs, Os, Es = arr
    t6 = t + 6
    loc = np.zeros(288, bool); loc[max(0, t6 - S.LOC):min(288, t6 + S.LOC + 1)] = True
    A = np.zeros(len(keys)); E = np.zeros(len(keys)); Al = np.zeros(len(keys)); El = np.zeros(len(keys))
    for dd in range(s, d + 1):
        keep = np.ones(288, bool)
        for tt in gaps.get(dd, []):
            keep[tt + 1:tt + 1 + GAP] = False
        if dd == d:
            keep[t:] = False
        A += Qs[dd][keep].sum(0); E += Es[dd][keep].sum(0)
        kl = keep & (loc | (dd == d))
        Al += Qs[dd][kl].sum(0); El += Es[dd][kl].sum(0)
    return A, E, Al, El


def main():
    sel = pd.read_parquet(ROOT / "artifacts/claude_ref/onset/selected_windows.parquet")
    trw = sel[(sel.split == "train") & (sel.layer == "raw")][["panel", "date", "t"]]
    tol = pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "artifacts/claude_ref/onset2").glob("tol_windows_*.parquet"))])
    trw = pd.concat([trw, tol[["panel", "date", "t"]]]).drop_duplicates()
    pops = {"train": [], "val": [], "pri": [], "off": []}
    for pi, p in enumerate(S.PANELS):
        keys, arrs = S.panel_arrays(p)
        for pop, split, layer in (("train", "train", "raw"), ("val", "validation", "masked"), ("pri", "private", "masked")):
            dates, cands = SEL.split_candidates(p, split, layer)
            di = {x: i for i, x in enumerate(dates)}
            if pop == "train":
                W = trw[trw.panel == p]
            else:
                W = pd.read_csv(ROOT / f"artifacts/claude_ref/onset/score_{split}_A2.csv"); W = W[W.panel == p]
            nmiss = 0
            for r in W.itertuples():
                d, t = di[r.date], int(r.t)
                if t not in cands[d][0]:
                    cands[d] = (sorted(set(cands[d][0]) | {t}), cands[d][1]); nmiss += 1  # keep the scored window selectable
                for j in range(J):
                    s = d - j
                    if s < 0 or dates[s][:7] != dates[d][:7]:
                        break
                    rank, gaps = emu_gaps(cands, dates, s, d, t, seed=pi * 100000 + s)
                    if rank is None:
                        continue
                    A, E, Al, El = stat(keys, arrs[split], s, d, t, gaps)
                    for k_, a, e, al, el in zip(keys, A, E, Al, El):
                        pops[pop].append(dict(wkey=f"{p}|{r.date}|{t}", panel=p, split=split, date=r.date, t=t, j=j, rank=rank,
                                              site=k_, A_all=a, E_all=e, A_loc=al, E_loc=el, kdays=j))
            print(p, pop, "windows", len(W), "not in replica", nmiss, flush=True)
            if pop != "train":
                wi = pd.read_csv(ROOT / f"data/raw/kaggle_public/task2/{p}/{split}/window_index.csv")
                wi = wi[wi.condition == "queue_onset"]
                for w in wi.itertuples():
                    T = pd.Timestamp(w.forecast_origin); t = T.hour * 12 + T.minute // 5; d = di[w.date]
                    A, E, Al, El = stat(keys, arrs[split], 0, d, t, {})
                    for k_, a, e, al, el in zip(keys, A, E, Al, El):
                        pops["off"].append(dict(wkey=w.window_id, panel=p, split=split, date=w.date, t=t, j=d, rank=-1,
                                                site=k_, A_all=a, E_all=e, A_loc=al, E_loc=el, kdays=d))
        del arrs
    for k, v in pops.items():
        pd.DataFrame(v).to_parquet(OUT / f"stat2_{k}.parquet")
        print(k, len(v))


if __name__ == "__main__":
    main()
