"""Claude onset5 helper 2026-10-04: per-window, per-site "bottleneck activity shift" statistic (online, same split).

For a window (panel, split, day d, origin t) and k earlier same-month days, history H = days max(1st of month, d-k)..d-1
(all 288 steps, or only steps within +-LOC of the T+30 time of day: 'loc') plus day d steps 0..t-1.
Masked layer only (same observation process in train and test): obs = finite & eligible, queued = obs & speed <= v_cut.
Per site s (Q03 site positions, incl. 'other'): A_s = queued cell-steps in H, E_s = sum over observed cell-steps in H of
the train expectation p(weekend, tod, cell) (tod-smoothed +-1 step). Expectation for train windows is leave-own-month-out
over Jun 2030-Feb 2031; for validation/private all train months. Nothing after t on day d, nothing from later days.
Populations: 'train' = all train selector windows (raw layer selection, onset_diag) plus the I-Q-TOL windows, 'val'/'pri' = onset_score pseudo
windows, 'off' = official onset windows. k in 0..4 for train/val/pri; official uses k = d-1 (all earlier days).
Output: artifacts/claude_ref/onset5/stat_{pop}.parquet (rows: wkey, panel, split, date, dom, t, k, site, A_all, E_all,
A_loc, E_loc, O_all).
Usage: python docs/reference_code/onset5_stat.py"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tfb.data import load_cache
OUT = ROOT / "artifacts/claude_ref/onset5"; OUT.mkdir(parents=True, exist_ok=True)
META = json.loads((ROOT / "artifacts/lane_q/0267/Q03_seed0_model/metadata.json").read_text())
PANELS = ["D12_I5_N", "D12_I5_S", "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S"]
LOC = 24  # +-2 h around the T+30 time of day on earlier days
KS = [0, 1, 2, 3, 4]


def layer(p, split):
    c = load_cache(p, split); st = c.stations.sort_values("idx")
    vc = st.v_cut.to_numpy(np.float32)
    sp = np.asarray(c["speed"], np.float32); el = np.asarray(c["elig"], bool)
    obs = np.isfinite(sp) & el; q = obs & (sp <= vc)
    return list(c.dates), obs, q


def smooth(x):
    """+-1 step tod smoothing along axis -2 (tod), circular."""
    return x + np.roll(x, 1, axis=-2) + np.roll(x, -1, axis=-2)


def panel_arrays(p):
    sites = META["sites"][p]
    out = {}
    dates, obs, q = layer(p, "train")
    we = np.array([pd.Timestamp(x).weekday() >= 5 for x in dates]); mon = np.array([x[:7] for x in dates])
    months = sorted(set(mon))
    QC = np.zeros((len(months), 2, 288, obs.shape[2])); OC = np.zeros_like(QC)
    for i, m in enumerate(months):
        for w in (0, 1):
            sel = (mon == m) & (we == w)
            QC[i, w] = q[sel].sum(0); OC[i, w] = obs[sel].sum(0)
    QCs, OCs = smooth(QC), smooth(OC)
    tq, to = QCs.sum(0), OCs.sum(0)
    p_all = (tq + .5) / (to + 1)
    pos = [list(s["positions"]) for s in sites]

    def site_sums(obs, q, pexp_by_day):
        D = obs.shape[0]
        Qs = np.stack([q[:, :, P].sum(-1) for P in pos], -1).astype(np.float32)
        Os = np.stack([obs[:, :, P].sum(-1) for P in pos], -1).astype(np.float32)
        Es = np.zeros_like(Qs)
        for d in range(D):
            pe = pexp_by_day(d)  # 288 x n
            x = obs[d] * pe
            Es[d] = np.stack([x[:, P].sum(-1) for P in pos], -1)
        return Qs, Os, Es

    mi = {m: i for i, m in enumerate(months)}
    lmo = {m: ((tq - QCs[i]) + .5) / ((to - OCs[i]) + 1) for m, i in mi.items()}
    out["train"] = (dates,) + site_sums(obs, q, lambda d: lmo[mon[d]][int(we[d])])
    del obs, q
    for split in ("validation", "private"):
        dts, o, qq = layer(p, split)
        w_ = np.array([pd.Timestamp(x).weekday() >= 5 for x in dts])
        out[split] = (dts,) + site_sums(o, qq, lambda d: p_all[int(w_[d])])
        del o, qq
    return [s["key"] for s in sites], out


def window_stats(keys, arr, d, t, k):
    dates, Qs, Os, Es = arr
    mon = dates[d][:7]
    lo = d
    while lo - 1 >= 0 and lo > d - k and dates[lo - 1][:7] == mon:
        lo -= 1
    t6 = t + 6
    loc = np.zeros(288, bool); loc[max(0, t6 - LOC):min(288, t6 + LOC + 1)] = True
    A_all = Qs[lo:d].sum((0, 1)) + Qs[d, :t].sum(0); E_all = Es[lo:d].sum((0, 1)) + Es[d, :t].sum(0)
    O_all = Os[lo:d].sum((0, 1)) + Os[d, :t].sum(0)
    A_loc = Qs[lo:d][:, loc].sum((0, 1)) + Qs[d, :t].sum(0); E_loc = Es[lo:d][:, loc].sum((0, 1)) + Es[d, :t].sum(0)
    return [dict(site=s, A_all=a, E_all=e, A_loc=al, E_loc=el, O_all=o, kdays=d - lo)
            for s, a, e, al, el, o in zip(keys, A_all, E_all, A_loc, E_loc, O_all)]


def main():
    sel = pd.read_parquet(ROOT / "artifacts/claude_ref/onset/selected_windows.parquet")
    trw = sel[(sel.split == "train") & (sel.layer == "raw")][["panel", "date", "t"]]
    tol = pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "artifacts/claude_ref/onset2").glob("tol_windows_*.parquet"))])
    trw = pd.concat([trw, tol[["panel", "date", "t"]]]).drop_duplicates()  # selector windows + I-Q-TOL windows
    pops = {"train": [], "val": [], "pri": [], "off": []}
    for p in PANELS:
        keys, arrs = panel_arrays(p)
        idx = {sp: {x: i for i, x in enumerate(arrs[sp][0])} for sp in arrs}
        for r in trw[trw.panel == p].itertuples():
            d = idx["train"][r.date]
            for k in KS:
                for x in window_stats(keys, arrs["train"], d, int(r.t), k):
                    x.update(wkey=f"{p}|{r.date}|{r.t}", panel=p, split="train", date=r.date, t=int(r.t), k=k)
                    pops["train"].append(x)
        for pop, split in (("val", "validation"), ("pri", "private")):
            sc = pd.read_csv(ROOT / f"artifacts/claude_ref/onset/score_{split}_A2.csv")
            for r in sc[sc.panel == p].itertuples():
                d = idx[split][r.date]
                for k in KS:
                    for x in window_stats(keys, arrs[split], d, int(r.t), k):
                        x.update(wkey=f"{p}|{r.date}|{r.t}", panel=p, split=split, date=r.date, t=int(r.t), k=k)
                        pops[pop].append(x)
            wi = pd.read_csv(ROOT / f"data/raw/kaggle_public/task2/{p}/{split}/window_index.csv")
            wi = wi[wi.condition == "queue_onset"]
            for w in wi.itertuples():
                T = pd.Timestamp(w.forecast_origin); t = T.hour * 12 + T.minute // 5; d = idx[split][w.date]
                for x in window_stats(keys, arrs[split], d, t, 40):
                    x.update(wkey=w.window_id, panel=p, split=split, date=w.date, t=t, k=-1)
                    pops["off"].append(x)
        print(p, {k: len(v) for k, v in pops.items()}, flush=True)
        del arrs
    for k, v in pops.items():
        df = pd.DataFrame(v); df["dom"] = df.date.str[-2:].astype(int)
        df.to_parquet(OUT / f"stat_{k}.parquet")


if __name__ == "__main__":
    main()
