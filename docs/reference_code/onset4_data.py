"""Claude onset4 helper 2026-10-04: per (window, site) physics features for the onset site model.

Online emulation per window (panel, split, date d, origin t; nothing after t):
  rows 0..11 = t-12..t-1: raw layer on train (official window_history is raw), masked layer on validation/private
  pseudo windows (no raw exists there); row 12 = t from the masked layer (Task 1 mask applies at T officially).
  Same-day earlier steps (< t-12) from the masked layer. Profiles from train fit days only (Jun-Dec 2030, raw).
Sites: Q03 metadata sites (artifacts/lane_q/0267/Q03_seed0_model/metadata.json), incl. 'other'.
Control cell c*(site): in region [min(P)-3, max(P)+3], the cell with the highest median flow/capacity at t over
  fit events where the site queued (pre-onset level); brk(site) = that median.
Populations (label = site has a queued truth cell at t+6):
  fit  = onset_diag selector windows, train raw layer, dates < 2031-01-01
  fitjf = same, 2031-01..02 (used only for the final March/April model)
  tol  = I-Q-TOL 513 windows (onset2 tol_windows_*.parquet; truth/valid there)
  val / pri = onset_score pseudo windows (score_{split}_A2.csv; labels masked queued & eligible at t+6)
Usage: python docs/reference_code/onset4_data.py  -> artifacts/claude_ref/onset4/feat_{pop}.parquet"""
import sys, json, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tfb.data import load_cache
warnings.simplefilter("ignore")
OUT = ROOT / "artifacts/claude_ref/onset4"
META = json.loads((ROOT / "artifacts/lane_q/0267/Q03_seed0_model/metadata.json").read_text())
PANELS = ["D12_I5_N", "D12_I5_S", "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S"]


def nanmean(a, axis=None):
    return np.nanmean(a, axis=axis) if np.size(a) else np.nan


def slope(y):
    x = np.arange(len(y), dtype=float); m = np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    x = x[m] - x[m].mean(); return float(np.dot(x, y[m] - y[m].mean()) / np.dot(x, x))


def weekend(date):
    return int(pd.Timestamp(date).weekday() >= 5)


class Panel:
    def __init__(self, p, split):
        self.p, self.split = p, split
        c = load_cache(p, split); st = c.stations.sort_values("idx")
        self.c = c; self.dates = list(c.dates); self.n = len(st)
        self.vf = st.free_speed_kmh.to_numpy(np.float32); self.cap = st.capacity_vph.to_numpy(np.float32)
        self.vcut = st.v_cut.to_numpy(np.float32)
        self.sm = np.asarray(c["speed"], np.float32) / self.vf; self.fm = np.asarray(c["flow"], np.float32) / self.cap
        self.om = np.asarray(c["occ"], np.float32)
        self.elig = np.asarray(c["elig"], bool)
        if split == "train":
            self.sr = np.asarray(c["speed_true"], np.float32) / self.vf; self.fr = np.asarray(c["flow_true"], np.float32) / self.cap
            self.orr = np.asarray(c["occ_true"], np.float32)

    def inputs(self, d, t):
        """S, F, O: 13 x n (t-12..t). Raw history on train, masked otherwise; row 12 masked."""
        if self.split == "train":
            S = np.vstack([self.sr[d, t - 12:t], self.sm[d, t:t + 1]]); F = np.vstack([self.fr[d, t - 12:t], self.fm[d, t:t + 1]])
            O = np.vstack([self.orr[d, t - 12:t], self.om[d, t:t + 1]])
        else:
            S, F, O = self.sm[d, t - 12:t + 1], self.fm[d, t - 12:t + 1], self.om[d, t - 12:t + 1]
        return S, F, O


def profiles():
    """Median raw flow ratio by (weekend, tod, cell) over fit days (train < 2031-01-01)."""
    out = {}
    for p in PANELS:
        P = Panel(p, "train"); fit = [i for i, x in enumerate(P.dates) if x < "2031-01-01"]
        we = np.array([weekend(P.dates[i]) for i in fit])
        f = P.fr[fit]
        out[p] = np.stack([np.nanmedian(f[we == w], 0) for w in (0, 1)])  # 2 x 288 x n
        print("profile", p, flush=True)
    np.savez_compressed(OUT / "profiles.npz", **out)
    return out


def site_list(p):
    return META["sites"][p]


def control_cells(prof):
    """Per (panel, site): control cell and pre-onset median flow ratio from fit events."""
    sel = pd.read_parquet(ROOT / "artifacts/claude_ref/onset/selected_windows.parquet")
    sel = sel[(sel.split == "train") & (sel.layer == "raw") & (sel.date < "2031-01-01")]
    ctrl = {}
    for p in PANELS:
        P = Panel(p, "train"); g = sel[sel.panel == p]
        for s in site_list(p):
            if s["other"]:
                continue
            Pos = s["positions"]; R = list(range(max(0, min(Pos) - 3), min(P.n, max(Pos) + 4)))
            vals = []
            for r in g.itertuples():
                q = {int(x) for x in r.qpos.split(";") if x}
                if q & set(Pos):
                    d = P.dates.index(r.date)
                    vals.append(nanmean(P.fr[d, r.t - 2:r.t + 1][:, R], 0))
            if not vals:
                continue
            med = np.nanmedian(np.array(vals), 0)
            j = int(np.nanargmax(med))
            ctrl[f"{p}/{s['key']}"] = dict(cell=R[j], brk=float(med[j]), n=len(vals), R=R, med=[float(x) for x in med])
        print("ctrl", p, {k: v for k, v in ctrl.items() if k.startswith(p)}, flush=True)
        ctrl[f"{p}/qmax"] = [float(x) for x in np.nanpercentile(P.fr[[i for i, x in enumerate(P.dates) if x < "2031-01-01"]].reshape(-1, P.n), 99.5, axis=0)]
    (OUT / "control_cells.json").write_text(json.dumps(ctrl, indent=1))
    return ctrl


def feats(P, d, t, prof, ctrl):
    S, F, O = P.inputs(d, t)
    date = P.dates[d]; we = weekend(date); pr = prof[P.p][we]
    t6 = t + 6
    # same-day earlier masked data (05:00 .. t-13)
    lo = 60
    rows = []
    sites = site_list(P.p)
    known_pos = set()
    for s in sites:
        if not s["other"]:
            known_pos |= set(s["positions"])
    site_minv = {}
    for s in sites:
        if s["other"]:
            continue
        site_minv[s["key"]] = np.nanmin(S[12, s["positions"]]) if np.isfinite(S[12, s["positions"]]).any() else np.nan
    for s in sites:
        Pos = list(s["positions"]); key = f"{P.p}/{s['key']}"
        if s["other"]:
            fT = nanmean(F[10:13, Pos], 0)
            c = Pos[int(np.nanargmax(fT))] if np.isfinite(fT).any() else Pos[0]
            brk = np.nan; R = Pos
        else:
            ci = ctrl.get(key)
            c = ci["cell"] if ci else max(Pos); brk = ci["brk"] if ci else np.nan
            R = list(range(max(0, min(Pos) - 3), min(P.n, max(Pos) + 4)))
        r = {"site_uid": key, "other": int(s["other"]), "n_cells": len(Pos), "cap_c": float(P.cap[c]) / 1000,
             "tod6": t6 / 12., "m30": t6 % 6, "weekend": we, "weekday": pd.Timestamp(date).weekday()}
        fc = F[:, c]
        r["fc_T"] = nanmean(fc[10:13]); r["fc_last"] = fc[12] if np.isfinite(fc[12]) else (fc[11] if np.isfinite(fc[11]) else np.nan)
        r["fc_h"] = nanmean(fc[0:6]); r["fc_slope"] = slope(fc)
        prT, pr6 = pr[t, c], pr[t6, c]
        r["prof_T"] = prT; r["prof_ratio"] = pr6 / prT if prT > 0 else np.nan
        r["fc_anom"] = r["fc_T"] / prT if prT > 0 else np.nan
        r["fc_proj"] = r["fc_T"] * r["prof_ratio"]
        r["brk"] = brk; r["head"] = r["fc_T"] - brk; r["head_proj"] = r["fc_proj"] - brk
        day = P.fm[d, lo:t - 12, c] / np.maximum(pr[lo:t - 12, c], 1e-3)
        r["day_anom"] = nanmean(day) if np.isfinite(day).any() else np.nan
        fR = nanmean(F[10:13][:, R], 0)
        r["fR_max"] = np.nanmax(fR) if np.isfinite(fR).any() else np.nan
        prR = pr[t][R]; r["fR_anom_max"] = np.nanmax(fR / np.maximum(prR, 1e-3)) if np.isfinite(fR).any() else np.nan
        r["vR_T"] = np.nanmin(S[12, R]) if np.isfinite(S[12, R]).any() else np.nan
        r["vR_T1"] = np.nanmin(S[11, R]) if np.isfinite(S[11, R]).any() else np.nan
        r["vR_h"] = np.nanmin(S[0:11][:, R]) if np.isfinite(S[0:11][:, R]).any() else np.nan
        vs = nanmean(S[:, R], 1); r["vR_slope"] = slope(vs)
        r["vP_T"] = np.nanmin(S[12, Pos]) if np.isfinite(S[12, Pos]).any() else np.nan
        r["oR_max"] = np.nanmax(nanmean(O[10:13][:, R], 0)) if np.isfinite(O[10:13][:, R]).any() else np.nan
        r["obsP_T"] = float(np.isfinite(S[12, Pos]).mean())
        others = [v for k, v in site_minv.items() if k != s["key"]]
        r["other_minv_T"] = np.nanmin(others) if others and np.isfinite(others).any() else np.nan
        # queued today earlier (masked, before history) at this site
        qd = (P.sm[d, lo:t - 12][:, Pos] <= .6)
        r["q_today"] = float(qd.any()) if qd.size else 0.
        # v2: empirical capacity (fit-day p99.5 flow ratio), per-cell heads over the region, upstream arrival
        qmax = np.asarray(ctrl[f"{P.p}/qmax"]); fT3 = nanmean(F[10:13], 0)
        u = fT3 / np.maximum(qmax, 1e-3)
        r["u_c"] = u[c]; r["u_R"] = np.nanmax(u[R]) if np.isfinite(u[R]).any() else np.nan
        ds_ = list(range(max(Pos) + 1, min(P.n, max(Pos) + 5)))
        r["u_ds"] = np.nanmax(u[ds_]) if ds_ and np.isfinite(u[ds_]).any() else np.nan
        r["u_P"] = np.nanmax(u[Pos]) if np.isfinite(u[Pos]).any() else np.nan
        prr = pr[t6] / np.maximum(pr[t], 1e-3)
        up_ = fT3 * prr / np.maximum(qmax, 1e-3)
        r["uproj_R"] = np.nanmax(up_[R]) if np.isfinite(up_[R]).any() else np.nan
        ci = ctrl.get(key)
        if ci and "R" in ci:
            hh = fT3[ci["R"]] - np.asarray(ci["med"])
            r["headR_max"] = np.nanmax(hh) if np.isfinite(hh).any() else np.nan
            r["headR_mean"] = nanmean(hh) if np.isfinite(hh).any() else np.nan
        else:
            r["headR_max"] = r["headR_mean"] = np.nan
        U = list(range(max(0, min(Pos) - 8), max(0, min(Pos) - 2)))
        an = fT3[U] / np.maximum(pr[t][U], 1e-3) if U else np.array([np.nan])
        r["up_anom"] = nanmean(an) if np.isfinite(an).any() else np.nan
        rows.append(r)
    return rows


def build(pop, prof, ctrl):
    out = []
    if pop in ("fit", "fitjf"):
        sel = pd.read_parquet(ROOT / "artifacts/claude_ref/onset/selected_windows.parquet")
        sel = sel[(sel.split == "train") & (sel.layer == "raw")]
        sel = sel[sel.date < "2031-01-01"] if pop == "fit" else sel[sel.date >= "2031-01-01"]
        W = [(r.panel, "train", r.date, int(r.t), [int(x) for x in r.qpos.split(";") if x], None) for r in sel.itertuples()]
    elif pop == "tol":
        T = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(ROOT / "artifacts/claude_ref/onset2/tol_windows_*.parquet")))])
        W = [(r.panel, "train", r.date, int(r.t), list(map(int, r.truth)), list(map(int, r.valid))) for r in T.itertuples()]
    else:
        split = {"val": "validation", "pri": "private"}[pop]
        sc = pd.read_csv(ROOT / f"artifacts/claude_ref/onset/score_{split}_A2.csv")
        W = [(r.panel, split, r.date, int(r.t), None, None) for r in sc.itertuples()]
    cache = {}
    for i, (p, split, date, t, truth, valid) in enumerate(W):
        if (p, split) not in cache:
            cache.clear(); cache[(p, split)] = Panel(p, split)
        P = cache[(p, split)]; d = P.dates.index(date)
        if truth is None:
            y = np.isfinite(P.sm[d, t + 6]) & P.elig[d, t + 6] & (P.sm[d, t + 6] <= .6)
            v = np.isfinite(P.sm[d, t + 6]) & P.elig[d, t + 6]
            truth = np.flatnonzero(y).tolist(); valid = np.flatnonzero(v).tolist()
        if valid is None:
            valid = np.flatnonzero(np.isfinite(P.sr[d, t + 6])).tolist()
        rows = feats(P, d, t, prof, ctrl)
        for s, r in zip(site_list(p), rows):
            r.update(panel=p, date=date, t=t, y=int(bool(set(truth) & set(s["positions"]))), wkey=f"{p}|{date}|{t}")
        out.extend(rows)
        if pop in ("tol", "val", "pri") or True:
            out[-1]["truth"] = ";".join(map(str, truth)); out[-1]["valid"] = ";".join(map(str, valid))
    df = pd.DataFrame(out)
    # truth/valid strings only on the last site row of each window: spread to all rows
    df[["truth", "valid"]] = df.groupby("wkey")[["truth", "valid"]].transform(lambda x: x.dropna().iloc[0] if x.notna().any() else None)
    df.to_parquet(OUT / f"feat_{pop}.parquet")
    print(pop, "windows", df.wkey.nunique(), "rows", len(df), flush=True)


if __name__ == "__main__":
    pf = OUT / "profiles.npz"
    prof = dict(np.load(pf)) if pf.exists() else profiles()
    cf = OUT / "control_cells.json"
    ctrl = json.loads(cf.read_text()) if cf.exists() else control_cells(prof)
    for pop in sys.argv[1:] or ["fit", "tol", "val", "pri", "fitjf"]:
        build(pop, prof, ctrl)
