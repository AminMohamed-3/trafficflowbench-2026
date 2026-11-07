"""asm2 (ledger 0475): truth-free difference report between two full submissions (award 1% reproduction gate).

Usage: python docs/reference_code/asm2_diff.py A B [--out X.json] [--splits private,validation]
  A, B = 'path[+delta[+delta...]]': a full submission (.csv, .csv.gz, or .parquet with all columns, 6,980,503 rows)
  followed by optional delta files (submission_id + any of speed_kmh, flow_vph, queue_pred, path_flow; csv/csv.gz/parquet),
  applied left to right. Example: c17.csv+phys5/c18b...delta.csv.gz+onset8/c19...delta.csv

What it reports per split (validation = public board, private = award board):
- state rows: max |dv|, max |dq|, rows changed, and a BOUND on |dS_state| per panel: the official state score is
  1 - rmse/25 (speed) and 1 - rmse_per_lane/600 (flow) per regime, and |rmse(A) - rmse(B)| <= rms(A - B) (triangle
  inequality), so |dS_state| <= mean_regime(.54 rms(dv)/25 + .46 rms(dq/lanes)/600). rms is taken over all target cells
  (the scored set drops cells with non-finite truth, a small share), so this is an approximate bound, never an estimate.
- physics: S_FD of A and B computed directly (needs no hidden truth except at non-target cells, where the released
  observation stands in); E_LWR bound = sum |d(n_A) - d(n_B)| over valid time transitions / sum |d(n_A)| (A's own density
  changes stand in for the hidden truth's in the denominator), all valid transitions (observable/ramp masks ignored).
  S_phys = (S_FD + 2 S_LWR)/3 per regime.
- queue rows: per window 1 - Jaccard(A, B) over all rows (Jaccard distance is a metric, so |IoU(A,T) - IoU(B,T)| <=
  1 - J(A,B) for any truth T; eligibility thinning keeps it a bound only approximately), aggregated as the official
  scorer (condition -> panel -> family, D12_I405 panels excluded).
- odme rows: relative L1 sum|df| / sum|f_A| per panel (bound proxy for S_od, S_link, S_attr; S_dev not bounded).
- bound on |dS_total| = .35 state + .15 physics + .30 queue + .20 odme (same aggregation: panel -> family -> mean).
Compare with the 1% gate: .01 x private score (~.0089).
"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json, argparse, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
KEY = ROOT / "data/raw/kaggle_public/submission_key.csv"
NROWS = 6980503
COLS = ["speed_kmh", "flow_vph", "queue_pred", "path_flow"]


def _read(p, cols=None):
    p = str(p)
    if p.endswith(".parquet"):
        return pd.read_parquet(p, columns=cols)
    return pd.read_csv(p, float_precision="round_trip", usecols=cols)


def load_spec(spec):
    parts = spec.split("+")
    base = _read(ROOT / parts[0] if not Path(parts[0]).is_absolute() else parts[0], ["submission_id"] + COLS)
    assert len(base) == NROWS, (parts[0], len(base))
    ids = base.submission_id.to_numpy()
    assert (np.diff(ids) > 0).all()
    out = {c: base[c].to_numpy(float).copy() for c in COLS}
    for dp in parts[1:]:
        dl = _read(ROOT / dp if not Path(dp).is_absolute() else dp)
        rows = np.searchsorted(ids, dl.submission_id.to_numpy())
        assert (ids[rows] == dl.submission_id.to_numpy()).all(), dp
        for c in dl.columns:
            if c in COLS:
                out[c][rows] = dl[c].to_numpy(float)
    return ids, out


def aggregate(panel_vals):
    s = pd.Series(panel_vals, dtype=float)
    if s.empty:
        return 0.0
    fam = s.groupby(s.index.str.rsplit("_", n=1).str[0]).mean()
    return float(fam.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--out", default=None)
    ap.add_argument("--splits", default="private,validation"); ap.add_argument("--no-physics", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    from tfb.data import load_cache
    from tfb import metric
    key = pd.read_csv(KEY, usecols=["submission_id"], dtype=np.int64)
    for col in ["task", "split", "panel", "timestamp", "link_id", "window_id", "mask_regime", "path_id"]:
        key[col] = pd.read_csv(KEY, usecols=[col], dtype=str)[col].astype("category")
    ida, A = load_spec(a.a); idb, B = load_spec(a.b)
    assert (ida == key.submission_id.to_numpy()).all() and (idb == ida).all()
    res = dict(a=a.a, b=a.b, splits={})
    wi = pd.concat([pd.read_csv(p, dtype=str) for p in (ROOT / "data/raw/kaggle_public/task2").glob("*/*/window_index.csv")])
    cond = dict(zip(wi.window_id, wi.condition))
    for split in a.splits.split(","):
        R = dict(state={}, physics={}, queue={}, odme={})
        ks = key[(key.split == split).to_numpy()]
        # ---- state + physics
        st = ks[(ks.task == "state").to_numpy()]
        sb_panel, ph_panel, fd_panel = {}, {}, {}
        for panel, g in st.groupby("panel", observed=True):
            idx = g.index.to_numpy()
            va, vb, qa, qb = A["speed_kmh"][idx], B["speed_kmh"][idx], A["flow_vph"][idx], B["flow_vph"][idx]
            c = load_cache(panel, split); sta = c.stations.sort_values("idx")
            ts = pd.to_datetime(g.timestamp.astype(str))
            d = np.array([c.dates.index(x) for x in ts.dt.strftime("%Y-%m-%d")]); t = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
            s = g.link_id.astype(str).map(dict(zip(sta.link_id.astype(str), sta.idx))).to_numpy().astype(int)
            prm = metric._parameters(c, ROOT)[0]
            lanes = np.maximum(prm.lanes.to_numpy(float), 1.)
            reg = np.asarray(c.regimes)[d]
            dv, dq = va - vb, (qa - qb) / lanes[s]
            row = dict(rows=int(len(idx)), changed=int(((va != vb) | (qa != qb)).sum()),
                       max_abs_dv=float(np.abs(va - vb).max()), max_abs_dq=float(np.abs(qa - qb).max()),
                       rms_dv=float(np.sqrt(np.mean(dv ** 2))), rms_dq_lane=float(np.sqrt(np.mean(dq ** 2))))
            bs = []
            for r in ("R1", "R2", "R3"):
                m = reg == r
                bs.append(.54 * np.sqrt(np.mean(dv[m] ** 2)) / 25 + .46 * np.sqrt(np.mean(dq[m] ** 2)) / 600 if m.any() else 0.)
            row["bound_S_state"] = float(np.mean(bs)); sb_panel[panel] = row["bound_S_state"]
            R["state"][panel] = row
            if a.no_physics:
                continue
            # physics: fields with the released observation at non-target cells
            tgt = np.asarray(c.target, bool); elig = np.asarray(c.elig, bool)
            vo, fo = np.asarray(c.speed, float), np.asarray(c.flow, float)
            L = prm.length_km.to_numpy(float); vf = prm.free_speed_kmh.to_numpy(float); cap = prm.capacity_vph.to_numpy(float)
            crit = prm.critical_density.to_numpy(float); jam = prm.k_jam.to_numpy(float)
            regd = np.asarray(c.regimes)
            out = {}
            for nm, (pv, pq) in (("A", (va, qa)), ("B", (vb, qb))):
                V = vo.copy(); Q = fo.copy(); V[d, t, s] = pv; Q[d, t, s] = pq
                valid = elig & np.isfinite(V) & np.isfinite(Q)
                k_l = Q / np.maximum(V, 1.) / lanes; q_l = Q / lanes
                qfd = np.where(k_l <= crit / lanes, vf * k_l, (cap / lanes / np.maximum(jam / lanes - crit / lanes, 1e-9)) * np.maximum(jam / lanes - k_l, 0))
                n = Q / np.maximum(V, 1.) * L
                out[nm] = (valid, np.abs(q_l - qfd), np.abs(q_l), n)
            (vA, eA, sA, nA), (vB, eB, sB, nB) = out["A"], out["B"]
            ph = []; fdd = []
            for r in ("R1", "R2", "R3"):
                dm = regd == r
                if not dm.any():
                    continue
                m = vA[dm] & vB[dm]
                fdA = 1 - eA[dm][m].sum() / (sA[dm][m].sum() + 1e-9); fdB = 1 - eB[dm][m].sum() / (sB[dm][m].sum() + 1e-9)
                tr = m[:, :-1] & m[:, 1:]
                dnA = np.diff(nA[dm], axis=1); dnB = np.diff(nB[dm], axis=1)
                num = np.abs(dnA - dnB)[tr].sum(); den = np.abs(dnA)[tr].sum()
                lw = min(1., num / max(den, 1e-9))
                fdd.append(abs(fdA - fdB)); ph.append((abs(fdA - fdB) + 2 * lw) / 3)
            row["dS_FD"] = float(np.mean(fdd)); row["bound_S_phys"] = float(np.mean(ph))
            ph_panel[panel] = row["bound_S_phys"]; fd_panel[panel] = row["dS_FD"]
        # ---- queue
        qk = ks[(ks.task == "queue").to_numpy()]
        wb = {}
        for w, g in qk.groupby("window_id", observed=True):
            idx = g.index.to_numpy(); pa = A["queue_pred"][idx] > .5; pb = B["queue_pred"][idx] > .5
            u = (pa | pb).sum(); jd = 0. if u == 0 else 1 - (pa & pb).sum() / u
            wb[str(w)] = (str(g.panel.iloc[0]), cond.get(str(w), "?"), float(jd), int((pa != pb).sum()))
        wdf = pd.DataFrame(wb.values(), index=list(wb.keys()), columns=["panel", "condition", "jd", "cells"])
        R["queue"] = dict(windows_changed={c_: int(((wdf.condition == c_) & (wdf.cells > 0)).sum()) for c_ in ("queue_onset", "queue_ongoing")},
                          cells_changed=int(wdf.cells.sum()), changed=wdf[wdf.cells > 0].to_dict("index"))
        pq_ = wdf[~wdf.panel.str.startswith("D12_I405_")].groupby(["panel", "condition"]).jd.mean().groupby("panel").mean().to_dict()
        # ---- odme
        ok = ks[(ks.task == "odme").to_numpy()]
        ob = {}
        for panel, g in ok.groupby("panel", observed=True):
            idx = g.index.to_numpy(); fa, fb = A["path_flow"][idx], B["path_flow"][idx]
            ob[panel] = float(np.abs(fa - fb).sum() / max(np.abs(fa).sum(), 1e-9))
            R["odme"][panel] = dict(rows=int(len(idx)), rel_l1=ob[panel], max_abs=float(np.abs(fa - fb).max()))
        tot = dict(bound_S_state=aggregate(sb_panel), bound_S_phys=aggregate(ph_panel) if ph_panel else None,
                   dS_FD=aggregate(fd_panel) if fd_panel else None, bound_S_queue=aggregate(pq_), odme_rel_l1=aggregate(ob))
        tot["bound_S_total"] = .35 * tot["bound_S_state"] + .15 * (tot["bound_S_phys"] or 0.) + .30 * tot["bound_S_queue"] + .20 * tot["odme_rel_l1"]
        R["total"] = tot
        res["splits"][split] = R
        print(split, json.dumps(tot), flush=True)
    res["runtime_s"] = round(time.time() - t0)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
