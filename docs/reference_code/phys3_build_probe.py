"""Claude helper 2026-10-04 (ledger 0372): Task 1 probe files on C03 (submissions/c03_queue_cnn_adapt_ongoing.csv). NOT uploaded.
Only state rows change; queue / odme rows equal C03. Pattern and row mapping as build_state_probe.py.
P1: 0360 (pooled OPT coefs) -> 0358 (pooled OPT band weights, gate on the 0360 speed) -> phys3 (r=3, a=.75, g=1).
P2: 0360 -> 0368 (state3 hedge, alpha 1, EPS 0, models m368_*.txt fitted on all OPT candidates) -> 0358 -> phys3.
Gap mask = cache.gap (official Task 2 windows: origin +5..+90 min, all stations, blank in the released layer); phys3 skips gap cells.
0368 features need a full speed field: target cells = current prediction, observed cells = released speed, other blank cells
filled by nearest-in-time value of the same station (F4 full-grid predictions are not in the submission).
Usage: python docs/reference_code/phys3_build_probe.py <P1|P2> <out.csv>"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
from tfb.data import load_cache
import flow_krige as FK
import state2_cm as CM
import phys3_proj as PJ
import phys3_stack as ST

PARENT = ROOT / "submissions/c03_queue_cnn_adapt_ongoing.csv"
R, A, G = ST.R, ST.A, ST.G


def fill_time(v):
    """Fill NaN along time (axis 1) with nearest earlier, then later, value of the same station."""
    D, T, L = v.shape
    x = np.moveaxis(v, 1, 0).reshape(T, -1)
    x = pd.DataFrame(x).ffill().bfill().to_numpy()
    return np.moveaxis(x.reshape(T, D, L), 0, 1)


def main(mode, out):
    sub = pd.read_csv(PARENT, float_precision="round_trip")
    key = pd.read_csv(ROOT / "data/raw/kaggle_public/submission_key.csv", dtype=str,
                      usecols=["submission_id", "task", "split", "panel", "timestamp", "link_id"])
    assert (sub.submission_id.astype(str).to_numpy() == key.submission_id.to_numpy()).all()
    w = np.asarray(json.loads((ROOT / "artifacts/claude_ref/flow/opt/opt_summary.json").read_text())["pooled_weights"])
    cf = json.loads((ROOT / "artifacts/claude_ref/state2/opt_coef.json").read_text()); cf = (cf["b_free"], cf["b_cong"])
    models = ST.load_368() if mode == "P2" else None
    speed, flow = sub.speed_kmh.to_numpy(float).copy(), sub.flow_vph.to_numpy(float).copy()
    st_rows = key.task == "state"; log = {}
    for (split, panel), g in key[st_rows].groupby(["split", "panel"]):
        c = load_cache(panel, split); st = c.stations.sort_values("idx")
        ts = pd.to_datetime(g.timestamp)
        d = np.array([c.dates.index(x) for x in ts.dt.strftime("%Y-%m-%d")]); t = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
        col = dict(zip(st.link_id.astype(str), st.idx)); s = g.link_id.map(col).to_numpy()
        assert not pd.isna(s).any(); s = s.astype(int)
        tgt = np.asarray(c.target, bool); gap = np.asarray(c.gap, bool)
        assert tgt[d, t, s].all() and tgt.sum() == len(g), (split, panel, tgt.sum(), len(g))
        pv = np.full(tgt.shape, np.nan); pq = pv.copy()
        idx = g.index.to_numpy(); pv[d, t, s] = speed[idx]; pq[d, t, s] = flow[idx]
        vobs = np.asarray(c.speed, float)
        observed = np.asarray(c.elig, bool) & ~tgt & ~gap & np.isfinite(vobs)
        vcut = st.v_cut.to_numpy(float)
        gr, _ = CM.common_factor(vobs, observed, c.stations)
        v = np.where(tgt, CM.apply(pv, gr, vcut, cf), np.nan)  # 0360
        nc = 0
        if mode == "P2":  # 0368 on the 0360 speed with F4 (C03) flow
            full = fill_time(np.where(tgt, v, np.where(observed, vobs, np.nan)))
            pqf = fill_time(np.where(tgt, pq, np.where(np.isfinite(np.asarray(c.flow, float)), np.asarray(c.flow, float), np.nan)))
            hv, nc = ST.apply_368(models, vobs, observed, c.stations, full, pqf, tgt)
            v = np.where(tgt, hv, np.nan)
        N, den = FK.krige_log_flow(np.asarray(c.flow, float), FK.positions(c.stations))
        q = FK.blend(np.where(tgt, v, 0.), pq, N, tgt, c.stations, w, den=den)  # 0358 gated on current speed
        nv, nq, m = PJ.project(np.where(tgt, v, np.nan), q, tgt, c.stations, R, A, G, gap=gap)  # phys3
        assert np.isfinite(nv[d, t, s]).all() and np.isfinite(nq[d, t, s]).all()
        speed[idx] = nv[d, t, s]; flow[idx] = nq[d, t, s]
        ch = (np.abs(flow[idx] - sub.flow_vph.to_numpy()[idx]) > 1e-9) | (np.abs(speed[idx] - sub.speed_kmh.to_numpy()[idx]) > 1e-9)
        log[f"{split}/{panel}"] = dict(rows=len(g), changed=int(ch.sum()), phys3_cells=int(m.sum()), hedge_cands=nc,
                                       gap_target=int((gap & tgt).sum()))
        print(split, panel, log[f"{split}/{panel}"], flush=True)
    new = sub.copy(); new["speed_kmh"] = speed; new["flow_vph"] = flow
    assert np.isfinite(new.loc[st_rows, ["speed_kmh", "flow_vph"]].to_numpy()).all()
    other = ~st_rows.to_numpy()
    assert new[other].equals(sub[other])
    assert len(new) == 6980503 and not new.isna().any().any()
    new.to_csv(out, index=False)
    h = hashlib.sha256(Path(out).read_bytes()).hexdigest()
    print("wrote", out, "rows", len(new), "state rows changed:", int(((new.speed_kmh != sub.speed_kmh) | (new.flow_vph != sub.flow_vph)).sum()), "sha256", h)
    Path(out).with_suffix(".json").write_text(json.dumps(dict(mode=mode, parent=str(PARENT.name), sha256=h, per_panel=log, r=R, a=A, g=G), indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
