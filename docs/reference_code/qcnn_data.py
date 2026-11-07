"""Claude 2026-10-01: data for a corridor CNN queue forecaster (Task 2), train split only.

Per Task 2 panel, from the audited cache:
- candidate windows on every train day with Q's offline selector (windows.candidate_windows logic:
  raw observed speed <= v_cut, coverage of eligible cells in T-60..T-5 >= .7, ongoing = some link with
  >= 2 queued steps in history and a queue in the horizon, onset = no history queue, none in T+5..T+25,
  some at T+30); labels = raw queued at T+5..T+30, valid = raw observed (ADV population convention);
- feature cubes (days, 288, 5, L): channels speed/vf, flow/capacity, occupancy, observed flag, queued
  flag; one cube from the masked layer (online same-day view) and one from the raw layer (window history).
A window at step t uses masked steps t-23..t-13, raw steps t-12..t-1 and the masked step t (RULES 2).
Output: artifacts/claude_ref/qcnn/<panel>.npz plus windows.parquet. Check: reproduces Q's frozen
I-Q-ADV candidate list (artifacts/lane_q/0106/iq_adv_windows.parquet) exactly on Jan-Feb 2031."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tfb.data import load_cache

PANELS = ["D12_I5_N", "D12_I5_S", "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S"]
OUT = ROOT / "artifacts/claude_ref/qcnn"


def cube(speed, flow, occ, st):
    vf = st.free_speed_kmh.to_numpy(np.float32); cap = st.capacity_vph.to_numpy(np.float32)
    vcut = st.v_cut.to_numpy(np.float32)
    obs = np.isfinite(speed)
    c = np.zeros(speed.shape[:2] + (5, speed.shape[2]), np.float16)
    c[:, :, 0] = np.where(obs, speed / vf, 0)
    c[:, :, 1] = np.where(np.isfinite(flow), flow / cap, 0)
    c[:, :, 2] = np.where(np.isfinite(occ), occ, 0)
    c[:, :, 3] = obs
    c[:, :, 4] = obs & (speed <= vcut)
    return c


def ramp_cube(c, st):
    """(days, 288, 3, L): on-ramp and off-ramp flow attached to each link in 1000 vph, and the
    observed share of the link's ramps. Released in every split (Q11 idea); used only at steps <= t."""
    rf = np.asarray(c["ramp_flow"], np.float32); r = c.ramps
    cap = st.capacity_vph.to_numpy(np.float32); L = len(st)
    out = np.zeros(rf.shape[:2] + (3, L), np.float32); cnt = np.zeros(L, np.float32)
    obs = np.isfinite(rf); v = np.where(obs, rf, 0)
    for ri, typ, j in zip(r.idx, r.ramp_type, r.station_idx):
        out[:, :, 0 if typ == "OR" else 1, j] += v[:, :, ri] / 1000
        out[:, :, 2, j] += obs[:, :, ri]; cnt[j] += 1
    out[:, :, 2] /= np.maximum(cnt, 1)
    return out.astype(np.float16)


def candidates(panel, dates, raw_speed, elig, vcut):
    rows, labels, valids = [], [], []
    for d, date in enumerate(dates):
        sp = raw_speed[d]; obs = np.isfinite(sp); queue = obs & (sp <= vcut)
        cum = np.vstack([np.zeros((1, queue.shape[1]), np.int32), np.cumsum(queue, 0, dtype=np.int32)])
        for t in range(12, 282):
            if elig[d, t - 12:t].mean() < .7:
                continue
            ongoing = bool(((cum[t] - cum[t - 12]) >= 2).any())
            fut = queue[t + 1:t + 7]
            if ongoing and fut.any():
                cond = "queue_ongoing"
            elif not ongoing and not fut[:5].any() and fut[5].any():
                cond = "queue_onset"
            else:
                continue
            valid = obs[t + 1:t + 7]
            pers = np.broadcast_to(queue[t - 1], fut.shape) & valid
            inter = int((pers & fut).sum()); union = int((pers | fut).sum())
            origin = pd.Timestamp(date, tz="UTC") + pd.Timedelta(minutes=5 * t)
            rows.append(dict(panel=panel, date=date, day=d, t=t, condition=cond,
                             forecast_origin=origin.strftime("%Y-%m-%dT%H:%M:%SZ"),
                             persistence_iou=inter / union if union else 1.0))
            labels.append(fut); valids.append(valid)
    return pd.DataFrame(rows), np.array(labels, bool), np.array(valids, bool)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    allw = []
    for panel in PANELS:
        c = load_cache(panel, "train")
        st = c.stations.sort_values("idx")
        assert (st.idx.to_numpy() == np.arange(len(st))).all()
        vcut = st.v_cut.to_numpy(np.float32)
        raw = [np.asarray(c[x + "_true"], np.float32) for x in ("speed", "flow", "occ")]
        msk = [np.asarray(c[x], np.float32) for x in ("speed", "flow", "occ")]
        elig = np.asarray(c.elig, bool)
        w, lab, val = candidates(panel, c.dates, raw[0], elig, vcut)
        static = np.stack([st.v_cut / st.free_speed_kmh, st.lanes / 5, np.log(st.length_km.clip(lower=.01)),
                           st.capacity_vph / 8000, st.has_on_ramp, st.has_off_ramp,
                           np.arange(len(st)) / len(st)]).astype(np.float32)
        np.savez(OUT / f"{panel}.npz", raw=cube(*raw, st), masked=cube(*msk, st), labels=lab, valid=val,
                 static=static, queued_raw=(np.isfinite(raw[0]) & (raw[0] <= vcut)),
                 observed_raw=np.isfinite(raw[0]), link_id=st.link_id.to_numpy().astype(str))
        w["row"] = np.arange(len(w)); allw.append(w)
        print(panel, len(w), w.condition.value_counts().to_dict(), flush=True)
    W = pd.concat(allw, ignore_index=True); W.to_parquet(OUT / "windows.parquet")
    # parity with Q's frozen ADV candidate list
    q = pd.read_parquet(ROOT / "artifacts/lane_q/0106/iq_adv_windows.parquet")
    mine = W[(W.date >= "2031-01-01") & (W.date <= "2031-02-28")]
    key = ["panel", "forecast_origin", "condition"]
    m = mine.merge(q, on=key, how="outer", indicator=True)
    print("ADV parity:", m._merge.value_counts().to_dict(),
          "max |persistence diff|", float((m.persistence_iou - m.history_last_step_persistence_proxy_iou).abs().max()))


if __name__ == "__main__":
    main()
