"""Claude onset5 2026-10-04: official window selector replica (onset AND ongoing) for official-day gap emulation.

Per day, origins t = 12..281 in time order, history coverage (eligible share of t-12..t-1) >= .7 (as lane-q candidate_windows):
  onset:   no link with >= 2 queued steps in t-12..t-1, nothing queued at t+1..t+5, something queued at t+6;
  ongoing: some link with >= 2 queued steps in t-12..t-1 and something queued in t+1..t+6, persistence proxy IoU
           (queue at t-1 repeated, valid = observed) <= .9 (synthetic_release_v1 maximum_persistence_iou).
Earliest-first, >= 72 steps between origins of the same condition (within a day). Official = first 5 per condition from
the start day. Officially the masked layer then hides steps T+1..T+18 (T+5..T+90) on all links for all 10 windows.
check(): replica vs released train/validation/private window_index (train: raw layer, test: masked layer).
Usage: python docs/reference_code/onset5_select.py   (runs check)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from tfb.data import load_cache
PANELS = ["D12_I5_N", "D12_I5_S", "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S"]
SPACING = 72


def day_candidates(q, obs, elig):
    """q, obs, elig: (288, L) bool for one day. Returns (onset_ts, ongoing_ts) after per-day spacing."""
    cum = np.vstack([np.zeros((1, q.shape[1]), np.int32), np.cumsum(q, 0, dtype=np.int32)])
    on, og = [], []; lo = lg = -10 ** 6
    for t in range(12, 282):
        if elig[t - 12:t].mean() < .7:
            continue
        ong = ((cum[t] - cum[t - 12]) >= 2).any(); fut = q[t + 1:t + 7]
        if ong and fut.any():
            v = obs[t + 1:t + 7]; pers = np.broadcast_to(q[t - 1], fut.shape) & v
            u = (pers | fut).sum(); iou = (pers & fut).sum() / u if u else 1.
            if iou <= .9 and t - lg >= SPACING:
                og.append(t); lg = t
        elif not ong and not fut[:5].any() and fut[5].any():
            if t - lo >= SPACING:
                on.append(t); lo = t
    return on, og


def split_candidates(p, split, layer):
    c = load_cache(p, split); st = c.stations.sort_values("idx"); vc = st.v_cut.to_numpy(np.float32)
    sp = np.asarray(c["speed_true" if layer == "raw" else "speed"], np.float32); elig = np.asarray(c["elig"], bool)
    obs = np.isfinite(sp); q = obs & (sp <= vc)
    out = {}
    for d in range(sp.shape[0]):
        out[d] = day_candidates(q[d], obs[d], elig[d])
    return list(c.dates), out


def official_from(cands, dates, s, n=5):
    """First n onset and n ongoing (d, t) from start day s (same month)."""
    mon = dates[s][:7]; on, og = [], []
    for d in range(s, len(dates)):
        if dates[d][:7] != mon or (len(on) >= n and len(og) >= n):
            break
        on += [(d, t) for t in cands[d][0]]; og += [(d, t) for t in cands[d][1]]
    return on[:n], og[:n]


def check():
    for p in PANELS:
        for split, layer in (("train", "raw"), ("validation", "masked"), ("private", "masked")):
            dates, cands = split_candidates(p, split, layer)
            on, og = official_from(cands, dates, 0)
            wi = pd.read_csv(ROOT / f"data/raw/kaggle_public/task2/{p}/{split}/window_index.csv")
            ts = pd.to_datetime(wi.forecast_origin); wi["t"] = ts.dt.hour * 12 + ts.dt.minute // 5
            wi["d"] = wi.date.map({x: i for i, x in enumerate(dates)})
            offo = set(zip(wi[wi.condition == "queue_onset"].d, wi[wi.condition == "queue_onset"].t))
            offg = set(zip(wi[wi.condition == "queue_ongoing"].d, wi[wi.condition == "queue_ongoing"].t))
            print(p, split, layer, "onset", len(set(on) & offo), "/5 ongoing", len(set(og) & offg), "/5",
                  "| ongoing replica", sorted(og), "official", sorted(offg), flush=True)


if __name__ == "__main__":
    check()
