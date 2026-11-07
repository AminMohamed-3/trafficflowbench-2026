"""ong2 2026-10-05: classify official ongoing windows by what the organizer's EARLIEST-origin selector implies.
  A 'crossing': origin t-1 was allowed (>= 72 steps after the day's onset window and earlier ongoing windows) and
    passed the pre-T gate (eligible coverage >= .7 over t-13..t-2, some link with >= 2 queued eligible steps there),
    so its persistence IoU was > .9 (its horizon t..t+5 held a queue: ongoing queue, assumed).
  B 'block-end': t-1 not allowed (t is the first allowed origin).
  C 'first-visible': t-1 allowed but failed the pre-T gate (queue newly visible).
Pre-T data only: window_history (t-12..t-1, with is_score_eligible) and the released masked layer at t-13.
Out: artifacts/claude_ref/ong2/official_types.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
from tfb.data import load_cache
from qcnn_train import PANELS

REL = ROOT / "data/raw/kaggle_public/task2"
OUT = ROOT / "artifacts/claude_ref/ong2"


def tstep(s):
    T = pd.Timestamp(s); return T.hour * 12 + T.minute // 5


def classify(panel, split):
    c = load_cache(panel, split); st = c.stations.sort_values("idx"); links = st.link_id.astype(str).tolist()
    col = {l: i for i, l in enumerate(links)}; vcut = st.v_cut.to_numpy(np.float32); L = len(links)
    wi = pd.read_csv(REL / panel / split / "window_index.csv"); wi["t"] = wi.forecast_origin.map(tstep)
    hist = pd.read_parquet(REL / panel / split / "window_history.parquet")
    sp_m = np.asarray(c["speed"], np.float32); el_m = np.asarray(c["elig"], bool)
    rows = []
    for w in wi[wi.condition == "queue_ongoing"].itertuples():
        t = w.t; same = wi[(wi.date == w.date) & (wi.t < t)]
        blocks = same.t.tolist(); allowed_t1 = all(t - 1 - b >= 72 for b in blocks)
        h = hist[hist.window_id == w.window_id]
        k = ((pd.to_datetime(h.timestamp) - pd.Timestamp(w.forecast_origin).normalize()).dt.total_seconds() // 300).astype(int) - (t - 12)
        q = np.zeros((13, L), bool); e = np.zeros((13, L), bool); ob = np.zeros((13, L), bool)   # rows: t-13..t-1
        j = h.link_id.astype(str).map(col).to_numpy(); kk = k.to_numpy() + 1
        v = h.speed_kmh.to_numpy(); fin = np.isfinite(v)
        ob[kk, j] = fin; q[kk, j] = fin & (v <= vcut[j]); e[kk, j] = h.is_score_eligible.to_numpy().astype(bool)
        d = c.day_index(str(w.date))
        if t - 13 >= 0:
            s13 = sp_m[d, t - 13]; ob[0] = np.isfinite(s13); q[0] = ob[0] & (s13 <= vcut); e[0] = el_m[d, t - 13]
        cov1 = e[0:12].mean(); vis1 = ((q[0:12] & e[0:12]).sum(0) >= 2).any()
        cov0 = e[1:13].mean(); vis0 = ((q[1:13] & e[1:13]).sum(0) >= 2).any()
        typ = "B" if not allowed_t1 else ("A" if (cov1 >= .7 and vis1) else "C")
        rows.append(dict(panel=panel, split=split, window_id=w.window_id, date=w.date, t=t, typ=typ,
                         gap=(t - max(blocks)) if blocks else np.nan, cov_t1=round(cov1, 3), vis_t1=vis1,
                         cov_t=round(cov0, 3), wi_cov=w.history_coverage, vis_t=vis0, obs_t13=int(ob[0].sum())))
    return rows


if __name__ == "__main__":
    rows = []
    for split in sys.argv[1:] or ("train", "validation", "private"):
        for p in PANELS:
            rows += classify(p, split)
    df = pd.DataFrame(rows); df.to_csv(OUT / "official_types.csv", index=False)
    print(df.groupby(["split", "typ"]).size().unstack(fill_value=0))
    print("pre-T gate replica check (cov_t vs window_index coverage, max abs diff):", (df.cov_t - df.wi_cov).abs().max(),
          "| vis_t share", df.vis_t.mean())
