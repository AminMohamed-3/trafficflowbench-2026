"""Claude qsel helper 2026-10-04: organizer selector replica for Task 2 ONGOING windows (train), checked on the 40
known train ongoing windows, and a selector-likeness score for every base-training candidate window.

Organizer config (ref/trafficflowbench-public/config/synthetic_release_v1.json, task2): 5 onset + 5 ongoing windows per
panel/split, minimum_spacing_minutes 360, minimum_history_coverage .7, maximum_persistence_iou .9, queue in horizon.
Replica rule (best of the variants tested on the 40 train windows, raw layer):
  gate   eligible share of history cells t-12..t-1 >= .7 (equals window_index history_coverage exactly)
  cond   some link with >= 2 queued ELIGIBLE steps in t-12..t-1 ("visible" queue), a queued observed cell in t+1..t+6
  pers   persistence IoU <= .9, persistence = last observed state per link in t-12..t-1, horizon = observed cells
  order  earliest first per day; >= 72 steps after the day's OFFICIAL onset window and earlier selected ongoing windows
Official = first 5 ongoing selections of the split.
Selector-likeness score s(d, t) of a training candidate (qcnn_data windows.parquet, ongoing rows): replica picks are
computed on every train day twice (no onset block; and blocked 72 steps after each onset-selector pick of that day,
the official-day situation); s = exp(-(t - pick) / TAU) for t >= pick (same day, nearest earlier pick; misses of the
replica are 1-3 steps late), times 1[pers <= .9]. Output: artifacts/claude_ref/qsel/train_selscore.parquet
(panel, date, t, pers_d, s) and replica_check.txt.
Usage: python docs/reference_code/qsel_replica.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
from tfb.data import load_cache

PANELS = ["D12_I5_N", "D12_I5_S", "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S"]
OUT = ROOT / "artifacts/claude_ref/qsel"
REL = ROOT / "data/raw/kaggle_public/task2"
SP, TAU, MAXP = 72, 4., .9


def onset_select(q, elig, spacing=SP, cov=.7):
    """Copy of onset_diag.select (onset selector replica, 39/40); onset_diag loads a Q artifact at import."""
    out = []
    for d in range(q.shape[0]):
        cum = np.vstack([np.zeros((1, q.shape[2]), np.int32), np.cumsum(q[d], 0, dtype=np.int32)])
        last = -10 ** 6
        for t in range(12, 282):
            if elig[d, t - 12:t].mean() < cov or ((cum[t] - cum[t - 12]) >= 2).any():
                continue
            if q[d, t + 1:t + 6].any() or not q[d, t + 6].any():
                continue
            if t - last < spacing:
                continue
            out.append((d, t)); last = t
    return out


def day_candidates(q, obs, elig, pers_def="lastobs", cond="elig"):
    """q, obs, elig: (288, L) bool for one day. Returns {t: pers} for gate+cond passing origins."""
    qc = q & elig if cond == "elig" else q
    cum = np.vstack([np.zeros((1, q.shape[1]), np.int32), np.cumsum(qc, 0, dtype=np.int32)])
    out = {}
    for t in range(12, 282):
        if elig[t - 12:t].mean() < .7 or not ((cum[t] - cum[t - 12]) >= 2).any():
            continue
        val = obs[t + 1:t + 7]; fut = q[t + 1:t + 7] & val
        if not fut.any():
            continue
        if pers_def == "last":
            last = q[t - 1]
        else:
            last = np.zeros(q.shape[1], bool)
            for s in range(t - 12, t):
                last = np.where(obs[s], q[s], last)
        p = np.broadcast_to(last, fut.shape) & val; u = (p | fut).sum()
        out[t] = (p & fut).sum() / u if u else 1.
    return out


def picks(cands, blocked=(), maxp=MAXP):
    sel = list(blocked); out = []
    for t in sorted(cands):
        if cands[t] <= maxp and all(abs(t - s) >= SP for s in sel):
            sel.append(t); out.append(t)
    return out


def official(panel, split, cond):
    w = pd.read_csv(REL / panel / split / "window_index.csv"); w = w[w.condition == cond].copy()
    ts = pd.to_datetime(w.forecast_origin); w["t"] = ts.dt.hour * 12 + ts.dt.minute // 5
    return list(zip(w.date.astype(str), w.t))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    W = pd.read_parquet(ROOT / "artifacts/claude_ref/qcnn/windows.parquet")
    W = W[W.condition == "queue_ongoing"]
    variants = {"chosen (elig cond, lastobs pers)": ("lastobs", "elig"), "last-step pers": ("last", "elig"),
                "obs cond": ("lastobs", "obs")}
    hits = {k: [0, 0] for k in variants}; lines, rows = [], []
    for p in PANELS:
        c = load_cache(p, "train"); st = c.stations.sort_values("idx"); vcut = st.v_cut.to_numpy(np.float32)
        sp = np.asarray(c["speed_true"], np.float32); obs = np.isfinite(sp); q = obs & (sp <= vcut); del sp
        elig = np.asarray(c["elig"], bool); dates = list(c.dates)
        on_off = official(p, "train", "queue_onset"); og_off = official(p, "train", "queue_ongoing")
        for name, (pdef, cond) in variants.items():
            sel = []
            for d in range(len(dates)):
                blk = [t for dd, t in on_off if dd == dates[d]]
                sel += [(dates[d], t) for t in picks(day_candidates(q[d], obs[d], elig[d], pdef, cond), blk)]
                if len(sel) >= 5:
                    break
            sel = sel[:5]
            hits[name][0] += len(set(sel) & set(og_off))
            hits[name][1] += sum(any(a == b and 0 <= tb - ta <= 3 for b, tb in og_off) for a, ta in sel)
            lines.append(f"{p} {name}: replica {sel} | official {og_off}")
        # selector-likeness on every train day
        on = {}
        for d, t in onset_select(q, elig):
            on.setdefault(d, []).append(t)
        wp = W[W.panel == p]
        for d, g in wp.groupby("day"):
            cd = day_candidates(q[d], obs[d], elig[d])
            pk = sorted(set(picks(cd)) | set(picks(cd, on.get(d, []))))
            for t in g.t.to_numpy():
                pr = [x for x in pk if x <= t]
                pd_ = cd.get(t, np.nan)
                s = np.exp(-(t - pr[-1]) / TAU) if pr and pd_ <= MAXP else 0.
                rows.append((p, dates[d], int(t), pd_, s, int((dates[d], int(t)) in set(og_off))))
        print(p, flush=True)
    for k, (a, b) in hits.items():
        lines.append(f"TOTAL {k}: exact {a}/40, official 0-3 steps after a replica pick {b}/40")
    (OUT / "replica_check.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines[-len(hits):]))
    df = pd.DataFrame(rows, columns=["panel", "date", "t", "pers_d", "s", "official"])
    df.to_parquet(OUT / "train_selscore.parquet")
    print("train ongoing windows", len(df), "s>0", int((df.s > 0).sum()), "sum s", round(df.s.sum(), 1),
          "| official windows s:", df[df.official == 1].s.round(2).tolist())
    print("pers_d by s>0.5:", df.groupby(df.s > .5).pers_d.describe().round(3).to_string())


if __name__ == "__main__":
    main()
