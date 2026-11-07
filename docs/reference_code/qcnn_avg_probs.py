"""Claude 2026-10-01: seed ensemble of adapted CNN predictions (qcnn_predict_adapt.py parquet outputs).
Averages prob per (window_id, timestamp, link_id), then expected-IoU decoding per window (Q04's rule).
Usage: python docs/reference_code/qcnn_avg_probs.py <out.csv> <a.parquet> <b.parquet> [...]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.append(str(Path(__file__).resolve().parent))
from qcnn_shift import eiou_mask

KEY = ["window_id", "timestamp", "link_id"]


def main(out, *parts):
    dfs = [pd.read_parquet(p) for p in parts]
    base = dfs[0][KEY].copy()
    for d in dfs[1:]:
        assert (d[KEY].to_numpy() == base.to_numpy()).all(), "row order differs"
    base["prob"] = np.mean([d.prob.to_numpy() for d in dfs], 0)
    base["queue_pred"] = 0
    for w, idx in base.groupby("window_id").indices.items():
        base.loc[base.index[idx], "queue_pred"] = eiou_mask(base.prob.to_numpy()[idx]).astype(int)
    for i, d in enumerate(dfs):
        print("agreement with part", i, round(float((d.queue_pred.to_numpy() == base.queue_pred.to_numpy()).mean()), 4))
    print("queued share", round(float(base.queue_pred.mean()), 4))
    base[KEY + ["queue_pred"]].to_csv(out, index=False); base.to_parquet(Path(out).with_suffix(".parquet"))


if __name__ == "__main__":
    main(*sys.argv[1:])
