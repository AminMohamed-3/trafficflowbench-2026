"""Expand Q0436's raw-data rebuilt site choices to complete onset rows, without a submission parent."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True, type=Path)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--delta", required=True, type=Path)
    a = ap.parse_args()
    choices = pd.read_csv(a.windows, keep_default_na=False).set_index("window_id")
    assert len(choices) == 80 and choices.index.is_unique
    wanted = set(choices.index)
    key = pd.concat([x[x.window_id.isin(wanted)] for x in pd.read_csv(
        ROOT / "data/raw/kaggle_public/submission_key.csv", chunksize=250000,
        usecols=["submission_id", "task", "window_id", "timestamp", "link_id"], dtype=str)], ignore_index=True)
    assert len(key) == 87000 and key.task.eq("queue").all()
    key["submission_id"] = key.submission_id.astype(np.int64)
    key["queue_pred"] = 0
    for wid, g in key.groupby("window_id", sort=False):
        row = choices.loc[wid]
        last = g.timestamp.eq(g.timestamp.max())
        selected = set(row.candidate.split(";")) - {""}
        assert selected <= set(g.link_id)
        key.loc[g.index, "queue_pred"] = (last & g.link_id.isin(selected)).astype(int)
    rows = key[["submission_id", "queue_pred"]].sort_values("submission_id").reset_index(drop=True)
    base = pd.read_parquet(a.base).set_index("submission_id").queue_pred.reindex(rows.submission_id)
    assert base.notna().all()
    delta = rows[rows.queue_pred.to_numpy() != base.to_numpy()]
    rows.to_parquet(a.out, index=False)
    # Retained onset8 uses CRLF; write it explicitly on every platform for byte parity.
    a.delta.write_bytes(delta.to_csv(index=False, lineterminator="\r\n").encode())
    receipt = dict(rows=len(rows), changed=len(delta), delta_sha256=hashlib.sha256(a.delta.read_bytes()).hexdigest())
    Path(str(a.out) + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
