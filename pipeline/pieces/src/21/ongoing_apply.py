"""Put new ongoing queue rows into a submission file (rows 21, 22, 23; worker R2 piece worker, 2026-10-21).

  python ongoing_apply.py PARENT.csv NEW_ROWS.parquet OUT.csv [--old-rows OLD_ROWS.parquet] [--report OUT.json]

NEW_ROWS / OLD_ROWS: the 87,000 official ongoing queue rows (submission_id, queue_pred), as written by
docs/reference_code/asm2_bbmember.py decode. OUT = PARENT with queue_pred replaced on the ids where NEW_ROWS differs from
PARENT; nothing else changes. This is what the development chain did (exp/L_lab/a3/a3.py delta: seed = changed rows,
tools/probe.py expand PARENT seed, then make + expand of the delta: the same values), done in one pass here: the parent is
read and written exactly as tools/probe.py cmd_expand does (read_full with float_precision round_trip, to_csv with the
parent's line ending; probe.py is copied unchanged next to this file, sha 2b5ec084).
Checks: NEW_ROWS covers exactly the 87,000 ongoing ids, all queue-task rows of PARENT, values 0/1; with --old-rows, the
parent's queue_pred on those ids equals OLD_ROWS (the parent chain's ongoing rows), so a wrong parent stops the run.
Prints one JSON line (changed cells by split via the key file is left to the window guard; here: total and ids)."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe as P                                             # unchanged copy of tools/probe.py


def rows(p):
    d = pd.read_parquet(p, use_threads=False)
    assert list(d.columns)[:2] == ["submission_id", "queue_pred"] and len(d) == 87000 and d.submission_id.is_unique, p
    s = d.set_index("submission_id").queue_pred.astype(float).sort_index()
    assert np.isin(s.to_numpy(), (0., 1.)).all(), f"{p}: queue_pred not 0/1"
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("parent"); ap.add_argument("new"); ap.add_argument("out")
    ap.add_argument("--old-rows"); ap.add_argument("--report")
    a = ap.parse_args()
    new = rows(a.new)
    p = P.read_full(a.parent)
    assert len(p) == P.N_ROWS, len(p)
    idx = pd.Index(p.submission_id).get_indexer(new.index)
    assert (idx >= 0).all(), "ongoing ids missing from the parent"
    assert (p.task.astype(str).to_numpy()[idx] == "queue").all(), "an ongoing id is not a queue row"
    cur = p.queue_pred.to_numpy()[idx].astype(float)
    rep = dict(parent=str(a.parent), new_rows=str(a.new), parent_sha256=P.sha256(a.parent))
    if a.old_rows:
        old = rows(a.old_rows).reindex(new.index)
        bad = int((old.to_numpy() != cur).sum())
        rep["parent_vs_old_rows_cells"] = bad
        assert bad == 0, f"parent queue_pred differs from OLD_ROWS in {bad} ongoing cells: wrong parent or wrong base-build rows"
    ch = new.to_numpy() != cur
    rep["changed_cells"] = int(ch.sum())
    col = p["queue_pred"].to_numpy().copy()
    col[idx[ch]] = new.to_numpy()[ch]
    p["queue_pred"] = col
    p.to_csv(a.out, index=False, lineterminator=P.line_ending(a.parent))
    rep.update(out=str(a.out), out_sha256=P.sha256(a.out))
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1))
    print("ONGOING_APPLY", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
