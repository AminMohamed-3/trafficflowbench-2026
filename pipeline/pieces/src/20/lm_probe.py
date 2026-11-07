"""Row 20 (C54) file step: parent C50 + the L_micro state parquet -> C54 (final_solution copy, R2 worker).

Parameterised copy of exp/L_micro/lm_probe.py (sha256 e823357ed6b59ec92d056a00bd1b5a8a92a75328d0111ff5987960e1ea995f22).
Parent state rows equal to the control build (|d| <= 1e-6) take the new values; the 128,465 gap2 rows of the parent and every
queue / ODME row stay. Differences, none of which changes the written file:
  - parent, control, output and info paths are arguments (original: C50 artifacts/claude_ref/probes/c_gap2w_on_c47.csv, control
    artifacts/claude_ref/c_b6long/p24k/official/t1.csv.state.parquet = the C47 state build, output under artifacts/claude_ref/l_micro);
  - the control may also be given as the C47 submission CSV (its state rows = that state build, written by repr round trip;
    reindexed to the new parquet's submission_id order); the original took the parquet;
  - the tools/probe.py make / diff calls after the write are left out (they only write a delta and print).
ROOT = $TFB_CODE_ROOT (for data/raw/kaggle_public/submission_key.csv).
Usage: python lm_probe.py STATE.parquet PARENT.csv CONTROL(.parquet|.csv) OUT.csv INFO.json"""
import os, sys, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(os.environ["TFB_CODE_ROOT"]).resolve()
KEY = ROOT / "data/raw/kaggle_public/submission_key.csv"; N_GAP2 = 128465; N = 6980503


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def control(path, ids):
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    c = pd.read_csv(path, float_precision="round_trip", usecols=["submission_id", "task", "speed_kmh", "flow_vph"], dtype={"task": "category"})
    c = c[c.task == "state"].set_index("submission_id").reindex(ids)
    assert c.speed_kmh.notna().all() and c.flow_vph.notna().all() and len(c) == len(ids), "control CSV lacks state rows of the build"
    return pd.DataFrame({"submission_id": ids, "speed_kmh": c.speed_kmh.to_numpy(), "flow_vph": c.flow_vph.to_numpy()})


def main(state_parquet, parent, ctl_path, full, info_path):
    C50 = Path(parent); full = Path(full)
    new = pd.read_parquet(state_parquet); ctl = control(ctl_path, new.submission_id.to_numpy())
    assert new.submission_id.equals(ctl.submission_id)
    par = pd.read_csv(C50, float_precision="round_trip"); assert len(par) == N
    pos = pd.Index(par.submission_id).get_indexer(ctl.submission_id); assert (pos >= 0).all() and (par.task.iloc[pos] == "state").all()
    assert (par.task == "state").sum() == len(ctl)
    same = (np.abs(par.speed_kmh.to_numpy()[pos] - ctl.speed_kmh.to_numpy()) <= 1e-6) & (np.abs(par.flow_vph.to_numpy()[pos] - ctl.flow_vph.to_numpy()) <= 1e-6)
    print("parent state rows equal to the control state build:", int(same.sum()), "kept (gap2 rows):", int((~same).sum()), flush=True)
    assert int((~same).sum()) == N_GAP2, "parent gap2 row count differs"
    other = (par.task != "state").to_numpy()
    keep = {c: par[c].to_numpy()[other].copy() for c in ("speed_kmh", "flow_vph", "queue_pred", "path_flow")}
    qp = par.queue_pred.to_numpy().copy(); pf = par.path_flow.to_numpy().copy(); ids0 = par.submission_id.to_numpy().copy(); task0 = par.task.to_numpy().copy()
    old = {c: par[c].to_numpy()[pos].copy() for c in ("speed_kmh", "flow_vph")}
    p2 = pos[same]
    par.loc[p2, "speed_kmh"] = new.speed_kmh.to_numpy()[same]; par.loc[p2, "flow_vph"] = new.flow_vph.to_numpy()[same]
    for c, v in keep.items():
        assert np.array_equal(par[c].to_numpy()[other], v, equal_nan=True), c
    assert np.array_equal(par.queue_pred.to_numpy(), qp, equal_nan=True) and np.array_equal(par.path_flow.to_numpy(), pf, equal_nan=True)
    assert np.array_equal(par.submission_id.to_numpy(), ids0) and (par.task.to_numpy() == task0).all() and len(par) == N
    sv, sq = par.speed_kmh.to_numpy()[pos], par.flow_vph.to_numpy()[pos]
    assert np.isfinite(sv).all() and np.isfinite(sq).all() and (sv > 0).all() and (sq >= 0).all()
    blanks = {c: int(par[c].isna().sum()) for c in par.columns}
    ch = {c: (par[c].to_numpy()[pos] != old[c]) for c in old}; anyc = ch["speed_kmh"] | ch["flow_vph"]
    key = pd.read_csv(KEY, usecols=["submission_id", "split"])
    ksp = key.split.to_numpy()[pd.Index(key.submission_id).get_indexer(par.submission_id.to_numpy()[pos])]; del key
    by_split = {sp: dict(state_rows=int((ksp == sp).sum()), rows_changed=int(anyc[ksp == sp].sum())) for sp in ("validation", "private")}
    gap2_changed = int(anyc[~same].sum()); assert gap2_changed == 0
    eol = "\r\n" if open(C50, "rb").readline().endswith(b"\r\n") else "\n"
    assert not full.exists(), full
    par.to_csv(full, index=False, lineterminator=eol)
    info = dict(probe=str(full), sha256=sha(full), parent=str(C50), parent_sha256=sha(C50), rows=len(par), blanks_by_column=blanks,
                state_rows_replaced=int(same.sum()), gap2_rows_kept=int((~same).sum()), rows_changed=int(anyc.sum()), rows_changed_by_split=by_split,
                queue_odme_rows_unchanged=True, state_parquet=str(state_parquet), control=str(ctl_path))
    Path(info_path).write_text(json.dumps(info, indent=1))
    print("PROBE", json.dumps(info), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:6])
