"""ong2 2026-10-05 (ledger 0469): selector-type-aware ongoing decode on official windows, delta on C17.
Type A official ongoing windows (ong2_official_types.py: t-1 allowed by the 72-step spacing and passing the pre-T gate,
so the organizer's persistence IoU at t-1 was > .9) get: C03 eiou mask -> selcond c .92 with the selector's persistence
(last observed state over t-12..t-1, not t-12..t) -> band raise (ong2_band_rule.raise_to) to IoU(mask, pers) >= .8.
Type B / C windows keep C17 (C03 + shipped selcond). Checks: C17 ongoing rows are reproduced from C03 probabilities
+ shipped selcond (gapres_selcond_official logic) on every official window; only queue_pred of type A ongoing rows change.
Out: artifacts/claude_ref/ong2/<name>.delta.csv.gz (submission_id, queue_pred; changed rows only), manifest json with
sha256 of the full CSV (C17 writer), per-window table <name>_windows.csv.
Options (ledger 0471): --sc=<parquet> type A windows use the selection-conditioned CNN probabilities (ong2_sccnn_official.py,
window_id/timestamp/link_id/prob) with their own eiou mask before selcond + raise; --noT selcond with t-12..t-1 persistence on
type B/C windows too (pseudo hard non-crossing windows +.0014, both months and halves).
Usage: python ong2_probe.py build [lo] [--sc=path] [--noT] [--name=x] | expand <name> <out.csv>"""
import os
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(k, "3")
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.append(str(ROOT / "docs/reference_code"))
from tfb.data import load_cache
from qcnn_data import cube
from gapres_selcond_rule import constrained, iou, eiou_order
from ong2_band_rule import raise_to
import asm_build as AB

OUT = ROOT / "artifacts/claude_ref/ong2"
C17 = ROOT / "artifacts/claude_ref/probes/c17_k6_s0447_state_on_c15.csv"
REL = ROOT / "data/raw/kaggle_public/task2"


def build(*args):
    lo = float(next((a for a in args if not a.startswith("--")), ".8"))
    scf = next((a[5:] for a in args if a.startswith("--sc=")), None); noT = "--noT" in args
    name = next((a[7:] for a in args if a.startswith("--name=")), None) or f"ong2_bandA{int(lo * 100)}_on_c17"
    # Serving mode uses only row probabilities and the released data, without a parent submission.
    rows_out = next((a.split("=", 1)[1] for a in args if a.startswith("--ongoing-out=")), None)
    base_probs = next((a.split("=", 1)[1] for a in args if a.startswith("--base-probs=")),
                      "artifacts/claude_ref/qcnn/c03_adapt_npt_ens2_ongoing.parquet")
    sc = pd.read_parquet(ROOT / scf).set_index(["window_id", "timestamp", "link_id"]).prob if scf else None
    types = pd.read_csv(OUT / "official_types.csv"); types = types[types.split != "train"].set_index("window_id")
    pr = pd.read_parquet(ROOT / base_probs)
    windows = set(pr.window_id)
    key = pd.concat([x[x.window_id.isin(windows)] for x in pd.read_csv(
        AB.KEY, usecols=["submission_id", "task", "window_id", "timestamp", "link_id"], dtype=str, chunksize=250000)], ignore_index=True)
    if rows_out:
        p17 = key[["submission_id"]].astype(np.int64).sort_values("submission_id").reset_index(drop=True)
        p17["queue_pred"] = np.nan
    else:
        p17 = pd.read_csv(C17, float_precision="round_trip")
        h17 = AB.sha(C17); assert AB.csv_sha(p17) == h17, "writer does not reproduce C17 bytes"
    sid = p17.submission_id.to_numpy()
    km = key.assign(submission_id=key.submission_id.astype(np.int64)).set_index(["window_id", "timestamp", "link_id"]).submission_id
    qnew = p17.queue_pred.to_numpy(float).copy(); wins = []
    for split in ("validation", "private"):
        for p in sorted(types[types.split == split].panel.unique()):
            c = load_cache(p, split); st = c.stations.sort_values("idx"); links = st.link_id.astype(str).tolist()
            c3 = cube(*[np.asarray(c[x], np.float32) for x in ("speed", "flow", "occ")], st)
            obs, q = c3[:, :, 3] > .5, c3[:, :, 4] > .5
            for wid, w in types[(types.split == split) & (types.panel == p)].iterrows():
                g = pr[pr.window_id == wid].sort_values(["timestamp", "link_id"])
                ts = sorted(g.timestamp.unique()); H = len(ts); L = len(g) // H
                idx = np.array([links.index(l) for l in g.link_id.to_numpy()[:L]])
                d = c.day_index(str(w.date)); t = int(w.t)
                lastT = np.zeros(c3.shape[-1], bool); last = None
                for k_ in range(t - 12, t + 1):
                    if k_ == t:
                        last = lastT.copy()
                    lastT = np.where(obs[d, k_], q[d, k_], lastT)
                pmT = np.broadcast_to(lastT[idx], (H, L)).ravel(); pm = np.broadcast_to(last[idx], (H, L)).ravel()
                f = g.prob.to_numpy(np.float64); base = g.queue_pred.to_numpy().astype(bool)
                o = np.argsort(-f); k = int(base.sum())
                m17, _ = constrained(f, o, k, pmT, .92)
                rows = np.searchsorted(sid, km.loc[list(zip(g.window_id, g.timestamp, g.link_id))].to_numpy())
                assert (sid[rows] == km.loc[list(zip(g.window_id, g.timestamp, g.link_id))].to_numpy()).all()
                if rows_out:
                    p17.loc[rows, "queue_pred"] = m17.astype(float)
                else:
                    assert np.array_equal(p17.queue_pred.to_numpy(float)[rows], m17.astype(float)), f"C17 not reproduced {wid}"
                new = m17
                if noT and w.typ != "A":
                    new, _ = constrained(f, o, k, pm, .92)
                if w.typ == "A":
                    if sc is not None:
                        f = sc.loc[list(zip(g.window_id, g.timestamp, g.link_id))].to_numpy(np.float64); k, o = eiou_order(f)
                    ms, _ = constrained(f, o, k, pm, .92)
                    new = raise_to(ms, f, pm, lo) if iou(ms, pm) < lo else ms
                qnew[rows] = new.astype(float)
                wins.append(dict(window_id=wid, split=split, panel=p, typ=w.typ, n_c17=int(m17.sum()), n_new=int(new.sum()),
                                 changed=int((new != m17).sum()), pp_c17=round(iou(m17, pm), 4), pp_new=round(iou(new, pm), 4),
                                 pp_base=round(iou(base, pm), 4)))
    wd = pd.DataFrame(wins)
    if rows_out:
        assert len(p17) == 87000 and not p17.queue_pred.isna().any() and np.isin(qnew, [0., 1.]).all()
        target = ROOT / rows_out; target.parent.mkdir(parents=True, exist_ok=True)
        p17["queue_pred"] = qnew
        p17.to_parquet(target, index=False)
        wd.to_csv(str(target) + ".windows.csv", index=False)
        print("wrote ongoing rows", target, len(p17), "changed vs C17", int(wd.changed.sum()), flush=True)
        return
    wd.to_csv(OUT / f"{name}_windows.csv", index=False)
    df = p17.copy(); df["queue_pred"] = qnew.astype(p17.queue_pred.dtype)
    ch = np.flatnonzero(qnew != p17.queue_pred.to_numpy(float))
    assert np.isfinite(qnew).all() and len(df) == 6980503 and not df[["speed_kmh", "flow_vph", "queue_pred"]].isna().any().any()
    dl = pd.DataFrame({"submission_id": sid[ch], "queue_pred": qnew[ch]})
    f = OUT / f"{name}.delta.csv.gz"; dl.to_csv(f, index=False, compression="gzip")
    sp = pd.read_csv(AB.KEY, usecols=["split"], dtype="category").split.to_numpy()[ch]
    cw = wd[wd.changed > 0]
    mf = dict(name=name, parent=str(C17.relative_to(ROOT)).replace("\\", "/"), parent_sha256=h17, sha256=AB.csv_sha(df),
              delta=str(f.relative_to(ROOT)).replace("\\", "/"), delta_sha256=AB.sha(f), delta_rows=int(len(ch)),
              delta_columns=["queue_pred"], changed_validation=int((sp == "validation").sum()), changed_private=int((sp == "private").sum()),
              changed_windows_validation=int((cw.split == "validation").sum()), changed_windows_private=int((cw.split == "private").sum()),
              type_counts=wd.groupby(["split", "typ"]).size().unstack(fill_value=0).to_dict("index"), rows=int(len(df)), lo=lo, sc=scf, noT=noT)
    (OUT / f"{name}.manifest.json").write_text(json.dumps(mf, indent=1, default=int))
    print(json.dumps(mf, indent=1, default=int)); print(cw.to_string())


def expand(name, out):
    mf = json.loads((OUT / f"{name}.manifest.json").read_text())
    assert AB.sha(C17) == mf["parent_sha256"] and AB.sha(ROOT / mf["delta"]) == mf["delta_sha256"]
    df = pd.read_csv(C17, float_precision="round_trip"); dl = pd.read_csv(ROOT / mf["delta"], float_precision="round_trip")
    rows = np.searchsorted(df.submission_id.to_numpy(), dl.submission_id.to_numpy())
    assert (df.submission_id.to_numpy()[rows] == dl.submission_id.to_numpy()).all()
    v = df.queue_pred.to_numpy(float).copy(); v[rows] = dl.queue_pred.to_numpy(float); df["queue_pred"] = v.astype(df.queue_pred.dtype)
    h = AB.write_csv(df, out); assert h == mf["sha256"], (h, mf["sha256"])
    print("wrote", out, "sha256", h, "(matches manifest)")


if __name__ == "__main__":
    build(*sys.argv[2:]) if sys.argv[1] == "build" else expand(sys.argv[2], sys.argv[3])
