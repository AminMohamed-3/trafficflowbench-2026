"""asm2 (ledger 0475): regenerate the C03 onset rows (Q03) and ODME rows (NNLS .05) from the raw release, through the
reviewed src/tfb pipeline functions of the 0193 native recipe (configs/incumbent_0193_native.yaml), and write them as a delta
(submission_id, queue_pred on onset rows, path_flow on odme rows) for comparison with C03 (asm2_cmp_delta.py).

Steps (as tfb.pipeline.build, without the state/ongoing parts):
  1. queue prior refit from raw train onsets 2030-06-01..12-31 (sha must equal the reviewed prior 17e1f842...)
  2. onset LightGBM raw refit (features from raw, seed 0); pipeline asserts its probabilities equal the reviewed model
     artifacts/lane_q/0121/full_model on the fit frame (exact)
  3. predict_queue per Task 2 panel/split with the refit onset model (onset windows kept; ongoing rows ignored)
  4. predict_odme per panel/split (lane D NNLS lambda .05, own released counts and prior)
Usage: python docs/reference_code/asm2_qo.py <out.parquet> [--threads 3]  (writes <out>_onset.parquet and <out>_odme.parquet)
"""
import os, sys, json, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, yaml


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(); t0 = time.time()
    os.environ["TFB_Q_THREADS"] = str(a.threads)
    from tfb import pipeline as PL
    from tfb.paths import release_root, panels
    cfg = yaml.safe_load((ROOT / "configs/incumbent_0193_native.yaml").read_text(encoding="utf-8-sig"))
    art = ROOT / "artifacts/claude_ref/asm2/qo"; art.mkdir(parents=True, exist_ok=True)
    # the reviewed sha 17e1f842 is of a CRLF file (Windows text mode); json.dumps writes LF on Linux. Check the LF-normalized
    # bytes against the reviewed prior's LF-normalized sha (7e4a6dbe...) instead of the platform-dependent raw sha.
    import hashlib
    qc = dict(cfg["queue"]); qc.pop("expected_prior_sha256", None)
    prior, records = PL.prepare_queue_prior(ROOT, qc, art / "prior_u1.json")
    lf = hashlib.sha256(Path(prior).read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()
    ref = hashlib.sha256((ROOT / "artifacts/lane_q/0121/full_model/prior.json").read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()
    print("prior LF sha", lf, "reviewed LF sha", ref, "equal", lf == ref, flush=True)
    if lf != ref:
        raise ValueError("prior refit differs from the reviewed prior (LF-normalized)")
    print("prior ok", time.time() - t0, flush=True)
    onset_model, receipt = PL.prepare_onset_model(ROOT, cfg["queue"], prior, records, art, a.threads)
    print("onset refit", json.dumps({k: v for k, v in receipt.items() if k != "sha256"}), flush=True)
    qcfg = {"prior_path": str(prior), "release_root": str(release_root(ROOT)), "onset_mode": cfg["queue"].get("onset_mode", "site_prior"),
            "onset_model": onset_model, "num_threads": a.threads}
    from tfb.lane_q.predict import predict_queue
    from tfb.lane_d.predict import predict_odme
    key = pd.read_csv(ROOT / "data/raw/kaggle_public/submission_key.csv", dtype=str, keep_default_na=False)
    key["submission_id"] = key.submission_id.astype(np.int64)
    out = []
    qk = key[key.task == "queue"].copy(); qk["ts"] = pd.to_datetime(qk.timestamp, utc=True)
    wi = pd.concat([pd.read_csv(p, dtype=str) for p in (ROOT / "data/raw/kaggle_public/task2").glob("*/*/window_index.csv")])
    onset_w = set(wi.window_id[wi.condition == "queue_onset"])
    for split in ("validation", "private"):
        for panel in panels(ROOT):
            f = predict_queue(panel, split, qcfg)
            if not len(f):
                continue
            f = f[f.window_id.isin(onset_w)].copy(); f["ts"] = pd.to_datetime(f.timestamp, utc=True)
            m = qk[qk.split == split].merge(f[["window_id", "ts", "link_id", "queue_pred"]], on=["window_id", "ts", "link_id"], how="inner", validate="one_to_one")
            assert len(m) == len(f), (panel, split, len(m), len(f))
            out.append(pd.DataFrame({"submission_id": m.submission_id, "queue_pred": m.queue_pred.astype(float)}))
            print("queue", split, panel, len(m), flush=True)
    ok = key[key.task == "odme"]
    # Use this checkout's release explicitly, as the queue loader already does.
    # Git-free source mirrors and organizer source archives have no .git marker.
    dcfg = dict(cfg["odme"], release_root=str(release_root(ROOT)))
    for split in ("validation", "private"):
        for panel in sorted(set(ok.panel[ok.split == split])):
            f = predict_odme(panel, split, dcfg)
            g = ok[(ok.split == split) & (ok.panel == panel)]
            m = g.merge(f.astype({"departure_time": str, "path_id": str}), on=["panel", "departure_time", "path_id"], how="left", validate="one_to_one")
            assert m.path_flow.notna().all() and len(m) == len(g), (panel, split)
            out.append(pd.DataFrame({"submission_id": m.submission_id, "path_flow": m.path_flow.astype(float)}))
            print("odme", split, panel, len(m), flush=True)
    for col in ("queue_pred", "path_flow"):
        d = pd.concat([x for x in out if col in x], ignore_index=True).sort_values("submission_id")
        fn = a.out.replace(".parquet", f"_{'onset' if col == 'queue_pred' else 'odme'}.parquet")
        d.to_parquet(fn, index=False)
        print("wrote", fn, len(d), "rows", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
