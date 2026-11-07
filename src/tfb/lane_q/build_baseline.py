"""Train and deliver Q-02 priors/predictions; sanity uses only I-Q-OFF."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import unittest

import numpy as np
import pandas as pd
import psutil

from .windows import TASK2_PANELS, Release, candidate_windows, online_view, project_root
from .predict import MODEL_VERSION, calendar_features, fit_panel_prior, predict_queue


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def peak_mb():
    info = psutil.Process().memory_info()
    value = getattr(info, "peak_wset", info.rss) / 1024**2
    if value > float(os.environ.get("TFB_Q_RAM_MB","4096")):
        raise MemoryError(f"Q memory cap {os.environ.get('TFB_Q_RAM_MB','4096')} MiB exceeded")
    return value


def collect_onsets(release, panel, start, end):
    rows = []
    network = release.network(panel)
    for date in release.files(panel, "train", "mainline_states"):
        if not start <= date <= end:
            continue
        day = release.train_grid(panel, date)
        candidates = candidate_windows(day)
        for w in candidates[candidates.condition.eq("queue_onset")].itertuples(index=False):
            view = online_view(panel, "train", date, w.forecast_origin, release=release)
            hour, day_type = calendar_features(view)
            step = view.origin.hour * 12 + view.origin.minute // 5
            target = [network.links[i] for i in np.flatnonzero(day.queued[step + 6])]
            rows.append({"panel": panel, "date": date, "forecast_origin": w.forecast_origin,
                         "hour": hour, "day_type": day_type, "target_links": target})
        peak_mb()
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--id", default="0114")
    cli.add_argument("--main-root", type=Path, default=project_root())
    cli.add_argument("--fit-only", action="store_true")
    args = cli.parse_args()
    root = args.main_root
    report = root / "exp" / f"{args.id}_q02_baseline"
    out = root / "artifacts/lane_q" / args.id
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    # Actual predictor future/ID/ordering tests, not just accessor tests.
    from . import test_predict, test_windows
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(module)
                               for module in [test_windows, test_predict]])
    if not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful():
        raise SystemExit("Prediction poison/output tests failed; no scores logged")
    release = Release(root / "data/raw/kaggle_public")
    fit = {"start": "2030-06-01", "end": "2030-12-31", "max_position_gap": 2}
    prior = {"model_version": MODEL_VERSION, "fit": fit, "panels": {}}
    off_prior = {"model_version": MODEL_VERSION, "fit": {**fit, "exclude": ["2030-06-01", "2030-06-10"]}, "panels": {}}
    for panel in TASK2_PANELS:
        checkpoint = out / f"{panel}_fit_onsets.parquet"
        if checkpoint.exists():
            frame = pd.read_parquet(checkpoint)
            rows = frame.to_dict("records")
            print(f"resume {panel}: {len(rows)} training onsets", flush=True)
        else:
            rows = collect_onsets(release, panel, fit["start"], fit["end"])
            pd.DataFrame(rows).to_parquet(checkpoint, index=False)
        if any(not fit["start"] <= row["date"] <= fit["end"] for row in rows):
            raise ValueError("Training checkpoint crossed the declared June-Dec fit dates")
        network = release.network(panel)
        prior["panels"][panel] = fit_panel_prior(rows, network.links, max_position_gap=fit["max_position_gap"])
        off_rows = [row for row in rows if not "2030-06-01" <= row["date"] <= "2030-06-10"]
        off_prior["panels"][panel] = fit_panel_prior(off_rows, network.links, max_position_gap=fit["max_position_gap"])
        # Checkpoint full deterministic model after every panel.
        (out / "prior_u1.json").write_text(json.dumps(prior, sort_keys=True, indent=2), encoding="utf-8")
        (out / "prior_off.json").write_text(json.dumps(off_prior, sort_keys=True, indent=2), encoding="utf-8")
        print(f"fit {panel}: {len(rows)} onsets, {len(prior['panels'][panel]['sites'])} sites; peak {peak_mb():.0f} MiB", flush=True)
    if args.fit_only:
        return
    # Import only after the lane path has been established.
    import tfb
    tfb.__path__.append(str(root / "src/tfb"))
    from tfb.metric import queue_score
    from .evaluate_baseline import sanity_report
    cfg = {"release_root": str(release.root), "prior_path": str(out / "prior_u1.json")}
    off_cfg = {**cfg, "prior_path": str(out / "prior_off.json")}
    prediction_chunks = {"train": [], "validation": [], "private": []}
    for split in prediction_chunks:
        for panel in TASK2_PANELS:
            values = predict_queue(panel, split, off_cfg if split == "train" else cfg)
            values.to_csv(out / f"{panel}_{split}_queue.csv", index=False)
            prediction_chunks[split].append(values)
        combined = pd.concat(prediction_chunks[split], ignore_index=True)
        combined.to_csv(out / f"queue_{split}.csv", index=False)
        print(f"prediction {split}: {len(combined)} exact binary template rows", flush=True)
    verified = json.loads((root / "exp/0110_q00_metric/result.json").read_text(encoding="utf-8"))
    if verified["metric_source_sha256"] != sha(root / "src/tfb/metric.py"):
        raise RuntimeError("Metric version changed: re-run Q-00 before baseline scores; prior and prediction checkpoints are preserved")
    score = sanity_report(release, pd.concat(prediction_chunks["train"], ignore_index=True), off_cfg, out, root, queue_score)
    result = {"id": args.id, "lane": "Q", "formulation": "Q-02 canonical persistence and onset site prior v0", "commit": None,
              "config": f"exp/{args.id}_q02_baseline/config.yaml", "utc": datetime.now(timezone.utc).isoformat(),
              "instrument": "I-Q-OFF sanity (80 official windows); ADV provisional, not read", "adv_read": 0, "incumbent_id": None,
              "panels": score["panels"], "onset_iou": score["onset_iou"], "ongoing_iou": score["ongoing_iou"],
              "S_queue": score["S_queue"], "delta_S_queue": score["delta_S_queue"], "ci95": score["ci95"],
              "frozen_subset": {"onset": None, "ongoing": None, "status": "not scored; selector provisional"},
              "official80": score, "poisoned_future_test": "PASS", "runtime_min": (time.monotonic()-started)/60,
              "peak_memory_mib": peak_mb(), "seed_sd": 0., "verdict": "REF",
              "fit": fit, "training_populations": {p: prior["panels"][p]["n_onsets"] for p in TASK2_PANELS},
              "artifacts": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
              "status": "canonical baseline delivered; Lead reproduction and ADV gates required before selection"}
    (report / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Q-02 REF complete, OFF S_queue {score['S_queue']:.6f}; ADV reads0", flush=True)


if __name__ == "__main__":
    main()
