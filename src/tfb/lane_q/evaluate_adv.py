"""Frozen I-Q-ADV reference, noise floor and known-worse gates; no selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest

import numpy as np
import pandas as pd
import yaml

from .windows import TASK2_PANELS, horizon_labels, online_view, project_root, iso
from .predict import forecast, MODEL_VERSION
from .evaluation_cache import EvaluationRelease
from .evaluate_baseline import bootstrap_macro, summarize
from .build_baseline import peak_mb, sha


def instrument_inputs(root):
    document = yaml.safe_load((root / "configs/instruments.yaml").read_text(encoding="utf-8"))
    if document["status"] != "FROZEN":
        raise ValueError("ADV instrument is not frozen")
    section = document["I-Q-ADV"]
    signature = hashlib.sha256(json.dumps(section, sort_keys=True).encode()).hexdigest()
    sources = {}
    for key in ["all_candidates", "frozen_subset", "first_per_day_diagnostic"]:
        item = section[key]
        path = root / item["path"]
        if sha(path) != item["sha256"]:
            raise AssertionError(f"Frozen {key} file hash changed")
        sources[key] = path
    return section, signature, sources


def macro(detail, value):
    conditions = detail.groupby(["panel", "condition"], sort=True)[value].mean()
    panels = conditions.groupby("panel", sort=True).mean().to_dict()
    families = {}
    for panel, score in panels.items():
        families.setdefault(panel.rsplit("_", 1)[0], []).append(score)
    return {"S_queue": float(np.mean([np.mean(v) for v in families.values()])),
            "panels": panels, "conditions": {f"{p}/{c}": float(v) for (p,c),v in conditions.items()},
            "windows": detail.assign(IoU_ST=detail[value]).to_dict("records")}


def score_grid(prediction, truth, valid):
    pv = prediction.astype(bool) & valid
    tv = truth.astype(bool) & valid
    intersection = int((pv & tv).sum())
    union = int((pv | tv).sum())
    return (intersection/union if union else (1. if valid.any() else 0.)), intersection, union


def population_summary(detail):
    scored = macro(detail, "baseline_iou")
    base = summarize(scored)
    ci_detail = detail.rename(columns={"baseline_iou":"IoU_ST"})
    base["ci95"] = bootstrap_macro(ci_detail, seed=1118)
    reference = summarize(macro(detail, "reference_iou"))
    zero = summarize(macro(detail, "zero_iou"))
    paired = detail.copy()
    paired["delta"] = paired.baseline_iou-paired.reference_iou
    base["delta_S_queue"] = base["S_queue"]-reference["S_queue"]
    base["paired_ci95"] = bootstrap_macro(paired, "delta", seed=1118)
    base["reference"] = reference
    base["all_zero"] = zero
    base["n_windows"] = len(detail)
    return base


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--baseline-id", default="0114")
    cli.add_argument("--control-id", default="0118")
    cli.add_argument("--main-root", type=Path, default=project_root())
    args = cli.parse_args()
    root = args.main_root
    out = root / "artifacts/lane_q" / args.control_id
    report = root / "exp" / f"{args.control_id}_q_adv_controls"
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    section, signature, sources = instrument_inputs(root)
    prior_path = root / "artifacts/lane_q" / args.baseline_id / "prior_u1.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    assert prior["model_version"] == MODEL_VERSION
    assert prior["fit"]["start"] == section["fit_start"] and prior["fit"]["end"] == section["fit_end"]
    metric_path = root / "src/tfb/metric.py"
    metric_hash = sha(metric_path)
    verified = json.loads((root / "exp/0110_q00_metric/result.json").read_text(encoding="utf-8"))
    if metric_hash != verified["metric_source_sha256"]:
        raise RuntimeError("Refresh Q-00 parity before the ADV reference read")
    manifest = {"instrument": signature, "prior": sha(prior_path), "metric": metric_hash,
                "populations": {key: sha(path) for key,path in sources.items()}}
    manifest_path = out / "inputs.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("Input version changed; prior checkpoints require explicit review")
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    from . import test_predict
    tests = unittest.defaultTestLoader.loadTestsFromModule(test_predict)
    if not unittest.TextTestRunner(verbosity=1).run(tests).wasSuccessful():
        raise SystemExit("Actual/cached predictor poison test failed; no ADV scores")
    import tfb
    tfb.__path__.append(str(root / "src/tfb"))
    from tfb.metric import queue_score
    populations = {"broad": pd.read_parquet(sources["all_candidates"]),
                   "frozen": pd.read_csv(sources["frozen_subset"]),
                   "first_per_day": pd.read_csv(sources["first_per_day_diagnostic"])}
    for frame in populations.values():
        frame["window_id"] = [f"sim_{p}_{iso(t)}" for p,t in zip(frame.panel,frame.forecast_origin)]
    joined = pd.concat(populations.values(),ignore_index=True).drop_duplicates(["panel","date","forecast_origin"])
    release = EvaluationRelease(root / "data/raw/kaggle_public")
    receipt_path = root / "artifacts/jobs" / f"{args.control_id}.json"
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    receipt = {"owner":"Q","pid":os.getpid(),"command":[sys.executable,"-m","tfb.lane_q.evaluate_adv"],
               "start_utc":datetime.now(timezone.utc).isoformat(),"eta_utc":(datetime.now(timezone.utc)+timedelta(minutes=25)).isoformat(),
               "checkpoint_path":str(out),"status":"RUNNING","adv_read":1,"threads":4,"ram_gb":4}
    receipt_path.write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    chunks = []
    parity = []
    for panel in TASK2_PANELS:
        checkpoint = out / f"{panel}_window_scores.csv"
        panel_windows = joined[joined.panel.eq(panel)].sort_values(["date","forecast_origin"])
        parity_path = out / f"{panel}_parity.json"
        if checkpoint.exists():
            if not parity_path.exists():
                raise RuntimeError("Checkpoint lacks its parity receipt")
            parity.extend(json.loads(parity_path.read_text(encoding="utf-8")))
            rows = pd.read_csv(checkpoint)
            assert len(rows)==len(panel_windows)
            chunks.append(rows)
            print(f"resume {panel} {len(rows)} windows",flush=True)
            continue
        rows = []
        tested_conditions = set()
        for date, windows in panel_windows.groupby("date",sort=True):
            day = release.train_grid(panel,date)
            for w in windows.itertuples(index=False):
                view = online_view(panel,"train",date,w.forecast_origin,release=release)
                condition = view.condition()
                if condition != w.condition:
                    raise AssertionError("Feature-visible condition disagrees with candidate selector")
                baseline = forecast(view,prior)
                reference = baseline.copy()
                if condition == "queue_onset":
                    reference[:] = 0
                k=view.origin.hour*12+view.origin.minute//5
                truth=day.queued[k+1:k+7]
                valid=day.observed[k+1:k+7]
                base_iou,inter,union=score_grid(baseline,truth,valid)
                ref_iou,_,_=score_grid(reference,truth,valid)
                zero_iou,_,_=score_grid(np.zeros_like(baseline),truth,valid)
                row={"window_id":w.window_id,"panel":panel,"date":date,"forecast_origin":w.forecast_origin,
                     "condition":condition,"baseline_iou":base_iou,"reference_iou":ref_iou,"zero_iou":zero_iou,
                     "intersection":inter,"union":union,"n_eligible":int(valid.sum())}
                rows.append(row)
                if condition not in tested_conditions:
                    # Keyed Lead parity for an onset and ongoing in every panel.
                    target=horizon_labels(day,view.origin)
                    target["window_id"]=w.window_id
                    pred=target[["window_id","timestamp","link_id"]].copy()
                    pred["queue_pred"]=baseline.reshape(-1)
                    window=pd.DataFrame({"window_id":[w.window_id],"panel":[panel],"condition":[condition]})
                    check=queue_score(pred,target,window)
                    error=abs(check["windows"][0]["IoU_ST"]-base_iou)
                    if error>1e-9:
                        raise AssertionError("Streaming/keyed metric parity failed")
                    parity.append({"panel":panel,"condition":condition,"max_abs_delta":error})
                    tested_conditions.add(condition)
            peak_mb()
        frame=pd.DataFrame(rows)
        frame.to_csv(checkpoint,index=False)
        parity_path.write_text(json.dumps([p for p in parity if p["panel"] == panel],indent=2),encoding="utf-8")
        chunks.append(frame)
        print(f"ADV reference {panel}: {len(frame)} windows; peak{peak_mb():.0f} MiB",flush=True)
        receipt["completed_panels"]=len(chunks)
        receipt_path.write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    details=pd.concat(chunks,ignore_index=True)
    details.to_csv(out / "all_window_scores.csv",index=False)
    assert instrument_inputs(root)[1]==signature and sha(prior_path)==manifest["prior"] and sha(metric_path)==metric_hash
    subset_details={}
    for name,frame in populations.items():
        subset=frame[["window_id"]].merge(details,on="window_id",validate="one_to_one")
        subset_details[name]=subset
        subset.to_csv(out / f"{name}_window_scores.csv",index=False)
    broad=subset_details["broad"]
    zero_onset=float(broad.loc[broad.condition.eq("queue_onset"),"zero_iou"].mean())
    persistence_ongoing=float(broad.loc[broad.condition.eq("queue_ongoing"),"reference_iou"].mean())
    zero_ongoing=float(broad.loc[broad.condition.eq("queue_ongoing"),"zero_iou"].mean())
    known_pass=zero_onset==0 and persistence_ongoing>zero_ongoing
    baseline_detail=broad.rename(columns={"baseline_iou":"IoU_ST"})
    floor_ci=bootstrap_macro(baseline_detail,seed=1118)
    floor={"seed_sd":0.,"ci95":floor_ci,"ci_half_width":(floor_ci[1]-floor_ci[0])/2,
           "instrument":"I-Q-ADV broad","B":1000,"bootstrap_unit":"windows stratified by panel/condition, macro family aggregation"}
    # Controls/noise record exists before the comparison summary is written.
    gates={"all_zero_onset_iou":zero_onset,"persistence_ongoing_micro_mean":persistence_ongoing,
           "all_zero_ongoing_iou":zero_ongoing,"known_worse_verdict":"PASS" if known_pass else "FAIL",
           "noise_floor":floor,"poisoned_future_test":"PASS","metric_parity":parity}
    (out / "gates.json").write_text(json.dumps(gates,indent=2),encoding="utf-8")
    if not known_pass:
        raise AssertionError("I-Q-ADV known-worse arm failed; no comparisons")
    result={"id":args.control_id,"lane":"Q","formulation":"I-Q-ADV first-use zero control and deterministic noise floor",
            "utc":datetime.now(timezone.utc).isoformat(),"commit":None,"baseline_id":args.baseline_id,"adv_read":1,
            "instrument":"I-Q-ADV","config":f"exp/{args.control_id}_q_adv_controls/config.yaml",
            "gates":gates,"populations":{name:population_summary(detail) for name,detail in subset_details.items()},
            "input_hashes":manifest,"peak_memory_mib":peak_mb(),"runtime_min":(time.monotonic()-started)/60,
            "verdict":"REF","selection":"none; baseline reference only; board calibration still required"}
    (report / "result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    receipt["status"]="COMPLETE"
    receipt["end_utc"]=datetime.now(timezone.utc).isoformat()
    receipt_path.write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(f"I-Q-ADV controls PASS, baseline reference measured once across{len(broad)} broad windows",flush=True)


if __name__=="__main__":
    main()
