"""I-Q-OFF sanity only. Never reads I-Q-ADV or selects an incumbent."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .windows import TASK2_PANELS, horizon_labels, interpolated_labels


def bootstrap_macro(detail, value="IoU_ST", *, seed=1104, B=1000):
    rng = np.random.default_rng(seed)
    groups = {}
    for (panel, condition), group in detail.groupby(["panel", "condition"], sort=True):
        values = group[value].to_numpy(dtype=float)
        draw = rng.integers(0, len(values), size=(B, len(values)))
        groups[(panel, condition)] = values[draw].mean(axis=1)
    panel_scores = {p: np.mean([v for (panel, _), v in groups.items() if panel == p], axis=0)
                    for p in sorted({p for p, _ in groups})}
    families = {}
    for panel, values in panel_scores.items():
        families.setdefault(panel.rsplit("_", 1)[0], []).append(values)
    scores = np.mean([np.mean(values, axis=0) for values in families.values()], axis=0)
    return [float(v) for v in np.quantile(scores, [.025, .975])]


def summarize(scored):
    detail = pd.DataFrame(scored["windows"])
    grouped = detail.groupby(["panel", "condition"]).IoU_ST.mean()
    onset = float(grouped.xs("queue_onset", level="condition").mean())
    ongoing = float(grouped.xs("queue_ongoing", level="condition").mean())
    panels = {}
    for panel, group in detail.groupby("panel"):
        panels[panel] = {condition.replace("queue_", ""): {"n": int(len(part)), "iou": float(part.IoU_ST.mean())}
                         for condition, part in group.groupby("condition")}
        panels[panel]["S_queue"] = scored["panels"][panel]
    return {"onset_iou": onset, "ongoing_iou": ongoing, "S_queue": scored["S_queue"], "panels": panels}


def sanity_report(release, prediction, cfg, out, root, queue_score):
    windows = pd.concat([release.window_index(p, "train") for p in TASK2_PANELS], ignore_index=True)
    truth = pd.read_parquet(root / "artifacts/lane_q/0106/official80_observed_labels.parquet")
    reference = prediction.copy()
    onset_ids = windows.loc[windows.condition.eq("queue_onset"), "window_id"]
    reference.loc[reference.window_id.isin(onset_ids), "queue_pred"] = 0
    reference.to_csv(out / "queue_off_persistence_zero_onset.csv", index=False)
    ref_score = queue_score(reference, truth, windows)
    ref_detail = pd.DataFrame(ref_score["windows"])
    floor_ci = bootstrap_macro(ref_detail)
    floor = {"instrument": "I-Q-OFF sanity only", "reference": "ongoing latest observation persistence, onset zero",
             "seed_sd": 0., "bootstrap_unit": "windows stratified by panel/condition; family macro mean", "B": 1000,
             "ci95": floor_ci, "ci_half_width": (floor_ci[1]-floor_ci[0])/2,
             "S_queue": ref_score["S_queue"], "adv_reads": 0}
    # Noise-floor record precedes computing/reporting any model delta.
    (out / "noise_floor.json").write_text(json.dumps(floor, indent=2), encoding="utf-8")
    scored = queue_score(prediction, truth, windows)
    detail = pd.DataFrame(scored["windows"])
    paired = detail.merge(ref_detail[["window_id", "IoU_ST"]].rename(columns={"IoU_ST": "reference_iou"}), on="window_id", validate="one_to_one")
    paired["delta"] = paired.IoU_ST-paired.reference_iou
    ci = bootstrap_macro(paired, "delta")
    delta = scored["S_queue"]-ref_score["S_queue"]
    detail.to_csv(out / "official80_window_scores.csv", index=False)
    paired.to_csv(out / "official80_paired_scores.csv", index=False)
    zero = prediction.copy()
    zero["queue_pred"] = 0
    zero_score = queue_score(zero, truth, windows)
    evidence = truth[truth.is_score_eligible.astype(bool)].groupby("window_id").queue_true.sum()
    no_queue = windows[windows.condition.eq("queue_onset") & windows.window_id.map(evidence).fillna(0).eq(0)]
    known = summarize(zero_score)
    known["caveat"] = "Official proxy labels contain an onset with no observed queue; true empty-union convention awards zero forecast1 there. Not an ADV known-worse gate."
    observable_windows = windows[~windows.window_id.isin(no_queue.window_id)]
    observable = summarize(queue_score(prediction, truth, observable_windows))
    observable["n_windows"] = len(observable_windows)
    observed_zero = summarize(queue_score(zero, truth, observable_windows))
    source_eligible, time_filled = [], []
    for panel in TASK2_PANELS:
        network = release.network(panel)
        for date, group in windows[windows.panel.eq(panel)].groupby("date"):
            day = release.train_grid(panel, date)
            for w in group.itertuples(index=False):
                for destination, labels in [(source_eligible, horizon_labels(day, w.forecast_origin, eligible_only=True)),
                                            (time_filled, interpolated_labels(day, w.forecast_origin, network))]:
                    labels["window_id"] = w.window_id
                    destination.append(labels)
    variants = {}
    for name, chunks in [("source_eligible_observed", source_eligible), ("time_interpolated_evaluation_only", time_filled)]:
        labels = pd.concat(chunks, ignore_index=True)
        target_path = out / f"official80_{name}_labels.parquet"
        labels.to_parquet(target_path, index=False)
        variant_score = queue_score(prediction, labels, windows)
        variants[name] = {**summarize(variant_score), "n_eligible_cells": int(labels.is_score_eligible.sum()),
                          "n_observed_cells": int(labels.label_observed.sum()),
                          "purpose": "label sensitivity; predictions unchanged; never features or selection"}
    result = {**summarize(scored), "n_windows": len(windows), "n_eligible_cells": int(truth.is_score_eligible.sum()),
              "fit_excludes": ["2030-06-01", "2030-06-10"], "reference": summarize(ref_score),
              "delta_S_queue": delta, "ci95": ci, "noise_floor": floor,
              "delta_floor_status": "above noise floor" if ci[0] > 0 and delta > floor["ci_half_width"] else "NULL relative to noise floor",
              "known_worse_off_diagnostic": known, "observable79": observable, "all_zero_observable79": observed_zero,
              "unobservable_onset_windows": no_queue[["panel", "date", "forecast_origin"]].to_dict("records"),
              "label_variants": variants, "selection": "none; I-Q-OFF sanity only, ADV selector provisional",
              "adv_reads": 0}
    (out / "sanity_off.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
