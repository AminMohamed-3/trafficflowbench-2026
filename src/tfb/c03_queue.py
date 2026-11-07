"""Native assembly adapter for C's pinned, causal C03 ongoing recipe.

The producer refits from released masked observations. This adapter accepts only
its bound receipt, combines those ongoing forecasts with the canonical Q03
onset predictor, and fills the released Task 2 template by natural key.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import project_root
from .submit import sha256

KEY = ["window_id", "timestamp", "link_id"]


def cache_metadata_digest(path):
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata.pop("runtime_seconds", None)
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_contract(root, cfg):
    path = root / cfg["contract"]
    if sha256(path) != cfg["contract_sha256"]:
        raise ValueError("C03 production contract changed")
    contract = json.loads(path.read_text(encoding="utf-8"))
    for name, digest in contract["pins"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"C03 source/checkpoint changed: {name}")
    if contract["recipe"] != {
        "device": "cpu", "threads": 2, "epochs": 3, "lr": 0.0003,
        "batch_size": 32, "weight_decay": 0.0001, "cap": None,
        "same_day_type": True, "seed": "sorted official date index within split",
        "ensemble": "mean float32 probabilities then expected IoU per whole window",
    }:
        raise ValueError("Unsupported C03 recipe")
    return contract


def prepare(root, cfg):
    contract = load_contract(root, cfg)
    path = root / cfg["receipt"]
    if sha256(path) != cfg["receipt_sha256"]:
        raise ValueError("C03 raw producer receipt changed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (receipt["verdict"] != "PASS" or not receipt["raw_production"] or
            receipt["contract_sha256"] != cfg["contract_sha256"] or
            receipt["recipe"] != contract["recipe"] or receipt["windows"] != 80 or receipt["rows"] != 87000):
        raise ValueError("C03 raw production is not complete/bound")
    for name, digest in receipt["source_hashes"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"C03 released source changed: {name}")
    for name, digest in receipt["cache_metadata_semantic_hashes"].items():
        if cache_metadata_digest(root / name) != digest:
            raise ValueError(f"C03 cache metadata semantics differ: {name}")
    for name, digest in receipt["outputs"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"C03 produced output changed: {name}")
    if receipt["ensemble_path"] not in receipt["outputs"]:
        raise ValueError("C03 ensemble is not pinned in its producer receipt")
    frame = pd.read_parquet(root / receipt["ensemble_path"])
    if (frame.duplicated(KEY).any() or not frame.queue_pred.isin([0, 1]).all() or
            not np.isfinite(frame.prob).all() or not frame.prob.between(0, 1).all()):
        raise ValueError("C03 invalid ensemble keys/binaries")
    groups = {(p, s): part[KEY + ["queue_pred"]].copy()
              for (p, s), part in frame.groupby(["panel", "split"], sort=False)}
    if len(groups) != 16 or len(frame) != receipt["rows"] or frame.window_id.nunique() != 80:
        raise ValueError("C03 population differs from official ongoing windows")
    return groups, receipt


def predict_queue(panel, split, cfg):
    from .lane_q.predict import predict_window
    from .lane_q.windows import Release, iso
    release = Release(cfg["release_root"])
    folder = release.root / "task2" / panel / split
    if not (folder / "sample_submission_queue.csv").exists():
        if panel not in {"D12_I405_N", "D12_I405_S"}:
            raise FileNotFoundError(folder)
        return pd.DataFrame(columns=KEY + ["queue_pred"])
    template = pd.read_csv(folder / "sample_submission_queue.csv", usecols=KEY, dtype=str)
    template["timestamp"] = [iso(t) for t in pd.to_datetime(template.timestamp, utc=True)]
    index = pd.read_csv(folder / "window_index.csv", dtype={"window_id": str})
    ongoing = cfg["c03_predictions"][(panel, split)]
    ids = set(ongoing.window_id)
    expected_ids = set(index.loc[index.condition.eq("queue_ongoing"), "window_id"])
    if ids != expected_ids:
        raise ValueError("C03 producer does not cover exactly the official ongoing IDs")
    # IDs route output components; the Q-owned accessor constructs onset inputs.
    parts = [ongoing]
    for w in index.loc[~index.window_id.isin(ids)].itertuples(index=False):
        parts.append(predict_window(panel, split, w.date, w.forecast_origin, w.window_id, cfg, release=release))
    prediction = pd.concat(parts, ignore_index=True)
    if prediction.duplicated(KEY).any() or template.duplicated(KEY).any():
        raise ValueError("Duplicate C03 native Task 2 keys")
    result = template.merge(prediction, on=KEY, how="outer", validate="one_to_one", indicator=True)
    if len(result) != len(template) or not result._merge.eq("both").all() or not result.queue_pred.isin([0, 1]).all():
        raise ValueError("C03/Q03 combination differs from the released template")
    result["queue_pred"] = result.queue_pred.astype(np.int8)
    return result[KEY + ["queue_pred"]]
