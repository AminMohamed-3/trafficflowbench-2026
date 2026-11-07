"""Config-driven whole-condition queue ablations for public calibration only.

Read decimal cells as text so the parent's state/ODME values are preserved
exactly, including float32 values expanded to float64 by the original assembler.
Released window conditions scope the diagnostic; lane models never see them.
"""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import yaml

from .data import guard
from .paths import panels, project_root, release_root
from .submit import COLUMNS, build_key_cache, key_parts, sha256, validate

CONDITIONS = {"queue_onset", "queue_ongoing"}


def load_window_conditions(index_paths):
    """Join by panel/split/window rather than relying on window-ID spelling."""
    conditions, sources = {}, {}
    for (panel, split), path in index_paths.items():
        path = Path(path)
        sources[str(path)] = sha256(path)
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        required = ["panel", "split", "window_id", "condition"]
        if not set(required) <= set(frame.columns):
            raise ValueError("Window index lacks diagnostic join fields")
        if not frame.panel.eq(panel).all() or not frame.split.eq(split).all():
            raise ValueError("Window index population mismatch")
        if frame.window_id.eq("").any() or not set(frame.condition) <= CONDITIONS:
            raise ValueError("Unknown or blank window condition")
        for row in frame[required].itertuples(index=False, name=None):
            key = row[:3]
            if key in conditions:
                raise ValueError("Duplicate window index key")
            conditions[key] = row[3]
        if sha256(path) != sources[str(path)]:
            raise ValueError("Window index changed while loading")
    if not conditions:
        raise ValueError("No released queue windows")
    return conditions, sources


def _reader(path, meta):
    return pd.read_csv(path, dtype=str, keep_default_na=False,
                       chunksize=meta["chunksize"])


def _aligned(reader, key):
    frame = next(reader, None)
    if frame is None or len(frame) != len(key) or list(frame.columns) != COLUMNS:
        raise ValueError("Probe/parent schema or row count mismatch")
    frame = frame.reset_index(drop=True)
    for column in ["submission_id", "task"]:
        if not np.array_equal(frame[column].to_numpy(), key[column].astype(str).to_numpy()):
            raise ValueError("Probe/parent IDs or tasks differ from global order")
    return frame


def _scope(key, conditions, condition):
    queue = key.task.eq("queue")
    mask = pd.Series(False, index=key.index)
    windows = list(key.loc[queue, ["panel", "split", "window_id"]].itertuples(index=False, name=None))
    labels = [conditions.get(window) for window in windows]
    if any(label is None for label in labels):
        raise ValueError("Queue key has no released window condition")
    if windows:
        mask.loc[queue] = np.asarray([label == condition for label in labels], dtype=bool)
    return mask, set(windows)


def verify_probe_difference(parent, probe, conditions, condition, *, root, key_cache):
    """Independent reread: every untargeted CSV cell must equal the parent text."""
    destination, meta = key_cache
    rows, target_rows, changed_rows = 0, 0, 0
    used, targeted = set(), set()
    populations = {}
    with _reader(parent, meta) as parent_reader, _reader(probe, meta) as probe_reader:
        for key in key_parts(destination, meta):
            before, after = _aligned(parent_reader, key), _aligned(probe_reader, key)
            mask, windows = _scope(key, conditions, condition)
            used.update(windows)
            protected = [column for column in COLUMNS if column != "queue_pred"]
            if not before[protected].equals(after[protected]):
                raise ValueError("Probe changed state, ODME, IDs or tasks")
            if not before.loc[~mask, "queue_pred"].equals(after.loc[~mask, "queue_pred"]):
                raise ValueError("Probe changed the other queue condition or unused cells")
            if not after.loc[mask, "queue_pred"].astype(float).eq(0).all():
                raise ValueError("Diagnostic target is not all zero")
            parent_values = before.loc[mask, "queue_pred"].astype(float)
            if not parent_values.isin([0, 1]).all():
                raise ValueError("Parent queue values are not binary")
            changed = before.queue_pred.ne(after.queue_pred)
            # Existing zeros must keep their original decimal spelling too.
            expected_changed = pd.Series(False, index=key.index)
            expected_changed.loc[mask] = parent_values.eq(1).to_numpy()
            if not changed.equals(expected_changed):
                raise ValueError("Unexpected changed-cell population")
            for (panel, split), group in key[mask].groupby(["panel", "split"], sort=False):
                label = f"{panel}/{split}"
                entry = populations.setdefault(label, {"target_rows": 0, "zeroed_ones": 0})
                entry["target_rows"] += len(group)
                entry["zeroed_ones"] += int(changed.loc[group.index].sum())
                targeted.update(group[["panel", "split", "window_id"]].itertuples(index=False, name=None))
            rows += len(key)
            target_rows += int(mask.sum())
            changed_rows += int(changed.sum())
        if next(parent_reader, None) is not None or next(probe_reader, None) is not None:
            raise ValueError("Parent/probe has extra rows")
    if used != set(conditions) or targeted != {key for key, label in conditions.items() if label == condition}:
        raise ValueError("Window-index/global-key population mismatch")
    if not target_rows or rows != meta["rows"]:
        raise ValueError("Empty target or truncated probe")
    return {"verdict": "PASS", "rows": rows, "target_rows": target_rows,
            "target_windows": len(targeted), "zeroed_ones": changed_rows,
            "populations": populations, "every_untargeted_cell_exact_text": True,
            "existing_target_zeros_exact_text": True, "all_target_queue_values_zero": True}


def build_probe(config_path, *, root=None, key_cache=None, index_paths=None):
    root = Path(root or project_root()).resolve()
    config_path = Path(config_path).resolve()
    config_hash = sha256(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    condition = config["condition"]
    if config["kind"] != "diagnostic_queue_zero" or condition not in CONDITIONS or config.get("upload") is not False:
        raise ValueError("Only non-uploading diagnostic queue-zero configs are supported")
    if not str(config["id"]).isdigit() or len(str(config["id"])) != 4:
        raise ValueError("Probe needs a four-digit ledger ID")
    parent = (root / config["parent"]).resolve()
    output = (root / config["output"]).resolve()
    submission_dir = (root / "submissions").resolve()
    if parent.parent != submission_dir or output.parent != submission_dir or parent == output:
        raise ValueError("Probe and parent must be distinct files in submissions")
    expected_name = f"{config['id']}_{config['slug']}.csv"
    if output.name != expected_name:
        raise ValueError("Probe output filename differs from ID/slug")
    temporary = output.with_suffix(output.suffix + ".building")
    if output.exists() or temporary.exists():
        raise FileExistsError("Refusing to replace a published or in-progress probe")
    if sha256(parent) != config["parent_sha256"]:
        raise ValueError("Parent differs from the pinned uploaded file")
    cache = key_cache or build_key_cache(root)
    destination, meta = cache
    validate(parent, root=root, key_cache=cache)
    if index_paths is None:
        index_paths = {(panel, split): release_root(root)/"task2"/panel/split/"window_index.csv"
                       for panel in panels(root) for split in ["validation", "private"]
                       if (release_root(root)/"task2"/panel/split/"window_index.csv").exists()}
    conditions, sources = load_window_conditions(index_paths)
    guard(root, output.parent, parent.stat().st_size + 1024**2)
    report_dir = root/"exp"/f"{config['id']}_{config['slug']}"
    report_dir.mkdir(parents=True, exist_ok=True)
    job = root/"artifacts/jobs"/f"{config['id']}.json"
    job.parent.mkdir(parents=True, exist_ok=True)
    job.write_text(json.dumps({"id": config["id"], "status": "RUNNING", "pid": os.getpid(),
                               "start_utc": datetime.now(timezone.utc).isoformat(),
                               "command": "tfb.pipeline probe", "config": str(config_path)}))
    start, written = time.monotonic(), 0
    try:
        with _reader(parent, meta) as reader:
            for key in key_parts(destination, meta):
                frame = _aligned(reader, key)
                mask, _ = _scope(key, conditions, condition)
                changed = mask & frame.queue_pred.astype(float).eq(1)
                frame.loc[changed, "queue_pred"] = "0.0"
                frame.to_csv(temporary, mode="w" if written == 0 else "a", header=written == 0,
                             index=False, lineterminator="\n")
                written += len(frame)
                if written % 1_000_000 == 0 or written == meta["rows"]:
                    print(f"probe {config['id']} wrote {written:,}/{meta['rows']:,}", flush=True)
            if next(reader, None) is not None:
                raise ValueError("Parent has extra rows")
        receipt = validate(temporary, root=root, key_cache=cache)
        difference = verify_probe_difference(parent, temporary, conditions, condition, root=root, key_cache=cache)
        if sha256(parent) != config["parent_sha256"] or sha256(config_path) != config_hash:
            raise ValueError("Parent/config changed during probe build")
        if any(sha256(path) != digest for path, digest in sources.items()):
            raise ValueError("Released diagnostic conditions changed during build")
        result = {**receipt, "id": config["id"], "file": str(output), "kind": config["kind"],
                  "condition": condition, "parent": config["parent"], "parent_sha256": config["parent_sha256"],
                  "parent_public": config["parent_public"], "config": str(config_path), "config_sha256": config_hash,
                  "pre_read_prediction": config["prediction"], "condition_index_sha256": sources,
                  "difference_audit": difference, "code_sha256": sha256(Path(__file__)),
                  "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root(), text=True).strip(),
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "runtime_minutes": (time.monotonic()-start)/60, "uploaded": False, "public": None,
                  "diagnostic_only": True, "eligible_for_final_selection": False}
        temporary.replace(output)
        output.with_suffix(output.suffix+".manifest.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        (report_dir/"result.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        job.write_text(json.dumps({"id": config["id"], "status": "COMPLETE", "file": str(output), "sha256": result["sha256"]}))
        print(json.dumps({key: result[key] for key in ["file", "sha256", "rows", "runtime_minutes"]}), flush=True)
        print(json.dumps(difference), flush=True)
        return output
    except Exception as error:
        job.write_text(json.dumps({"id": config["id"], "status": "FAILED", "error": str(error)}))
        raise
