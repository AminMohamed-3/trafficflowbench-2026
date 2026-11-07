"""Bounded-memory assembly and strict validation in the released global order."""
from __future__ import annotations
from collections import Counter, OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import guard
from .paths import project_root, release_root

KEYS = {
    "state": ["panel", "timestamp", "station_id", "link_id", "mask_regime"],
    "queue": ["window_id", "timestamp", "link_id"],
    "odme": ["panel", "departure_time", "path_id"],
}
VALUES = {"state": ["speed_kmh", "flow_vph"], "queue": ["queue_pred"], "odme": ["path_flow"]}
COLUMNS = ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(frame, task):
    result = frame.copy()
    for key in KEYS[task]:
        if result[key].isna().any() or result[key].astype(str).eq("").any():
            raise ValueError(f"{task}: blank natural key {key}")
        if key == "timestamp":
            result[key] = pd.to_datetime(result[key], utc=True, format="mixed", errors="raise").astype(str)
        else:
            result[key] = result[key].astype(str)
    return result


def build_key_cache(root=None, *, chunksize=100_000, key_path=None, sample_path=None, destination=None):
    root = Path(root or project_root())
    key = Path(key_path or release_root(root) / "submission_key.csv")
    sample = Path(sample_path or release_root(root) / "sample_submission.csv")
    dest = Path(destination or root / "artifacts/key")
    sources = {"key": sha256(key), "sample": sha256(sample)}
    meta_path = dest / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["sources"] != sources:
            raise ValueError("Global key cache belongs to different source files")
        return dest, meta
    guard(root, dest, key.stat().st_size)
    dest.mkdir(parents=True, exist_ok=True)
    sample_reader = iter(pd.read_csv(sample, usecols=["submission_id", "task"], dtype=str,
                                     keep_default_na=False, chunksize=chunksize))
    before = (key.stat().st_size, key.stat().st_mtime_ns)
    parts, populations, counts = [], Counter(), Counter()
    total, previous_id = 0, None
    for number, frame in enumerate(pd.read_csv(key, dtype=str, keep_default_na=False, chunksize=chunksize)):
        reference = next(sample_reader, None)
        if reference is None or len(reference) != len(frame):
            raise ValueError("Key/sample row counts disagree")
        for column in ("submission_id", "task"):
            if not np.array_equal(frame[column].to_numpy(), reference[column].to_numpy()):
                raise ValueError(f"Key/sample order mismatch in {column}")
        if not set(frame.task) <= set(KEYS):
            raise ValueError("Unknown task in global key")
        ids = pd.to_numeric(frame.submission_id, errors="raise").to_numpy(dtype=np.int64)
        if (np.diff(ids) <= 0).any() or (previous_id is not None and ids[0] <= previous_id):
            raise ValueError("Released submission IDs must be unique and increasing")
        previous_id = int(ids[-1])
        frame["submission_id"] = ids
        for task in KEYS:
            rows = frame.task.eq(task)
            if rows.any():
                frame.loc[rows, KEYS[task]] = normalize(frame.loc[rows], task)[KEYS[task]]
        for (task, panel, split), size in frame.groupby(["task", "panel", "split"], sort=False).size().items():
            populations[f"{task}/{panel}/{split}"] += int(size)
        counts.update(frame.task)
        path = dest / f"part-{number:05d}.parquet"
        guard(root, dest, max(frame.memory_usage(deep=True).sum(), 1))
        frame.to_parquet(path, index=False, compression="zstd")
        parts.append({"file": path.name, "rows": len(frame), "sha256": sha256(path)})
        total += len(frame)
        print(f"key cache {total:,} rows", flush=True)
    if next(sample_reader, None) is not None:
        raise ValueError("Sample has extra rows")
    if before != (key.stat().st_size, key.stat().st_mtime_ns):
        raise ValueError("Key changed during conversion")
    meta = {"schema": 1, "rows": total, "chunksize": chunksize, "sources": sources, "parts": parts,
            "counts": dict(counts), "populations": dict(populations),
            "order": "released key, verified against sample IDs and tasks"}
    temporary = dest / "meta.building.json"
    temporary.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    temporary.replace(meta_path)
    return dest, meta


def key_parts(destination, meta, *, verify=True):
    for part in meta["parts"]:
        path = Path(destination) / part["file"]
        if verify and sha256(path) != part["sha256"]:
            raise ValueError(f"Corrupt global key shard: {path}")
        frame = pd.read_parquet(path)
        if len(frame) != part["rows"]:
            raise ValueError("Key shard row count changed")
        yield frame


class _Predictions:
    def __init__(self, sources, populations):
        self.sources = sources
        self.populations = populations
        self.tables = OrderedDict()

    def get(self, task, panel, split):
        group = (task, panel, split)
        if group in self.tables:
            return self.tables[group]
        source = self.sources[task]
        if callable(source):
            frame = source(panel, split)
        elif isinstance(source, dict):
            frame = source[(panel, split)]
        else:
            frame = source
        if isinstance(frame, (str, Path)):
            frame = pd.read_csv(frame, dtype={k: str for k in KEYS[task]})
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{task} predictor must return a DataFrame")
        needed = KEYS[task] + VALUES[task]
        missing = set(needed) - set(frame.columns)
        if missing:
            raise ValueError(f"{task} missing columns: {sorted(missing)}")
        if not callable(source) and not isinstance(source, dict):
            if "panel" in frame:
                frame = frame[frame.panel.astype(str).eq(panel)]
            if "split" in frame:
                frame = frame[frame.split.astype(str).eq(split)]
        frame = normalize(frame[needed], task)
        if frame.duplicated(KEYS[task]).any():
            raise ValueError(f"{task}/{panel}/{split}: duplicate natural keys")
        expected = self.populations[f"{task}/{panel}/{split}"]
        if len(frame) != expected:
            raise ValueError(f"{task}/{panel}/{split}: {len(frame)} predictions, expected {expected}")
        for column in VALUES[task]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        table = frame.set_index(KEYS[task])
        # Keep only the current population. Predictors are invoked lazily.
        self.tables.clear()
        self.tables[group] = table
        return table


def _check_values(frame):
    if list(frame.columns) != COLUMNS or frame.isna().any().any():
        raise ValueError("Submission has wrong columns or blank cells")
    numeric = frame[COLUMNS[2:]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Submission has nonfinite values")
    for task in KEYS:
        rows = frame.task.eq(task)
        unused = [column for column in COLUMNS[2:] if column not in VALUES[task]]
        if frame.loc[rows, unused].ne(0).any().any():
            raise ValueError(f"Unused {task} columns must be zero")
    state = frame.task.eq("state")
    if not frame.loc[state, "speed_kmh"].between(1, 130).all() or not frame.loc[state, "flow_vph"].ge(50).all():
        raise ValueError("State predictions violate speed [1,130] or flow >=50")
    if not frame.loc[frame.task.eq("queue"), "queue_pred"].isin([0, 1]).all():
        raise ValueError("Queue predictions must be binary")
    if not frame.loc[frame.task.eq("odme"), "path_flow"].ge(0).all():
        raise ValueError("Path flows must be nonnegative")


def validate(path, *, root=None, key_cache=None):
    root = Path(root or project_root())
    destination, meta = key_cache or build_key_cache(root)
    reader = iter(pd.read_csv(path, chunksize=meta["chunksize"]))
    try:
        total, counts = 0, Counter()
        for key in key_parts(destination, meta):
            out = next(reader, None)
            if out is None or len(out) != len(key):
                raise ValueError("Submission row count differs from sample")
            if not np.array_equal(out.submission_id.to_numpy(), key.submission_id.to_numpy()):
                raise ValueError("Submission IDs/order differ from sample")
            if not np.array_equal(out.task.to_numpy(), key.task.to_numpy()):
                raise ValueError("Submission tasks/order differ from sample")
            _check_values(out)
            total += len(out)
            counts.update(out.task)
        if next(reader, None) is not None or total != meta["rows"]:
            raise ValueError("Submission has extra rows")
        return {"verdict": "PASS", "rows": total, "counts": dict(counts), "sha256": sha256(path),
                "key_sha256": meta["sources"]["key"], "sample_sha256": meta["sources"]["sample"]}
    finally:
        reader.close()

def assemble(state, queue, odme, *, output=None, experiment_id=None, slug="submission", root=None, key_cache=None):
    root = Path(root or project_root())
    if output is None:
        if experiment_id is None:
            raise ValueError("Supply output or experiment_id")
        output = root / "submissions" / f"{experiment_id}_{slug}.csv"
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace a published artifact: {output}")
    destination, meta = key_cache or build_key_cache(root)
    guard(root, output.parent, meta["rows"] * 110)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".building")
    sources = _Predictions({"state": state, "queue": queue, "odme": odme}, meta["populations"])
    written = 0
    for key in key_parts(destination, meta):
        out = key[["submission_id", "task"]].copy()
        for column in COLUMNS[2:]:
            out[column] = 0.0
        for (task, panel, split), group in key.groupby(["task", "panel", "split"], sort=False):
            table = sources.get(task, panel, split)
            wanted = pd.MultiIndex.from_frame(group[KEYS[task]])
            found = table.reindex(wanted)
            if found.isna().any().any():
                raise ValueError(f"{task}/{panel}/{split}: missing predictions")
            out.loc[group.index, VALUES[task]] = found[VALUES[task]].to_numpy(dtype=float)
        out = out[COLUMNS]
        _check_values(out)
        out.to_csv(temporary, mode="w" if written == 0 else "a", header=written == 0, index=False)
        written += len(out)
        print(f"assembled {written:,}/{meta['rows']:,}", flush=True)
    receipt = validate(temporary, root=root, key_cache=(destination, meta))
    temporary.replace(output)
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return output
