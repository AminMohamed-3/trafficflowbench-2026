"""Build and load audited, upstream-ordered TrafficFlowBench arrays."""
from dataclasses import dataclass
from datetime import date
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.lib.format import open_memmap

from .paths import cache_root, project_root, release_root

VERSION = 1
CHANNELS = {"speed": "speed_kmh", "flow": "flow_vph", "occ": "occupancy"}
DTYPES = {**{k: np.float32 for k in CHANNELS}, "pct": np.uint8,
          **{k: np.bool_ for k in ("elig", "target", "gap", "hist")},
          **{k + "_true": np.float32 for k in CHANNELS}}
SPLITS = ("train", "validation", "private")


def guard(root, destination, expected_bytes=0):
    spec = importlib.util.spec_from_file_location("tfb_diskguard", Path(root) / "tools" / "diskguard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_free_space(destination, expected_bytes=expected_bytes)


def read_parquet(path, columns=None):
    # ParquetFile avoids interpreting Hive directory names as duplicate columns.
    return pq.ParquetFile(path).read(columns=columns, use_threads=False).to_pandas()


def dated_files(folder):
    result = {}
    for path in sorted(Path(folder).rglob("*.parquet")):
        match = re.search(r"(\d{4})_(\d{2})_(\d{2})\.parquet$", path.name)
        if not match:
            raise ValueError(f"Unexpected daily filename: {path}")
        day = "-".join(match.groups())
        if day in result:
            raise ValueError(f"Duplicate source date: {day}")
        result[day] = path
    return dict(sorted(result.items()))


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def station_table(network, first_masked):
    base = read_parquet(first_masked, ["station_id", "link_id", "milepost"]).drop_duplicates()
    if base.station_id.duplicated().any():
        raise ValueError("Station mapping changes within the first day")
    links = pd.read_csv(network / "links.csv")
    topo = pd.read_csv(network / "lwr_mainline_topology.csv")
    result = base.merge(topo[["mainline_link_id", "order_index"]].rename(
        columns={"mainline_link_id": "link_id"}), on="link_id", validate="many_to_one")
    if len(result) != len(base):
        raise ValueError("Some stations have no topology mapping")
    result = result.merge(links, on="link_id", validate="many_to_one")
    if len(result) != len(base) or result[["order_index", "length_km", "lanes"]].isna().any().any():
        raise ValueError("Missing network parameters")
    if (result[["lanes", "length_km", "free_speed_kmh", "capacity_vph"]] <= 0).any().any():
        raise ValueError("Nonpositive network parameter")
    result["critical_density"] = result.capacity_vph / result.free_speed_kmh
    fd = pd.read_csv(network / "fd_parameters.csv")
    jam = fd.set_index("link_id")["k_jam"] if "k_jam" in fd else pd.Series(dtype=float)
    result["k_jam"] = result.link_id.map(jam).fillna(100 * result.lanes)
    result["k_jam"] = np.maximum(result.k_jam, 1.05 * result.critical_density)
    result["v_cut"] = 0.6 * result.free_speed_kmh
    attachments = pd.read_csv(network / "ramp_attachment_map.csv")
    for column, kind in (("has_on_ramp", "OR"), ("has_off_ramp", "FR")):
        result[column] = result.link_id.isin(attachments.loc[
            attachments.ramp_type == kind, "nearest_mainline_link_id"])
    result = result.sort_values(["order_index", "milepost"], kind="stable").reset_index(drop=True)
    result.insert(0, "idx", np.arange(len(result), dtype=np.int32))
    one_to_one = bool(not result.link_id.duplicated().any() and len(result) == len(links)
                      and set(result.link_id) == set(links.link_id))
    return result, one_to_one


def ramp_table(network, stations):
    result = pd.read_csv(network / "ramp_attachment_map.csv").rename(
        columns={"ramp_link_id": "ramp_id", "nearest_mainline_link_id": "link_id"})
    if result.ramp_id.duplicated().any() or not result.ramp_type.isin(["OR", "FR"]).all():
        raise ValueError("Invalid ramp attachment map")
    lookup = stations.groupby("link_id", sort=False)["idx"].first()
    result["station_idx"] = result.link_id.map(lookup)
    if result.station_idx.isna().any():
        raise ValueError("Ramp is not attached to an observed mainline station")
    result["station_idx"] = result.station_idx.astype(np.int32)
    result = result.sort_values(["station_idx", "ramp_type", "ramp_id"], kind="stable").reset_index(drop=True)
    result.insert(0, "idx", np.arange(len(result), dtype=np.int32))
    return result


def indices(frame, days, entities, id_column):
    timestamps = frame.timestamp.astype(str)
    if not timestamps.str.endswith("Z").all():
        raise ValueError("Timestamp is not explicitly UTC")
    ts = pd.to_datetime(timestamps, utc=True, errors="raise")
    if ((ts.dt.minute % 5 != 0) | (ts.dt.second != 0) | (ts.dt.microsecond != 0)).any():
        raise ValueError("Timestamp is not on the five-minute grid")
    day = timestamps.str[:10].map(days)
    entity = frame[id_column].map(entities)
    if day.isna().any() or entity.isna().any():
        raise ValueError("Unknown date or entity key")
    step = ts.dt.hour.to_numpy(dtype=np.int16) * 12 + ts.dt.minute.to_numpy(dtype=np.int16) // 5
    return day.to_numpy(dtype=np.int32), step, entity.to_numpy(dtype=np.int32)


def assert_daily(frame, keys, n_entities, day_index):
    day, step, entity = keys
    flat = step.astype(np.int64) * n_entities + entity
    if len(frame) != 288 * n_entities or len(np.unique(flat)) != len(frame) or not (day == day_index).all():
        raise ValueError("Daily parquet does not contain exactly one full entity grid")


def window_masks(windows, days, gap, hist, split):
    for row in windows.itertuples(index=False):
        origin = pd.Timestamp(row.forecast_origin)
        if origin.tzinfo is None:
            raise ValueError("Window origin is not UTC")
        for array, offsets in ((hist, range(-60, 0, 5)), (gap, range(5, 95, 5))):
            if array is gap and split == "train":
                continue
            for minutes in offsets:
                ts = origin + pd.Timedelta(minutes=minutes)
                key = ts.strftime("%Y-%m-%d")
                if key in days:
                    array[days[key], ts.hour * 12 + ts.minute // 5, :] = True


def template_targets(template, days, entities, shape, regimes, station_links, destination=None):
    mask = destination if destination is not None else np.zeros(shape, dtype=bool)
    counts = {r: 0 for r in ("R1", "R2", "R3")}
    count = 0
    for chunk in pd.read_csv(template, usecols=["timestamp", "station_id", "link_id", "mask_regime"],
                             chunksize=200_000):
        key = indices(chunk, days, entities, "station_id")
        if not np.array_equal(chunk.link_id.to_numpy(), chunk.station_id.map(station_links).to_numpy()):
            raise ValueError("Template station/link mapping differs from the network")
        flat = np.ravel_multi_index(key, shape)
        if len(np.unique(flat)) != len(chunk) or mask[key].any():
            raise ValueError("Duplicate Task 1 template cell")
        if not np.array_equal(chunk.mask_regime.to_numpy(), np.asarray(regimes, dtype=object)[key[0]]):
            raise ValueError("Template regime differs from its source day")
        mask[key] = True
        count += len(chunk)
        for regime, number in chunk.mask_regime.value_counts().items():
            if regime not in counts:
                raise ValueError(f"Unknown regime {regime}")
            counts[regime] += int(number)
    return mask, count, counts


def audit_roundtrip(sources, truth_sources, ramp_sources, arrays, days, station_ids, ramp_ids, sampled):
    for day in sampled:
        index = days[day]
        frame = read_parquet(sources[day])
        keys = indices(frame, days, station_ids, "station_id")
        for name, column in CHANNELS.items():
            expected = frame[column].to_numpy(dtype=np.float32)
            if not np.array_equal(arrays[name][keys], expected, equal_nan=True):
                raise AssertionError(f"{day} {name} round-trip failed")
        for name, column in (("pct", "pct_observed"), ("elig", "is_score_eligible")):
            if not np.array_equal(arrays[name][keys], frame[column].to_numpy(dtype=DTYPES[name])):
                raise AssertionError(f"{day} {name} round-trip failed")
        if truth_sources:
            truth = read_parquet(truth_sources[day])
            truth_keys = indices(truth, days, station_ids, "station_id")
            for name, column in CHANNELS.items():
                if not np.array_equal(arrays[name + "_true"][truth_keys],
                                      truth[column].to_numpy(dtype=np.float32), equal_nan=True):
                    raise AssertionError(f"{day} {name}_true round-trip failed")
        ramps = read_parquet(ramp_sources[day])
        ramp_keys = indices(ramps, days, ramp_ids, "ramp_link_id")
        if not np.array_equal(arrays["ramp_flow"][ramp_keys], ramps.flow_vph.to_numpy(dtype=np.float32),
                              equal_nan=True):
            raise AssertionError(f"{day} ramp flow round-trip failed")
        if not np.array_equal(arrays["ramp_elig"][ramp_keys], ramps.is_score_eligible.to_numpy(dtype=bool)):
            raise AssertionError(f"{day} ramp eligibility round-trip failed")


def build_cache(panel, split, root=None):
    root = Path(root or project_root())
    if split not in SPLITS:
        raise ValueError(f"Invalid split {split}")
    release = release_root(root)
    output = cache_root(root) / panel / split
    meta_path = output / "meta.json"
    if meta_path.exists():
        cached = load_cache(panel, split, root=root)
        if cached.meta["format_version"] != VERSION:
            raise ValueError("Cache version changed; rebuild explicitly")
        return cached.meta
    if meta_path.exists():
        raise FileExistsError("An audited cache already exists; use a new versioned cache root")
    start = time.monotonic()
    corridor = release / "corridors" / panel
    network = corridor / "network"
    sources = dated_files(corridor / split / "mainline_states_masked")
    truths = dated_files(corridor / split / "mainline_states") if split == "train" else {}
    ramps = dated_files(corridor / split / "ramp_states")
    if not sources or set(sources) != set(ramps) or (truths and set(sources) != set(truths)):
        raise ValueError("Source calendars differ or are empty")
    dates = list(sources)
    calendar_path = release / "config" / "synthetic_release_v1.json"
    calendar = json.loads(calendar_path.read_text(encoding="utf-8"))["synthetic_calendar"][split]
    expected = pd.date_range(calendar["start"], calendar["end_exclusive"], inclusive="left").strftime("%Y-%m-%d").tolist()
    if dates != expected:
        raise ValueError("Noncontiguous source calendar")
    days = {day: i for i, day in enumerate(dates)}
    stations, one_to_one = station_table(network, next(iter(sources.values())))
    ramp_metadata = ramp_table(network, stations)
    station_ids = dict(zip(stations.station_id, stations.idx))
    ramp_ids = dict(zip(ramp_metadata.ramp_id, ramp_metadata.idx))
    shape = (len(dates), 288, len(stations))
    ramp_shape = (len(dates), 288, len(ramp_metadata))
    names = {k: v for k, v in DTYPES.items() if split == "train" or not k.endswith("_true")}
    estimated = int(np.prod(shape)) * sum(np.dtype(v).itemsize for v in names.values())
    estimated += int(np.prod(ramp_shape)) * 5 + 10 * 1024**2
    guard(root, output, estimated)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "building.json"
    marker.write_text(json.dumps({"owner": "L", "pid": os.getpid(), "panel": panel, "split": split}),
                      encoding="utf-8")
    arrays = {}
    for name, dtype in names.items():
        arrays[name] = open_memmap(output / (name + ".npy"), mode="w+", dtype=dtype, shape=shape)
        arrays[name][:] = np.nan if np.issubdtype(dtype, np.floating) else 0
    for name, dtype in (("ramp_flow", np.float32), ("ramp_elig", np.bool_)):
        arrays[name] = open_memmap(output / (name + ".npy"), mode="w+", dtype=dtype, shape=ramp_shape)
        arrays[name][:] = np.nan if name == "ramp_flow" else False
    regimes = []
    entity_links = stations.set_index("station_id").link_id.to_dict()
    for day, source in sources.items():
        frame = read_parquet(source)
        key = indices(frame, days, station_ids, "station_id")
        assert_daily(frame, key, len(stations), days[day])
        if not np.array_equal(frame.link_id.to_numpy(), frame.station_id.map(entity_links).to_numpy()):
            raise ValueError("Daily station/link mapping changed")
        unique_regime = frame.mask_regime.unique()
        if len(unique_regime) != 1 or unique_regime[0] not in ("R1", "R2", "R3"):
            raise ValueError("Each day must have one valid regime")
        regimes.append(str(unique_regime[0]))
        for name, column in CHANNELS.items():
            arrays[name][key] = frame[column].to_numpy(dtype=np.float32)
        pct = frame.pct_observed.to_numpy()
        if ((pct < 0) | (pct > 100)).any():
            raise ValueError("Quality percentage outside [0, 100]")
        arrays["pct"][key] = pct.astype(np.uint8)
        arrays["elig"][key] = frame.is_score_eligible.to_numpy(dtype=bool)
        if split == "train":
            truth = read_parquet(truths[day])
            true_key = indices(truth, days, station_ids, "station_id")
            assert_daily(truth, true_key, len(stations), days[day])
            for name, column in CHANNELS.items():
                arrays[name + "_true"][true_key] = truth[column].to_numpy(dtype=np.float32)
            d = days[day]
            arrays["target"][d] = (arrays["elig"][d] & np.isnan(arrays["speed"][d])
                                    & np.isfinite(arrays["speed_true"][d])
                                    & np.isfinite(arrays["flow_true"][d]))
        ramp_frame = read_parquet(ramps[day])
        ramp_key = indices(ramp_frame, days, ramp_ids, "ramp_link_id")
        assert_daily(ramp_frame, ramp_key, len(ramp_metadata), days[day])
        arrays["ramp_flow"][ramp_key] = ramp_frame.flow_vph.to_numpy(dtype=np.float32)
        arrays["ramp_elig"][ramp_key] = ramp_frame.is_score_eligible.to_numpy(dtype=bool)
    template = release / "task1" / panel / split / "sample_submission_state.csv"
    templated, template_rows, template_regimes = template_targets(
        template, days, station_ids, shape, regimes, entity_links,
        destination=arrays["target"] if split != "train" else None)
    if split == "train" and not np.array_equal(templated, arrays["target"]):
        differing = int(np.count_nonzero(templated != arrays["target"]))
        raise AssertionError(f"Train target mask differs from template in {differing} cells")
    if np.any(arrays["target"] & ~arrays["elig"]):
        raise AssertionError("Template includes ineligible cells")
    if any(np.any(arrays["target"] & ~np.isnan(arrays[name])) for name in CHANNELS):
        raise AssertionError("A target measurement is visible")
    window_path = release / "task2" / panel / split / "window_index.csv"
    windows = pd.read_csv(window_path) if window_path.exists() else pd.DataFrame()
    window_masks(windows, days, arrays["gap"], arrays["hist"], split)
    if split != "train":
        for name in CHANNELS:
            if not np.isnan(arrays[name][arrays["gap"]]).all():
                raise AssertionError(f"Observed {name} inside a forecast gap")
    actual_regimes = {r: int(sum(np.count_nonzero(arrays["target"][i])
                                    for i, v in enumerate(regimes) if v == r))
                      for r in ("R1", "R2", "R3")}
    if actual_regimes != template_regimes:
        raise AssertionError("Target regime counts differ from template")
    rng = np.random.default_rng(104)
    sampled = sorted(rng.choice(dates, size=min(3, len(dates)), replace=False).tolist())
    audit_roundtrip(sources, truths, ramps, arrays, days, station_ids, ramp_ids, sampled)
    for array in arrays.values():
        array.flush()
    guard(root, output)
    stations.to_parquet(output / "stations.parquet", index=False)
    ramp_metadata.to_parquet(output / "ramps.parquet", index=False)
    source_paths = list(sources.values()) + list(truths.values()) + list(ramps.values())
    source_paths += [network / name for name in ("links.csv", "fd_parameters.csv",
                      "lwr_mainline_topology.csv", "ramp_attachment_map.csv")]
    source_paths += [template, calendar_path]
    if window_path.exists():
        source_paths.append(window_path)
    manifest = [{"path": str(p.relative_to(release)).replace("\\", "/"),
                 "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(source_paths)]
    metadata = {
        "format_version": VERSION, "panel": panel, "split": split, "verified": True,
        "dates": dates, "regime_per_day": regimes,
        "weekday": [date.fromisoformat(day).weekday() for day in dates],
        "shape": list(shape), "ramp_shape": list(ramp_shape),
        "arrays": {k: str(v.dtype) for k, v in arrays.items()},
        "stations_links_one_to_one": one_to_one, "n_stations": len(stations),
        "n_ramps": len(ramp_metadata), "target_rows": template_rows,
        "target_rows_by_regime": actual_regimes, "n_windows": len(windows),
        "gap_cells": int(np.count_nonzero(arrays["gap"])),
        "history_cells": int(np.count_nonzero(arrays["hist"])),
        "audit": {"roundtrip_days": sampled, "measurements_compare_as": "float32",
                  "template_membership": "exact", "forecast_gaps_blank": True},
        "sources": manifest, "runtime_seconds": round(time.monotonic() - start, 3),
    }
    temporary = output / "meta.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, meta_path)
    marker.unlink()
    return metadata


@dataclass
class PanelCache:
    path: Path
    meta: dict
    stations: pd.DataFrame
    ramps: pd.DataFrame
    arrays: dict

    def __getitem__(self, name):
        return self.arrays[name]

    def __getattr__(self, name):
        if name in self.arrays:
            return self.arrays[name]
        raise AttributeError(name)

    @property
    def dates(self):
        return self.meta["dates"]

    @property
    def regimes(self):
        return np.asarray(self.meta["regime_per_day"])

    def day_index(self, day):
        return self.dates.index(str(day))


def load_cache(panel, split, root=None, mmap_mode="r"):
    path = cache_root(root) / panel / split
    metadata = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if not metadata.get("verified") or metadata["format_version"] != VERSION:
        raise ValueError("Cache is not a verified compatible version")
    arrays = {name: np.load(path / (name + ".npy"), mmap_mode=mmap_mode)
              for name in metadata["arrays"]}
    for name, array in arrays.items():
        expected = metadata["ramp_shape"] if name.startswith("ramp_") else metadata["shape"]
        if list(array.shape) != expected or str(array.dtype) != metadata["arrays"][name]:
            raise ValueError(f"Invalid cache array: {name}")
    return PanelCache(path, metadata, pd.read_parquet(path / "stations.parquet"),
                      pd.read_parquet(path / "ramps.parquet"), arrays)
