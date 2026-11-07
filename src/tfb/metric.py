"""Independent numpy scoring replicas. Raw float64 truth is used for parity."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from .data import guard, indices, read_parquet
from .paths import project_root, release_root

REGIMES = ("R1", "R2", "R3")
EPS = 1e-9


def aggregate(scores, task="state"):
    """Equal panel -> directional family -> family mean, not a cell-weighted mean."""
    if isinstance(scores, dict):
        data = pd.DataFrame({"panel": list(scores), "value": list(scores.values())})
    else:
        data = pd.DataFrame(scores).copy()
        column = {"state": "S_state", "physics": "S_physics", "queue": "S_queue",
                  "odme": "S_ODME", "link": "S_link"}[task]
        if column not in data and task == "queue":
            column = "IoU_ST"
        if "level" in data and (data.level == "panel").any():
            data = data.loc[data.level == "panel"]
        data = data.rename(columns={column: "value"})[["panel", "value"]]
    data = data[data.panel.notna()]
    if task == "queue":
        data = data[~data.panel.str.startswith("D12_I405_")]
    if data.empty:
        return 0.0
    means = data.groupby("panel", sort=True).value.mean().reset_index()
    means["family"] = means.panel.str.rsplit("_", n=1).str[0]
    return float(means.groupby("family", sort=True).value.mean().mean())


def _shapes(speed_true, flow_true, pred_speed, pred_flow, target, lanes, regimes):
    shape = np.shape(speed_true)
    if len(shape) != 3 or any(np.shape(a) != shape for a in (flow_true, pred_speed, pred_flow, target)):
        raise ValueError("State arrays must have one common [days,steps,stations] shape")
    if len(lanes) != shape[2] or len(regimes) != shape[0]:
        raise ValueError("Lane or regime dimensions do not match state arrays")
    if set(regimes) - set(REGIMES):
        raise ValueError("Invalid regime")
    return shape


def state_score(speed_true, flow_true, pred_speed, pred_flow, *, target, lanes, regimes):
    """Score aligned arrays; each missing prediction channel independently becomes zero."""
    _shapes(speed_true, flow_true, pred_speed, pred_flow, target, lanes, regimes)
    lanes = np.maximum(np.asarray(lanes, dtype=np.float64), 1.0)
    stats = {r: {"n_target_cells": 0, "n_missing_predictions": 0, "n_invalid_truth": 0,
                 "sum_sq_speed": 0.0, "sum_sq_flow_per_lane": 0.0} for r in REGIMES}
    for day, regime in enumerate(regimes):
        mask = np.asarray(target[day], dtype=bool)
        tv = np.asarray(speed_true[day], dtype=np.float64)
        tq = np.asarray(flow_true[day], dtype=np.float64)
        pv = np.asarray(pred_speed[day], dtype=np.float64)
        pq = np.asarray(pred_flow[day], dtype=np.float64)
        usable = mask & np.isfinite(tv) & np.isfinite(tq)
        missing = ~np.isfinite(pv) | ~np.isfinite(pq)
        dv = np.where(np.isfinite(pv), pv, 0.0) - tv
        dq = (np.where(np.isfinite(pq), pq, 0.0) - tq) / lanes
        row = stats[regime]
        row["n_target_cells"] += int(usable.sum())
        row["n_missing_predictions"] += int((usable & missing).sum())
        row["n_invalid_truth"] += int((mask & ~usable).sum())
        row["sum_sq_speed"] += float(np.sum(dv[usable] ** 2))
        row["sum_sq_flow_per_lane"] += float(np.sum(dq[usable] ** 2))
    for row in stats.values():
        n = row["n_target_cells"]
        row["rmse_speed_kmh"] = float(np.sqrt(row["sum_sq_speed"] / n)) if n else float("nan")
        row["rmse_flow_vph_per_lane"] = float(np.sqrt(row["sum_sq_flow_per_lane"] / n)) if n else float("nan")
        row["S_speed"] = max(0.0, 1 - row["rmse_speed_kmh"] / 25) if n else 0.0
        row["S_flow"] = max(0.0, 1 - row["rmse_flow_vph_per_lane"] / 600) if n else 0.0
        row["S_state"] = .54 * row["S_speed"] + .46 * row["S_flow"]
    return {"regimes": stats, "S_state": float(np.mean([stats[r]["S_state"] for r in REGIMES])),
            "n_target_cells": sum(x["n_target_cells"] for x in stats.values())}


def source_truth(cache, root=None):
    """Prepare a reusable float64 truth mmap without loading a whole panel into RAM."""
    if cache.meta["split"] != "train":
        raise ValueError("Released truth exists only for train")
    root = Path(root or project_root())
    prefix = f"corridors/{cache.meta['panel']}/train/mainline_states/"
    sources = [s for s in cache.meta["sources"] if s["path"].startswith(prefix)]
    import hashlib
    fingerprint = hashlib.sha256("".join(s["sha256"] for s in sources).encode()).hexdigest()
    output = root / "artifacts" / "metric_truth" / cache.meta["panel"] / fingerprint[:16]
    marker = output / "meta.json"
    if not marker.exists():
        shape = tuple(cache.meta["shape"])
        guard(root, output, int(np.prod(shape)) * 16 + 1024**2)
        output.mkdir(parents=True, exist_ok=True)
        arrays = {name: open_memmap(output / (name + ".npy"), mode="w+", dtype=np.float64, shape=shape)
                  for name in ("speed", "flow")}
        for array in arrays.values():
            array[:] = np.nan
        days = {day: i for i, day in enumerate(cache.dates)}
        entities = dict(zip(cache.stations.station_id, cache.stations.idx))
        for source in sources:
            frame = read_parquet(release_root(root) / source["path"],
                                 ["timestamp", "station_id", "speed_kmh", "flow_vph"])
            keys = indices(frame, days, entities, "station_id")
            arrays["speed"][keys] = frame.speed_kmh.to_numpy(dtype=np.float64)
            arrays["flow"][keys] = frame.flow_vph.to_numpy(dtype=np.float64)
        for array in arrays.values():
            array.flush()
        marker.write_text(json.dumps({"sha256": fingerprint, "shape": list(shape)}), encoding="utf-8")
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if metadata["sha256"] != fingerprint or metadata["shape"] != cache.meta["shape"]:
        raise ValueError("Exact truth cache does not match the release")
    return tuple(np.load(output / (name + ".npy"), mmap_mode="r") for name in ("speed", "flow"))


def state_score_cache(cache, pred_speed, pred_flow, exact=True, root=None):
    truth = source_truth(cache, root=root) if exact else (cache.speed_true, cache.flow_true)
    result = state_score(*truth, pred_speed, pred_flow, target=cache.target,
                         lanes=cache.stations.lanes.to_numpy(), regimes=cache.regimes)
    result["truth_precision"] = "source_float64" if exact else "cache_float32"
    return result


def _parameters(cache, root):
    parameters = cache.stations.copy()
    network = release_root(root) / "corridors" / cache.meta["panel"] / "network"
    fd = pd.read_csv(network / "fd_parameters.csv")
    numeric = [x for x in ("free_speed_kmh", "capacity_vph", "lanes", "k_jam") if x in fd]
    fd = fd.groupby("link_id", sort=False)[numeric].mean()
    for column in numeric:
        parameters[column] = parameters.link_id.map(fd[column]).fillna(parameters[column])
    parameters["lanes"] = parameters.lanes.clip(lower=1)
    parameters["critical_density"] = parameters.capacity_vph / parameters.free_speed_kmh.clip(lower=1)
    parameters["k_jam"] = np.maximum(parameters.k_jam.fillna(100 * parameters.lanes),
                                    1.05 * parameters.critical_density)
    topo = pd.read_csv(network / "lwr_mainline_topology.csv", dtype=str).set_index("link_id")
    on = set(topo.index[topo.on_ramp_link_ids.fillna("").str.len() > 0])
    off = set(topo.index[topo.off_ramp_link_ids.fillna("").str.len() > 0])
    return parameters, topo, on, off


def physics_score(cache, pred_speed, pred_flow, *, truth_speed=None, truth_flow=None,
                  mode=None, root=None, ramp_filter=True):
    """Truth-projected physics replica. This is a local training instrument, not hidden flux."""
    if not cache.meta["stations_links_one_to_one"]:
        raise ValueError("Fast physics path requires audited one-station-per-link panels")
    root = Path(root or project_root())
    if truth_speed is None or truth_flow is None:
        truth_speed, truth_flow = source_truth(cache, root=root)
    _shapes(truth_speed, truth_flow, pred_speed, pred_flow, cache.target,
            cache.stations.lanes, cache.regimes)
    if mode is None:
        config = root / "ref" / "trafficflowbench-public" / "config" / "task3_lwr_modes.json"
        mode = json.loads(config.read_text(encoding="utf-8"))["panels"][cache.meta["panel"]]["mode"]
    if mode not in ("A", "B", "C"):
        raise ValueError("Invalid physics coverage mode")
    p, topo, on_attached, off_attached = _parameters(cache, root)
    lanes = p.lanes.to_numpy(dtype=float)
    vf = p.free_speed_kmh.to_numpy(dtype=float)
    cap = p.capacity_vph.to_numpy(dtype=float)
    crit = p.critical_density.to_numpy(dtype=float)
    jam = p.k_jam.to_numpy(dtype=float)
    length = p.length_km.to_numpy(dtype=float)
    links = p.link_id.to_numpy()
    topo_ids = set(topo.index)
    stats = {r: {"fd_error": 0.0, "fd_scale": 0.0, "n_state_rows": 0,
                 "n_submitted": 0, "n_below_floor": 0, "lwr_error": 0.0,
                 "lwr_scale": 0.0, "n_rows": 0} for r in REGIMES}
    available = {r: np.zeros(len(links), bool) for r in REGIMES}

    def field(day):
        target = np.asarray(cache.target[day], dtype=bool)
        tv = np.asarray(truth_speed[day], dtype=float)
        tq = np.asarray(truth_flow[day], dtype=float)
        v = np.where(target, pred_speed[day], tv)
        q = np.where(target, pred_flow[day], tq)
        valid = np.asarray(cache.elig[day], dtype=bool) & np.isfinite(v) & np.isfinite(q)
        n = q / np.maximum(v, 1.0) * length
        true_n = tq / np.maximum(tv, 1.0) * length
        return v, q, valid, n, true_n, target

    for day, regime in enumerate(cache.regimes):
        _, _, valid, _, _, _ = field(day)
        available[regime] |= valid.any(axis=0)
    observable = {}
    for regime, seen in available.items():
        if mode == "C":
            seen = seen & ~np.isin(links, list(on_attached | off_attached))
        seen_links = set(links[seen])
        observable[regime] = np.asarray([
            link in topo.index and
            ({x.strip() for x in str(topo.loc[link, "incoming_link_ids"]).split(";")} & topo_ids).issubset(seen_links)
            and ({x.strip() for x in str(topo.loc[link, "outgoing_link_ids"]).split(";")} & topo_ids).issubset(seen_links)
            for link in links], dtype=bool) & seen

    def ramp_valid(day):
        result = np.ones((cache.meta["shape"][1], len(links)), dtype=bool)
        if not ramp_filter or mode == "C":
            return result
        for kind, attached in (("OR", on_attached), ("FR", off_attached)):
            keep = np.ones_like(result)
            keep[:, np.isin(links, list(attached))] = False
            for ramp in cache.ramps.loc[cache.ramps.ramp_type == kind].itertuples(index=False):
                valid = cache.ramp_elig[day, :, ramp.idx] & np.isfinite(cache.ramp_flow[day, :, ramp.idx])
                keep[:, ramp.station_idx] |= valid
            result &= keep
        return result

    def accumulate(row, current_n, next_n, current_true, next_true, valid):
        if valid.any():
            delta = next_n - current_n
            rhs = next_true - current_true
            row["lwr_error"] += float(np.abs(delta[valid] - rhs[valid]).sum())
            row["lwr_scale"] += float(np.abs(rhs[valid]).sum())
            row["n_rows"] += int(valid.sum())

    for day, regime in enumerate(cache.regimes):
        v, q, valid, n, true_n, target = field(day)
        row = stats[regime]
        density = q / np.maximum(v, 1.0)
        k_lane = density / lanes
        q_lane = q / lanes
        q_fd = np.where(k_lane <= crit / lanes, vf * k_lane,
                        (cap / lanes / np.maximum(jam / lanes - crit / lanes, EPS))
                        * np.maximum(jam / lanes - k_lane, 0))
        row["fd_error"] += float(np.abs(q_lane[valid] - q_fd[valid]).sum())
        row["fd_scale"] += float(np.abs(q_lane[valid]).sum())
        row["n_state_rows"] += int(valid.sum())
        supplied = target & valid
        row["n_submitted"] += int(supplied.sum())
        row["n_below_floor"] += int((supplied & (q < 50)).sum())
        ok = observable[regime][None, :]
        ramps = ramp_valid(day)
        transitions = valid[:-1] & valid[1:] & ok & ramps[:-1]
        accumulate(row, n[:-1], n[1:], true_n[:-1], true_n[1:], transitions)
        if day + 1 < len(cache.dates) and cache.regimes[day + 1] == regime:
            delta_day = (pd.Timestamp(cache.dates[day + 1]) - pd.Timestamp(cache.dates[day])).days
            if delta_day == 1 and cache.meta["shape"][1] == 288:
                _, _, next_valid, next_n, next_true, _ = field(day + 1)
                crossing = valid[-1] & next_valid[0] & observable[regime] & ramps[-1]
                accumulate(row, n[-1], next_n[0], true_n[-1], next_true[0], crossing)
    for row in stats.values():
        row["S_FD"] = max(0.0, 1 - row["fd_error"] / (row["fd_scale"] + EPS))
        if not row["n_state_rows"] or (row["n_submitted"] and
                row["n_below_floor"] / row["n_submitted"] > .20):
            row["S_FD"] = 0.0
        row["E_LWR"] = row["lwr_error"] / (row["lwr_scale"] + EPS) if row["n_rows"] else 1.0
        row["S_LWR"] = max(0.0, 1 - min(1.0, row["E_LWR"])) if row["n_rows"] else 0.0
        row["S_physics"] = (row["S_FD"] + 2 * row["S_LWR"]) / 3
    return {"regimes": stats, "S_physics": float(np.mean([stats[r]["S_physics"] for r in REGIMES])),
            "mode": mode, "boundary": "truth_projected", "truth_precision": str(truth_speed.dtype)}


def queue_score(prediction, truth, windows):
    """Keyed queue score with equal condition, panel and family aggregation."""
    keys = ["window_id", "timestamp", "link_id"]
    prediction, truth, windows = prediction.copy(), truth.copy(), windows.copy()
    for frame in (prediction, truth):
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
        for column in ("window_id", "link_id"):
            frame[column] = frame[column].astype(str)
        if frame.duplicated(keys).any():
            raise ValueError("Duplicate queue cell key")
    values = pd.to_numeric(prediction.queue_pred, errors="coerce")
    if not values.dropna().isin([0, 1]).all():
        raise ValueError("Queue predictions must be binary")
    prediction["queue_pred"] = values
    rows = []
    for window in windows.itertuples(index=False):
        target = truth[(truth.window_id == window.window_id) & truth.is_score_eligible.astype(bool)]
        pred = prediction[prediction.window_id == window.window_id]
        if target.empty:
            iou, missing, union, intersection = 0.0, 0, 0, 0
        else:
            if target.queue_true.isna().any():
                raise ValueError("Eligible queue labels must not be null")
            merged = target.merge(pred[keys + ["queue_pred"]], on=keys, how="left", validate="one_to_one")
            missing = int(merged.queue_pred.isna().sum())
            pv = merged.queue_pred.fillna(0).to_numpy(dtype=bool)
            tv = merged.queue_true.to_numpy(dtype=bool)
            intersection = int((pv & tv).sum())
            union = int((pv | tv).sum())
            iou = intersection / union if union else 1.0
        rows.append({"window_id": window.window_id, "panel": window.panel,
                     "condition": window.condition, "IoU_ST": iou,
                     "n_eligible": len(target), "missing_predictions": missing,
                     "intersection": intersection, "union": union})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return {"S_queue": 0.0, "windows": [], "panels": {}, "conditions": {}}
    conditions = detail.groupby(["panel", "condition"], sort=True).IoU_ST.mean()
    panel_scores = conditions.groupby("panel", sort=True).mean().to_dict()
    return {"S_queue": aggregate(panel_scores, "queue"), "windows": rows, "panels": panel_scores,
            "conditions": {f"{p}/{c}": float(v) for (p, c), v in conditions.items()}}


def odme_link_score(A, path_flow, counts):
    """Counted-link fit; invalid/nonfinite/negative paths invalidate the corridor."""
    f, c = np.asarray(path_flow, dtype=float), np.asarray(counts, dtype=float)
    if A.shape != (len(c), len(f)):
        raise ValueError("Operator and flow/count dimensions differ")
    if not np.isfinite(f).all() or (f < 0).any():
        return 0.0
    loaded = np.asarray(A @ f).reshape(-1)
    return max(0.0, 1 - float(np.abs(loaded - c).sum()) / max(float(c.sum()), EPS))


def odme_score(A, path_flow, counts, *, truth, prior, destinations):
    """All ODME terms for synthetic tests only; no released path-flow truth exists."""
    f, star, b = (np.asarray(x, dtype=float) for x in (path_flow, truth, prior))
    if f.shape != star.shape or f.shape != b.shape or len(destinations) != len(f):
        raise ValueError("ODME path dimensions differ")
    if not np.isfinite(f).all() or (f < 0).any():
        return {k: 0.0 for k in ("S_od", "S_link", "S_dev", "S_attr", "S_ODME")}
    s_od = max(0.0, 1 - float(np.abs(f - star).sum()) / max(float(star.sum()), EPS))
    s_link = odme_link_score(A, f, counts)
    dhat, dstar = float(np.abs(f - b).sum()), float(np.abs(star - b).sum())
    # Continuous extension for the synthetic degenerate case truth == prior.
    s_dev = float(np.exp(-abs(dhat / dstar - 1))) if dstar > 0 else float(dhat == 0)
    _, inverse = np.unique(np.asarray(destinations), return_inverse=True)
    attraction = np.bincount(inverse, weights=f) / max(float(f.sum()), EPS)
    reference = np.bincount(inverse, weights=star) / max(float(star.sum()), EPS)
    s_attr = max(0.0, 1 - .5 * float(np.abs(attraction - reference).sum()))
    return {"S_od": s_od, "S_link": s_link, "S_dev": s_dev, "S_attr": s_attr,
            "S_ODME": .45 * s_od + .25 * s_link + .15 * s_dev + .15 * s_attr}


def write_truth_projected_flux(cache, output, root=None):
    """Write a train-only flux from released truth; never read organizer-only flux."""
    if cache.meta["split"] != "train" or not cache.meta["stations_links_one_to_one"]:
        raise ValueError("Flux projection requires train and audited one-to-one station/link mapping")
    import os
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .data import dated_files
    root = Path(root or project_root())
    output = Path(output)
    guard(root, output, int(np.prod(cache.meta["shape"])) * 70)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".building.parquet")
    tv, tq = source_truth(cache, root=root)
    length = cache.stations.length_km.to_numpy(dtype=float)
    days = {day: i for i, day in enumerate(cache.dates)}
    ramp_ids = dict(zip(cache.ramps.ramp_id, cache.ramps.idx))
    ramp_sources = dated_files(release_root(root) / "corridors" / cache.meta["panel"] / "train" / "ramp_states")
    attached = cache.ramps.station_idx.to_numpy()
    links = cache.stations.link_id.to_numpy()
    writer = None
    written = 0
    try:
        for day, date in enumerate(cache.dates):
            v, q = np.asarray(tv[day]), np.asarray(tq[day])
            valid = cache.elig[day] & np.isfinite(v) & np.isfinite(q)
            n = q / np.maximum(v, 1) * length
            dn = np.zeros_like(n)
            adjacent = valid[:-1] & valid[1:]
            dn[:-1] = np.where(adjacent, n[1:] - n[:-1], 0)
            if day + 1 < len(cache.dates) and cache.regimes[day + 1] == cache.regimes[day]:
                next_v, next_q = np.asarray(tv[day + 1][0]), np.asarray(tq[day + 1][0])
                next_valid = cache.elig[day + 1][0] & np.isfinite(next_v) & np.isfinite(next_q)
                dn[-1] = np.where(valid[-1] & next_valid,
                                  next_q / np.maximum(next_v, 1) * length - n[-1], 0)
            ramps = read_parquet(ramp_sources[date])
            keys = indices(ramps, days, ramp_ids, "ramp_link_id")
            eligible = ramps.is_score_eligible.astype(bool).to_numpy() & (ramps.pct_observed.to_numpy() >= 75)
            eligible &= np.isfinite(ramps.flow_vph.to_numpy())
            on, off = np.zeros_like(n), np.zeros_like(n)
            for flow, kinds in ((on, ("OR", "ON")), (off, ("FR", "OFF"))):
                chosen = eligible & ramps.ramp_type.str.upper().isin(kinds).to_numpy()
                np.add.at(flow, (keys[1][chosen], attached[keys[2][chosen]]),
                          ramps.flow_vph.to_numpy(dtype=float)[chosen])
            incoming = dn / (5 / 60) - on + off
            timestamps = np.asarray([f"{date}T{step//12:02d}:{(step%12)*5:02d}:00Z" for step in range(288)])
            rows, columns = np.where(valid)
            table = pa.table({"panel": [cache.meta["panel"]] * len(rows),
                              "timestamp": timestamps[rows], "link_id": links[columns],
                              "inflow_vph": incoming[rows, columns],
                              "outflow_vph": np.zeros(len(rows))})
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            written += len(rows)
    finally:
        if writer is not None:
            writer.close()
    os.replace(temporary, output)
    return {"path": str(output), "n_rows": written, "source": "released train truth",
            "identity": "inflow=dN/dt-on+off; outflow=0"}
