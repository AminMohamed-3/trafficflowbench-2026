"""Observed-only station constants with explicit period and hidden-cell scope."""
from datetime import date
import warnings

import numpy as np
import pandas as pd


VERSION = 1
NIGHT_TIMEZONE = "America/Los_Angeles"


def _median(values, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def estimate_constants(cache, dates=None, *, hidden=None, check=None):
    """Return constants from exactly the named dates' released observations.

    Slots and weekday/weekend indices use UTC, matching existing lane features.
    The free-speed night sample uses 02:00 <= local hour < 05:00 Pacific time.
    Hidden cells are excluded from both fields before any statistic. No truth,
    target or occupancy array is read. Missing historical bins remain NaN.
    """
    chosen = list(cache.dates if dates is None else dates)
    if len(set(chosen)) != len(chosen) or chosen != sorted(chosen):
        raise ValueError("Constant dates must be unique and chronological")
    lookup = {d: i for i, d in enumerate(cache.dates)}
    if any(d not in lookup for d in chosen):
        raise ValueError("Constant period lies outside the supplied view")
    indices = np.asarray([lookup[d] for d in chosen], dtype=int)
    shape = cache.speed.shape
    if cache.flow.shape != shape or shape[1] != 288:
        raise ValueError("Incompatible observed grids")
    if hidden is not None and (hidden.shape != shape or hidden.dtype != np.bool_):
        raise ValueError("Hidden-cell mask must match the observed grids")
    ns = shape[2]
    nominal = cache.stations.free_speed_kmh.to_numpy(dtype=np.float32)
    lanes = cache.stations.lanes.to_numpy(dtype=np.float32)
    if len(nominal) != ns or not np.all(lanes > 0):
        raise ValueError("Station geometry differs from the observed grids")
    weekend = np.asarray([date.fromisoformat(d).weekday() >= 5 for d in chosen])
    ticks = pd.DatetimeIndex([pd.Timestamp(d, tz="UTC") + pd.Timedelta(minutes=5*t)
                             for d in chosen for t in range(288)], tz="UTC")
    hours = np.asarray(ticks.tz_convert(NIGHT_TIMEZONE).hour).reshape(len(chosen), 288)
    night = (hours >= 2) & (hours < 5)
    arrays = {"free_speed": nominal.copy(), "free_speed_high": np.full(ns, np.nan, np.float32),
              "free_speed_night": np.full(ns, np.nan, np.float32),
              "free_speed_high_count": np.zeros(ns, np.int32),
              "free_speed_night_count": np.zeros(ns, np.int32),
              "free_speed_used_count": np.zeros(ns, np.int32),
              "history_speed": np.full((2, 288, ns), np.nan, np.float32),
              "history_flow_lane": np.full((2, 288, ns), np.nan, np.float32),
              "history_speed_count": np.zeros((2, 288, ns), np.int32),
              "history_flow_lane_count": np.zeros((2, 288, ns), np.int32)}
    for station in range(ns):
        v = np.array(cache.speed[indices, :, station], dtype=np.float32, copy=True)
        q = np.array(cache.flow[indices, :, station], dtype=np.float32, copy=True) / lanes[station]
        valid_v = np.isfinite(v) & (v >= 1)
        valid_q = np.isfinite(q) & (q >= 0)
        if hidden is not None:
            excluded = hidden[indices, :, station]
            valid_v &= ~excluded
            valid_q &= ~excluded
        v[~valid_v] = np.nan
        q[~valid_q] = np.nan
        high = valid_v & (v > .85*nominal[station])
        night_high = high & night
        for key, mask in (("high", high), ("night", night_high)):
            count = int(mask.sum())
            arrays[f"free_speed_{key}_count"][station] = count
            if count:
                arrays[f"free_speed_{key}"][station] = _median(v[mask])
        # Prefer the quiet-night estimate; use all high-speed observations if
        # fewer than two hours of samples survive. Empty periods use geometry.
        used = "night" if night_high.sum() >= 24 else "high"
        if arrays[f"free_speed_{used}_count"][station] >= 24:
            arrays["free_speed"][station] = arrays[f"free_speed_{used}"][station]
            arrays["free_speed_used_count"][station] = arrays[f"free_speed_{used}_count"][station]
        for kind in (0, 1):
            rows = weekend == bool(kind)
            if rows.any():
                arrays["history_speed"][kind, :, station] = _median(v[rows], axis=0)
                arrays["history_flow_lane"][kind, :, station] = _median(q[rows], axis=0)
                arrays["history_speed_count"][kind, :, station] = valid_v[rows].sum(axis=0)
                arrays["history_flow_lane_count"][kind, :, station] = valid_q[rows].sum(axis=0)
        if check is not None and station % 16 == 0:
            check()
    return arrays


def own_month_constants(cache, *, hidden=None, check=None):
    """One independent observed table per UTC calendar month."""
    return {month: estimate_constants(cache, [d for d in cache.dates if d.startswith(month)],
                                       hidden=hidden, check=check)
            for month in sorted({d[:7] for d in cache.dates})}


def online_constants(cache, query_date, *, check=None):
    """Strict earlier-date prefix; same-day inputs belong to online_view.

    This API never reads the query date or any later date. The queue owner's
    accessor supplies same-day history through T separately. Training priors
    may be used as a separately recorded fallback under the Task 2 rules.
    """
    query_date = str(query_date)
    if query_date not in cache.dates:
        raise ValueError("Query date is outside the supplied split")
    return estimate_constants(cache, [d for d in cache.dates if d < query_date], check=check)
