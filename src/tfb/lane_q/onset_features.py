"""Canonical Q-03 precursor features. No labels or future values are inputs."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .windows import candidate_windows, grid_from_frame, timestamp


STATIC_FEATURES = ["target_hour", "weekday", "weekend", "site_km", "site_free_speed", "site_capacity", "site_cells", "other_site"]
LOCAL_FEATURES = [*[f"flow_ratio_mean_{j}" for j in range(4)], *[f"flow_ratio_slope_{j}" for j in range(4)],
                  "site_speed_min_ratio", "site_speed_slope_ratio", "site_last_speed_ratio", "other_sites_min_speed_ratio", "other_sites_queued_now"]
EXTRA_FEATURES = ["characteristic_flow_ratio", "characteristic_speed_ratio", "characteristic_available", "minutes_since_queued",
                  "prior_month_first_share", "incident_min_speed_ratio", "incident_speed_drop_ratio"]
NUMERIC_FEATURES = STATIC_FEATURES + LOCAL_FEATURES + EXTRA_FEATURES


def site_table(panel_prior, *, minimum_onsets=3):
    known, rare = [], []
    covered = set()
    links = panel_prior["links"]
    for site in panel_prior["sites"]:
        if site["n_windows"] >= minimum_onsets:
            item = {**site, "key": f"site_{min(site['positions'])}", "other": False}
            known.append(item)
            covered.update(site["positions"])
        else:
            rare.extend(site["cell_sets"])
    other_positions = [i for i in range(len(links)) if i not in covered]
    if other_positions:
        rare_counts = defaultdict(int)
        for entry in rare:
            rare_counts[tuple(entry["links"])] += entry["count"]
        known.append({"key": "other", "other": True, "positions": other_positions,
                      "links": [links[i] for i in other_positions], "n_windows": sum(rare_counts.values()),
                      "cell_sets": [{"links":list(cells),"count":n} for cells,n in sorted(rare_counts.items(),key=lambda item:(-item[1],item[0]))]})
    return known


def finite_min(values):
    values = np.asarray(values,dtype=float)
    values = values[np.isfinite(values)]
    return float(values.min()) if len(values) else np.nan


def finite_mean(values):
    values = np.asarray(values,dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def slope(frame, column, origin):
    valid = frame[np.isfinite(frame[column])]
    if len(valid) < 2:
        return np.nan
    x = (valid.timestamp-origin).dt.total_seconds().to_numpy()/60
    y = valid[column].to_numpy(dtype=float)
    x = x-x.mean()
    return float(np.dot(x,y-y.mean())/np.dot(x,x)) if np.dot(x,x)>0 else np.nan


class PriorDayContext:
    """Cached earlier-day summaries; explicit date filter precedes aggregation."""
    def __init__(self):
        self._days = {}

    def shares(self, view, sites):
        month_start = view.date[:7]+"-01"
        key = (view.panel,view.split,month_start,tuple(s["key"] for s in sites))
        days = self._days.setdefault(key,{})
        start = str((timestamp(max(days))+pd.Timedelta(days=1)).date()) if days else month_start
        for date,frame in view.previous_days(start_date=start):
            grid = grid_from_frame(frame,view.network,date)
            candidates = candidate_windows(grid)
            onsets = candidates[candidates.condition.eq("queue_onset")].sort_values("forecast_origin")
            first = set()
            if len(onsets):
                t = timestamp(onsets.iloc[0].forecast_origin)
                k = t.hour*12+t.minute//5+6
                for site in sites:
                    if grid.queued[k,site["positions"]].any():
                        first.add(site["key"])
            days[date] = first
        allowed = [first for date,first in days.items() if month_start<=date<view.date]
        return {s["key"]:sum(s["key"] in first for first in allowed)/len(allowed) if allowed else 0. for s in sites}


def feature_frame(view, sites, *, prior_context=None):
    """One row/site using the accessor's same-day and earlier-day observations."""
    network = view.network
    fd = network.fd_parameters
    if fd is None or not {"capacity_vph","length_km"}.issubset(fd.columns):
        raise ValueError("Q-03 requires released FD capacities and lengths")
    capacity = fd.capacity_vph.to_numpy(dtype=float)
    length = fd.length_km.to_numpy(dtype=float)
    if not (np.isfinite(capacity).all() and (capacity>0).all() and np.isfinite(length).all()):
        raise ValueError("Invalid released FD parameters")
    km = np.cumsum(length)
    travel_hours = np.cumsum(length/network.free_speed)
    observations = view.observations()
    last30 = observations[observations.timestamp.ge(view.origin-pd.Timedelta(minutes=30))].copy()
    last30["speed_ratio"] = last30.speed_kmh/last30.link_id.map(dict(zip(network.links,network.free_speed)))
    last30["flow_ratio"] = last30.flow_vph/last30.link_id.map(dict(zip(network.links,capacity)))
    valid = observations[np.isfinite(observations.speed_kmh)]
    latest = valid.sort_values(["timestamp","link_id"]).groupby("link_id",sort=False).tail(1)
    latest_ratio = latest.set_index("link_id").speed_kmh.reindex(network.links).to_numpy(dtype=float)/network.free_speed
    site_last = {site["key"]:finite_min(latest_ratio[site["positions"]]) for site in sites if not site["other"]}
    site_queued = {site["key"]:bool(np.any(latest_ratio[site["positions"]]<=.6)) for site in sites if not site["other"]}
    shares = (prior_context or PriorDayContext()).shares(view,sites)
    outside = [link for site in sites if site["other"] for link in site["links"]]
    unknown = last30[last30.link_id.isin(outside)]
    incident_min = finite_min(unknown.speed_ratio)
    before = unknown[unknown.timestamp.le(view.origin-pd.Timedelta(minutes=15))].groupby("link_id").speed_ratio.mean()
    after = unknown[unknown.timestamp.gt(view.origin-pd.Timedelta(minutes=15))].groupby("link_id").speed_ratio.mean()
    incident_drop = finite_min((after-before).dropna())
    target = view.origin+pd.Timedelta(minutes=30)
    rows = []
    for site in sites:
        positions = site["positions"]
        if site["other"]:
            ratios = latest_ratio[positions]
            anchor = positions[int(np.nanargmin(ratios))] if np.isfinite(ratios).any() else positions[0]
        else:
            anchor = max(positions)
        site_rows = last30[last30.link_id.isin(site["links"])]
        average_speed = site_rows.groupby("timestamp",as_index=False).speed_ratio.mean()
        row = {"site_uid":f"{view.panel}/{site['key']}", "target_hour":target.hour+target.minute/60,
               "weekday":target.weekday(),"weekend":int(target.weekday()>=5),"site_km":float(km[anchor]),
               "site_free_speed":float(network.free_speed[anchor]),"site_capacity":float(capacity[anchor]),
               "site_cells":len(positions),"other_site":int(site["other"]),
               "site_speed_min_ratio":finite_min(site_rows.speed_ratio),"site_speed_slope_ratio":slope(average_speed,"speed_ratio",view.origin),
               "site_last_speed_ratio":finite_min(latest_ratio[positions]),
               "other_sites_min_speed_ratio":finite_min([value for key,value in site_last.items() if key!=site["key"]]),
               "other_sites_queued_now":sum(value for key,value in site_queued.items() if key!=site["key"]),
               "incident_min_speed_ratio":incident_min,"incident_speed_drop_ratio":incident_drop,
               "prior_month_first_share":shares[site["key"]]}
        for j in range(4):
            idx = max(0,anchor-j)
            local = last30[last30.link_id.eq(network.links[idx])]
            row[f"flow_ratio_mean_{j}"] = finite_mean(local.flow_ratio)
            row[f"flow_ratio_slope_{j}"] = slope(local,"flow_ratio",view.origin)
        wanted = travel_hours[anchor]-.5
        if wanted<0:
            upstream,available=0,0
        else:
            upstream=int(np.abs(travel_hours[:anchor+1]-wanted).argmin())
            available=1
        arrival = last30[last30.link_id.eq(network.links[upstream])]
        arrival_recent=arrival[arrival.timestamp.ge(view.origin-pd.Timedelta(minutes=10))]
        row["characteristic_flow_ratio"]=finite_mean(arrival_recent.flow_vph)/capacity[anchor]
        row["characteristic_speed_ratio"]=finite_mean(arrival_recent.speed_ratio)
        row["characteristic_available"]=available
        queued=observations[observations.link_id.isin(site["links"]) & observations.speed_kmh.le(observations.link_id.map(network.cutoff_by_link))]
        row["minutes_since_queued"]=(view.origin-queued.timestamp.max()).total_seconds()/60 if len(queued) else -1.
        rows.append(row)
    frame = pd.DataFrame(rows,columns=["site_uid",*NUMERIC_FEATURES])
    frame[NUMERIC_FEATURES]=frame[NUMERIC_FEATURES].astype(np.float32)
    return frame
