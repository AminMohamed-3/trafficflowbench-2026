"""Canonical S-01 state baselines and the required submission contract."""
from functools import lru_cache
import numpy as np
import pandas as pd
from tfb.data import load_cache, indices
from tfb.paths import release_root
from .reference_data import interp_time, interp_space

METHODS = ('B1', 'B2', 'B3')


def _fit_dates(cfg):
    dates = cfg.get('fit_dates')
    return tuple(sorted(str(d) for d in dates)) if dates is not None else None


@lru_cache(maxsize=40)
def free_flow_means(panel, fit_dates=None):
    """Estimate station free-flow speeds from explicitly selected released train days."""
    train = load_cache(panel, 'train')
    selected = np.arange(len(train.dates)) if fit_dates is None else np.asarray([train.day_index(d) for d in fit_dates])
    if not len(selected):
        raise ValueError('Free-flow fitting population is empty')
    vf = train.stations.free_speed_kmh.to_numpy(np.float32)
    total = np.zeros(len(vf), dtype=float)
    count = np.zeros(len(vf), dtype=np.int64)
    for d in selected:
        values = train.speed_true[d]
        keep = np.isfinite(values) & (values > 0.85 * vf)
        total += np.where(keep, values, 0.).sum(axis=0, dtype=float)
        count += keep.sum(axis=0)
    return np.divide(total, count, out=vf.astype(float), where=count > 0).astype(np.float32)


def _complete_day_missing(prediction, observations, fallback):
    """Explicit fallback only for station-days without a single observed value."""
    out = prediction
    for s in range(out.shape[2]):
        invalid = ~np.isfinite(out[:, :, s])
        if invalid.any():
            values = observations[:, :, s]
            usable = values[np.isfinite(values)]
            value = float(np.median(usable)) if len(usable) else float(fallback[s])
            out[:, :, s][invalid] = value
    return out


def predict_arrays(cache, cfg):
    """Return [day,step,station] float32 arrays using released masked inputs only."""
    method = cfg.get('method', 'B1')
    if method not in METHODS:
        raise ValueError(f'Unknown S-01 method: {method}')
    pv, pq = interp_time(cache.speed), interp_time(cache.flow)
    vf = cache.stations.free_speed_kmh.to_numpy(np.float32)
    cap = cache.stations.capacity_vph.to_numpy(float)
    pv = _complete_day_missing(pv, cache.speed, vf)
    pq = _complete_day_missing(pq, cache.flow, 0.4 * cap)
    if method in ('B2', 'B3'):
        if cache.meta['split'] == 'train' and cfg.get('fit_dates') is None:
            raise ValueError('Training-fold predictions require explicit fit_dates for B2/B3')
        means = free_flow_means(cache.meta['panel'], _fit_dates(cfg))
        times = np.arange(cache.speed.shape[1])
        for d in range(len(cache.dates)):
            for s in range(cache.speed.shape[2]):
                y = cache.speed[d, :, s]
                valid = np.isfinite(y)
                before = np.maximum.accumulate(np.where(valid, times, -1))
                after = np.minimum.accumulate(np.where(valid, times, len(times))[::-1])[::-1]
                interior = (before >= 0) & (after < len(times)) & cache.target[d, :, s]
                bi, ai = np.maximum(before, 0), np.minimum(after, len(times) - 1)
                free = interior & (y[bi] > .85 * vf[s]) & (y[ai] > .85 * vf[s])
                pv[d, free, s] = means[s]
    if method == 'B3':
        length = cache.stations.length_km.to_numpy(float)
        positions = length.cumsum() - length / 2
        spatial = interp_space(cache.flow, positions)
        # With no observations at all in a gap row, spatial interpolation is
        # undefined. Preserve the canonical B1 total-flow estimate there.
        pq = np.where(np.isfinite(spatial), spatial, pq).astype(np.float32)
    np.clip(pv, 3., 120.75, out=pv)
    np.maximum(pq, 50., out=pq)
    # Round the ceiling inward so float32 output never exceeds the exact bound.
    upper = 1.3 * cap
    upper32 = upper.astype(np.float32)
    upper32 = np.where(upper32.astype(float) > upper, np.nextafter(upper32, np.float32(-np.inf)), upper32)
    np.minimum(pq, upper32, out=pq)
    if not np.isfinite(pv[cache.target]).all() or not np.isfinite(pq[cache.target]).all():
        raise ValueError('Nonfinite prediction on a published target')
    return pv.astype(np.float32, copy=False), pq.astype(np.float32, copy=False)


def predict_state(panel, split, cfg):
    """One keyed row per published template, preserving its timestamp text."""
    cache = load_cache(panel, split)
    pv, pq = predict_arrays(cache, cfg)
    template = release_root() / 'task1' / panel / split / 'sample_submission_state.csv'
    columns = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']
    output = pd.read_csv(template, usecols=columns, dtype=str)
    if len(output) != cache.meta['target_rows'] or output.duplicated(columns).any():
        raise ValueError('Published template population differs from audited cache')
    days = {d: i for i, d in enumerate(cache.dates)}
    stations = dict(zip(cache.stations.station_id, cache.stations.idx))
    key = indices(output, days, stations, 'station_id')
    if not cache.target[key].all():
        raise ValueError('Output includes a cell outside the published target mask')
    link_map = cache.stations.set_index('station_id').link_id
    if not np.array_equal(output.link_id.to_numpy(), output.station_id.map(link_map).to_numpy()):
        raise ValueError('Template station/link mapping differs from cache')
    if not np.array_equal(output.mask_regime.to_numpy(), cache.regimes[key[0]]):
        raise ValueError('Template regime differs from cache')
    output['speed_kmh'], output['flow_vph'] = pv[key], pq[key]
    return output[columns + ['speed_kmh', 'flow_vph']]
