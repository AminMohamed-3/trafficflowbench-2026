"""S07 observation-only targets and fixed stationary-prior stress.

No helper in this module accesses *_true arrays. Original template targets and
supplied history remain unavailable as adaptation labels.
"""
from copy import deepcopy
from types import SimpleNamespace
import hashlib
import numpy as np
import pandas as pd

OBSERVATION_ARRAYS = ('speed', 'flow', 'occ', 'pct', 'elig', 'target', 'gap', 'hist',
                      'ramp_flow')
WINDOW_COLUMNS = ('panel', 'date', 'forecast_origin', 'condition', 'gap_start', 'gap_end')


def observation_view(cache):
    arrays = {name: np.asarray(getattr(cache, name)).copy()
              for name in OBSERVATION_ARRAYS}
    view = SimpleNamespace(meta=deepcopy(cache.meta), stations=cache.stations.copy(),
                           ramps=cache.ramps.copy(), dates=list(cache.dates),
                           regimes=np.asarray(cache.regimes).copy(), **arrays)
    view.s_windows = getattr(cache, 's_windows', pd.DataFrame(columns=WINDOW_COLUMNS)).copy()
    assert not any(name.endswith('_true') for name in vars(view))
    return view


def auxiliary_windows(panel, dates, published, original):
    """Select using semantic origins, never IDs, file order, or traffic truth."""
    if published.empty:
        return pd.DataFrame(columns=WINDOW_COLUMNS)
    candidates = published.copy()
    candidates['panel'] = panel
    candidates['date'] = candidates.forecast_origin.astype(str).str[:10]
    candidates = candidates[candidates.date.isin(dates)]
    candidates = candidates[['panel', 'date', 'forecast_origin', 'condition']].drop_duplicates()
    # Duplicate conditions for one origin would be ambiguous released metadata.
    assert not candidates.forecast_origin.duplicated().any()
    occupied = list(pd.to_datetime(
        original.loc[original.panel.eq(panel) & original.date.isin(dates),
                     'forecast_origin'], utc=True))
    chosen = []
    for row in candidates.sort_values(['forecast_origin', 'condition']).itertuples(index=False):
        origin = pd.Timestamp(row.forecast_origin)
        assert origin.tzinfo is not None and origin.minute % 5 == 0
        step = origin.hour * 12 + origin.minute // 5
        if not 12 <= step <= 268:
            continue
        assert row.condition in ('queue_onset', 'queue_ongoing')
        if any(abs(origin - other) < pd.Timedelta(minutes=155) for other in occupied):
            continue
        occupied.append(origin)
        chosen.append(dict(panel=panel, date=row.date,
                           forecast_origin=origin.strftime('%Y-%m-%dT%H:%M:%SZ'),
                           condition=row.condition,
                           gap_start=(origin + pd.Timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                           gap_end=(origin + pd.Timedelta(minutes=90)).strftime('%Y-%m-%dT%H:%M:%SZ')))
    return pd.DataFrame(chosen, columns=WINDOW_COLUMNS)


def make_pseudo_observations(cache, auxiliary, rate=.10, seed=8853):
    """Snapshot known labels, then omit selected cells and full auxiliary horizons."""
    assert 0 < rate < 1
    view = observation_view(cache)
    original_target = view.target.copy()
    original_history = view.hist.copy()
    eligible_known = (view.elig & np.isfinite(view.speed) & np.isfinite(view.flow)
                      & ~original_target & ~original_history & ~view.gap)
    extra_gap = np.zeros_like(view.target)
    extra_history = np.zeros_like(view.target)
    dates = {day: i for i, day in enumerate(view.dates)}
    for row in auxiliary.itertuples(index=False):
        assert row.panel == view.meta['panel'] and row.date in dates
        origin = pd.Timestamp(row.forecast_origin)
        step = origin.hour * 12 + origin.minute // 5
        assert 12 <= step <= 268
        d = dates[row.date]
        extra_gap[d, step+1:step+19] = True
        extra_history[d, step-12:step] = True
    assert not (extra_gap & (view.gap | view.hist | extra_history)).any()
    assert not (extra_history & view.gap).any()
    selected = np.zeros_like(view.target)
    for d, day in enumerate(view.dates):
        token = f'S07 observed masks{seed}|{view.meta["panel"]}|{day}'
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'little'))
        selected[d] = eligible_known[d] & (rng.random(view.target[d].shape) < rate)
    selected |= eligible_known & extra_gap
    selected &= ~extra_history
    # Labels are copied from previously released measurements, not the truth cache.
    label_v = view.speed[selected].copy()
    station_indices = np.nonzero(selected)[2]
    label_qpl = view.flow[selected] / view.stations.lanes.to_numpy(np.float32)[station_indices]
    assert np.isfinite(label_v).all() and np.isfinite(label_qpl).all()
    omitted = selected | extra_gap
    for name in ('speed', 'flow', 'occ'):
        getattr(view, name)[omitted] = np.nan
    view.gap |= extra_gap
    view.hist |= extra_history
    view.target = selected
    original = view.s_windows.reindex(columns=WINDOW_COLUMNS)
    view.s_windows = pd.concat([original, auxiliary], ignore_index=True)
    assert not (view.target & (original_target | original_history | extra_history)).any()
    assert np.isnan(view.speed[view.target]).all() and np.isnan(view.flow[view.target]).all()
    report = dict(n_pseudo_targets=int(selected.sum()), n_original_targets=int(original_target.sum()),
                  n_auxiliary_windows=len(auxiliary), n_auxiliary_gap_cells=int(extra_gap.sum()),
                  n_auxiliary_gap_labels=int((selected & extra_gap).sum()),
                  original_target_label_intersection=0, supplied_history_label_intersection=0,
                  hidden_truth_access=False)
    return view, label_v, label_qpl.astype(np.float32), report


def stressed_summaries(summaries, stations, panel, dose, seed=8842):
    """Fit-only sigma and station shifts; no held measurements enter the stress."""
    assert dose in (0., .5, 1.)
    result = {key: np.asarray(value).copy() for key, value in summaries.items()}
    vf = stations.free_speed_kmh.to_numpy(float)
    finite = np.isfinite(result['mean'])
    assert finite.any()
    sigma = float(np.clip(np.sqrt(np.mean((result['mean'][finite] - vf[finite]) ** 2)), .5, 3.))
    token = f'S07 prior shift{seed}|{panel}'
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], 'little'))
    shift = rng.normal(0., sigma, len(vf)) * dose
    old_mean = result['mean'].copy()
    result['mean'] = np.clip(result['mean'] + shift, 3., 120.75)
    old_history = result['history_speed'].copy()
    free = old_history > .85 * vf
    result['history_speed'] = np.where(
        free, np.clip(old_history + shift[None, None, :], 3., 120.75), old_history)
    for key in summaries:
        if key not in ('mean', 'history_speed'):
            assert np.array_equal(result[key], summaries[key], equal_nan=True)
    report = dict(dose=dose, sigma_fit_only_kmh=sigma,
                  requested_shift_rms_kmh=float(np.sqrt(np.mean(shift**2))),
                  actual_mean_shift_rms_kmh=float(np.sqrt(np.nanmean((result['mean']-old_mean)**2))),
                  free_history_cells_shifted=int((free & np.isfinite(old_history)).sum()),
                  truth_or_measurement_perturbation=False)
    return result, report
