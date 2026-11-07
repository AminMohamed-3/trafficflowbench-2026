"""Apply only frozen S instrument populations to released train caches."""
import numpy as np
import pandas as pd


def apply_frozen_gaps(cache, tv, tq, windows):
    """Blank all stations in horizons, release raw history, preserve template targets."""
    dates={d:i for i,d in enumerate(cache.dates)}
    gap=np.zeros_like(cache.target,dtype=bool)
    history=np.zeros_like(cache.target,dtype=bool)
    own=windows[windows.panel==cache.meta['panel']]
    for row in own.itertuples(index=False):
        origin=pd.Timestamp(row.forecast_origin)
        assert origin.tzinfo is not None
        assert str(row.gap_start)==(origin+pd.Timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%SZ')
        assert str(row.gap_end)==(origin+pd.Timedelta(minutes=90)).strftime('%Y-%m-%dT%H:%M:%SZ')
        for mask,offsets in [(gap,range(5,95,5)),(history,range(-60,0,5))]:
            for offset in offsets:
                stamp=origin+pd.Timedelta(minutes=offset)
                day=stamp.strftime('%Y-%m-%d')
                if day not in dates: raise ValueError('Frozen window extends outside score dates')
                mask[dates[day],stamp.hour*12+stamp.minute//5,:]=True
    if (gap&history).any():
        raise ValueError('Frozen history and horizon masks overlap; instrument needs Lead review')
    original_targets=int(cache.target.sum())
    cache.speed[gap]=np.nan;cache.flow[gap]=np.nan;cache.occ[gap]=np.nan
    cache.speed[history]=tv[history].astype(np.float32)
    cache.flow[history]=tq[history].astype(np.float32)
    cache.occ[history]=cache.occ_true[history]
    cache.target[history]=False
    cache.gap=gap;cache.hist=history
    assert np.isnan(cache.speed[gap]).all() and np.isnan(cache.flow[gap]).all()
    assert not(cache.target&history).any()
    assert np.isnan(cache.speed[cache.target]).all() and np.isnan(cache.flow[cache.target]).all()
    return {'n_windows':len(own),'gap_target_count':int((cache.target&gap).sum()),
            'history_targets_removed':original_targets-int(cache.target.sum()),'gap_cells':int(gap.sum())}
