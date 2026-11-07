"""Independent raw-release reader for S-00; model tensors are float32."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa

pa.set_cpu_count(8)
pa.set_io_thread_count(2)

REGIMES = ('R1', 'R2', 'R3')


def main_root():
    root = Path(__file__).resolve().parents[3]
    marker = root / '.git'
    if marker.is_file():
        gitdir = Path(marker.read_text().partition(':')[2].strip())
        if not gitdir.is_absolute():
            gitdir = (root / gitdir).resolve()
        return gitdir.parent.parent.parent
    return root


def release_root():
    return main_root() / 'data' / 'raw' / 'kaggle_public'


def network(panel, release=None):
    base = (release or release_root()) / 'corridors' / panel / 'network'
    topo = pd.read_csv(base / 'lwr_mainline_topology.csv', dtype=str)
    topo['order_index'] = topo.order_index.astype(int)
    topo = topo.sort_values('order_index')
    links = pd.read_csv(base / 'links.csv', dtype={'link_id': str}).drop_duplicates('link_id')
    fd = pd.read_csv(base / 'fd_parameters.csv', dtype={'link_id': str})
    params = links.set_index('link_id').copy()
    for c in ('lanes', 'free_speed_kmh', 'capacity_vph', 'k_jam'):
        if c in fd:
            values = fd.groupby('link_id')[c].mean()
            params[c] = values.reindex(params.index).combine_first(params.get(c, pd.Series(index=params.index, dtype=float)))
    params['lanes'] = params.lanes.clip(lower=1)
    params['critical_density'] = params.capacity_vph / params.free_speed_kmh.clip(lower=1)
    params['k_jam'] = np.maximum(params.k_jam.fillna(100 * params.lanes), 1.05 * params.critical_density)
    return topo, params, fd


def steps(timestamps):
    text = timestamps.astype(str).str.slice(11, 16)
    return (text.str.slice(0, 2).astype(int) * 12 + text.str.slice(3, 5).astype(int) // 5).to_numpy()


def selected_paths(panel, months=('2030-06', '2030-07'), release=None):
    base = (release or release_root()) / 'corridors' / panel / 'train'
    truth = sorted(p for m in months for p in (base / 'mainline_states' / ('year_month=' + m)).glob('*.parquet'))
    masked = {p.name: p for p in (base / 'mainline_states_masked').glob('*/*.parquet')}
    return truth, masked


def read_template(panel, dates, release=None):
    path = (release or release_root()) / 'task1' / panel / 'train' / 'sample_submission_state.csv'
    columns = ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']
    pieces = []
    for part in pd.read_csv(path, usecols=columns, dtype=str, chunksize=200_000):
        part = part[part.timestamp.str.slice(0, 10).isin(dates)]
        if len(part):
            pieces.append(part)
    frame = pd.concat(pieces, ignore_index=True)
    assert not frame.duplicated(columns).any(), 'Duplicate template keys'
    return frame


def load_panel(panel, months=('2030-06', '2030-07'), release=None, exact_truth=False):
    raw = Path(release) if release is not None else release_root()
    topo, params, fd = network(panel, raw)
    order = topo.link_id.tolist()
    truth_paths, masked_paths = selected_paths(panel, months, raw)
    assert truth_paths, 'No released train days'
    li = {link: i for i, link in enumerate(order)}
    shape = (len(truth_paths), 288, len(order))
    a = {key: np.full(shape, np.nan, np.float64 if exact_truth else np.float32) for key in ('V', 'Q', 'VM', 'QM')}
    a['E'] = np.zeros(shape, bool)
    a['on_valid'] = np.zeros(shape, bool)
    a['off_valid'] = np.zeros(shape, bool)
    a['reg'], a['dates'] = [], []
    amap = pd.read_csv(raw / 'corridors' / panel / 'network' / 'ramp_attachment_map.csv', dtype=str)
    ramp_links = dict(zip(amap.ramp_link_id, amap.nearest_mainline_link_id))
    on = set(topo.loc[topo.on_ramp_link_ids.fillna('').str.len() > 0, 'link_id'])
    off = set(topo.loc[topo.off_ramp_link_ids.fillna('').str.len() > 0, 'link_id'])
    base = raw / 'corridors' / panel / 'train'
    ramp_paths = {p.name: p for p in (base / 'ramp_states').glob('*/*.parquet')}
    for d, path in enumerate(truth_paths):
        mpath = masked_paths[path.name]
        cols = ['date', 'timestamp', 'station_id', 'link_id', 'speed_kmh', 'flow_vph', 'is_score_eligible']
        truth = pd.read_parquet(path, columns=cols)
        masked = pd.read_parquet(mpath, columns=cols + ['mask_regime'])
        assert truth.station_id.nunique() == truth.link_id.nunique(), 'S-00 expects one station per link'
        for frame, vk, qk in ((truth, 'V', 'Q'), (masked, 'VM', 'QM')):
            ti, si = steps(frame.timestamp), frame.link_id.map(li).to_numpy()
            assert np.isfinite(si.astype(float)).all(), 'Unknown network link'
            a[vk][d, ti, si] = frame.speed_kmh.to_numpy(a[vk].dtype)
            a[qk][d, ti, si] = frame.flow_vph.to_numpy(a[qk].dtype)
        a['E'][d, steps(masked.timestamp), masked.link_id.map(li).to_numpy()] = masked.is_score_eligible.to_numpy() == 1
        assert masked.mask_regime.nunique() == 1
        a['reg'].append(str(masked.mask_regime.iloc[0]))
        a['dates'].append(str(masked.date.iloc[0]))
        ramp = pd.read_parquet(ramp_paths[path.name.replace('mainline', 'ramp')])
        ramp = ramp[(ramp.is_score_eligible == 1) & (ramp.pct_observed >= 75) & ramp.flow_vph.notna()].copy()
        ramp['link_id'] = ramp.ramp_link_id.map(ramp_links)
        ramp = ramp[ramp.link_id.isin(li)]
        for typ, dest in ((('OR', 'ON'), 'on_valid'), (('FR', 'OFF'), 'off_valid')):
            rows = ramp[ramp.ramp_type.str.upper().isin(typ)]
            a[dest][d, steps(rows.timestamp), rows.link_id.map(li).to_numpy()] = True
    a['reg'] = np.asarray(a['reg'])
    a['dates'] = np.asarray(a['dates'])
    a['on_valid'] |= np.asarray([link not in on for link in order])[None, None, :]
    a['off_valid'] |= np.asarray([link not in off for link in order])[None, None, :]
    a['target'] = a['E'] & np.isnan(a['VM']) & np.isnan(a['QM']) & np.isfinite(a['V']) & np.isfinite(a['Q'])
    a['template_target'] = np.zeros(shape, bool)
    template = read_template(panel, set(a['dates']), raw)
    di = {date: i for i, date in enumerate(a['dates'])}
    ds = template.timestamp.str.slice(0, 10).map(di).to_numpy()
    ts, ls = steps(template.timestamp), template.link_id.map(li).to_numpy()
    assert np.isfinite(ls.astype(float)).all()
    assert (template.mask_regime.to_numpy() == a['reg'][ds]).all(), 'Template regime mismatch'
    a['template_target'][ds, ts, ls] = True
    assert (a['template_target'] & ~a['target']).sum() == 0, 'Template includes a nonmasked or ineligible cell'
    a['template_count'] = len(template)
    a['order'] = np.asarray(order)
    a['params'] = params.reindex(order)
    length = a['params'].length_km.to_numpy(np.float32)
    a['position'] = np.cumsum(length, dtype=np.float64) - length / 2
    return a


def interp_time(values):
    out = values.copy()
    ts = np.arange(values.shape[1])
    for d in range(len(values)):
        for s in range(values.shape[2]):
            y = values[d, :, s]
            valid = np.isfinite(y)
            if valid.any():
                out[d, :, s] = np.interp(ts, ts[valid], y[valid])
    return out


def interp_space(values, positions):
    out = values.copy()
    for d in range(len(values)):
        for t in range(values.shape[1]):
            y = values[d, t]
            valid = np.isfinite(y)
            if valid.any():
                out[d, t] = np.interp(positions, positions[valid], y[valid])
    return out


