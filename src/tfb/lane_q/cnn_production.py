"""Q-owned C03/C05 numerical recipes and bounded online CNN inputs.

Functions marked _fit_* are copied from the assigned C recipes; adaptation data
is constructed only through OnlineView.previous_days. Candidate selectors inspect
labels only on these entirely allowed earlier dates. No same-date training.
"""
import copy, math, os, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
PANELS = ['D12_I5_N','D12_I5_S','D7_I10_E','D7_I10_W','D7_I210_E','D7_I210_W','D7_I405_N','D7_I405_S']
CTX, ONSET_W, RAMP_C = 24, 20., 3
EPOCHS, LR, BS, CAP = 3, 3e-4, 32, 1500
GROUPS = {'prior': list(range(127,133)), 'pos':[126], 'tod':[133,134], 'statics':list(range(120,127))}


def apply_drop(x, drop):
    """Zero channel groups that carry train-scenario structure (split-robust variants, C02 finding)."""
    for g in drop:
        x[:, GROUPS[g]] = 0
    return x

class Block(nn.Module):
    def __init__(self, c, dil):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, c), nn.Conv1d(c, c, 3, padding=dil, dilation=dil)
        self.n2, self.c2 = nn.GroupNorm(8, c), nn.Conv1d(c, c, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.gelu(self.n1(x)))
        return x + self.c2(F.gelu(self.n2(h)))

class Net(nn.Module):
    def __init__(self, cin=CTX * 5 + 7 + 6 + 3, c=128, dils=(1, 2, 4, 8, 16, 32, 1, 4, 16)):
        super().__init__()
        self.inp = nn.Conv1d(cin, c, 1)
        self.blocks = nn.Sequential(*[Block(c, d) for d in dils])
        self.out = nn.Conv1d(c, 6, 1)

    def forward(self, x):
        return self.out(F.gelu(self.blocks(self.inp(x))))

def cube(speed, flow, occ, st):
    vf = st.free_speed_kmh.to_numpy(np.float32); cap = st.capacity_vph.to_numpy(np.float32)
    vcut = st.v_cut.to_numpy(np.float32)
    obs = np.isfinite(speed)
    c = np.zeros(speed.shape[:2] + (5, speed.shape[2]), np.float16)
    c[:, :, 0] = np.where(obs, speed / vf, 0)
    c[:, :, 1] = np.where(np.isfinite(flow), flow / cap, 0)
    c[:, :, 2] = np.where(np.isfinite(occ), occ, 0)
    c[:, :, 3] = obs
    c[:, :, 4] = obs & (speed <= vcut)
    return c

def statics(st):
    L = len(st)
    return np.stack([st.v_cut / st.free_speed_kmh, st.lanes / 5, np.log(st.length_km.clip(lower=.01)),
                     st.capacity_vph / 8000, st.has_on_ramp, st.has_off_ramp, np.arange(L) / L]).astype(np.float32)

def make_x(c3, rc, stat, B, drop):
    L = c3.shape[-1]
    ctx = np.stack([c3[d, t - CTX + 1:t + 1].astype(np.float32).reshape(CTX * 5, L) for d, t in B])
    tod = np.array([2 * math.pi * t / 288 for d, t in B])
    glob = np.stack([np.sin(tod), np.cos(tod), np.ones(len(B))], 1).astype(np.float32)
    x = torch.cat([torch.tensor(ctx), torch.tensor(stat)[None].expand(len(B), -1, -1), torch.zeros(len(B), 6, L),
                   torch.tensor(glob)[:, :, None].expand(len(B), 3, L)], 1)
    if rc is not None:
        x = torch.cat([x, torch.tensor(np.stack([rc[d, t - CTX + 1:t + 1].astype(np.float32).reshape(CTX * RAMP_C, L) for d, t in B]))], 1)
    return apply_drop(x, drop)

def weekend(date_str):
    return pd.Timestamp(str(date_str)).dayofweek >= 5

def windows(c3, step=3):
    q, obs = c3[:, :, 4] > .5, c3[:, :, 3] > .5
    out = []
    for d in range(c3.shape[0]):
        for t in range(CTX, 282, step):
            if obs[d, t - 12:t].mean() < .5:
                continue
            if not ((q[d, t - 12:t].sum(0) >= 2).any() and (q[d, t + 1:t + 7] & obs[d, t + 1:t + 7]).any()):
                continue
            out.append((d, t))
    return out

def eiou_mask(pr):
    """Top-m cells with m maximising expected IoU ~ sum_top p / (m + sum_all p - sum_top p) (Q04's rule)."""
    f = pr.ravel(); o = np.argsort(-f); c = np.cumsum(f[o]); m = np.arange(1, len(f) + 1)
    e = c / (m + f.sum() - c); k = int(np.argmax(e)) + 1
    out = np.zeros(len(f), bool); out[o[:k]] = True
    return out.reshape(pr.shape)

def onsets(c3):
    obs, q = c3[..., 3, :] > .5, c3[..., 4, :] > .5; qq = obs & q; ev = []
    for d in range(c3.shape[0]):
        for t in range(12, 282):
            if obs[d, t - 11:t + 1].mean() < .5 or qq[d, t - 11:t + 1].any() or qq[d, t + 1:t + 6].any() or not qq[d, t + 6].any():
                continue
            ev.append((d, t, qq[d, t + 6].copy(), obs[d, t + 1:t + 7].copy()))
    return ev

def _fit_ongoing(base, data, before, drop, dev, seed, daytype=True, cap=None):
    """Fine-tune on pseudo windows of dates < before (same day type as `before` when daytype).
    cap: optional max windows per epoch (random subset; C03 official predictions used no cap)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    kind = weekend(before)
    pool = {p: [(d, t) for d, t in v["W"] if v["dates"][d] < before and (not daytype or weekend(v["dates"][d]) == kind)]
            for p, v in data.items()}
    if sum(len(w) for w in pool.values()) == 0:
        return base
    model = copy.deepcopy(base); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    for ep in range(EPOCHS):
        sub = pool
        if cap:
            n = sum(len(w) for w in pool.values()); keep = rng.random(n) < min(1., cap / n); it = iter(keep)
            sub = {p: [w for w in pool[p] if next(it)] for p in PANELS}
        order = [(p, i) for p in PANELS for i in range(0, len(sub[p]), BS)]
        rng.shuffle(order)
        for p, i in order:
            v = data[p]; W = sub[p]; B = [W[j] for j in rng.permutation(len(W))[:BS]]
            x = make_x(v["c3"], v["rc"], v["stat"], B, drop).to(dev)
            obs = np.stack([v["c3"][d, t + 1:t + 7, 3] > .5 for d, t in B]); q = np.stack([v["c3"][d, t + 1:t + 7, 4] > .5 for d, t in B])
            y = torch.tensor(q & obs, dtype=torch.float32, device=dev); m = torch.tensor(obs, dtype=torch.float32, device=dev)
            loss = (F.binary_cross_entropy_with_logits(model(x), y, reduction="none") * m).sum() / (m.sum() + 1)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model

def _fit_onset(base, pool, data, drop, seed):
    if not pool:
        return base
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = copy.deepcopy(base); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    for ep in range(EPOCHS):
        sel = [pool[i] for i in rng.permutation(len(pool))[:CAP]]
        for i in range(0, len(sel), BS):
            B = sel[i:i + BS]
            by = {}
            for j, (p, d, t, kind) in enumerate(B):
                by.setdefault(p, []).append((j, d, t, kind))
            loss_sum, den = 0., 0.
            for p, items in by.items():
                v = data[p]
                x = make_x(v["c3"], None, v["stat"], [(d, t) for _, d, t, _ in items], drop)
                x[:, 135] = torch.tensor([0. if kind == "onset" else 1. for _, _, _, kind in items])[:, None]
                obs = np.stack([v["obs"][d, t + 1:t + 7] for _, d, t, _ in items]); q = np.stack([v["qq"][d, t + 1:t + 7] for _, d, t, _ in items])
                m = obs.astype(np.float32); w = np.ones(len(items), np.float32)
                for k, (_, d, t, kind) in enumerate(items):
                    if kind == "onset":
                        m[k, :5] = 0; w[k] = ONSET_W
                y = torch.tensor(q, dtype=torch.float32); mm = torch.tensor(m)
                l = (F.binary_cross_entropy_with_logits(model(x), y, reduction="none") * mm).sum((1, 2))
                loss_sum = loss_sum + (l * torch.tensor(w)).sum(); den += float(mm.sum())
            loss = loss_sum / (den + 1)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model

from .windows import OnlineView, timestamp

DROP = ('prior', 'pos', 'tod')


def _ordered_stations(stations):
    stations = stations.sort_values('idx').reset_index(drop=True)
    assert np.array_equal(stations.idx.to_numpy(), np.arange(len(stations)))
    assert not stations.link_id.duplicated().any()
    return stations


def _frame_arrays(frame, stations, date, end=None):
    """Build one day's grid from an already permitted accessor frame."""
    stations = _ordered_stations(stations)
    links = stations.link_id.astype(str).tolist()
    assert frame.empty or set(frame.link_id.astype(str)).issubset(set(links))
    times = pd.to_datetime(frame.timestamp, utc=True)
    day = timestamp(date)
    assert times.empty or times.dt.normalize().eq(day).all()
    if end is not None:
        assert times.empty or times.max() <= timestamp(end)
    assert not frame.duplicated(['timestamp', 'link_id']).any()
    steps = ((times - day).dt.total_seconds() / 300).to_numpy()
    assert np.equal(steps, np.floor(steps)).all() and ((steps >= 0) & (steps < 288)).all()
    positions = frame.link_id.astype(str).map({link: i for i, link in enumerate(links)}).to_numpy(int)
    arrays = []
    for name in ['speed_kmh', 'flow_vph', 'occupancy']:
        values = np.full((1, 288, len(links)), np.nan, np.float32)
        values[0, steps.astype(int), positions] = frame[name].to_numpy(np.float32)
        arrays.append(values)
    return arrays


def input_from_view(view, stations, condition='queue_ongoing'):
    """C03/C05 input, constructed entirely from a bounded OnlineView."""
    assert isinstance(view, OnlineView)
    assert condition in ('queue_ongoing', 'queue_onset')
    stations = _ordered_stations(stations)
    assert set(stations.link_id.astype(str)) == set(view.network.links)
    origin = view.origin
    t = origin.hour * 12 + origin.minute // 5
    links = stations.link_id.astype(str).tolist()
    masked = view.mainline
    assert masked.empty or pd.to_datetime(masked.timestamp, utc=True).max() <= origin
    masked_cube = cube(*_frame_arrays(masked, stations, view.date, end=origin), stations)[0]
    lo, length = t - CTX + 1, len(links)
    context = np.zeros((CTX, 5, length), np.float32)
    for step in range(max(lo, 0), t + 1):
        if not t - 12 <= step <= t - 1:
            context[step - lo] = masked_cube[step]
    history = view.history
    times = pd.to_datetime(history.timestamp, utc=True)
    assert not times.empty
    assert times.min() >= origin - pd.Timedelta(minutes=60)
    assert times.max() <= origin - pd.Timedelta(minutes=5)
    history_cube = cube(*_frame_arrays(history, stations, view.date, end=origin), stations)[0]
    assert t >= 12
    context[11:23] = history_cube[t - 12:t]
    tod = 2 * np.pi * t / 288
    global_channels = torch.tensor([np.sin(tod), np.cos(tod), float(condition == 'queue_ongoing')],
                                   dtype=torch.float32)[None, :, None].expand(1, 3, length)
    x = torch.cat([torch.tensor(context.reshape(1, CTX * 5, length)),
                   torch.tensor(statics(stations))[None], torch.zeros(1, 6, length), global_channels], 1)
    return apply_drop(x, DROP), links


def adaptation_data(release, split, before, stations_by_panel):
    """All numerical adaptation data comes from earlier dates of this split."""
    before = str(timestamp(before).date())
    assert split in ('validation', 'private')
    data = {}
    for panel in PANELS:
        stations = _ordered_stations(stations_by_panel[panel])
        view = OnlineView(release, panel, split, before, timestamp(before))
        dates, cubes = [], []
        for date, frame in view.previous_days():
            assert date < before
            assert frame.empty or pd.to_datetime(frame.timestamp, utc=True).max() < timestamp(before)
            dates.append(date)
            cubes.append(cube(*_frame_arrays(frame, stations, date), stations)[0])
        c3 = np.stack(cubes) if cubes else np.empty((0, 288, 5, len(stations)), np.float16)
        obs, queue = c3[:, :, 3] > .5, c3[:, :, 4] > .5
        data[panel] = {'c3': c3, 'rc': None, 'stat': statics(stations), 'dates': np.asarray(dates),
                       'W': windows(c3, step=1), 'ev': onsets(c3), 'obs': obs, 'qq': obs & queue}
    return data


def adapt_for_date(base, data, before, date_seed, condition='queue_ongoing'):
    """Original C03 uncapped or C05 capped recipe; never fits on the origin date."""
    if os.environ.get('TFB_VM') != '1':
        raise RuntimeError('CNN adaptation fits require the registered VM launcher')
    before = str(timestamp(before).date())
    assert list(data) == PANELS
    assert condition in ('queue_ongoing', 'queue_onset')
    for v in data.values():
        assert all(str(date) < before for date in v['dates'])
        assert v['c3'].shape[0] == len(v['dates'])
    if condition == 'queue_ongoing':
        return _fit_ongoing(base, data, before, DROP, 'cpu', date_seed, daytype=True, cap=None)
    kind = weekend(before)
    pool = [(p, d, t, 'on') for p, v in data.items() for d, t in v['W']
            if weekend(v['dates'][d]) == kind]
    pool += [(p, d, t, 'onset') for p, v in data.items() for d, t, _, _ in v['ev']
             if weekend(v['dates'][d]) == kind]
    assert all(t >= CTX - 1 for _, _, t, _ in pool), 'Original onset recipe needs a full24-step context'
    return _fit_onset(base, pool, data, DROP, date_seed)


def predict_online(models, view, stations, condition='queue_ongoing'):
    """Mean seed probabilities before C's whole-window expected-IoU decoder."""
    if os.environ.get('TFB_VM') != '1':
        raise RuntimeError('New CNN serving inference requires the registered VM launcher')
    assert len(models) == 2
    x, links = input_from_view(view, stations, condition)
    for model in models:
        model.eval()
    with torch.no_grad():
        probabilities = torch.stack([torch.sigmoid(model(x)) for model in models]).mean(0)[0].numpy()
    if condition == 'queue_onset':
        binary = np.zeros_like(probabilities, bool)
        binary[5] = eiou_mask(probabilities[5])
    else:
        binary = eiou_mask(probabilities)
    return probabilities, binary, links


class CNNProduction:
    """Object accepted by Q's existing cfg.ongoing_model/onset_model interface."""
    def __init__(self, bases, stations_by_split, release, prior, date_seeds,
                 condition='queue_ongoing'):
        assert len(bases) == 2
        assert condition in ('queue_ongoing', 'queue_onset')
        self.bases, self.stations = bases, stations_by_split
        self.release, self.prior, self.date_seeds = release, prior, date_seeds
        self.condition, self._models = condition, {}

    def forecast(self, view):
        assert isinstance(view, OnlineView)
        assert view.split in ('validation', 'private')
        assert view._release.root == self.release.root
        key = view.split, view.date, self.condition
        stations = self.stations[view.split]
        if key not in self._models:
            data = adaptation_data(self.release, view.split, view.date, stations)
            seed = self.date_seeds[view.split][view.date]
            self._models[key] = [adapt_for_date(base, data, view.date, seed, self.condition)
                                 for base in self.bases]
        _, binary, links = predict_online(self._models[key], view, stations[view.panel], self.condition)
        positions = {link: i for i, link in enumerate(links)}
        return binary[:, [positions[link] for link in view.network.links]].astype(np.int8)

    @classmethod
    def load(cls, directory, *, root, release, num_threads=2):
        """Load VM-regenerated base checkpoints and public station metadata."""
        if os.environ.get('TFB_VM') != '1':
            raise RuntimeError('CNN production loading/serving requires its registered VM job')
        assert num_threads == 2, 'Preserve the fixed C03/C05 CPU recipe'
        torch.set_num_threads(2)
        metadata = json.loads((Path(directory) / 'metadata.json').read_text())
        assert metadata['feature_builder'] == 'C03_C05_online_cnn_v1'
        assert metadata['base_seeds'] == [0, 1]
        assert metadata['drop'] == list(DROP)
        assert metadata['condition'] in ('queue_ongoing', 'queue_onset')
        bases = []
        assert len(metadata['base_models']) == 2
        for path in metadata['base_models']:
            with (root / path).open('rb') as model_file:
                assert hashlib.file_digest(model_file, 'sha256').hexdigest() == metadata['base_model_sha256'][path]
            base = Net(cin=CTX * 5 + 16)
            base.load_state_dict(torch.load(root / path, map_location='cpu', weights_only=True))
            base.eval()
            bases.append(base)
        assert len(bases) == 2
        stations = {split: {panel: _ordered_stations(pd.read_parquet(root / path))
                           for panel, path in paths.items()}
                    for split, paths in metadata['stations'].items()}
        with (root / metadata['canonical_q03_prior']).open('rb') as prior_file:
            assert hashlib.file_digest(prior_file, 'sha256').hexdigest() == metadata['canonical_q03_prior_sha256']
        prior = json.loads((root / metadata['canonical_q03_prior']).read_text())
        return cls(bases, stations, release, prior, metadata['official_date_seeds'], metadata['condition'])
