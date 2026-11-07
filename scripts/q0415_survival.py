"""Q0415: interval-censored first-event hazard; train-null calibrated on S1+F.

Feature code receives only an OnlineHistory. Outcome/selector code is separate.
Earlier-day first events have intervals (last observed clear, first queued].
Gaps are missing exposure, never negative queue labels. The fixed statistic is
the expected first-event mass within +/-30 min of T+30 divided by the probability
of still being at risk 30 min earlier. No shift of Q03's train time prior.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import numpy as np
import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('TFB_ROOT', SOURCE_ROOT))
sys.path.insert(0, str(SOURCE_ROOT / 'src'))
sys.path.insert(0, str(SOURCE_ROOT / 'docs/reference_code'))
OUT = ROOT / 'artifacts/lane_q/0415_survival'
START = 48
BAND = 6
WIDTH = 12.


@dataclass(frozen=True)
class OnlineHistory:
    dates: tuple[str, ...]
    queue: tuple[np.ndarray, ...]
    observed: tuple[np.ndarray, ...]
    origin: int


def online_history(dates, qs, obs, start, day, origin, gaps):
    """Single accessor: slices before copying; never reads >=T or later dates."""
    assert 0 <= start <= day < len(dates) and 0 <= origin < 288
    assert dates[start][:7] == dates[day][:7]
    queue, observed = [], []
    for dd in range(start, day + 1):
        end = origin if dd == day else 288  # strict no-T fallback by construction
        q = np.array(qs[dd, :end], copy=True)
        o = np.array(obs[dd, :end], copy=True)
        for tt in gaps.get(dd, ()):
            lo, hi = max(0, tt + 1), min(end, tt + 19)
            if lo < hi:
                q[lo:hi] = 0
                o[lo:hi] = 0
        q[o <= 0] = 0
        q.setflags(write=False)
        o.setflags(write=False)
        queue.append(q)
        observed.append(o)
    return OnlineHistory(tuple(dates[start:day + 1]), tuple(queue), tuple(observed), origin)


def interval(q, observed):
    """First sustained visible queue and its last observed clear predecessor.

    Returns None for no detected first event. Missing sites have no exposure.
    A second visible queued observation within two steps rejects isolated noise.
    """
    yes = (q > 0) & (observed > 0)
    nxt = np.zeros_like(yes)
    nxt[:-1] |= yes[1:]
    nxt[:-2] |= yes[2:]
    ix = np.flatnonzero(yes & nxt & (np.arange(len(q)) >= START))
    if not len(ix):
        return None
    upper = int(ix[0])
    clear = np.flatnonzero((observed[:upper] > 0) & ~yes[:upper] & (np.arange(upper) >= START))
    lower = int(clear[-1] + 1) if len(clear) else START
    return lower, upper


def survival_stat(view: OnlineHistory):
    """Previous days only estimate event distribution; origin conditions hazard.

    Current-day survival is guaranteed by the onset task selection; its values
    are deliberately not used as extra negative labels. Fractional interval
    events and at-risk exposure are smoothed by one fixed Gaussian hour kernel.
    """
    nk = view.queue[-1].shape[1]
    target = view.origin + 6
    grid = np.arange(START, 288)
    events, risk, n = np.zeros(nk), np.zeros(nk), np.zeros(nk)
    for q, o in zip(view.queue[:-1], view.observed[:-1]):
        for k in range(nk):
            visible = o[START:, k] > 0
            if visible.mean() < .25:
                continue
            n[k] += 1
            iv = interval(q[:, k], o[:, k])
            if iv is None:
                # No observed event supplies right-censored exposure only up to
                # last visible time. Entirely missing local bands contribute 0.
                band = (grid >= target - BAND) & (grid <= target + BAND)
                risk[k] += float(visible[band].mean()) if band.any() else 0.
                continue
            lo, hi = iv
            latent = np.arange(lo, hi + 1)
            density = np.exp(-.5 * ((grid[:, None] - latent) / WIDTH) ** 2).mean(axis=1)
            density /= density.sum()
            events[k] += density[(grid >= target - BAND) & (grid <= target + BAND)].sum()
            risk[k] += density[grid >= target - BAND].sum()
    hazard = (events + .5) / (risk + 1.)
    return dict(lr=np.log(hazard), events=events, risk=risk, days=n)


def poison_fixture():
    rng = np.random.default_rng(415)
    dates = ['2031-03-01', '2031-03-02', '2031-03-03', '2031-03-04']
    q = np.zeros((4, 288, 3)); o = np.ones_like(q)
    q[0, 103:130, 0] = 1; q[1, 111:140, 0] = 1
    q[0, 160:180, 1] = 1; q[1, 164:190, 1] = 1
    o[:, :, 2] = 0
    gaps = {0: [96], 1: [104]}
    ref = survival_stat(online_history(dates, q, o, 0, 2, 100, gaps))
    for fill in ('nan', 'garbage'):
        qq, oo = q.copy(), o.copy()
        qq[2, 100:] = np.nan if fill == 'nan' else rng.normal(100, 5, qq[2, 100:].shape)
        oo[2, 100:] = np.nan if fill == 'nan' else 777
        qq[3:] = np.nan if fill == 'nan' else 888
        oo[3:] = np.nan if fill == 'nan' else 999
        got = survival_stat(online_history(dates, qq, oo, 0, 2, 100, gaps))
        for k in ref:
            np.testing.assert_array_equal(ref[k], got[k])
    # Hidden queue cells have no effect, and the gap widens the start interval.
    qq = q.copy(); qq[0, 97:115] = rng.normal(200, 10, qq[0, 97:115].shape)
    view = online_history(dates, qq, o, 0, 2, 100, gaps)
    got = survival_stat(view)
    for k in ref:
        np.testing.assert_array_equal(ref[k], got[k])
    assert interval(view.queue[0][:, 0], view.observed[0][:, 0]) == (97, 115)
    assert ref['days'][2] == 0 and ref['risk'][2] == 0
    empty = survival_stat(online_history(dates, q, o, 2, 2, 100, {}))
    np.testing.assert_array_equal(empty['lr'], np.full(3, np.log(.5)))
    return dict(status='PASS', checks=['future values', 'future dates', 'hidden gap values',
                                     'interval censoring', 'missing exposure', 'zero-history neutrality'])


def site_arrays(panel, split, sites):
    """Compact released-layer aggregates; all feature access is through accessor."""
    from tfb.data import load_cache
    c = load_cache(panel, split)
    vc = c.stations.sort_values('idx').v_cut.to_numpy(np.float32)
    speed, elig = c['speed'], c['elig']
    qs = np.empty((len(c.dates), 288, len(sites)), dtype=np.float32)
    obs = np.empty_like(qs)
    for d in range(len(c.dates)):
        for k, site in enumerate(sites):
            pos = list(site['positions'])
            sp = np.asarray(speed[d])[:, pos]
            ok = np.isfinite(sp) & np.asarray(elig[d])[:, pos]
            obs[d, :, k] = ok.sum(1)
            qs[d, :, k] = (ok & (sp <= vc[pos])).sum(1)
    return tuple(c.dates), qs, obs


def build(pop, panels):
    """Replay identical existing official-gap instrument, features kept separate."""
    import onset5_select as SEL
    import onset5_stat2 as S2
    meta = json.loads((ROOT / 'artifacts/lane_q/0267/Q03_seed0_model/metadata.json').read_text())
    split = {'train': 'train', 'val': 'validation', 'pri': 'private'}[pop]
    if pop == 'train':
        sel = pd.read_parquet(ROOT / 'artifacts/claude_ref/onset/selected_windows.parquet')
        windows = sel[(sel.split == 'train') & (sel.layer == 'raw')][['panel', 'date', 't']]
        tol = pd.concat([pd.read_parquet(f)[['panel', 'date', 't']] for f in
                         sorted((ROOT / 'artifacts/claude_ref/onset2').glob('tol_windows_*.parquet'))])
        windows = pd.concat([windows, tol]).drop_duplicates()
    else:
        windows = pd.read_csv(ROOT / f'artifacts/claude_ref/onset/score_{split}_A2.csv')
    all_panels = SEL.PANELS
    for panel in panels or all_panels:
        sites = meta['sites'][panel]
        dates, qs, obs = site_arrays(panel, split, sites)
        _, candidates = SEL.split_candidates(panel, split, 'raw' if pop == 'train' else 'masked')
        idx = {date: d for d, date in enumerate(dates)}
        rows, poison_n = [], 0
        for row in windows[windows.panel == panel].itertuples():
            d, t = idx[row.date], int(row.t)
            if t not in candidates[d][0]:
                candidates[d] = (sorted(set(candidates[d][0]) | {t}), candidates[d][1])
            for j in range(9):
                s = d - j
                if s < 0 or dates[s][:7] != dates[d][:7]:
                    break
                rank, gaps = S2.emu_gaps(candidates, dates, s, d, t,
                                       seed=all_panels.index(panel) * 100000 + s)
                if rank is None:
                    continue
                view = online_history(dates, qs, obs, s, d, t, gaps)
                stat = survival_stat(view)
                if poison_n < 3 and j > 0:
                    qq, oo = qs.copy(), obs.copy()
                    qq[d, t:] = 9e8; oo[d, t:] = 0; qq[d + 1:] = np.nan; oo[d + 1:] = np.nan
                    poisoned = survival_stat(online_history(dates, qq, oo, s, d, t, gaps))
                    for key in stat:
                        np.testing.assert_array_equal(stat[key], poisoned[key])
                    poison_n += 1
                for k, site in enumerate(sites):
                    rows.append(dict(wkey=f'{panel}|{row.date}|{t}', panel=panel, date=row.date,
                                     t=t, j=j, kdays=j, site=site['key'], mon=row.date[:7],
                                     **{key: float(values[k]) for key, values in stat.items()}))
        pd.DataFrame(rows).to_parquet(OUT / f'stat_{pop}_{panel}.parquet')
        (OUT / f'poison_{pop}_{panel}.json').write_text(json.dumps(
            dict(status='PASS', origins_checked=poison_n, population=pop, panel=panel,
                 source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))
        print(json.dumps(dict(panel=panel, population=pop, rows=len(rows), poison_checks=poison_n)), flush=True)


def evaluate(pop, phase):
    import onset6_common as C
    import onset6_evalF as F
    import onset5_eval2 as V
    import onset4_eval as E
    expected = set(C.SITES)
    for source in ('train',) if pop in ('tol', 'tolm') else ('train', pop):
        for panel in expected:
            check = json.loads((OUT / f'poison_{source}_{panel}.json').read_text())
            assert check['status'] == 'PASS' and check['origins_checked'] >= 3
    if pop == 'val' and phase == 'confirm':
        dev = json.loads((OUT / 'result_val_dev.json').read_text())
        assert dev['dev_odd']['delta_ci'][0] > 0, 'March development failed; confirmation remains unread'
    if pop in ('tol', 'tolm', 'pri'):
        val = json.loads((OUT / 'result_val_confirm.json').read_text())
        assert val['confirm_even']['delta_ci'][1] > 0, 'March confirmation did not pass'
    if pop == 'pri':
        tol = json.loads((OUT / 'result_tol_all.json').read_text())
        assert all(value['delta_ci'][0] >= 0 for value in tol.values()), 'I-Q-TOL worsened'
    tr = pd.concat([pd.read_parquet(f) for f in sorted(OUT.glob('stat_train_*.parquet'))], ignore_index=True)
    query = tr[tr.date >= '2031-01-01'].copy() if pop in ('tol', 'tolm') else pd.concat(
        [pd.read_parquet(f) for f in sorted(OUT.glob(f'stat_{pop}_*.parquet'))], ignore_index=True)
    u, med, nn = V.null_quantiles(tr, query)
    query = query.assign(uH=u, medH=med, nnH=nn).rename(columns={'lr': 'lrH'})
    fp, ren = F.extra_frames()
    extra = fp[pop][['wkey', 'site', 'j', 'lr', 'u', 'med']].rename(columns=ren).merge(
        query[['wkey', 'site', 'j', 'lrH', 'uH', 'medH', 'nnH']], on=['wkey', 'site', 'j'], validate='one_to_one')
    if pop == 'val':
        assert phase in ('dev', 'confirm'), 'Read March in separate development and confirmation stages'
        wanted = query[pd.to_datetime(query.date).dt.day % 2 == (1 if phase == 'dev' else 0)].wkey
        extra = extra[extra.wkey.isin(wanted)].copy()
    baseline = lambda g: F.adjF(g.assign(pq=C.s1(g)), .02, 1.)
    candidate = lambda g: V.adjust(baseline(g), g.lrH.to_numpy(), g.uH.to_numpy(), g.medH.to_numpy(), .02, 1.)[0]
    # Do not use C.score's left-join of extras: it would silently score dates
    # excluded from the requested half with NaN hazard adjustments.
    frame, st = C.frames()[pop]
    frame = frame[frame.wkey.isin(extra.wkey)].copy()
    windows = E.windows(frame.drop_duplicates('wkey')).set_index('wkey')
    rows = []
    for j in range(9):
        g = frame.merge(st[st.j == j][['wkey', 'site', 'lr', 'u', 'med']],
                        on=['wkey', 'site'], how='inner', validate='one_to_one')
        g = g.merge(extra[extra.j == j].drop(columns='j'), on=['wkey', 'site'],
                    how='inner', validate='one_to_one')
        if g.empty:
            continue
        assert g[['lrH', 'uH', 'medH']].notna().all().all()
        decoded = {'s1f': C.decode(g, baseline(g)), 'survival': C.decode(g, candidate(g))}
        for key in decoded['s1f']:
            truth, valid = windows.loc[key, ['truth_l', 'valid_l']]
            row = dict(wkey=key, j=j,
                       unobs=float(any(not (cells[key] & set(valid)) for cells in decoded.values())),
                       changed=float(decoded['s1f'][key] != decoded['survival'][key]))
            row.update({name: E.iou(cells[key], truth, valid) for name, cells in decoded.items()})
            rows.append(row)
    scores = pd.DataFrame(rows).groupby('wkey')[['s1f', 'survival', 'unobs', 'changed']].mean()
    scored = windows[['panel', 'date', 't']].join(scores, how='inner').reset_index()
    if pop == 'val':
        assert (pd.to_datetime(scored.date).dt.day % 2 == (1 if phase == 'dev' else 0)).all()
    scored.to_parquet(OUT / f'scores_{pop}_{phase}.parquet')
    if pop in ('tol', 'tolm'):
        subsets = {'Jan': scored[scored.date < '2031-02-01'], 'Feb': scored[scored.date >= '2031-02-01']}
    else:
        parity = pd.to_datetime(scored.date).dt.day % 2
        subsets = ({'dev_odd' if phase == 'dev' else 'confirm_even': scored}
                   if pop == 'val' else {'odd': scored[parity == 1], 'even': scored[parity == 0], 'all': scored})
        subsets['observed_only'] = scored[scored.unobs == 0]
    result = {}
    for key, frame in subsets.items():
        result[key] = dict(n=len(frame), delta_ci=E.boot(frame, 'survival', 's1f'),
                           reference=E.macro(frame, 's1f'), candidate=E.macro(frame, 'survival'))
    (OUT / f'result_{pop}_{phase}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['check', 'build', 'eval'])
    p.add_argument('--pop', choices=['train', 'val', 'pri', 'tol', 'tolm'])
    p.add_argument('--panels', nargs='*')
    p.add_argument('--phase', choices=['dev', 'confirm', 'all'], default='dev')
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    check = poison_fixture()
    check['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT / 'fixture.json').write_text(json.dumps(check, indent=2))
    if args.mode == 'check':
        print(json.dumps(check))
    elif args.mode == 'build':
        build(args.pop, args.panels)
    else:
        evaluate(args.pop, args.phase)


if __name__ == '__main__':
    main()
