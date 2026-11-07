"""Q0436: fixed agreement of current physical and online historical demand evidence."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import q0415_survival as H

sys.path.insert(0, str(H.SOURCE_ROOT / 'src'))
ROOT = H.ROOT
OUT = ROOT / 'artifacts/lane_q/0436_demand_agreement'
MODEL_DIR = ROOT / 'artifacts/lane_q/0425_demand_rank'
import q0434_flow_share as FS
V2 = ['u_c', 'u_R', 'u_ds', 'u_P', 'uproj_R', 'headR_max', 'headR_mean', 'up_anom']


def agreement(a, b):
    return np.where(np.sign(a) == np.sign(b), np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.)


def agreement_fixture():
    a=np.array([2.,-2.,1.,-1.,0.]); b=np.array([1.,-3.,-1.,1.,3.])
    np.testing.assert_array_equal(agreement(a,b),[1.,-2.,0.,0.,0.])
    np.testing.assert_array_equal(agreement(a,np.zeros_like(a)),np.zeros_like(a))
    np.testing.assert_array_equal(agreement(a,b),agreement(b,a))
    return dict(status='PASS',checks=['same-sign weaker evidence','no-history identity','symmetric sources'])


def columns():
    import onset4_eval as E
    return E.PHYS + V2  # common absolute clock excluded; every input becomes a difference


def source_frame(pop):
    return pd.read_parquet(ROOT / f'artifacts/claude_ref/onset4/feat_{pop}.parquet')


def differences(x):
    with np.errstate(invalid='ignore'):
        return x[:, None, :] - x[None, :, :]


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def latent(log_odds):
    anti = .5 * (log_odds - log_odds.T)
    return anti.sum(axis=1) / len(anti)


def fixture():
    x = np.arange(24.).reshape(4, 6)
    np.testing.assert_array_equal(differences(x), differences(x + np.arange(6.) * 100))
    score = np.array([-2., .5, 1., 2.])
    pair = score[:, None] - score[None, :]
    np.testing.assert_allclose(latent(pair), score - score.mean())
    order = [2, 0, 3, 1]
    np.testing.assert_array_equal(latent(pair[np.ix_(order, order)]), latent(pair)[order])
    return dict(status='PASS', checks=['common-level cancellation', 'latent-score recovery', 'permutation equivariance'],
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def causal_check(pop):
    """Verify actual cached columns and future-poison on two origins per panel.

    Features are recomputed from released arrays, with known train-only profiles.
    Raw outcome columns and selector membership are never supplied to features.
    """
    import onset4_data as D
    frame = source_frame(pop)
    if pop == 'val':
        frame = frame[pd.to_datetime(frame.date).dt.day % 2 == 1]
    cols = columns()
    prof = dict(np.load(ROOT / 'artifacts/claude_ref/onset4/profiles.npz'))
    ctrl = json.loads((ROOT / 'artifacts/claude_ref/onset4/control_cells.json').read_text())
    receipts = []
    for panel, group in frame.groupby('panel'):
        obj = D.Panel(panel, {'val': 'validation', 'pri': 'private'}[pop])
        for _, block in list(group.groupby('wkey', sort=True))[:2]:
            query = block.iloc[0]
            d, t = obj.dates.index(query.date), int(query.t)
            direct = pd.DataFrame(D.feats(obj, d, t, prof, ctrl)).set_index('site_uid')
            expected = block.set_index('site_uid')[cols].sort_index().to_numpy(dtype=float)
            actual = direct[cols].sort_index().to_numpy(dtype=float)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7, equal_nan=True)
            poisoned = copy.copy(obj)
            for attr in ('sm', 'fm', 'om', 'elig'):
                array = np.array(getattr(obj, attr), copy=True)
                fill = False if array.dtype.kind == 'b' else np.nan
                array[d, t + 1:] = fill
                array[d + 1:] = fill
                setattr(poisoned, attr, array)
            changed = pd.DataFrame(D.feats(poisoned, d, t, prof, ctrl)).set_index('site_uid')
            got = changed[cols].sort_index().to_numpy(dtype=float)
            np.testing.assert_array_equal(actual, got)
            np.testing.assert_array_equal(differences(actual), differences(got))
            receipts.append(dict(panel=panel, date=query.date, t=t, source_parity='PASS', future_poison='PASS'))
            del poisoned
        del obj
    result = dict(status='PASS', population=pop, n=len(receipts), checks=receipts)
    (OUT / f'causal_{pop}.json').write_text(json.dumps(result, indent=2))
    print(f'{pop}: direct-source parity and future poison PASS on {len(receipts)} origins', flush=True)


def fit():
    import lightgbm as lgb
    import onset4_eval as E
    df = source_frame('fit')
    assert (df.date < '2031-01-01').all()
    cols = columns()
    xs, ys, weights = [], [], []
    for _, group in df.groupby('wkey', sort=False):
        x, y = group[cols].to_numpy(np.float32), group.y.to_numpy()
        select = y[:, None] != y[None, :]
        if not select.any():
            continue
        xs.append(differences(x)[select])
        ys.append((y[:, None] > y[None, :])[select].astype(float))
        weights.append(np.full(select.sum(), 1. / select.sum()))
    x, y, weight = np.vstack(xs), np.concatenate(ys), np.concatenate(weights)
    ds = lgb.Dataset(x, label=y, weight=weight, feature_name=cols, free_raw_data=False)
    (OUT / 'columns.json').write_text(json.dumps(cols))
    for seed in (0, 1, 2):
        params = dict(E.PARAMS, seed=seed, num_threads=4)
        model = lgb.train(params, ds, num_boost_round=300)
        model.save_model(str(OUT / f'ranker_{seed}.txt'))
        print(f'fit seed {seed}: {len(y)} ordered pairs, {len(cols)} difference features', flush=True)


def probabilities(pop):
    import lightgbm as lgb
    frame = source_frame(pop)
    cols = json.loads((MODEL_DIR / 'columns.json').read_text())
    models = [lgb.Booster(model_file=str(MODEL_DIR / f'ranker_{seed}.txt')) for seed in (0, 1, 2)]
    out = frame[['wkey', 'site_uid']].copy()
    values = np.empty((3, len(frame)))
    for _, indices in frame.groupby('wkey', sort=False).indices.items():
        block = frame.iloc[indices]
        n = len(block)
        x = differences(block[cols].to_numpy(np.float32)).reshape(n * n, len(cols))
        for seed, model in enumerate(models):
            pair = logit(model.predict(x, num_threads=4)).reshape(n, n)
            z = latent(pair)
            probability = np.exp(z - z.max()); probability /= probability.sum()
            values[seed, indices] = logit(probability)
    for seed in (0, 1, 2):
        out[f'logit_rank_{seed}'] = values[seed]
    out['logit_rank'] = values.mean(axis=0)
    return out


def summary(frame):
    import onset4_eval as E
    delta = E.boot(frame, 'rank', 's1f')
    seed = [E.macro(frame, f'rank_{s}') - E.macro(frame, 's1f') for s in (0, 1, 2)]
    return dict(n=len(frame), reference=E.macro(frame, 's1f'), candidate=E.macro(frame, 'rank'),
                delta_ci=delta, q03=E.macro(frame, 'q03'), vs_q03_delta_ci=E.boot(frame, 'rank', 'q03'), seed_delta_sd=float(np.std(seed, ddof=1)), seed_deltas=seed,
                panels=frame.groupby('panel')[['s1f', 'rank']].mean().to_dict(orient='index'))


def passed_march():
    r = json.loads((OUT / 'result_val_confirm.json').read_text())
    return r['delta_ci'][1] > 0 and r['seed_delta_sd'] < .5 * r['delta_ci'][0]


def evaluate(pop, phase):
    import onset6_common as C
    import onset6_evalF as F
    import onset4_eval as E
    if pop == 'val' and phase == 'confirm':
        assert json.loads((OUT / 'result_val_dev.json').read_text())['delta_ci'][0] > 0
    if pop in ('tol', 'tolm', 'pri'):
        assert passed_march(), 'March CI/seed gate did not pass'
    if pop == 'pri':
        r = json.loads((OUT / 'result_tol_all.json').read_text())
        assert all(r[k]['delta_ci'][0] >= 0 for k in ('Jan', 'Feb')), 'Raw TOL worsened'
    check = json.loads((OUT / f'causal_{"pri" if pop == "pri" else "val"}.json').read_text())
    assert check['status'] == 'PASS' and check['n'] == 16
    flow = FS.calibrate('tol' if pop == 'tolm' else pop)
    pred = probabilities(pop)
    frame, stats = C.frames()[pop]
    frame = frame.merge(pred, on=['wkey', 'site_uid'], validate='one_to_one')
    if pop == 'val':
        assert phase in ('dev', 'confirm')
        frame = frame[pd.to_datetime(frame.date).dt.day % 2 == (1 if phase == 'dev' else 0)]
    fp, ren = F.extra_frames()
    fstat = fp[pop][['wkey', 'site', 'j', 'lr', 'u', 'med']].rename(columns=ren)
    windows = E.windows(frame.drop_duplicates('wkey')).set_index('wkey')
    rows = []
    for j in range(9):
        g = frame.merge(stats[stats.j == j][['wkey', 'site', 'lr', 'u', 'med']], on=['wkey', 'site'], how='inner')
        g = g.merge(fstat[fstat.j == j].drop(columns='j'), on=['wkey', 'site'], how='inner', validate='one_to_one')
        g = g.merge(flow[flow.j == j][['wkey','site','correction']], on=['wkey','site'], how='left', validate='one_to_one')
        assert g.loc[g.site != 'other','correction'].notna().all()
        if g.empty:
            continue
        baseline = F.adjF(g.assign(pq=C.s1(g)), .02, 1.)
        decoded = {'q03': C.decode(g, g.pq.to_numpy()), 's1f': C.decode(g, baseline)}
        for method in ('rank', 'rank_0', 'rank_1', 'rank_2'):
            physical = .5 * (g[f'logit_{method}'].to_numpy() - logit(g.pq.to_numpy()))
            correction = agreement(physical, g.correction.fillna(0).to_numpy())
            z = logit(baseline) + correction
            probability = np.where(correction == 0, baseline, 1 / (1 + np.exp(-z)))
            decoded[method] = C.decode(g, probability)
        for key in decoded['s1f']:
            truth, valid = windows.loc[key, ['truth_l', 'valid_l']]
            rows.append(dict(wkey=key, j=j, changed=float(decoded['s1f'][key] != decoded['rank'][key]),
                             unobs=float(not (decoded['s1f'][key] & set(valid)) or not (decoded['rank'][key] & set(valid))),
                             **{name: E.iou(cells[key], truth, valid) for name, cells in decoded.items()}))
    replay = pd.DataFrame(rows)
    replay.to_parquet(OUT / f'replay_{pop}_{phase}.parquet')
    scores = replay.groupby('wkey').mean(numeric_only=True).drop(columns='j')
    scores = windows[['panel', 'date', 't']].join(scores, how='inner').reset_index()
    scores.to_parquet(OUT / f'scores_{pop}_{phase}.parquet')
    if pop in ('tol', 'tolm'):
        result = {key: summary(f) for key, f in [('Jan', scores[scores.date < '2031-02-01']), ('Feb', scores[scores.date >= '2031-02-01'])]}
    else:
        result = {**summary(scores), 'observed_only': summary(scores[scores.unobs == 0])}
    (OUT / f'result_{pop}_{phase}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['check', 'dev', 'confirm', 'april', 'tolerance'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    result = fixture(); result['agreement_fixture'] = agreement_fixture()
    assert json.loads((FS.OUT / 'bounded_accessor_audit.json').read_text())['status'] == 'PASS'
    (OUT / 'fixture.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if args.mode == 'dev':
        causal_check('val'); evaluate('val', 'dev')
    elif args.mode == 'confirm':
        evaluate('val', 'confirm')
        if passed_march():
            evaluate('tol', 'all'); evaluate('tolm', 'all')
    elif args.mode == 'april':
        assert passed_march()
        r = json.loads((OUT / 'result_tol_all.json').read_text())
        assert all(r[k]['delta_ci'][0] >= 0 for k in ('Jan', 'Feb'))
        causal_check('pri'); evaluate('pri', 'all')
    elif args.mode == 'tolerance':
        assert passed_march()
        evaluate('tol', 'all'); evaluate('tolm', 'all')


if __name__ == '__main__':
    main()
