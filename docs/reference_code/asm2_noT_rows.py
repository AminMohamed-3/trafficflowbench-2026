"""Verify T's strict retrained delivery before replacing all Task 2 rows."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PARENT_SHA = 'bccfbf4a541558cd4edb2e048ec4972300a90ab9b05b1dc0923b837c2b3757d0'
JOIN = ['window_id', 'timestamp', 'link_id']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate(root, rows, receipt, key=None):
    root = Path(root).resolve()
    boundary = root / 'artifacts/T_noT'
    hashes = {}

    def checked(path, expected=None):
        path = Path(path)
        path = (path if path.is_absolute() else root / path).resolve()
        assert path.is_relative_to(boundary), 'no-T delivery input must be under artifacts/T_noT'
        assert path.is_file(), f'missing no-T delivery input: {path}'
        name = path.relative_to(root).as_posix()
        digest = hashes.get(name) or sha(path)
        assert expected is None or digest == expected, f'no-T SHA mismatch: {path}'
        hashes[name] = digest
        return path

    receipt = checked(receipt)
    proof = json.loads(receipt.read_text())
    assert proof['status'] == 'PASS' and proof['parent_sha'] == PARENT_SHA
    assert proof['task2_rows'] == 174000 and proof['ongoing_windows'] == proof['onset_windows'] == 80
    rows = checked(rows, proof['task2_sha'])
    replacement = pd.read_parquet(rows)
    assert list(replacement.columns) == ['submission_id', 'queue_pred']
    assert len(replacement) == 174000 and replacement.submission_id.is_unique
    assert pd.api.types.is_integer_dtype(replacement.submission_id)
    assert np.isin(replacement.queue_pred, [0, 1]).all()
    projection_meta = checked(boundary / 'parent/C19_queue.json')
    meta = json.loads(projection_meta.read_text())
    assert meta['parent_sha'] == PARENT_SHA and meta['raw_data'] is False and meta['rows'] == 174000
    projection = pd.read_parquet(checked(boundary / 'parent/C19_queue.parquet', meta['projection_sha']))
    assert len(projection) == 174000 and projection.submission_id.is_unique
    assert set(projection.submission_id) == set(replacement.submission_id)

    ongoing, onset = [], []
    counts = dict(ongoing=0, onset=0, q03=0)
    receipts = proof['receipt_shas']
    assert receipts, 'missing qualified shard receipts'
    for name, digest in sorted(receipts.items()):
        path = checked(name, digest)
        assert path.suffix == '.json'
        shard_proof = json.loads(path.read_text())
        assert shard_proof['status'] == 'PASS'
        shard = checked(path.with_suffix('.parquet'), shard_proof['sha256'])
        contract = shard_proof['contract']
        if path.name.startswith(('validation_', 'private_')):
            assert contract['zeroed'] is False and contract['withT'] is False, 'requires retrained strict no-T bases'
            assert shard_proof['poison_checks'] > 0 and shard_proof['max_abs_dprob'] == 0
            assert len(contract['model_shas']) == 4
            for folder, model_sha in contract['model_shas'].items():
                checked(Path(folder) / 'model.pt', model_sha)
                cfg = json.loads(checked(Path(folder) / 'config.json').read_text())
                assert cfg.get('strict_noT') is True, 'T-trained weights cannot serve a no-T delivery'
            frame = pd.read_parquet(shard, columns=JOIN + ['queue_pred'])
            assert shard_proof['poison_checks'] == 2 * frame.window_id.nunique(), 'all ongoing windows require both poison checks'
            ongoing.append(frame)
            counts['ongoing'] += 1
        else:
            assert contract['arm'] == 'N', 'requires refitted no-T onset, not inference-zeroed reference'
            assert shard_proof['poison_checks'] == 2 * shard_proof['windows']
            if path.name.startswith('official_'):
                assert shard_proof['max_abs_dinput'] == 0
                assert set(contract['models']) == {'ranker_0', 'ranker_1', 'ranker_2'}
                for model, model_sha in contract['models'].items():
                    checked(boundary / 'onset/rank_model' / (model + '.txt'), model_sha)
                onset.append(pd.read_parquet(shard, columns=JOIN + ['queue_pred']))
                counts['onset'] += 1
            else:
                assert path.name.startswith('q03_off_'), 'unexpected no-T shard receipt'
                assert shard_proof['max_abs_dprob'] == 0
                folder = boundary / 'onset/Q03_model'
                checked(folder / 'model.txt', contract['model_sha'])
                checked(folder / 'prior.json', contract['prior_sha'])
                metadata = json.loads(checked(folder / 'metadata.json', contract['metadata_sha']).read_text())
                assert metadata.get('strict_noT') is True, 'requires retrained strict no-T Q03'
                counts['q03'] += 1
    assert counts['ongoing'] > 0 and counts['onset'] == counts['q03'] == 8
    og, on = pd.concat(ongoing, ignore_index=True), pd.concat(onset, ignore_index=True)
    assert len(og) == len(on) == 87000 and og.window_id.nunique() == on.window_id.nunique() == 80
    assert set(og.window_id).isdisjoint(on.window_id)
    certified = pd.concat([og, on], ignore_index=True)
    assert not certified.duplicated(JOIN).any()
    if key is None:
        parts = []
        for block in pd.read_csv(root / 'data/raw/kaggle_public/submission_key.csv',
                                 usecols=['submission_id', 'task', *JOIN], chunksize=250000,
                                 dtype={'submission_id':np.int64, **{c:str for c in ['task', *JOIN]}}):
            q = block[block.task == 'queue']
            if len(q):
                parts.append(q)
        key = pd.concat(parts, ignore_index=True)
    key = key[['submission_id', 'task', *JOIN]].copy()
    for col in ['task', *JOIN]:
        key[col] = key[col].astype(str)
    assert len(key) == 174000 and key.submission_id.is_unique and set(key.task) == {'queue'}
    assert key.window_id.nunique() == 160 and set(key.submission_id) == set(replacement.submission_id)
    rebuilt = key.merge(certified, on=JOIN, how='left', validate='one_to_one')
    assert rebuilt.queue_pred.notna().all()
    pd.testing.assert_frame_equal(rebuilt[['submission_id', 'queue_pred']].sort_values('submission_id').reset_index(drop=True),
                                  replacement.sort_values('submission_id').reset_index(drop=True), check_dtype=False, check_exact=True)
    return replacement, dict(status='PASS', rows=174000, windows=160, parent_sha=PARENT_SHA,
                             input_sha256=hashes, shard_counts=counts,
                             semantics='retrained strict no-T; exact qualified shards; Task 2 replacement only')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--rows', default='artifacts/T_noT/delivery/task2_noT.parquet')
    ap.add_argument('--receipt', default='artifacts/T_noT/delivery/rows_receipt.json')
    args = ap.parse_args()
    _, report = validate(args.root, args.rows, args.receipt)
    print(json.dumps(report), flush=True)
