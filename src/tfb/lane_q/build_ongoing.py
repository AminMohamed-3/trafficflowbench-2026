"""Fit Q-04 panel models, with train-only canonical feature dose comparisons."""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta,timezone
import gc
import json
import os
from pathlib import Path
import shutil
import time
import unittest
import numpy as np
import pandas as pd
from .ongoing_model import OngoingModel,train,expected_iou_set
from .ongoing_features import feature_frame,static_sites,DOSES,FEATURES
from .ongoing_poison import assert_fitted_poison
from .windows import TASK2_PANELS,online_view,project_root,candidate_windows,iso
from .evaluation_cache import EvaluationRelease
from .build_baseline import sha,peak_mb
from .build_onset import guard_disk
from .evaluate_adv import score_grid,macro
from .evaluate_baseline import bootstrap_macro
from .source_provenance import source_changes,retain_sources


def run_tests():
    from . import test_ongoing
    if not unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_ongoing)).wasSuccessful():
        raise RuntimeError('Q04 poisoned-future/API/decision gate failed; no scores')


def remap_sites(frame,sites,network):
    result=frame.copy()
    length=network.fd_parameters.length_km.to_numpy(dtype=float)
    km=np.cumsum(length)-.5*length
    positions=frame.link_position.to_numpy(dtype=int)
    if sites:
        anchors=np.asarray([s['head'] for s in sites])
        mapping=np.abs(km[:,None]-km[anchors]).argmin(axis=1)
        uids=np.asarray([s['key'] for s in sites],dtype=object)
        result['site_uid']=uids[mapping[positions]]
        result['site_distance_km']=(km[positions]-km[anchors[mapping[positions]]]).astype(np.float32)
    else:
        result['site_uid']='other';result['site_distance_km']=np.float32(np.nan)
    return result


def extract_panel(release,panel,sites,folds,out):
    rng=np.random.default_rng(404)
    rows=[];labels=[];windows=[]
    network=release.network(panel);n=len(network.links)
    for date in release.files(panel,'train','mainline_states'):
        if not '2030-06-01'<=date<='2030-12-31':continue
        day=release.train_grid(panel,date)
        candidates=candidate_windows(day)
        ongoing=candidates[candidates.condition.eq('queue_ongoing')].sort_values('forecast_origin')
        choice=rng.choice(len(ongoing),min(2,len(ongoing)),replace=False)
        selected=ongoing.iloc[choice].sort_values('forecast_origin')
        for w in selected.itertuples(index=False):
            view=online_view(panel,'train',date,w.forecast_origin,release=release)
            if view.condition()!='queue_ongoing':raise AssertionError('Training selector/online condition mismatch')
            frame=feature_frame(view,sites)
            k=view.origin.hour*12+view.origin.minute//5
            truth=day.queued[k+1:k+7].reshape(-1);valid=day.observed[k+1:k+7].reshape(-1)
            frame['target']=np.where(valid,truth,-1).astype(np.int8)
            frame['weight']=np.float32(1/max(1,valid.sum()))
            key=f'{panel}/{iso(view.origin)}'
            frame['window_key']=key;frame['date']=date;frame['fold']=folds[date]
            frame['link_position']=np.tile(np.arange(n,dtype=np.int16),6)
            rows.append(frame)
            labels.append({'window_key':key,'panel':panel,'date':date,'forecast_origin':iso(view.origin),'fold':folds[date],
                'truth_positions':np.flatnonzero(truth).tolist(),'observed_positions':np.flatnonzero(valid).tolist()})
            windows.append({'panel':panel,'date':date,'forecast_origin':iso(view.origin),'condition':'queue_ongoing','fold':folds[date]})
        if len(windows) and len(windows)%80==0:
            print(f'features {panel}: {len(windows)} sampled windows; peak{peak_mb():.0f}MiB',flush=True)
        peak_mb()
    guard_disk(out)
    frame=pd.concat(rows,ignore_index=True);label=pd.DataFrame(labels);window=pd.DataFrame(windows)
    return frame,label,window


def cache_paths(out,panel):
    return [out/f'{panel}_{name}.parquet' for name in ['features','labels','windows']]


def save_model_checkpoint(out,name,panel,booster,meta):
    directory=out/name;directory.mkdir(parents=True,exist_ok=True)
    booster.save_model(str(directory/f'{panel}.txt'))
    (directory/f'{panel}_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')


def scores(model,frame,labels,panel):
    probability=model.probabilities(frame,panel)
    groups=frame.groupby('window_key',sort=False).indices
    n=len(model.prior['panels'][panel]['links']);rows=[]
    for w in labels.itertuples(index=False):
        ix=groups[w.window_key]
        block=frame.iloc[ix]
        if len(ix)!=6*n:raise AssertionError('Holdout forecasts must include every future cell, even unobserved labels')
        pred=expected_iou_set(probability[ix])
        valid=np.zeros(6*n,dtype=bool);valid[w.observed_positions]=True
        truth=np.zeros(6*n,dtype=bool);truth[w.truth_positions]=True
        base=block.last_queued.to_numpy().astype(bool)
        iou,_,_=score_grid(pred,truth,valid);ref,_,_=score_grid(base,truth,valid)
        rows.append({'window_key':w.window_key,'panel':panel,'condition':'queue_ongoing','IoU_ST':iou,'baseline_iou':ref,'delta':iou-ref})
    return pd.DataFrame(rows)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--id',default='0137');cli.add_argument('--main-root',type=Path,default=project_root())
    args=cli.parse_args();root=args.main_root;out=root/'artifacts/lane_q'/args.id;report=root/'exp'/f'{args.id}_q04_ongoing_f1'
    out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    guard_disk(root);run_tests();started=time.monotonic()
    folds=pd.read_csv(root/'configs/date_folds.csv');foldmap=dict(zip(folds.date,folds.fold))
    fullprior_path=root/'artifacts/lane_q/0114/prior_u1.json'
    innerprior_path=root/'artifacts/lane_q/0121/inner_Ffull_model/prior.json'
    fullprior=json.loads(fullprior_path.read_text());innerprior=json.loads(innerprior_path.read_text())
    release=EvaluationRelease(root/'data/raw/kaggle_public')
    manifest={'folds':sha(root/'configs/date_folds.csv'),'full_prior':sha(fullprior_path),'inner_prior':sha(innerprior_path),
              'features_source':sha(Path(__file__).with_name('ongoing_features.py')),'learner_source':sha(Path(__file__).with_name('ongoing_model.py')),
              'runtime_config':sha(root/'configs/lightgbm_runtime.yaml'),'sampling_seed':404,'windows_per_panel_day':2,
              'fd_sources':{p:sha(release.root/f'corridors/{p}/network/fd_parameters.csv') for p in TASK2_PANELS}}
    path=out/'fit_inputs.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous!=manifest:
            changed={key for key in set(previous)|set(manifest) if previous.get(key)!=manifest.get(key)}
            runtime=json.loads((root/'exp/0144_l_shared_runtime/result.json').read_text())
            fixture=json.loads((root/'artifacts/lane_q/0121/shared_runtime_fixture.json').read_text())
            allowed_sources=source_changes(out,previous,manifest)
            if changed-allowed_sources-{'runtime_config'} or runtime['version']!='4.6.0' or fixture['result']!='PASS':
                raise RuntimeError('Changed Q04 fit inputs; refuse stale cache')
            # D0012 verified every installed native/Python file against the
            # original runtime. Preserve original fit provenance and record the
            # resource-only equivalent migration separately.
            (out/'runtime_migration_equivalence.json').write_text(json.dumps({
                'old_config_sha256':previous['runtime_config'],'new_config_sha256':manifest['runtime_config'],
                'shared_receipt_sha256':sha(root/'exp/0144_l_shared_runtime/result.json'),
                'q03_shared_fixture_sha256':sha(root/'artifacts/lane_q/0121/shared_runtime_fixture.json'),
                'decision':'D0012','learned_inputs_unchanged':True},indent=2),encoding='utf-8')
            manifest=previous
    retain_sources(out,manifest)
    path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    receipt_path=root/'artifacts/jobs'/f'{args.id}.json'
    receipt={'owner':'Q','pid':os.getpid(),'command':[__import__('sys').executable,'-m','tfb.lane_q.build_ongoing'],
        'start_utc':datetime.now(timezone.utc).isoformat(),'eta_utc':(datetime.now(timezone.utc)+timedelta(minutes=90)).isoformat(),
        'checkpoint_path':str(out),'status':'RUNNING','threads':3,'ram_gb':3.5,'adv_read':0}
    receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    all_scores={dose:[] for dose in DOSES};full_meta={};inner_meta={dose:{} for dose in DOSES};poison=[]
    counts={}
    for panel in TASK2_PANELS:
        paths=cache_paths(out,panel);receipt_hash=out/f'{panel}_cache_hashes.json'
        if all(p.exists() for p in paths):
            if not receipt_hash.exists():raise RuntimeError('Missing feature-cache integrity receipt')
            expected=json.loads(receipt_hash.read_text())
            if any(sha(p)!=expected[p.name] for p in paths):raise RuntimeError('Q04 cached feature/label/window checksum mismatch')
            frame,labels,windows=[pd.read_parquet(p) for p in paths]
            print(f'resume {panel} {len(labels)} windows',flush=True)
        else:
            frame,labels,windows=extract_panel(release,panel,static_sites(fullprior['panels'][panel]),foldmap,out)
            for p,value in zip(paths,[frame,labels,windows]):value.to_parquet(p,index=False)
            receipt_hash.write_text(json.dumps({p.name:sha(p) for p in paths},indent=2),encoding='utf-8')
        counts[panel]={'windows':len(labels),'rows':len(frame),'observed_rows':int(frame.target.ge(0).sum())}
        inner=remap_sites(frame,static_sites(innerprior['panels'][panel]),release.network(panel))
        fit=inner[inner.fold.lt(4)&inner.target.ge(0)]
        held=inner[inner.fold.eq(4)];heldlabels=labels[labels.fold.eq(4)]
        # Record matched persistence noise before any candidate comparisons.
        n=len(release.network(panel).links);refrows=[]
        groups=held.groupby('window_key',sort=False).indices
        for w in heldlabels.itertuples(index=False):
            block=held.iloc[groups[w.window_key]]
            truth=np.zeros(6*n,dtype=bool);truth[w.truth_positions]=True
            valid=np.zeros(6*n,dtype=bool);valid[w.observed_positions]=True
            iou,_,_=score_grid(block.last_queued.to_numpy().astype(bool),truth,valid)
            refrows.append({'window_key':w.window_key,'panel':panel,'condition':'queue_ongoing','IoU_ST':iou})
        ref=pd.DataFrame(refrows);ci=bootstrap_macro(ref,seed=137)
        (out/f'{panel}_inner_noise_before_comparison.json').write_text(json.dumps({'seed_sd':0.,'B':1000,'ci95':ci,'ci_half_width':(ci[1]-ci[0])/2,'baseline':'persistence'},indent=2),encoding='utf-8')
        del refrows,ref
        for dose,features in DOSES.items():
            checkpoint=out/f'{panel}_inner_{dose}_scores.csv'
            directory=f'inner_{dose}_model'
            meta_path=out/directory/f'{panel}_metadata.json'
            if checkpoint.exists() and meta_path.exists():
                detail=pd.read_csv(checkpoint);meta=json.loads(meta_path.read_text())
            else:
                booster,categories=train(fit,features)
                meta={'categories':categories,'sites':static_sites(innerprior['panels'][panel])}
                model=OngoingModel({panel:booster},{'features':features,'panels':{panel:meta}},innerprior)
                if dose=='F0':
                    fixture=windows.iloc[0]
                    gate=assert_fitted_poison(model,release,panel,fixture.date,fixture.forecast_origin)
                    (out/f'{panel}_inner_actual_poison.json').write_text(json.dumps(gate,indent=2),encoding='utf-8')
                detail=scores(model,held,heldlabels,panel)
                detail.to_csv(checkpoint,index=False)
                save_model_checkpoint(out,directory,panel,booster,meta)
                del model,booster
            all_scores[dose].append(detail);inner_meta[dose][panel]=meta
            print(f'inner {panel}/{dose}: {detail.IoU_ST.mean():.6f}, delta{detail.delta.mean():+.6f}; n{len(detail)}; peak{peak_mb():.0f}MiB',flush=True)
            gc.collect()
        del fit,held,heldlabels,inner
        full_path=out/'full_model'/f'{panel}.txt';full_meta_path=out/'full_model'/f'{panel}_metadata.json'
        if full_path.exists() and full_meta_path.exists():
            meta=json.loads(full_meta_path.read_text());booster=lgb_booster(full_path)
        else:
            booster,categories=train(frame[frame.target.ge(0)],FEATURES)
            meta={'categories':categories,'sites':static_sites(fullprior['panels'][panel])}
            save_model_checkpoint(out,'full_model',panel,booster,meta)
        model=OngoingModel({panel:booster},{'features':FEATURES,'panels':{panel:meta}},fullprior)
        fixture=windows.iloc[0]
        gate=assert_fitted_poison(model,release,panel,fixture.date,fixture.forecast_origin)
        (out/f'{panel}_full_actual_poison.json').write_text(json.dumps(gate,indent=2),encoding='utf-8')
        poison.append(gate);full_meta[panel]=meta
        print(f'full {panel} saved; sampled{len(labels)} windows; peak{peak_mb():.0f}MiB',flush=True)
        del frame,labels,windows,model,booster;gc.collect()
        receipt['completed_panels']=len(full_meta);receipt['elapsed_min']=(time.monotonic()-started)/60
        receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    for name,meta,prior in [('full',full_meta,fullprior),*[(f'inner_{dose}',inner_meta[dose],innerprior) for dose in DOSES]]:
        directory=out/f'{name}_model'
        features=FEATURES if name=='full' else DOSES[name.removeprefix('inner_')]
        metadata={'model_version':1,'features':features,'panels':meta,'fit_start':'2030-06-01','fit_end':'2030-12-31' if name=='full' else '2030-11-19',
                  'input_hashes':manifest,'primary_dose':'Ffull','selection':'none'}
        (directory/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        (directory/'prior.json').write_text(json.dumps(prior,indent=2),encoding='utf-8')
    dose_results={}
    for dose,parts in all_scores.items():
        detail=pd.concat(parts,ignore_index=True);detail.to_csv(out/f'inner_{dose}_all_scores.csv',index=False)
        dose_results[dose]={'ongoing_iou':macro(detail,'IoU_ST')['S_queue'],'delta_ongoing_vs_persistence':macro(detail,'delta')['S_queue'],
             'paired_ci95':bootstrap_macro(detail,'delta',seed=137),'n_windows':len(detail),
             'panels':detail.groupby('panel').agg(n=('IoU_ST','size'),iou=('IoU_ST','mean'),delta=('delta','mean')).to_dict('index')}
    result={'id':args.id,'lane':'Q','formulation':'Q-04 canonical ongoing per-cell LightGBM Ffull','commit':None,
        'config':f'exp/{args.id}_q04_ongoing_f1/config.yaml','utc':datetime.now(timezone.utc).isoformat(),
        'instrument':'train chronological fold4 screen; ADV not read','adv_read':0,'incumbent_id':'0114',
        'inner_doses':dose_results,'fit_inputs':manifest,'training_counts':counts,'poisoned_future_test':'PASS',
        'actual_model_poison':poison,'runtime_min':(time.monotonic()-started)/60,'peak_memory_mib':peak_mb(),
        'verdict':'REF','selection':'none; predeclared full primary for first frozen ADV read'}
    result_path=report/'result.json'
    if result_path.exists():
        previous_result=json.loads(result_path.read_text())
        if previous_result.get('adv_read',0)>0:
            previous_result['fit_resume_verification']={'counts':counts,'poisoned_future_test':'PASS','fit_inputs':manifest,
                'runtime_min':result['runtime_min'],'selection':'none; completed ADV report preserved'}
            result=previous_result
    result_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    receipt['status']='COMPLETE';receipt['end_utc']=datetime.now(timezone.utc).isoformat()
    receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(f'Q04 fitting complete, elapsed{(time.monotonic()-started)/60:.2f}min',flush=True)


def lgb_booster(path):
    import lightgbm as lgb
    return lgb.Booster(model_file=str(path))

if __name__=='__main__':main()
