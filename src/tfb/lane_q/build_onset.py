"""Reproducible Q-03 training and predeclared train-only feature dose check."""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import unittest
import numpy as np
import pandas as pd

from .windows import project_root,TASK2_PANELS,online_view,iso
from .evaluation_cache import EvaluationRelease
from .predict import fit_panel_prior
from .onset_features import site_table,feature_frame,PriorDayContext,NUMERIC_FEATURES
from .onset_model import OnsetModel,train,choose_cells,DOSES
from .build_baseline import sha,peak_mb
from .evaluate_adv import score_grid,macro
from .evaluate_baseline import bootstrap_macro


def guard_disk(root):
    if shutil.disk_usage(root).free<10*1024**3:raise OSError('Disk guard: less than 10 GiB free')


def make_prior(records,links):
    return fit_panel_prior(records.to_dict('records'),links)


def extract(release,panel,records,sites,fold_map):
    context=PriorDayContext()
    features=[]
    labels=[]
    for date,windows in records.groupby('date',sort=True):
        day=release.train_grid(panel,date)
        for w in windows.sort_values('forecast_origin').itertuples(index=False):
            view=online_view(panel,'train',date,w.forecast_origin,release=release)
            if view.condition()!='queue_onset':raise AssertionError('Fit candidate condition mismatch')
            frame=feature_frame(view,sites,prior_context=context)
            step=view.origin.hour*12+view.origin.minute//5+6
            truth,observed=day.queued[step],day.observed[step]
            actual={day.links[i] for i in np.flatnonzero(truth)}
            if actual!=set(w.target_links):raise AssertionError('Fit checkpoint no longer matches raw labels')
            valid=[bool(observed[s['positions']].any()) for s in sites]
            frame['target']=[int(truth[s['positions']].any()) for s in sites]
            frame['weight']=1/max(1,sum(valid))
            key=f'{panel}/{iso(view.origin)}'
            frame['window_key']=key
            frame['date']=date
            frame['panel']=panel
            frame['fold']=fold_map[date]
            frame=frame[np.asarray(valid)]
            features.append(frame)
            labels.append({'window_key':key,'panel':panel,'date':date,'forecast_origin':iso(view.origin),
                           'fold':fold_map[date],'truth_positions':np.flatnonzero(truth).tolist(),
                           'observed_positions':np.flatnonzero(observed).tolist()})
        peak_mb()
    return pd.concat(features,ignore_index=True),pd.DataFrame(labels)


def inner_scores(model,features,labels):
    probabilities=model.probabilities(features)
    prediction=features[['window_key','site_uid']].copy()
    prediction['probability']=probabilities
    rows=[]
    for panel,part in labels.groupby('panel',sort=True):
        sites=model.metadata['sites'][panel]
        order=[f'{panel}/{s["key"]}' for s in sites]
        links=model.prior['panels'][panel]['links']
        for w in part.itertuples(index=False):
            p=prediction[prediction.window_key.eq(w.window_key)].set_index('site_uid').probability.reindex(order).fillna(0).to_numpy()
            cells=set(choose_cells(sites,p))
            guess=np.asarray([link in cells for link in links])
            truth=np.zeros(len(links),dtype=bool);truth[w.truth_positions]=True
            valid=np.zeros(len(links),dtype=bool);valid[w.observed_positions]=True
            iou,_,_=score_grid(guess,truth,valid)
            # Matched prior baseline uses exactly the same training folds.
            from .windows import timestamp
            t=timestamp(w.forecast_origin)+pd.Timedelta(minutes=30)
            context=[f'hour_day:{t.hour}/{"weekend" if t.weekday()>=5 else "weekday"}',f'hour:{t.hour}',f'day:{"weekend" if t.weekday()>=5 else "weekday"}','all']
            prior=model.prior['panels'][panel]
            baseline=next((set(prior['contexts'][c]['target_links']) for c in context if c in prior['contexts']),set())
            base,_,_=score_grid(np.asarray([x in baseline for x in links]),truth,valid)
            modal=prior['sites'][prior['contexts']['all']['site']]
            nonmodal=bool(set(np.flatnonzero(truth))-set(modal['positions']))
            rows.append({'window_key':w.window_key,'panel':panel,'condition':'queue_onset','IoU_ST':iou,'baseline_iou':base,'delta':iou-base,'nonmodal_true_site':nonmodal})
    return pd.DataFrame(rows)


def run_tests():
    from . import test_onset
    if not unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_onset)).wasSuccessful():
        raise RuntimeError('Q-03 future poison/decision test failed; no scores permitted')


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--id',default='0121')
    cli.add_argument('--main-root',type=Path,default=project_root())
    args=cli.parse_args();root=args.main_root
    out=root/'artifacts/lane_q'/args.id
    report=root/'exp'/f'{args.id}_q03_onset_f1'
    out.mkdir(parents=True,exist_ok=True)
    guard_disk(root);run_tests()
    started=time.monotonic()
    folds=pd.read_csv(root/'configs/date_folds.csv')
    fold_map=dict(zip(folds.date,folds.fold))
    assert min(fold_map)=='2030-06-01' and max(fold_map)=='2030-12-31'
    prior_path=root/'artifacts/lane_q/0114/prior_u1.json'
    full_prior=json.loads(prior_path.read_text())
    records={p:pd.read_parquet(root/f'artifacts/lane_q/0114/{p}_fit_onsets.parquet') for p in TASK2_PANELS}
    manifest={'prior':sha(prior_path),'folds':sha(root/'configs/date_folds.csv'),
              'fit_windows':{p:sha(root/f'artifacts/lane_q/0114/{p}_fit_onsets.parquet') for p in TASK2_PANELS},
              'features_source':sha(Path(__file__).with_name('onset_features.py')),
              'decision_source':sha(Path(__file__).with_name('onset_model.py')),
              'runtime_manifest':sha(out/'runtime/manifest.json'),
              'fit_start':'2030-06-01','fit_end':'2030-12-31','monthly_prior_context':True}
    manifest_path=out/'fit_inputs.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:raise RuntimeError('Fit inputs changed; refuse stale checkpoints')
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    release=EvaluationRelease(root/'data/raw/kaggle_public')
    inner_prior={'model_version':1,'panels':{}}
    inner_sites={};full_sites={}
    receipt_path=root/'artifacts/jobs'/f'{args.id}.json'
    receipt={'owner':'Q','pid':os.getpid(),'command':[__import__('sys').executable,'-m','tfb.lane_q.build_onset'],
             'start_utc':datetime.now(timezone.utc).isoformat(),'eta_utc':(datetime.now(timezone.utc)+timedelta(minutes=40)).isoformat(),
             'checkpoint_path':str(out),'status':'RUNNING','threads':4,'ram_gb':4,'adv_read':0}
    receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    all_inner=[];all_full=[];all_labels=[]
    for panel in TASK2_PANELS:
        fit=records[panel]
        assert fit.date.between('2030-06-01','2030-12-31').all()
        inner_records=fit[fit.date.map(fold_map).lt(4)]
        inner_prior['panels'][panel]=make_prior(inner_records,release.network(panel).links)
        inner_sites[panel]=site_table(inner_prior['panels'][panel])
        full_sites[panel]=site_table(full_prior['panels'][panel])
        paths=[out/f'{panel}_{kind}.parquet' for kind in ['inner_features','full_features','labels']]
        if all(path.exists() for path in paths):
            a,b,c=[pd.read_parquet(path) for path in paths]
            print(f'resume {panel}: {len(c)} onsets',flush=True)
        else:
            a,c=extract(release,panel,fit,inner_sites[panel],fold_map)
            geometry=lambda ss:[(site['key'],site['positions'],site['other']) for site in ss]
            if geometry(inner_sites[panel])==geometry(full_sites[panel]):
                b=a.copy()
            else:
                b,c2=extract(release,panel,fit,full_sites[panel],fold_map)
                pd.testing.assert_frame_equal(c,c2)
            guard_disk(root)
            for path,frame in zip(paths,[a,b,c]):frame.to_parquet(path,index=False)
            print(f'fit features {panel}: {len(c)} onsets; sites inner/full {len(inner_sites[panel])}/{len(full_sites[panel])}; peak{peak_mb():.0f} MiB',flush=True)
        all_inner.append(a);all_full.append(b);all_labels.append(c)
        receipt['completed_panels']=len(all_inner)
        receipt['elapsed_min']=(time.monotonic()-started)/60
        receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    inner=pd.concat(all_inner,ignore_index=True);full=pd.concat(all_full,ignore_index=True);labels=pd.concat(all_labels,ignore_index=True)
    train_rows=inner[inner.fold.lt(4)]
    heldout=inner[inner.fold.eq(4)]
    holdout_labels=labels[labels.fold.eq(4)]
    dose_results={}
    for dose,features in DOSES.items():
        booster,categories=train(train_rows,features)
        metadata={'model_version':1,'features':features,'categories':categories,'sites':inner_sites,'close_ratio':.8,'dose':dose,'fit_end':'2030-11-19'}
        model=OnsetModel(booster,metadata,inner_prior)
        detail=inner_scores(model,heldout,holdout_labels)
        detail.to_csv(out/f'inner_{dose}_scores.csv',index=False)
        # Deterministic fit; no stochastic sampling in the learner.
        ci=bootstrap_macro(detail,seed=121)
        if dose=='F0':
            floor={'seed_sd':0.,'ci95_onset':ci,'ci_half_width_onset':(ci[1]-ci[0])/2,'B':1000,'instrument':'train chronological fold4 onset only'}
            (out/'inner_noise_floor.json').write_text(json.dumps(floor,indent=2),encoding='utf-8')
        score=macro(detail,'IoU_ST')['S_queue']
        result={'onset_iou':score,'ci95_onset':ci,'delta_onset_vs_matched_prior':macro(detail,'delta')['S_queue'],
                'paired_ci95_onset':bootstrap_macro(detail,'delta',seed=121),'n_windows':len(detail),
                'panels':detail.groupby('panel').IoU_ST.mean().to_dict(),
                'nonmodal_true_site':detail[detail.nonmodal_true_site].groupby('panel').agg(n=('IoU_ST','size'),iou=('IoU_ST','mean')).to_dict('index')}
        dose_results[dose]=result
        model.save(out/f'inner_{dose}_model')
        print(f'inner {dose}: onsetIoU {score:.6f}; n{len(detail)}',flush=True)
    booster,categories=train(full,DOSES['Ffull'])
    metadata={'model_version':1,'features':DOSES['Ffull'],'categories':categories,'sites':full_sites,'close_ratio':.8,'dose':'Ffull',
              'fit_start':'2030-06-01','fit_end':'2030-12-31','input_hashes':manifest,'model_selection':'none; predeclared primary candidate'}
    model=OnsetModel(booster,metadata,full_prior)
    model.save(out/'full_model')
    run_tests()
    result={'id':args.id,'lane':'Q','formulation':'Q-03 canonical onset site LightGBM Ffull',
            'utc':datetime.now(timezone.utc).isoformat(),'commit':None,'config':f'exp/{args.id}_q03_onset_f1/config.yaml',
            'instrument':'train chronological fold4 dose screen; I-Q-ADV not yet read','adv_read':0,'incumbent_id':'0114',
            'poisoned_future_test':'PASS','inner_doses':dose_results,'inner_noise_floor':floor,
            'n_train_windows':len(labels),'n_site_rows':len(full),'fit_inputs':manifest,'model_sha256':sha(out/'full_model/model.txt'),
            'runtime_min':(time.monotonic()-started)/60,'peak_memory_mib':peak_mb(),'verdict':'REF','selection':'none; Ffull predeclared for ADV'}
    (report/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    receipt['status']='COMPLETE';receipt['end_utc']=datetime.now(timezone.utc).isoformat()
    receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(f'Q-03 full fit saved; {len(labels)} onset windows; elapsed {result["runtime_min"]:.2f}min',flush=True)

if __name__=='__main__':main()
