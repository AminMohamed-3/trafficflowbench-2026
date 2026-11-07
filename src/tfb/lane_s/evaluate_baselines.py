"""Frozen I-S-OPT fold evaluation for S-01 canonical baselines."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import yaml
from tfb import metric
from tfb.data import load_cache, indices, read_parquet
from tfb.paths import project_root, release_root, panels
from .predict import predict_arrays
from .reference_data import interp_space
from .instruments import apply_frozen_gaps
from .day_stats import day_statistics
from .bootstrap import aggregate, paired_ci, gate


def subset_cache(cache, selected):
    meta=dict(cache.meta)
    meta['shape']=[len(selected),cache.meta['shape'][1],cache.meta['shape'][2]]
    arrays={k:np.asarray(v[selected]) for k,v in cache.arrays.items()}
    meta['target_rows']=int(arrays['target'].sum())
    return SimpleNamespace(meta=meta,stations=cache.stations,ramps=cache.ramps,
        dates=[cache.dates[i] for i in selected],regimes=cache.regimes[selected],**arrays)


def raw_truth(cache,root):
    arrays=[np.full(cache.meta['shape'],np.nan,float) for _ in range(2)]
    dates={d:i for i,d in enumerate(cache.dates)}
    stations=dict(zip(cache.stations.station_id,cache.stations.idx))
    base=release_root(root)/'corridors'/cache.meta['panel']/'train'/'mainline_states'
    for path in base.glob('*/*.parquet'):
        day='-'.join(path.stem.split('_')[-3:])
        if day not in dates: continue
        data=read_parquet(path,['timestamp','station_id','speed_kmh','flow_vph'])
        key=indices(data,dates,stations,'station_id')
        for a,c in zip(arrays,['speed_kmh','flow_vph']): a[key]=data[c].to_numpy(float)
    assert np.isfinite(arrays[0][cache.target]).all() and np.isfinite(arrays[1][cache.target]).all()
    return arrays


def panel_scores(rows):
    result={}
    for panel,x in rows.groupby('panel'):
        result[panel]=aggregate(x,{panel:panel})
        regimes={}
        for r in metric.REGIMES:
            v=x[x.regime==r]
            n=int(v.n_target.sum())
            rv=float(np.sqrt(v.sq_v.sum()/n)) if n else None
            rq=float(np.sqrt(v.sq_q_lane.sum()/n)) if n else None
            fd=max(0,1-v.fd_error.sum()/(v.fd_flow.sum()+1e-9))
            lwr=max(0,1-v.lwr_error.sum()/(v.lwr_rhs.sum()+1e-9)) if v.n_transition.sum() else 0
            regimes[r]={'n':n,'rmse_v':rv,'rmse_q_lane':rq,'S_state':.54*max(0,1-rv/25)+.46*max(0,1-rq/600) if n else 0,
                        'S_FD':float(fd),'S_LWR':float(lwr)}
        result[panel].update(regimes)
    return result


def breakdown(cache,tv,tq,pv,pq):
    target=cache.target
    vf=cache.stations.free_speed_kmh.to_numpy(float)
    classes={'gap':target&cache.gap,'congested':target&(tv<=.6*vf),'free':target&(tv>.85*vf)}
    result={}
    for name,mask in classes.items():
        dv=(pv.astype(float)-tv)[mask]
        dq=((pq.astype(float)-tq)/cache.stations.lanes.to_numpy(float))[mask]
        result[name]={'n':int(mask.sum()),'sq_v':float((dv**2).sum()),'sq_q_lane':float((dq**2).sum())}
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fold',type=int,default=0)
    ap.add_argument('--instrument',choices=['I-S-OPT','I-S-ADV'],default='I-S-OPT')
    ap.add_argument('--control-id',default='0115')
    ap.add_argument('--id',default='0111')
    args=ap.parse_args()
    root=project_root()
    cfg=yaml.safe_load((root/'configs/instruments.yaml').read_text(encoding='utf-8-sig'))
    fold_path=root/cfg['date_folds']['path']
    assert hashlib.sha256(fold_path.read_bytes()).hexdigest()==cfg['date_folds']['sha256'],'Frozen fold hash mismatch'
    folds=pd.read_csv(fold_path,dtype={'date':str})
    windows=None
    if args.instrument=='I-S-OPT':
        held=folds.loc[folds.fold==args.fold,'date'].tolist()
        fit=folds.loc[folds.fold!=args.fold,'date'].tolist()
    else:
        assert cfg['status']=='FROZEN'
        held=pd.date_range(cfg['I-S-ADV']['score_start'],cfg['I-S-ADV']['score_end']).strftime('%Y-%m-%d').tolist()
        fit=pd.date_range(cfg['I-S-ADV']['fit_start'],cfg['I-S-ADV']['fit_end']).strftime('%Y-%m-%d').tolist()
        wpath=root/cfg['I-S-ADV']['windows']['path']
        assert hashlib.sha256(wpath.read_bytes()).hexdigest()==cfg['I-S-ADV']['windows']['sha256']
        windows=pd.read_csv(wpath)
        subprocess.run([sys.executable,str(root/'tools/ledger.py'),'close','--id',args.id,'--owner','S','--verdict','RUNNING',
            '--reads','ADV reads 1','--instrument','I-S-ADV (590 panel-days)',
            '--result','Frozen B1/B2/B3 plus known-worse control, first ADV read in progress; no tuning'],check=True)
    assert fit and held and not(set(fit)&set(held))
    assert min(fit)>=cfg['I-S-OPT']['start'] and max(fit)<=cfg['I-S-OPT']['end']
    manifest=json.loads((release_root(root)/'config/corridors.json').read_text())
    families={p['corridor_id']:p['family_id'] for p in manifest['panels']}
    dest=root/'artifacts/lane_s'/args.id/('opt_fold'+str(args.fold) if args.instrument=='I-S-OPT' else 'adv_read1')
    dest.mkdir(parents=True,exist_ok=True)
    methods=['B1','B2','B3']+(['SPACE'] if args.instrument=='I-S-ADV' else [])
    all_rows={m:[] for m in methods}
    all_breakdown={m:{name:{'n':0,'sq_v':0.,'sq_q_lane':0.} for name in ('gap','congested','free')} for m in methods}
    start=time.monotonic()
    for panel in panels():
        checkpoint=dest/(panel+'.json')
        if checkpoint.exists():
            content=json.loads(checkpoint.read_text())
            assert content['fold_sha256']==cfg['date_folds']['sha256']
            assert content['predict_sha256']==hashlib.sha256((Path(__file__).parent/'predict.py').read_bytes()).hexdigest()
            for method in methods:
                all_rows[method].append(pd.DataFrame(content['rows'][method]))
                for name,record in content['breakdown'][method].items():
                    for key,val in record.items(): all_breakdown[method][name][key]+=val
            print('Resumed',panel,flush=True)
            continue
        cache=load_cache(panel,'train')
        cache=subset_cache(cache,[cache.day_index(d) for d in held])
        tv,tq=raw_truth(cache,root)
        population=apply_frozen_gaps(cache,tv,tq,windows) if windows is not None else {'n_windows':0,'gap_target_count':0}
        content={'fold_sha256':cfg['date_folds']['sha256'],
            'predict_sha256':hashlib.sha256((Path(__file__).parent/'predict.py').read_bytes()).hexdigest(),'rows':{},'breakdown':{},'population':population}
        for method in methods:
            if method!='SPACE':
                pv,pq=predict_arrays(cache,{'method':method,'fit_dates':fit})
            else:
                pv,pq=predict_arrays(cache,{'method':'B1','fit_dates':fit})
                length=cache.stations.length_km.to_numpy(float)
                pos=length.cumsum()-length/2
                vs,qs=interp_space(cache.speed,pos),interp_space(cache.flow,pos)
                pv=np.where(np.isfinite(vs),vs,pv).astype(np.float32)
                pq=np.where(np.isfinite(qs),qs,pq).astype(np.float32)
                np.clip(pv,3,120.75,out=pv)
                upper=1.3*cache.stations.capacity_vph.to_numpy(float)
                upper32=upper.astype(np.float32)
                upper32=np.where(upper32.astype(float)>upper,np.nextafter(upper32,np.float32(-np.inf)),upper32)
                np.maximum(pq,50,out=pq);np.minimum(pq,upper32,out=pq)
            rows=day_statistics(cache,pv,pq,tv,tq,root=root)
            total=aggregate(rows,{panel:families[panel]})
            # Every bootstrap must aggregate exactly to the independently verified shared scorer.
            st=metric.state_score(tv,tq,pv,pq,target=cache.target,lanes=cache.stations.lanes,regimes=cache.regimes)
            ph=metric.physics_score(cache,pv,pq,truth_speed=tv,truth_flow=tq,mode='A',root=root)
            assert abs(total['S_state']-st['S_state'])<1e-9 and abs(total['S_physics']-ph['S_physics'])<1e-9
            content['rows'][method]=rows.to_dict('records')
            content['breakdown'][method]=breakdown(cache,tv,tq,pv,pq)
            all_rows[method].append(rows)
            for name,record in content['breakdown'][method].items():
                for key,val in record.items(): all_breakdown[method][name][key]+=val
            print(json.dumps({'panel':panel,'method':method,**total}),flush=True)
            del pv,pq
        checkpoint.write_text(json.dumps(content,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        del cache,tv,tq
    rows={m:pd.concat(all_rows[m],ignore_index=True) for m in methods}
    noise=paired_ci(rows['B1'],rows['B1'],families)
    # Log noise before any paired method comparison, through the required ledger tool.
    subprocess.run([sys.executable,str(root/'tools/ledger.py'),'close','--id',args.id,'--owner','S','--verdict','RUNNING',
        '--instrument',f'{args.instrument} ({len(held)*len(families)} panel-days)',
        '--result',f"Precomparison B1 seed SD 0; instrument J_S CI half-width {noise['incumbent_instrument_ci_half_width']:.8f}",
        '--reads','ADV reads 0' if args.instrument=='I-S-OPT' else 'ADV reads 1'],check=True)
    (dest/'noise_floor.json').write_text(json.dumps(noise,indent=2)+'\n',encoding='utf-8')
    control=None
    if 'SPACE' in methods:
        control={'time':aggregate(rows['B1'],families),'spatial':aggregate(rows['SPACE'],families)}
        control['verdict']='PASS' if control['spatial']['S_state']<control['time']['S_state'] else 'FAIL'
        (dest/'known_worse_control.json').write_text(json.dumps(control,indent=2)+'\n',encoding='utf-8')
        subprocess.run([sys.executable,str(root/'tools/ledger.py'),'close','--id',args.control_id,'--owner','S',
            '--verdict',control['verdict'],'--result',f"time S_state {control['time']['S_state']:.8f}; spatial {control['spatial']['S_state']:.8f}",
            '--reads','ADV reads 1 control, shared with S-01 first read'],check=True)
        assert control['verdict']=='PASS','Known-worse control failed; stop candidate comparisons'
    results={}
    for method in methods:
        pair=paired_ci(rows['B1'],rows[method],families)
        parts=all_breakdown[method]
        for name,val in parts.items():
            n=val['n']; val['rmse_v']=float(np.sqrt(val['sq_v']/n)) if n else None
            val['rmse_q_lane']=float(np.sqrt(val['sq_q_lane']/n)) if n else None
        results[method]={'panels':panel_scores(rows[method]),**aggregate(rows[method],families),
            'comparison_vs_B1':pair,'verdict':gate(pair) if method!='B1' else 'REF','breakdown':parts}
    best=max(['B1','B2','B3'],key=lambda m:results[m]['J_S'])
    report={'id':args.id,'lane':'S','instrument':args.instrument,'fold':args.fold,'fit_dates':fit,'held_dates':held,
        'n_panel_days':len(held)*len(families),'methods':results,'noise_floor':noise,'seed_sd_J_S':0.,
        'adv_read':0 if args.instrument=='I-S-OPT' else 1,'best_joint_method':best,'known_worse_control':control,'selection':'No incumbent promotion; Lead reproduction and three-board sign calibration pending',
        'runtime_min':(time.monotonic()-start)/60,'fold_sha256':cfg['date_folds']['sha256'],
        'instrument_yaml_sha256':hashlib.sha256((root/'configs/instruments.yaml').read_bytes()).hexdigest(),
        'utc':datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')}
    (dest/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'best_joint_method':best,'methods':{m:{k:results[m][k] for k in ['S_state','S_physics','J_S','verdict']} for m in methods},'runtime_min':report['runtime_min']}),flush=True)


if __name__=='__main__': main()
