"""Resumable S-03 training data from frozen OPT dates and published targets."""
import argparse
from datetime import datetime,timezone,date
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from numpy.lib.format import open_memmap
import yaml
from tfb.data import load_cache,guard
from tfb.paths import project_root,panels
from .predict import predict_arrays
from .reference_data import interp_space
from .asm import predict_asm
from .evaluate_baselines import subset_cache
from .features import day_features,training_sample,geometry


def rng_for(panel,day):
    seed=int.from_bytes(hashlib.sha256(('S03 uniform sample4701|'+panel+'|'+day).encode()).digest()[:8],'little')
    return np.random.default_rng(seed)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare_bases(cache,asm_parameters,asm_path=None,asm_values=None):
    pv,pq=predict_arrays(cache,{'method':'B1'})
    pos=geometry(cache)['position'];vs,qs=interp_space(cache.speed,pos),interp_space(cache.flow,pos)
    vs=np.where(np.isfinite(vs),vs,pv).astype(np.float32);qs=np.where(np.isfinite(qs),qs,pq).astype(np.float32)
    stamp={'parameters':asm_parameters,'dates':cache.dates,'panel':cache.meta['panel'],'split':cache.meta['split'],'asm_source':sha(Path(__file__).parent/'asm.py')}
    stamp_path=asm_path.with_suffix('.json') if asm_path is not None else None
    if asm_values is not None:
        av,aq=asm_values;assert av.shape==cache.speed.shape and aq.shape==cache.flow.shape
    elif asm_path is not None and asm_path.exists() and stamp_path.exists():
        assert json.loads(stamp_path.read_text())==stamp,'Stale ASM feature cache'
        asm=np.load(asm_path,mmap_mode='r'); assert asm.shape==(2,*cache.speed.shape);av,aq=asm[0],asm[1]
    else:
        av,aq=predict_asm(cache,asm_parameters)
        if asm_path is not None:
            guard(project_root(),asm_path,expected_bytes=2*cache.speed.size*4)
            out=open_memmap(asm_path,mode='w+',dtype=np.float32,shape=(2,*cache.speed.shape));out[0]=av;out[1]=aq;out.flush();del out
            stamp_path.write_text(json.dumps(stamp,indent=2)+'\n')
    return {'time':(pv,pq),'space':(vs,qs),'asm':(av,aq)}


def build_panel(panel,code,dates,cfg,destination):
    cache0=load_cache(panel,'train');chosen=[cache0.day_index(d) for d in dates]
    contract={'source_sha256':{name:sha(Path(__file__).parent/name) for name in ['features.py','asm.py','build_lgb_data.py','predict.py']},'cache_meta_sha256':sha(cache0.path/'meta.json'),'dates':dates,'asm_parameters':cfg['asm_parameters'],'panel_code':code,'sample_seed':'S03 uniform sample4701; canonical date/geometry coordinates','row_sampling':'all nonfree and within1km/15min of congestion, plus10percent other free targets; weight10 for sampled free'}
    folder=destination/panel;folder.mkdir(parents=True,exist_ok=True);manifest=folder/'manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text())
        assert saved['contract']==contract,'Stale S-03 dataset: sources/population changed'
        for name,expected in saved['file_sha256'].items():assert sha(folder/name)==expected,'Dataset integrity failed '+name
        print(json.dumps({'panel':panel,'resumed':True,'n_rows':saved['n_rows']}),flush=True);return saved
    count=0;day_counts=[];total_targets=0;count_free=0;count_cong=0
    for d,raw_d in enumerate(chosen):
        target=cache0.target[raw_d];truth=cache0.speed_true[raw_d]
        # Use a one-day view so sampling cannot depend on excluded dates.
        view=type('DayView',(),{})();view.stations=cache0.stations;view.target=target[None]
        selected,w=training_sample(view,0,truth,rng_for(panel,dates[d]));n=int(selected.sum());count+=n;day_counts.append(n)
        total_targets+=int(target.sum());count_free+=int((target&(truth>.85*cache0.stations.free_speed_kmh.to_numpy())).sum());count_cong+=int((target&(truth<=cache0.stations.v_cut.to_numpy())).sum())
    cache=subset_cache(cache0,chosen);del cache0
    guard(project_root(),folder,expected_bytes=2*cache.speed.size*4)
    bases=prepare_bases(cache,cfg['asm_parameters'],folder/'asm.npy')
    example,flag=day_features(cache,0,bases,code);columns=list(example);ncol=len(columns)
    expected=count*(ncol*4+2*4+5*2+4+1)+2*cache.speed.size*4
    guard(project_root(),folder,expected_bytes=expected)
    X=open_memmap(folder/'X.npy',mode='w+',dtype=np.float32,shape=(count,ncol))
    truth_out=open_memmap(folder/'truth.npy',mode='w+',dtype=np.float32,shape=(count,2))
    metadata=open_memmap(folder/'metadata.npy',mode='w+',dtype=np.int16,shape=(count,4))
    weights=open_memmap(folder/'weight.npy',mode='w+',dtype=np.float32,shape=(count,))
    override=open_memmap(folder/'override.npy',mode='w+',dtype=np.bool_,shape=(count,))
    start=time.monotonic();cursor=0;lanes=cache.stations.lanes.to_numpy(np.float32)
    for d,day in enumerate(dates):
        feat,flag=day_features(cache,d,bases,code);assert list(feat)==columns
        selected,w=training_sample(cache,d,cache.speed_true[d],rng_for(panel,day));ti,si=np.nonzero(selected);n=len(ti);assert n==day_counts[d]
        sl=slice(cursor,cursor+n);X[sl]=np.column_stack([a[selected] for a in feat.values()]);truth_out[sl,0]=cache.speed_true[d][selected];truth_out[sl,1]=(cache.flow_true[d]/lanes)[selected]
        metadata[sl]=np.column_stack([np.full(n,d),ti,si,np.full(n,int(date.fromisoformat(day).weekday()>=5))]);weights[sl]=w[selected];override[sl]=flag[selected];cursor+=n
        if d%30==0:print(json.dumps({'panel':panel,'date':day,'rows_written':cursor}),flush=True)
    assert cursor==count and np.isfinite(truth_out).all()
    for a in [X,truth_out,metadata,weights,override]:a.flush()
    del X,truth_out,metadata,weights,override,bases,cache
    names=['X.npy','truth.npy','metadata.npy','weight.npy','override.npy','asm.npy','asm.json']
    result={'contract':contract,'columns':columns,'n_rows':count,'n_template_targets':total_targets,'n_free_targets':count_free,'n_congested_targets':count_cong,'file_sha256':{name:sha(folder/name) for name in names},'runtime_min':(time.monotonic()-start)/60,'utc':datetime.now(timezone.utc).isoformat()}
    manifest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'panel':panel,'n_rows':count,'n_columns':ncol,'completed':True}),flush=True);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',required=True);ap.add_argument('--panels',nargs='*');args=ap.parse_args()
    root=project_root();exp=Path(__file__).resolve().parents[3]/'exp'/f'{args.id}_s03_lgb_f1';cfg=yaml.safe_load((exp/'config.yaml').read_text());assert cfg['fit_end']=='2030-12-31','This driver is OPT-only'
    import pandas as pd
    dates=pd.date_range(cfg['fit_start'],cfg['fit_end']).strftime('%Y-%m-%d').tolist()
    destination=root/'artifacts/lane_s'/args.id/'data_opt';destination.mkdir(parents=True,exist_ok=True)
    result={}
    for code,panel in enumerate(panels()):
        if args.panels and panel not in args.panels:continue
        result[panel]=build_panel(panel,code,dates,cfg,destination)
    (destination/'summary.json').write_text(json.dumps({'panels':{p:{k:a[k] for k in ['n_rows','n_template_targets','n_free_targets','n_congested_targets']} for p,a in result.items()},'columns':next(iter(result.values()))['columns']},indent=2)+'\n')
if __name__=='__main__':main()
