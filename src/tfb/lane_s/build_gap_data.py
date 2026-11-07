"""Resumable F2 fit data on June-Dec inputs with simulated full-station gaps."""
import argparse,gc,json,time
from datetime import date,datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np,pandas as pd,yaml
from numpy.lib.format import open_memmap
from tfb.data import load_cache,guard
from tfb.paths import project_root,panels
from .build_lgb_data import sha,rng_for,prepare_bases
from .evaluate_baselines import subset_cache
from .instruments import apply_frozen_gaps
from .features import training_sample
from .gap_features import day_features,feature_groups
from .asm import predict_asm
from .jobs import owned_job

def take_days(cache,selected):
    arrays={k:np.asarray(v[selected]) for k,v in vars(cache).items() if isinstance(v,np.ndarray) and v.shape and v.shape[0]==len(cache.dates)}
    meta=dict(cache.meta);meta['shape']=[len(selected),*cache.speed.shape[1:]]
    return SimpleNamespace(meta=meta,stations=cache.stations,ramps=cache.ramps,dates=[cache.dates[i] for i in selected],**arrays)

def gap_bases(cache,cfg,folder):
    path=folder/'asm.npy';stamp=folder/'asm.json'
    signature={'parameters':cfg['asm_parameters'],'dates':cache.dates,'gap_windows_sha256':cfg['training_windows_sha256'],'source':sha(Path(__file__).parent/'asm.py'),'panel':cache.meta['panel']}
    if path.exists() and stamp.exists():
        assert json.loads(stamp.read_text())==signature
        values=np.load(path,mmap_mode='r');assert values.shape==(2,*cache.speed.shape)
    else:
        original=project_root()/'artifacts/lane_s/0131/data_opt'/cache.meta['panel']
        manifest=json.loads((original/'manifest.json').read_text())
        assert manifest['contract']['dates']==cache.dates and manifest['contract']['asm_parameters']==cfg['asm_parameters']
        assert sha(original/'asm.npy')==manifest['file_sha256']['asm.npy']
        guard(project_root(),path,expected_bytes=2*cache.speed.size*4)
        values=open_memmap(path,mode='w+',dtype=np.float32,shape=(2,*cache.speed.shape))
        old=np.load(original/'asm.npy',mmap_mode='r');values[:]=old;del old
        affected=np.flatnonzero(cache.gap.any(axis=(1,2)))
        if len(affected):
            c=take_days(cache,affected);av,aq=predict_asm(c,cfg['asm_parameters']);values[0,affected]=av;values[1,affected]=aq;del c,av,aq
        values.flush();stamp.write_text(json.dumps(signature,indent=2)+'\n')
    return prepare_bases(cache,cfg['asm_parameters'],asm_values=(values[0],values[1]))

def choose_rows(cache,d,rng):
    selected,w=training_sample(cache,d,cache.speed_true[d],rng)
    # All gap targets are represented; sampling probability there is one.
    selected|=cache.target[d]&cache.gap[d]
    w=np.where(cache.gap[d],1.,w)
    return selected,w

def build_panel(panel,code,dates,cfg,dest,windows):
    raw=load_cache(panel,'train');raw_meta=sha(raw.path/'meta.json')
    cache=subset_cache(raw,[raw.day_index(d) for d in dates]);del raw
    assert dates and min(dates)>='2030-06-01' and max(dates)<='2030-12-31'
    population=apply_frozen_gaps(cache,cache.speed_true,cache.flow_true,windows);cache.s_windows=windows
    contract={'dates':dates,'panel_code':code,'cache_meta_sha256':raw_meta,'training_windows_sha256':cfg['training_windows_sha256'],'asm_parameters':cfg['asm_parameters'],'source_sha256':{name:sha(Path(__file__).parent/name) for name in ['gap_features.py','build_gap_data.py','features.py','predict.py','asm.py','instruments.py']},'row_sampling':'all gap/nonfree/near-congestion; other free10percent weight10; gap probability1 weight1'}
    folder=dest/panel;folder.mkdir(parents=True,exist_ok=True);manifest=folder/'manifest.json'
    if manifest.exists():
        report=json.loads(manifest.read_text());assert report['contract']==contract,'Stale F2 data'
        for name,h in report['file_sha256'].items():assert sha(folder/name)==h
        print(json.dumps({'panel':panel,'resumed':True,'rows':report['n_rows']}),flush=True);return report
    counts=[int(choose_rows(cache,d,rng_for(panel,day))[0].sum()) for d,day in enumerate(dates)];count=sum(counts)
    bases=gap_bases(cache,cfg,folder);example,_=day_features(cache,0,bases,code);columns=list(example)
    guard(project_root(),folder,expected_bytes=count*(len(columns)*4+2*4+4*2+4+1))
    X=open_memmap(folder/'X.npy',mode='w+',dtype=np.float32,shape=(count,len(columns)))
    truth=open_memmap(folder/'truth.npy',mode='w+',dtype=np.float32,shape=(count,2))
    metadata=open_memmap(folder/'metadata.npy',mode='w+',dtype=np.int16,shape=(count,4))
    weight=open_memmap(folder/'weight.npy',mode='w+',dtype=np.float32,shape=(count,))
    override=open_memmap(folder/'override.npy',mode='w+',dtype=np.bool_,shape=(count,))
    cursor=0;lanes=cache.stations.lanes.to_numpy(float);start=time.monotonic()
    for d,day in enumerate(dates):
        features,flag=day_features(cache,d,bases,code);assert list(features)==columns
        selected,w=choose_rows(cache,d,rng_for(panel,day));ti,si=np.nonzero(selected);n=len(ti);assert n==counts[d]
        sl=slice(cursor,cursor+n);X[sl]=np.column_stack([a[selected] for a in features.values()])
        truth[sl,0]=cache.speed_true[d][selected];truth[sl,1]=(cache.flow_true[d]/lanes)[selected]
        metadata[sl]=np.column_stack([np.full(n,d),ti,si,np.full(n,int(date.fromisoformat(day).weekday()>=5))])
        weight[sl]=w[selected];override[sl]=flag[selected];cursor+=n
        if d%30==0:print(json.dumps({'panel':panel,'day':day,'rows':cursor}),flush=True)
    assert cursor==count and np.isfinite(truth).all()
    for a in [X,truth,metadata,weight,override]:a.flush()
    report={'contract':contract,'columns':columns,'feature_groups':feature_groups(columns),'n_rows':count,'n_template_targets':int(cache.target.sum()),'gap_target_count':int((cache.gap&cache.target).sum()),'population':population,'runtime_min':(time.monotonic()-start)/60}
    del X,truth,metadata,weight,override,bases,cache;gc.collect()
    names=['X.npy','truth.npy','metadata.npy','weight.npy','override.npy','asm.npy','asm.json']
    report['file_sha256']={name:sha(folder/name) for name in names};manifest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'panel':panel,'completed':True,'n_rows':count,'gap_targets':report['gap_target_count']}),flush=True);return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--id',default='0142');ap.add_argument('--panels',nargs='*');args=ap.parse_args()
    root=project_root();exp=Path(__file__).resolve().parents[3]/'exp'/f'{args.id}_s03_lgb_gap'
    cfg=yaml.safe_load((exp/'config.yaml').read_text());windows=pd.read_csv(root/cfg['training_windows_path'])
    assert sha(root/cfg['training_windows_path'])==cfg['training_windows_sha256'];assert windows.date.between('2030-06-01','2030-12-31').all()
    dates=pd.date_range(cfg['fit_start'],cfg['fit_end']).strftime('%Y-%m-%d').tolist()
    dest=root/'artifacts/lane_s'/args.id/'data_opt';dest.mkdir(parents=True,exist_ok=True);result={}
    with owned_job(args.id,dest,eta_hours=1,ram_gib=7) as receipt:
        for code,panel in enumerate(panels()):
            if args.panels and panel not in args.panels:continue
            result[panel]=build_panel(panel,code,dates,cfg,dest,windows)
        report={'id':args.id,'columns':next(iter(result.values()))['columns'],'panels':{p:{k:a[k] for k in ['n_rows','n_template_targets','gap_target_count']} for p,a in result.items()},'peak_rss_gib':receipt['peak_rss_gib'],'adv_access':False}
        (dest/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()

