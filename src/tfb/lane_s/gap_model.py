"""Canonical F1 residual training, fit-only summaries and prediction helpers."""
from pathlib import Path
import gc,hashlib,json
import numpy as np
from .lgb_runtime import load_lightgbm
lgb=load_lightgbm()
from tfb.data import load_cache
from .features import DYNAMIC,CATEGORICAL,fit_summaries,fill_dynamic
from .gap_features import day_features
from .build_lgb_data import sha,prepare_bases


def save_summaries(folder,panels,fit_dates):
    summaries={}
    for panel in panels:
        cache=load_cache(panel,'train');s=fit_summaries(cache,fit_dates);del cache
        np.savez_compressed(folder/f'{panel}_summaries.npz',**s);summaries[panel]=s
    return summaries


def load_summaries(folder,panel):
    with np.load(Path(folder)/f'{panel}_summaries.npz',allow_pickle=False) as a:return {k:a[k] for k in a.files}


def matrix(data_root,selected_panels,selected_dates,summaries):
    manifests={p:json.loads((data_root/p/'manifest.json').read_text()) for p in selected_panels}
    columns=next(iter(manifests.values()))['columns'];counts={}
    for panel,a in manifests.items():
        assert a['columns']==columns
        metadata=np.load(data_root/panel/'metadata.npy',mmap_mode='r')
        is_date=np.asarray([d in selected_dates for d in a['contract']['dates']]);counts[panel]=int(is_date[metadata[:,0]].sum());del metadata
    n=sum(counts.values());X=np.empty((n,len(columns)),np.float32);y=np.empty((n,2),np.float32);weights=np.empty(n,np.float32)
    cursor=0;index={k:i for i,k in enumerate(columns)}
    for panel,a in manifests.items():
        folder=data_root/panel;metadata=np.load(folder/'metadata.npy',mmap_mode='r');is_date=np.asarray([d in selected_dates for d in a['contract']['dates']]);keep=is_date[metadata[:,0]]
        meta=np.asarray(metadata[keep]);part=np.asarray(np.load(folder/'X.npy',mmap_mode='r')[keep]);truth=np.asarray(np.load(folder/'truth.npy',mmap_mode='r')[keep]);flag=np.asarray(np.load(folder/'override.npy',mmap_mode='r')[keep]);w=np.asarray(np.load(folder/'weight.npy',mmap_mode='r')[keep]);npart=len(meta)
        fill_dynamic(part,columns,meta[:,2],meta[:,1],meta[:,3],flag,summaries[panel])
        truth[:,0]-=part[:,index['base_B2_speed']];truth[:,1]-=part[:,index['base_time_flow_lane']]
        for r in [0,1,2]:
            group=part[:,index['regime']]==r
            if group.any():w[group]*=n/(len(selected_panels)*3*float(w[group].sum()))
        sl=slice(cursor,cursor+npart);X[sl]=part;y[sl]=truth;weights[sl]=w;cursor+=npart
        del metadata,meta,part,truth,flag,w;gc.collect()
    assert cursor==n and np.isfinite(y).all() and np.isfinite(weights).all()
    return X,y,weights,columns


def train_case(folder,data_root,selected_panels,fit_dates,held_dates,cfg,fold):
    folder.mkdir(parents=True,exist_ok=True)
    for panel in selected_panels:
        manifest=json.loads((data_root/panel/'manifest.json').read_text())
        for name,h in manifest['file_sha256'].items():assert sha(data_root/panel/name)==h,'Dataset tamper '+panel+'/'+name
    integrity={'configuration':cfg,'fit_dates':fit_dates,'held_dates':held_dates,'fold':fold,'panels':selected_panels,'dataset_manifests':{p:sha(data_root/p/'manifest.json') for p in selected_panels},'model_source':sha(Path(__file__)),'features_source':sha(Path(__file__).parent/'gap_features.py'),'runtime_source':sha(Path(__file__).parent/'lgb_runtime.py'),'lightgbm_version':lgb.__version__,'lightgbm_dll_sha256':sha(lgb.basic._LIB._name)}
    stamp=folder/'contract.json'
    if stamp.exists():assert json.loads(stamp.read_text())==integrity,'Stale residual model case'
    else:stamp.write_text(json.dumps(integrity,indent=2)+'\n')
    checkpoints={}
    for target in ['speed','flow_lane']:
        checkpoint=folder/f'{target}_complete.json'
        if checkpoint.exists():
            a=json.loads(checkpoint.read_text());assert a['model_sha256']==sha(folder/f'{target}.txt'),'Model integrity failed';checkpoints[target]=a
    if len(checkpoints)==2 and (folder/'metadata.json').exists():
        metadata=json.loads((folder/'metadata.json').read_text());assert metadata['integrity']==integrity
        for panel,h in metadata['summary_sha256'].items():assert sha(folder/f'{panel}_summaries.npz')==h
        return checkpoints
    summaries=save_summaries(folder,selected_panels,fit_dates)
    X,y,w,columns=matrix(data_root,selected_panels,set(fit_dates),summaries)
    xv,yv,wv,_=matrix(data_root,selected_panels,set(held_dates),summaries) if held_dates else (None,None,None,None)
    for k,target in enumerate(['speed','flow_lane']):
        if target in checkpoints:continue
        used_columns=[c for c in columns if c not in cfg.get('drop_features',[])]; used_index=[columns.index(c) for c in used_columns]
        model_X=X[:,used_index] if len(used_index)!=len(columns) else X
        model_xv=xv[:,used_index] if xv is not None and len(used_index)!=len(columns) else xv
        train=lgb.Dataset(model_X,label=np.ascontiguousarray(y[:,k]),weight=w,feature_name=used_columns,categorical_feature=[c for c in CATEGORICAL if c in used_columns],free_raw_data=True)
        valid=lgb.Dataset(model_xv,label=np.ascontiguousarray(yv[:,k]),weight=wv,reference=train,feature_name=used_columns,categorical_feature=[c for c in CATEGORICAL if c in used_columns],free_raw_data=True) if held_dates else None
        callbacks=[lgb.log_evaluation(50)]
        if valid is not None:callbacks.append(lgb.early_stopping(cfg['early_stopping_rounds'],verbose=True))
        model=lgb.train(cfg['parameters'],train,num_boost_round=cfg.get('num_boost_round_by_target',{}).get(target,cfg['num_boost_round']),valid_sets=[valid] if valid is not None else None,valid_names=['OPT held'] if valid is not None else None,callbacks=callbacks)
        model.save_model(str(folder/f'{target}.txt'))
        result={'best_iteration':int(model.best_iteration or model.current_iteration()),'validation_score':model.best_score,'model_sha256':sha(folder/f'{target}.txt')}
        (folder/f'{target}_complete.json').write_text(json.dumps(result,indent=2)+'\n');checkpoints[target]=result
        del model,train,valid,model_X,model_xv;gc.collect()
    (folder/'metadata.json').write_text(json.dumps({'columns':columns,'fit_dates':fit_dates,'panels':selected_panels,'asm_parameters':cfg['asm_parameters'],'model_cases':checkpoints,'integrity':integrity,'summary_sha256':{p:sha(folder/f'{p}_summaries.npz') for p in selected_panels}},indent=2)+'\n')
    return checkpoints


def predict_residual_arrays(cache,folder,panel_code,asm_values=None):
    assert not cache.gap.any() or getattr(cache,'s_windows',None) is not None,'F2 requires released window context for every gap'
    folder=Path(folder);metadata=json.loads((folder/'metadata.json').read_text());columns=metadata['columns'];panel=cache.meta['panel']
    models={k:lgb.Booster(model_file=str(folder/f'{k}.txt')) for k in ['speed','flow_lane']}
    for k,model in models.items():assert sha(folder/f'{k}.txt')==metadata['model_cases'][k]['model_sha256']
    assert sha(folder/f'{panel}_summaries.npz')==metadata['summary_sha256'][panel]
    summaries=load_summaries(folder,panel);bases=prepare_bases(cache,metadata['asm_parameters'],asm_values=asm_values)
    pv=bases['time'][0].copy();pq=bases['time'][1].copy();st=cache.stations;lanes=st.lanes.to_numpy(np.float32)
    for d in range(len(cache.dates)):
        features,override=day_features(cache,d,bases,panel_code);assert list(features)==columns
        target=cache.target[d];ti,si=np.nonzero(target)
        if not len(ti):continue
        x=np.column_stack([a[target] for a in features.values()]);weekend=np.full(len(ti),int(__import__('datetime').date.fromisoformat(cache.dates[d]).weekday()>=5))
        fill_dynamic(x,columns,si,ti,weekend,override[target],summaries)
        index={k:i for i,k in enumerate(columns)}
        pv[d][target]=x[:,index['base_B2_speed']]+models['speed'].predict(x[:,[columns.index(c) for c in models['speed'].feature_name()]],num_threads=8)
        pq[d][target]=lanes[si]*(x[:,index['base_time_flow_lane']]+models['flow_lane'].predict(x[:,[columns.index(c) for c in models['flow_lane'].feature_name()]],num_threads=8))
    np.clip(pv,3.,120.75,out=pv)
    upper=1.3*st.capacity_vph.to_numpy(float);upper32=upper.astype(np.float32);upper32=np.where(upper32.astype(float)>upper,np.nextafter(upper32,np.float32(-np.inf)),upper32)
    np.maximum(pq,50.,out=pq);np.minimum(pq,upper32,out=pq)
    assert np.isfinite(pv[cache.target]).all() and np.isfinite(pq[cache.target]).all()
    return pv,pq

