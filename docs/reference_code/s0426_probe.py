"""Candidate builder for S0424 plus S0426 on exact K6; run only after physics passes."""
import sys,json,os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key,'3')
from pathlib import Path
import numpy as np
import lightgbm as lgb
import s0424_probe as P
BASE=P.OUT; OUT=P.ROOT/'artifacts/lane_s/0426_probe'; original_predict=P.predict
LOCAL=P.ROOT/'artifacts/lane_s/0426_local'; loaded={}

class LocalEnsemble:
    def __init__(self,models):
        self.models=models
        assert all(m.feature_name()==models[0].feature_name() for m in models)
    def feature_name(self): return self.models[0].feature_name()
    def predict(self,X,num_threads=3):
        return np.clip(np.mean([m.predict(X,num_threads=num_threads) for m in self.models],axis=0),-.01,.01)

def predict(c,parent,pf,global_models):
    first,a=original_predict(c,parent,pf,global_models)
    updated=parent.set_index('submission_id',drop=False).copy()
    cols=['speed_kmh','flow_vph']; updated.loc[first.submission_id,cols]=first[cols].to_numpy()
    split=c.meta['split']
    if split not in loaded:
        loaded[split]={k:[LocalEnsemble([lgb.Booster(model_file=str(LOCAL/split/f'seed{s}'/f'occupancy_{k}.txt'))
            for s in (4701,4702,4703)])] for k in ('v','q')}
    second,b=original_predict(c,updated.reset_index(drop=True),pf,loaded[split])
    updated.loc[second.submission_id,cols]=second[cols].to_numpy()
    final=updated.reset_index(drop=True); assert np.array_equal(final.submission_id,parent.submission_id)
    changed=np.any(final[cols].to_numpy()!=parent[cols].to_numpy(),axis=1)
    delta=final.loc[changed,['submission_id']+cols].copy()
    stats=dict(rows=len(parent),changed_rows=len(delta),global_stage=a,local_stage=b,
        max_log_speed_change=float(np.max(np.abs(np.log(final.speed_kmh.to_numpy()/parent.speed_kmh.to_numpy())))),
        max_log_flow_change=float(np.max(np.abs(np.log(final.flow_vph.to_numpy()/parent.flow_vph.to_numpy())))))
    return delta,stats

def contract():
    import types,pandas as pd
    from tfb.data import load_cache
    meta=json.loads((P.B.INP/'metadata.json').read_text()); _,pf,_=P.B.modules(meta)
    pn='D12_I405_S'; split='validation'; c=load_cache(pn,split,root=P.ROOT)
    z=np.load(P.ROOT/f'artifacts/lane_s/0419_own_inputs/{split}_{pn}.npz')
    a=np.load(LOCAL/split/f'{pn}.npz'); T=z['target']; visible=z['visible']; ix=a['ix']; I=tuple(ix.T)
    h=types.SimpleNamespace(meta=c.meta,dates=c.dates,stations=c.stations,target=T,gap=z['gap'],
        speed=np.where(visible,np.asarray(c.speed,float),np.nan),flow=np.where(visible,np.asarray(c.flow,float),np.nan),
        occ=np.where(visible,np.asarray(c.occ,float),np.nan))
    v=np.full(T.shape,3.); q=np.full(T.shape,50.); v[I]=a['v']; q[I]=a['q']
    X,xx,names=P.C.features(h,v,q,pf)
    assert np.array_equal(ix,xx) and np.array_equal(X,a['X'],equal_nan=True)
    allix=np.argwhere(T); AI=tuple(allix.T); st=c.stations.sort_values('idx')
    ts=pd.to_datetime(np.asarray(c.dates)[allix[:,0]],utc=True)+pd.to_timedelta(5*allix[:,1],unit='m')
    frame=pd.DataFrame(dict(submission_id=np.arange(len(allix)),timestamp=ts.strftime('%Y-%m-%dT%H:%M:%SZ'),
        link_id=st.link_id.astype(str).to_numpy()[allix[:,2]],speed_kmh=v[AI],flow_vph=q[AI]))
    ms={k:[lgb.Booster(model_file=str(LOCAL/split/f'seed{s}'/f'occupancy_{k}.txt')) for s in (4701,4702,4703)] for k in ('v','q')}
    result,_=original_predict(h,frame,pf,{k:[LocalEnsemble(mm)] for k,mm in ms.items()})
    actual=frame.copy(); actual.loc[result.submission_id,['speed_kmh','flow_vph']]=result[['speed_kmh','flow_vph']].to_numpy()
    pp={k:np.mean([m.predict(X,num_threads=3) for m in ms[k]],axis=0) for k in ('v','q')}
    v[I]=np.clip(a['v']*np.exp(.5*np.clip(pp['v'],-.01,.01)),3.,a['clip'][ix[:,2]])
    q[I]=np.clip(a['q']*np.exp(.5*np.clip(pp['q'],-.01,.01)),50.,1.3*a['cap'][ix[:,2]])
    error=float(np.max(np.abs(actual[['speed_kmh','flow_vph']].to_numpy()-np.stack([v[AI],q[AI]],1))))
    assert error<1e-9
    receipt=dict(view='March OWN D12_I405_S',nongap_rows=len(ix),feature_parity_exact=True,
        evaluation_vs_serving_max_abs_diff=error,scope='Same observed view and target-only parent; adapter clipping/averaging checked against evaluated recipe.')
    (OUT/'serving_contract.json').write_text(json.dumps(receipt,indent=2)); print(json.dumps(receipt),flush=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('parent_state.parquet','parent.json'):
        if not (OUT/name).exists(): os.link(BASE/name,OUT/name)
    P.OUT=OUT; P.predict=predict
    if sys.argv[1]=='contract': return contract()
    if sys.argv[1]=='validate': return P.validate_delta()
    P.build()
    if '--reproduce' in sys.argv or '--sample' in sys.argv: return
    mf=json.loads((OUT/'manifest.json').read_text())
    mf['recipe']='S0424 canonical occupancy followed by split-local S0426: odd OWN date fits, seeds4701/2/3 equal, half mean log residual clipped +/-0.01. Consolidated final values on K6.'
    mf['increment_reference']='S0424; base K6 retained as the named delta parent. All final non-gap values include both stages.'
    for split in ('validation','private'):
        for seed in (4701,4702,4703):
            for k in ('v','q'):
                p=LOCAL/split/f'seed{seed}'/f'occupancy_{k}.txt'
                mf['models'][str(p.relative_to(P.ROOT))]=P.sha(p)
    for p in (Path(__file__),Path(__file__).with_name('s0426_local.py')): mf['sources'][p.name]=P.sha(p)
    (OUT/'manifest.json').write_text(json.dumps(mf,indent=2))

if __name__=='__main__': main()
