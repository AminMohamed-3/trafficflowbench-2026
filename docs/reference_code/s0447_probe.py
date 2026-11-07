"""Consolidated S0424+S0426+S0447 delta on exact K6, gated before serving."""
import os,sys,json,shutil
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ.setdefault(k,'3')
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import s0426_probe as Q
import s0447_joint_flow as F
P=Q.P; OUT=P.ROOT/'artifacts/lane_s/0447_probe'; BASE=Q.OUT; loaded={}

def stage(c,parent,pf):
    from tfb.data import indices
    st=c.stations.sort_values('idx'); I=indices(parent,dict(zip(c.dates,range(len(c.dates)))),dict(zip(st.link_id.astype(str),st.idx)),'link_id')
    v=np.asarray(c.speed,float).copy(); q=np.asarray(c.flow,float).copy()
    v[I]=parent.speed_kmh.to_numpy(float); q[I]=parent.flow_vph.to_numpy(float)
    T=np.zeros(c.target.shape,bool); T[I]=True; assert np.array_equal(T,c.target)
    X,ix,names,gate=F.features(c,v,q,pf); J=tuple(ix[gate].T); split=c.meta['split']
    if split not in loaded: loaded[split]=[lgb.Booster(model_file=str(F.OUT/split/f'seed{s}/joint.txt')) for s in (4701,4702,4703)]
    assert all(m.feature_name()==names for m in loaded[split])
    pred=np.mean([m.predict(X,num_threads=3) for m in loaded[split]],axis=0)
    nq=q.copy(); nq[J]=np.clip(q[J]*np.exp(.5*np.clip(pred[gate],-.01,.01)),50.,1.3*st.capacity_vph.to_numpy(float)[ix[gate,2]])
    assert np.array_equal(nq[c.gap],q[c.gap],equal_nan=True)
    changed=nq[I]!=q[I]; assert not c.gap[I][changed].any()
    delta=parent.loc[changed,['submission_id','speed_kmh']].copy(); delta['flow_vph']=nq[I][changed]
    return delta,dict(rows=len(parent),eligible_rows=int(gate.sum()),changed_rows=int(changed.sum()),features_sha=P.B.digest(X),
        gate_sha=P.B.digest(gate),max_abs_log_flow_change=float(np.max(np.abs(np.log(nq[J]/q[J])))) if gate.any() else 0.)

def predict(c,parent,pf,global_models):
    first,a=Q.predict(c,parent,pf,global_models); cols=['speed_kmh','flow_vph']
    updated=parent.set_index('submission_id',drop=False).copy(); updated.loc[first.submission_id,cols]=first[cols].to_numpy()
    second,b=stage(c,updated.reset_index(drop=True),pf); updated.loc[second.submission_id,cols]=second[cols].to_numpy()
    final=updated.reset_index(drop=True); assert np.array_equal(final.submission_id,parent.submission_id)
    changed=np.any(final[cols].to_numpy()!=parent[cols].to_numpy(),axis=1)
    return final.loc[changed,['submission_id']+cols].copy(),dict(rows=len(parent),changed_rows=int(changed.sum()),occupancy_stages=a,joint_flow_stage=b)

def contract():
    meta,_,pf=F.D.setup(); pn='D12_I405_S'; split='validation'
    c,_,_,_,_,_=F.D.layer(pn,split); z=dict(np.load(F.OUT/split/f'{pn}.npz'))
    ix=z['ix']; I=tuple(ix.T); v=np.nan_to_num(z['v'],nan=3.); q=np.nan_to_num(z['q'],nan=50.)
    X,oi,names,gate=F.features(c,v,q,pf)
    assert np.array_equal(oi,ix) and np.array_equal(X,z['X'],equal_nan=True) and np.array_equal(gate,z['gate'])
    allix=np.argwhere(c.target); AI=tuple(allix.T); st=c.stations.sort_values('idx')
    ts=pd.to_datetime(np.asarray(c.dates)[allix[:,0]],utc=True)+pd.to_timedelta(5*allix[:,1],unit='m')
    frame=pd.DataFrame(dict(submission_id=np.arange(len(allix)),timestamp=ts.strftime('%Y-%m-%dT%H:%M:%SZ'),
        link_id=st.link_id.astype(str).to_numpy()[allix[:,2]],speed_kmh=v[AI],flow_vph=q[AI]))
    delta,stats=stage(c,frame,pf); actual=frame.copy(); actual.loc[delta.submission_id,['speed_kmh','flow_vph']]=delta[['speed_kmh','flow_vph']].to_numpy()
    pp=np.mean([m.predict(X,num_threads=3) for m in loaded[split]],axis=0); J=tuple(ix[gate].T)
    q[J]=np.clip(q[J]*np.exp(.5*np.clip(pp[gate],-.01,.01)),50.,1.3*z['cap'][ix[gate,2]])
    err=float(np.max(np.abs(actual[['speed_kmh','flow_vph']].to_numpy()-np.stack([v[AI],q[AI]],1))))
    assert err<1e-9
    (OUT/'serving_contract.json').write_text(json.dumps(dict(panel=pn,split=split,rows=len(ix),feature_parity_exact=True,
        evaluation_vs_serving_max_abs_diff=err,stage=stats),indent=2)); print('SERVING CONTRACT',err,flush=True)

def main():
    for split in ('validation','private','opt'): F.gate(split)
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('parent_state.parquet','parent.json'):
        if not (OUT/name).exists(): os.link(BASE/name,OUT/name)
    P.OUT=OUT; P.predict=predict
    if sys.argv[1]=='contract': return contract()
    if sys.argv[1]=='validate': return P.validate_delta()
    P.build()
    if '--reproduce' in sys.argv or '--sample' in sys.argv: return
    mf=json.loads((OUT/'manifest.json').read_text())
    mf['recipe']='S0424 canonical occupancy + S0426 split-local occupancy + S0447 joint demand-block/clock flow correction, equal3 odd OWN fits; final values consolidated on K6.'
    mf['increment_reference']='S0426; this delta supersedes S0424 and S0426, never apply them again afterward.'
    for split in ('validation','private'):
        for seed in (4701,4702,4703):
            paths=[Q.LOCAL/split/f'seed{seed}'/f'occupancy_{k}.txt' for k in ('v','q')]+[F.OUT/split/f'seed{seed}/joint.txt']
            for p in paths: mf['models'][str(p.relative_to(P.ROOT))]=P.sha(p)
    for name in ('s0447_probe.py','s0447_joint_flow.py','s0445_density_loss.py','s0426_probe.py','s0426_local.py',
                 'phys4_lsa.py','phys4_feat.py','state3_feat.py','gen_blk.py',
                 's0447_cumulative.py','s0418_finalphys.py','state5_lib.py','flow_screen.py'):
        p=Path(__file__).with_name(name); mf['sources'][name]=P.sha(p)
    (OUT/'source').mkdir(exist_ok=True)
    for name in mf['sources']: shutil.copy2(Path(__file__).with_name(name),OUT/'source'/name)
    (OUT/'manifest.json').write_text(json.dumps(mf,indent=2))

if __name__=='__main__': main()
