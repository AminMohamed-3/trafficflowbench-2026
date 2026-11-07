"""S0419: all-panel other-date stage-3 parent, using frozen occupancy correctors.
prepare exports only residual predictions from existing S13/F13 (no raw data).
build reconstructs F14/S14 and features on VM. score fits S3L on odd dates of F14,
applies to even S14, then compares frozen occupancy/control. Upstream historical
cross-fit design is retained; this specifically removes stage-3 in-sample optimism.
"""
import os,sys,json,time,shutil
from pathlib import Path
import numpy as np
import s0416_density as B, s0419_occ as O, s0418_finalphys as E
ROOT=B.ROOT; INP=ROOT/'artifacts/lane_s/0419_xparent_inputs'; OUT=ROOT/'artifacts/lane_s/0419_xparent'

def prepare():
    from tfb.data import load_cache
    meta=json.loads((B.INP/'metadata.json').read_text()); meta['exported_panels']=meta['panels']; meta['panels_info']={}
    INP.mkdir(parents=True,exist_ok=True)
    for pn in meta['panels']:
        c=load_cache(pn,'train',root=ROOT); ii=[c.dates.index(d) for d in meta['OPT_held_dates']]
        ov=np.asarray(c.speed[ii],float); oq=np.asarray(c.flow[ii],float)
        S=np.load(ROOT/f'artifacts/claude_ref/asm/xfit_final_{pn}.npz'); hpS=np.load(ROOT/f'artifacts/claude_ref/asm/xfit_s1_{pn}.npz')['hp']
        F=np.load(ROOT/f'artifacts/claude_ref/state5/opt_{pn}.npz'); hpF=np.load(ROOT/f'artifacts/claude_ref/polish/c12_opt_{pn}.npz')['p']
        v=S['v'].astype(float); q=S['q'].astype(float)
        np.savez_compressed(INP/f'{pn}.npz',dv=v-np.nan_to_num(ov,nan=0.),dq=q-np.nan_to_num(oq,nan=0.),hp=hpS,
            Fdv=F['v13'].astype(float)-np.nan_to_num(ov,nan=0.),Fdq=F['q13'].astype(float)-np.nan_to_num(oq,nan=0.),Fhp=hpF)
        meta['panels_info'][pn]=dict(target_sha=B.digest(c.target[ii]),v_sha=B.digest(v),q_sha=B.digest(q))
        print('prepared OPT',pn,round((INP/f'{pn}.npz').stat().st_size/2**20,1),'MiB',flush=True)
    (INP/'metadata.json').write_text(json.dumps(meta,indent=2))

def build():
    from tfb.data import load_cache
    from tfb.lane_s.evaluate_baselines import subset_cache,raw_truth
    meta=json.loads((INP/'metadata.json').read_text()); fs,pf,ls=B.modules(meta); OUT.mkdir(parents=True,exist_ok=True)
    for pn in meta['panels']:
        if (OUT/f'parent_{pn}.npz').exists():
            print('reuse complete parent',pn,flush=True); continue
        c=load_cache(pn,'train',root=ROOT); h=subset_cache(c,[c.dates.index(d) for d in meta['OPT_held_dates']]); del c
        tv,tq=raw_truth(h,ROOT); z=dict(np.load(INP/f'{pn}.npz'))
        assert B.digest(h.target)==meta['panels_info'][pn]['target_sha']
        # Preserve identical original B.INP model and names; only prediction inputs changed.
        f=B.reconstruct(h,tv,tq,{'dv':z['Fdv'],'dq':z['Fdq'],'hp':z['Fhp']},pf,ls,features_only=True)
        s=B.reconstruct(h,tv,tq,z,pf,ls,features_only=True); I=tuple(f['ix'].T)
        assert np.array_equal(f['ix'],s['ix'])
        np.savez_compressed(OUT/f'parent_{pn}.building.npz',FX=f['X'],SX=s['X'],ix=f['ix'],
            Fyq=np.log(tq[I]/f['q14'][I]).astype(np.float32),Fyv=np.log(tv[I]/f['v14'][I]).astype(np.float32),
            Sv=s['v14'].astype(np.float32),Sq=s['q14'].astype(np.float32),clip=s['clip'])
        (OUT/f'parent_{pn}.building.npz').replace(OUT/f'parent_{pn}.npz')
        print('built',pn,flush=True)

def score():
    import lightgbm as lgb
    from tfb.data import load_cache
    from tfb.lane_s.evaluate_baselines import subset_cache,raw_truth
    meta=json.loads((INP/'metadata.json').read_text()); fs,pf,ls=B.modules(meta); rng=np.random.default_rng(1)
    Z={pn:dict(np.load(OUT/f'parent_{pn}.npz')) for pn in meta['panels']}
    count=sum(np.sum(d['ix'][:,0]%2==1) for d in Z.values()); prob=min(1.,3_000_000/count)
    sel={pn:np.flatnonzero((d['ix'][:,0]%2==1)&(rng.random(len(d['ix']))<prob)) for pn,d in Z.items()}
    X=np.concatenate([d['FX'][sel[pn]] for pn,d in Z.items()]); parent_models={}
    for label,alpha in [('q',.03),('v',.01)]:
        y=np.concatenate([d['Fy'+label][sel[pn]] for pn,d in Z.items()])
        prm=dict(objective='huber',alpha=alpha,learning_rate=.05,num_leaves=127,min_data_in_leaf=1000,
            feature_fraction=.8,bagging_fraction=.7,bagging_freq=1,lambda_l2=1.,num_threads=4,verbosity=-1,seed=0)
        m=lgb.train(prm,lgb.Dataset(X,y),600); m.save_model(str(OUT/f'S3L_odd_{label}.txt')); parent_models[label]=m
        print('fit other-date S3L',label,'rows',len(y),flush=True)
    del X
    models={a:{k:[lgb.Booster(model_file=str((O.BASE_OUT if s==4701 else O.BASE_OUT/f'seed{s}')/f'{a}_{k}.txt'))
        for s in (4701,4702,4703)] for k in ('v','q')} for a in ('control','occupancy')}
    comps={}
    for pn in meta['panels']:
        d=Z.pop(pn); c=load_cache(pn,'train',root=ROOT); h=subset_cache(c,[c.dates.index(x) for x in meta['OPT_held_dates']]); del c
        tv,tq=raw_truth(h,ROOT); I=tuple(d['ix'].T); v=d['Sv'].astype(float); q=d['Sq'].astype(float)
        v[I]*=np.exp(parent_models['v'].predict(d['SX'],num_threads=4)); q[I]*=np.exp(parent_models['q'].predict(d['SX'],num_threads=4))
        v[I]=np.minimum(v[I],d['clip'][d['ix'][:,2]])
        XX,ix,names=O.features(h,v,q,pf); J=tuple(ix.T)
        comps[pn]={'base':fs.day_components(h,tv,tq,v,q)}
        for arm in ('control','occupancy'):
            use=np.array([arm=='occupancy' or not x.startswith('occ_') for x in names]); nv=v.copy(); nq=q.copy()
            pv=np.mean([m.predict(XX[:,use],num_threads=4) for m in models[arm]['v']],axis=0)
            pq=np.mean([m.predict(XX[:,use],num_threads=4) for m in models[arm]['q']],axis=0)
            nv[J]*=np.exp(.5*np.clip(pv,-.05,.05)); nq[J]*=np.exp(.5*np.clip(pq,-.05,.05))
            nv[J]=np.clip(nv[J],3.,d['clip'][ix[:,2]]); nq[J]=np.clip(nq[J],50.,1.3*h.stations.sort_values('idx').capacity_vph.to_numpy(float)[ix[:,2]])
            comps[pn][arm]=fs.day_components(h,tv,tq,nv,nq)
        np.savez_compressed(OUT/f'stats_{pn}.npz',regimes=np.asarray(comps[pn]['base'][1]),**{k:x[0] for k,x in comps[pn].items()})
        print('scored',pn,flush=True)
    D=len(meta['OPT_held_dates']); held=np.flatnonzero(np.arange(D)%2==0)
    result=E.bootstrap(fs,comps,held); result['scope']='Stage3L fitted only odd dates; scored even dates. Upstream retained historical cross-fit parents, frozen two-panel occupancy correctors.'
    (OUT/'result.json').write_text(json.dumps(result,indent=2)); print('RESULT',json.dumps(result),flush=True)

if __name__=='__main__': {'prepare':prepare,'build':build,'score':score}[sys.argv[1]]()
