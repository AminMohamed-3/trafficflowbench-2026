"""S0419 fixed-model transfer to OWN. Export derived residuals only; raw stays local.
prepare validation: laptop short inference/export. evaluate validation: VM.
Models are the frozen two-panel odd-date OPT fits, never refitted on OWN labels.
"""
import os,sys,json,types,time
from pathlib import Path
import numpy as np
import s0416_density as B
import s0419_occ as O
ROOT=B.ROOT; INP=ROOT/'artifacts/lane_s/0419_own_inputs'; OUT=ROOT/'artifacts/lane_s/0419_own'

def prepare(split):
    import lightgbm as lgb
    from tfb.data import load_cache
    meta=json.loads((B.INP/'metadata.json').read_text()); B.modules(meta)
    import state2_cm as CM, s2stage as S2
    coef=json.loads((ROOT/'artifacts/claude_ref/state2/opt_coef.json').read_text()); cf=(coef['b_free'],coef['b_cong'])
    models=tuple(lgb.Booster(model_file=str(ROOT/f'artifacts/claude_ref/phys3/m368_{n}.txt')) for n in ('clf','gc','gf'))
    tag={'validation':'mar','private':'apr'}[split]; info={}; INP.mkdir(parents=True,exist_ok=True)
    panels=meta['panels'] if '--all' in sys.argv else meta['exported_panels']
    for pn in panels:
        c=load_cache(pn,split,root=ROOT); z=dict(np.load(ROOT/f'artifacts/claude_ref/state5/own_{tag}_s08_{pn}.npz'))
        s=np.load(ROOT/f'artifacts/lane_s/0398_s08/own_{split}/{pn}.npz')
        T=z['target']; gap=z['gap']; vo=z['vobs'].astype(float); qo=z['qobs'].astype(float)
        observed=np.isfinite(vo)&~T&~gap; st=c.stations.sort_values('idx')
        gr,_=CM.common_factor(vo,observed,c.stations)
        rv=np.where(T,CM.apply(s['speed'].astype(float),gr,st.v_cut.to_numpy(float),cf),s['speed'].astype(float))
        hp=S2.hedge_368(models,vo,observed,c.stations,rv,s['flow'].astype(float),T)[1]
        v=z['v13'].astype(float); q=z['q13'].astype(float)
        # All arrays exported are prediction residuals or derived masks. No labels or observations.
        np.savez_compressed(INP/f'{split}_{pn}.npz',dv=v-np.nan_to_num(vo,nan=0.),dq=q-np.nan_to_num(qo,nan=0.),
            hp=hp,target=T,gap=gap,visible=np.isfinite(vo)&np.isfinite(qo))
        rawv=np.asarray(c.speed,float); rawq=np.asarray(c.flow,float)
        assert np.allclose(rawv[T],z['tv'][T],equal_nan=True) and np.allclose(rawq[T],z['tq'][T],equal_nan=True)
        assert np.allclose(rawv[np.isfinite(vo)],vo[np.isfinite(vo)]) and np.allclose(rawq[np.isfinite(qo)],qo[np.isfinite(qo)])
        info[pn]=dict(dates=z['dates'].tolist(),regimes=z['regimes'].tolist(),shape=list(T.shape),target_sha=B.digest(T),
            v_sha=B.digest(v),q_sha=B.digest(q),raw_speed_sha=B.digest(np.asarray(c.speed)),raw_flow_sha=B.digest(np.asarray(c.flow)),
            stations=st.station_id.astype(str).tolist())
        print('prepared OWN',split,pn,flush=True)
    (INP/f'{split}.json').write_text(json.dumps(info,indent=2))

def evaluate(split):
    import lightgbm as lgb
    from tfb.data import load_cache
    meta=json.loads((B.INP/'metadata.json').read_text()); _,pf,ls=B.modules(meta)
    import state5_lib as L
    # Restrict aggregation to declared two-panel screen; each is a distinct family.
    info=json.loads((INP/f'{split}.json').read_text()); OUT.mkdir(parents=True,exist_ok=True)
    L.PANELS=list(info)
    L.GROUPS=[[i for i,pn in enumerate(L.PANELS) if meta['families'][pn]==fam] for fam in sorted({meta['families'][p] for p in L.PANELS})]
    ensemble='--ensemble' in sys.argv
    seeds=[4701,4702,4703] if ensemble else [4701]
    arms=['control','occupancy']+[f'occupancy_{s}' for s in seeds] if ensemble else ['control','occupancy']
    stats={k:{} for k in ['base']+arms}; t0=time.time()
    models={s:{arm:{label:lgb.Booster(model_file=str((O.BASE_OUT if s==4701 else O.BASE_OUT/f'seed{s}')/f'{arm}_{label}.txt'))
        for label in ('v','q')} for arm in ('control','occupancy')} for s in seeds}
    for pn in L.PANELS:
        c=load_cache(pn,split,root=ROOT); z=dict(np.load(INP/f'{split}_{pn}.npz')); inf=info[pn]
        assert c.dates==inf['dates'] and B.digest(c.speed)==inf['raw_speed_sha'] and B.digest(c.flow)==inf['raw_flow_sha']
        assert B.digest(z['target'])==inf['target_sha']; visible=z['visible']; T=z['target']; gap=z['gap']
        tv=np.asarray(c.speed,float); tq=np.asarray(c.flow,float)
        h=types.SimpleNamespace(meta=c.meta,stations=c.stations,dates=c.dates,regimes=np.asarray(inf['regimes']),
            target=T,gap=gap,elig=np.asarray(c.elig,bool),speed=np.where(visible,tv,np.nan),flow=np.where(visible,tq,np.nan),
            occ=np.where(visible,np.asarray(c.occ,float),np.nan))
        assert not (visible&T).any(), 'OWN labels visible'
        v,q,sv,sq=B.reconstruct(h,None,None,z,pf,ls,own=True)
        assert B.digest(sv)==inf['v_sha'] and B.digest(sq)==inf['q_sha']
        X,ix,names=O.features(h,v,q,pf); I=tuple(ix.T); arr={'base':(v,q)}
        import pandas as pd
        st=c.stations.sort_values('idx'); topo=pd.read_csv(ROOT/f'data/raw/kaggle_public/corridors/{pn}/network/lwr_mainline_topology.csv')
        clip=1.15*topo.set_index(topo.link_id.astype(str)).free_flow_speed_kmh.reindex(st.link_id.astype(str)).to_numpy(float)
        pred={}
        for s in seeds:
            for arm in ('control','occupancy'):
                use=np.array([arm=='occupancy' or not k.startswith('occ_') for k in names])
                assert np.array(names)[use].tolist()==models[s][arm]['v'].feature_name()
                pred[s,arm]={k:models[s][arm][k].predict(X[:,use],num_threads=4) for k in ('v','q')}
        for arm in arms:
            kind=arm.split('_')[0]; selected=[int(arm.split('_')[1])] if '_' in arm else seeds
            pp={k:np.mean([pred[s,kind][k] for s in selected],axis=0) for k in ('v','q')}; nv=v.copy(); nq=q.copy()
            nv[I]*=np.exp(.5*np.clip(pp['v'],-.05,.05)); nq[I]*=np.exp(.5*np.clip(pp['q'],-.05,.05))
            nv[I]=np.maximum(3.,np.minimum(nv[I],clip[ix[:,2]])); nq[I]=np.clip(nq[I],50.,1.3*st.capacity_vph.to_numpy(float)[ix[:,2]])
            assert np.array_equal(nv[gap],v[gap],equal_nan=True) and np.array_equal(nq[gap],q[gap],equal_nan=True)
            arr[arm]=(nv,nq)
        zz=dict(tv=tv,tq=tq,target=T,gap=gap,regimes=h.regimes)
        for arm,(nv,nq) in arr.items(): stats[arm][pn]=L.stats(pn,zz,nv,nq,st.lanes.to_numpy(float))
        print(split,pn,'scored',round(time.time()-t0),'seconds',flush=True)
    shares=L.share_array(split)
    res={arm:L.compare(stats['base'],stats[arm],shares) for arm in arms}
    res['occupancy_vs_control']=L.compare(stats['control'],stats['occupancy'],shares)
    res['scope']=f'{len(L.PANELS)}-panel OWN; non-block evidence. Gap predictions remain S13; mix score is diagnostic only, not full K6 gap specialist.'
    if ensemble: res['seed_sd_nonblock']=float(np.std([res[f'occupancy_{s}']['nb']['delta'] for s in seeds],ddof=1))
    suffix='_ens3' if ensemble else ''
    (OUT/f'{split}_{len(L.PANELS)}{suffix}.json').write_text(json.dumps(res,indent=2,default=float)); print('RESULT',json.dumps(res),flush=True)

if __name__=='__main__': {'prepare':prepare,'evaluate':evaluate}[sys.argv[1]](sys.argv[2])
