"""S0419 incremental released occupancy on a K6 non-gap parent."""
import os,sys,json,time
import numpy as np
from scipy.ndimage import convolve1d
import s0416_density as B
import s0418_finalphys as E
ROOT=B.ROOT; BASE_OUT=ROOT/'artifacts/lane_s/0419_occ'; SEED=int(os.environ.get('S0419_SEED','4701'))
OUT=BASE_OUT if SEED==4701 else BASE_OUT/f'seed{SEED}'

def features(h,v,q,pf):
    import state3_feat as F
    T=np.asarray(h.target,bool); gap=np.asarray(h.gap,bool)
    obs=np.isfinite(h.speed)&np.isfinite(h.flow)&np.isfinite(h.occ)&(h.speed>0)&(h.flow>0)&(h.occ>0)&~T&~gap
    # Only released values enter calibration and features; truths are unavailable here.
    lk=np.log(np.maximum(h.flow,1.))-np.log(np.maximum(h.speed,1.))
    lo=np.log(np.maximum(h.occ,1e-6)); cal=np.nanmedian(np.where(obs,lk-lo,np.nan),axis=(0,1))
    occ=lo+cal[None,None,:]; parent=np.log(np.maximum(q,1.))-np.log(np.maximum(v,1.))
    D,n,S=T.shape; di=np.arange(D)[:,None,None]; ti=np.arange(n)[None,:,None]; si=np.arange(S)[None,None,:]
    ib,ia=F.nearest_axis(obs,1); su,sd=F.nearest_axis(obs,2)
    cols={}; st=h.stations.sort_values('idx'); vf=st.free_speed_kmh.to_numpy(float)
    for tag,x in [('qv',lk),('occ',occ)]:
        for name,idx,axis in [('before',ib,1),('after',ia,1),('up',su,2),('down',sd,2)]:
            vals=x[di,np.maximum(idx,0),si] if axis==1 else x[di,ti,np.maximum(idx,0)]
            cols[tag+'_'+name]=np.where(idx>=0,vals-parent,np.nan)
        fre=obs&(h.speed>.9*vf)
        # Residual relative to the neighboring prediction removes spatial demand gradients.
        residual=np.where(fre,x-parent,0.)
        for width in (3,12):
            ker=np.ones(2*width+1); ker[width]=0
            den=convolve1d(fre.astype(float),ker,axis=2,mode='constant')
            num=convolve1d(residual,ker,axis=2,mode='constant')
            cols[f'{tag}_local{width}']=np.where(den>0,num/np.maximum(den,1),np.nan)
    cols.update(vrel=v/vf,qrel=q/st.capacity_vph.to_numpy(float),lk=parent,
        before_dist=np.where(ib>=0,ti-ib,99),after_dist=np.where(ia>=0,ia-ti,99),
        up_dist=np.where(su>=0,si-su,99),down_dist=np.where(sd>=0,sd-si,99))
    run,pos=pf.runinfo(T); cols.update(run=run,pos=pos)
    m=T&~gap&np.isfinite(v)&np.isfinite(q); ix=np.argwhere(m); names=list(cols)
    X=np.stack([np.broadcast_to(cols[k],T.shape)[m] for k in names],1).astype(np.float32)
    return X,ix,names

def main():
    import lightgbm as lgb
    from tfb.data import load_cache
    from tfb.lane_s.evaluate_baselines import subset_cache,raw_truth
    meta=json.loads((B.INP/'metadata.json').read_text()); fs,pf,ls=B.modules(meta)
    OUT.mkdir(parents=True,exist_ok=True); data={}; t0=time.time()
    for pn in meta['exported_panels']:
        c=load_cache(pn,'train',root=ROOT); h=subset_cache(c,[c.dates.index(d) for d in meta['OPT_held_dates']]); del c
        tv,tq=raw_truth(h,ROOT); z=np.load(B.INP/f'{pn}.npz')
        assert B.digest(h.target)==meta['panels_info'][pn]['target_sha']
        v,q,sv,sq=B.reconstruct(h,tv,tq,z,pf,ls)
        assert B.digest(sv)==meta['panels_info'][pn]['v_sha'] and B.digest(sq)==meta['panels_info'][pn]['q_sha']
        X,ix,names=features(h,v,q,pf); I=tuple(ix.T)
        data[pn]=dict(h=h,tv=tv,tq=tq,v=v,q=q,X=X,ix=ix,yv=np.log(tv[I]/v[I]),yq=np.log(tq[I]/q[I]))
        print(pn,'features',X.shape,'seconds',round(time.time()-t0),flush=True)
    prm=dict(objective='huber',learning_rate=.05,num_leaves=31,min_data_in_leaf=500,
        feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambda_l2=10.,num_threads=4,verbosity=-1,seed=SEED)
    comps={pn:{'base':fs.day_components(d['h'],d['tv'],d['tq'],d['v'],d['q'])} for pn,d in data.items()}
    imps={}
    for arm in ('control','occupancy'):
        use=np.array([arm=='occupancy' or not k.startswith('occ_') for k in names])
        X=np.concatenate([d['X'][d['ix'][:,0]%2==1][:,use] for d in data.values()])
        preds={pn:{} for pn in data}
        for label,alpha in [('q',.03),('v',.01)]:
            y=np.concatenate([d['y'+label][d['ix'][:,0]%2==1] for d in data.values()])
            model=lgb.train(dict(prm,alpha=alpha),lgb.Dataset(X,y,feature_name=np.array(names)[use].tolist()),200)
            model.save_model(str(OUT/f'{arm}_{label}.txt'))
            imps[arm+'_'+label]=dict(zip(np.array(names)[use].tolist(),model.feature_importance(importance_type='gain').tolist()))
            for pn,d in data.items(): preds[pn][label]=model.predict(d['X'][:,use],num_threads=4)
        for pn,d in data.items():
            nv=d['v'].copy(); nq=d['q'].copy(); I=tuple(d['ix'].T)
            nv[I]*=np.exp(.5*np.clip(preds[pn]['v'],-.05,.05)); nq[I]*=np.exp(.5*np.clip(preds[pn]['q'],-.05,.05))
            # Existing topology generator cap, reproduced from the reference utility.
            import pandas as pd
            st=d['h'].stations.sort_values('idx'); topo=pd.read_csv(ROOT/f'data/raw/kaggle_public/corridors/{pn}/network/lwr_mainline_topology.csv')
            clip=1.15*topo.set_index(topo.link_id.astype(str)).free_flow_speed_kmh.reindex(st.link_id.astype(str)).to_numpy(float)
            nv[I]=np.maximum(3.,np.minimum(nv[I],clip[d['ix'][:,2]])); nq[I]=np.clip(nq[I],50.,1.3*st.capacity_vph.to_numpy(float)[d['ix'][:,2]])
            comps[pn][arm]=fs.day_components(d['h'],d['tv'],d['tq'],nv,nq)
        print('fitted',arm,'seconds',round(time.time()-t0),flush=True)
    D=len(meta['OPT_held_dates']); held=np.flatnonzero(np.arange(D)%2==0)
    res={'vs_K6':E.bootstrap(fs,comps,held),'feature_gain':imps}
    against={pn:{'base':c['control'],'occupancy':c['occupancy']} for pn,c in comps.items()}
    res['vs_control']=E.bootstrap(fs,against,held)
    for pn,c in comps.items(): np.savez_compressed(OUT/f'stats_{pn}.npz',regimes=np.asarray(c['base'][1]),**{k:x[0] for k,x in c.items()})
    (OUT/'screen.json').write_text(json.dumps(res,indent=2)); print('RESULT',json.dumps({k:v for k,v in res.items() if k!='feature_gain'}),flush=True)

if __name__=='__main__': main()
