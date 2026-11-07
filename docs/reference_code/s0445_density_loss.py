"""S0445: native-boundary-weighted density median, incremental on S0426."""
import os,sys,json,types
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ.setdefault(k,'4')
import numpy as np
import s0416_density as B
import s0424_canonical as C
import s0419_own as W
import s0418_finalphys as E
ROOT=B.ROOT; OUT=ROOT/'artifacts/lane_s/0445_density'

def setup():
    from tfb.data import load_cache
    meta=json.loads((B.INP/'metadata.json').read_text()); fs,pf,_=B.modules(meta)
    OUT.mkdir(parents=True,exist_ok=True); return meta,fs,pf

def boundary(h,ix):
    ob=~np.asarray(h.target,bool)&~np.asarray(h.gap,bool)&np.asarray(h.elig,bool)&np.isfinite(h.speed)&np.isfinite(h.flow)
    edge=np.zeros(ob.shape,np.int8); edge[:,1:]+=ob[:,:-1]; edge[:,:-1]+=ob[:,1:]
    edge*=np.asarray(h.elig,bool); return edge[tuple(ix.T)]

def layer(pn,split):
    from tfb.data import load_cache
    from tfb.lane_s.evaluate_baselines import subset_cache,raw_truth
    c=load_cache(pn,'train' if split=='opt' else split,root=ROOT)
    if split=='opt':
        meta=json.loads((B.INP/'metadata.json').read_text()); h=subset_cache(c,[c.dates.index(d) for d in meta['OPT_held_dates']])
        tv,tq=raw_truth(h,ROOT); z=dict(np.load(ROOT/f'artifacts/lane_s/0426_opt/{pn}.npz'))
        v=z['v'].copy(); q=z['q'].copy(); return h,tv,tq,z,v,q
    info=json.loads((W.INP/f'{split}.json').read_text())[pn]; r=dict(np.load(W.INP/f'{split}_{pn}.npz'))
    h=subset_cache(c,list(range(len(c.dates)))); h.target=r['target']; h.gap=r['gap']; h.regimes=np.asarray(info['regimes'])
    tv=np.asarray(c.speed,float); tq=np.asarray(c.flow,float)
    h.speed=np.where(r['visible'],tv,np.nan); h.flow=np.where(r['visible'],tq,np.nan); h.occ=np.where(r['visible'],c.occ,np.nan)
    z=dict(np.load(ROOT/f'artifacts/lane_s/0426_local/{split}/{pn}.npz')); I=tuple(z['ix'].T)
    v=tv.copy(); q=tq.copy(); v[I]=z['v']; q[I]=z['q']; return h,tv,tq,z,v,q

def selected(h,days):
    # PanelCache subsets cannot be subset_cache'd again; slice explicit day arrays.
    out=types.SimpleNamespace(**vars(h)); D=len(h.dates)
    for k,val in vars(h).items():
        if isinstance(val,np.ndarray) and val.ndim>=1 and val.shape[0]==D and k not in ('stations','ramps'):
            setattr(out,k,val[days])
    out.dates=[h.dates[d] for d in days]; out.regimes=np.asarray(h.regimes)[days]
    return out

def proxy_view(h):
    hh=types.SimpleNamespace(**vars(h)); hh.target=np.asarray(h.target,bool)&~np.asarray(h.gap,bool)
    hh.elig=np.asarray(h.elig,bool)&~np.asarray(h.gap,bool); return hh

def build(split):
    import lightgbm as lgb
    meta,fs,pf=setup(); import state5_lib as L
    dest=OUT/split; dest.mkdir(exist_ok=True)
    modeldir=ROOT/'artifacts/lane_s/0426_opt' if split=='opt' else ROOT/f'artifacts/lane_s/0426_local/{split}'
    ms={k:[lgb.Booster(model_file=str(modeldir/(f'occupancy_{k}_{s}.txt' if split=='opt' else f'seed{s}/occupancy_{k}.txt'))) for s in (4701,4702,4703)] for k in ('v','q')}
    for pn in meta['panels']:
        path=dest/f'{pn}.npz'
        if path.exists(): print('reuse',split,pn,flush=True); continue
        h,tv,tq,z,v,q=layer(pn,split); ix=z['ix']; I=tuple(ix.T); st=h.stations.sort_values('idx'); cap=st.capacity_vph.to_numpy(float)
        for k,out,low,upper in [('v',v,3.,z['clip']),('q',q,50.,1.3*cap)]:
            pp=np.mean([m.predict(z['X'],num_threads=4) for m in ms[k]],axis=0)
            out[I]=np.clip(out[I]*np.exp(.5*np.clip(pp,-.01,.01)),low,upper[ix[:,2]])
        X,oi,names=C.features(h,v,q,pf); assert np.array_equal(ix,oi)
        edges=boundary(h,ix); gate=edges>0; yy=np.log(tq[I]/tv[I])-np.log(q[I]/v[I])
        hh=proxy_view(h); comp=fs.day_components(hh,tv,tq,v,q)[0]
        weights=np.zeros(len(ix)); odd=ix[:,0]%2==1
        length=st.length_km.to_numpy(float); count=q[I]/v[I]*length[ix[:,2]]
        for r in ('R1','R2','R3'):
            ds=(np.arange(len(h.dates))%2==1)&(h.regimes==r); m=odd&(h.regimes[ix[:,0]]==r)&gate
            den=comp[ds,6].sum()
            if m.any(): assert den>0; weights[m]=count[m]*edges[m]/den
        stats=z['stats'].copy() if split!='opt' else np.empty(0)
        if split!='opt':
            for d,r in enumerate(h.regimes):
                m=ix[:,0]==d; rr=L.REG.index(str(r)); assert int(stats[rr,d,0,0])==int(m.sum())
                stats[rr,d,0,1]=np.sum((v[I][m]-tv[I][m])**2)
                stats[rr,d,0,2]=np.sum(((q[I][m]-tq[I][m])/z['lanes'][ix[m,2]])**2)
        else:
            prior=np.load(ROOT/f'artifacts/lane_s/0426_opt/stats_{pn}.npz')['occupancy']
            assert np.allclose(comp,prior,rtol=1e-10,atol=1e-7)
        if pn==meta['panels'][0]:
            vv=v.copy(); qq=q.copy(); vv[~h.target]=123456.; qq[~h.target]=987654.
            xp,ip,_=C.features(h,vv,qq,pf); assert np.array_equal(X,xp,equal_nan=True)
            h64=types.SimpleNamespace(**vars(h)); h64.speed=np.asarray(h.speed,float); h64.flow=np.asarray(h.flow,float); h64.occ=np.asarray(h.occ,float)
            xp,ip,_=C.features(h64,v,q,pf); assert np.array_equal(X,xp,equal_nan=True)
            (dest/'contract.json').write_text(json.dumps(dict(non_target_parent_invariant=True,input_precision_invariant=True)))
        np.savez_compressed(dest/f'{pn}.building.npz',X=X,ix=ix,names=np.asarray(names),gate=gate,y=yy,w=weights,
            v=v,q=q,cap=cap,stats=stats,regimes=h.regimes)
        (dest/f'{pn}.building.npz').replace(path); print('built',split,pn,int((gate&odd).sum()),flush=True)

def score(split,seed):
    import lightgbm as lgb
    meta,fs,pf=setup(); import state5_lib as L
    dest=OUT/split; modeldir=dest/(f'seed{seed}' if seed else 'ensemble3'); modeldir.mkdir(exist_ok=True)
    Z={pn:dict(np.load(dest/f'{pn}.npz')) for pn in meta['panels']}; names=next(iter(Z.values()))['names'].tolist()
    models={}; sel={pn:z['gate']&(z['ix'][:,0]%2==1) for pn,z in Z.items()}
    if seed:
        X=np.concatenate([z['X'][sel[pn]] for pn,z in Z.items()]); y=np.concatenate([z['y'][sel[pn]] for pn,z in Z.items()])
        w=np.concatenate([z['w'][sel[pn]] for pn,z in Z.items()]); w/=w.mean()
        for arm in ('median','huber'):
            prm=dict(objective='quantile' if arm=='median' else 'huber',alpha=.5 if arm=='median' else .03,
                learning_rate=.05,num_leaves=31,min_data_in_leaf=500,lambda_l2=10.,feature_fraction=.8,bagging_fraction=.8,
                bagging_freq=1,num_threads=4,verbosity=-1,seed=seed)
            m=lgb.train(prm,lgb.Dataset(X,y,weight=w if arm=='median' else None,feature_name=names),200)
            m.save_model(str(modeldir/f'{arm}.txt')); models[arm]=[m]; print('fit',split,seed,arm,flush=True)
        del X,y,w
    else:
        models={a:[lgb.Booster(model_file=str(dest/f'seed{s}/{a}.txt')) for s in (4701,4702,4703)] for a in ('median','huber')}
    comps={}; bases={}; states={a:{} for a in models}
    for pn in meta['panels']:
        z=Z.pop(pn); h,tv,tq,_,_,_=layer(pn,split); hh=proxy_view(h); even=np.arange(len(h.dates))[::2]; he=selected(hh,even)
        ix=z['ix']; I=tuple(ix.T); v=z['v']; q=z['q']; g=z['gate']; J=tuple(ix[g].T)
        comps[pn]={'base':fs.day_components(he,tv[even],tq[even],v[even],q[even])}
        if split!='opt': bases[pn]=z['stats'][:,::2]
        for arm,ms in models.items():
            pp=np.mean([m.predict(z['X'],num_threads=4) for m in ms],axis=0); nq=q.copy()
            nq[J]=np.clip(q[J]*np.exp(.5*np.clip(pp[g],-.01,.01)),50.,1.3*z['cap'][ix[g,2]])
            comps[pn][arm]=fs.day_components(he,tv[even],tq[even],v[even],nq[even])
            if split!='opt':
                ss=z['stats'].copy()
                for d,r in enumerate(h.regimes):
                    m=ix[:,0]==d; rr=L.REG.index(str(r)); lanes=h.stations.sort_values('idx').lanes.to_numpy(float)
                    ss[rr,d,0,2]=np.sum(((nq[I][m]-tq[I][m])/lanes[ix[m,2]])**2)
                states[arm][pn]=ss[:,::2]
        np.savez_compressed(modeldir/f'stats_{pn}.npz',regimes=he.regimes,**{k:x[0] for k,x in comps[pn].items()})
        print('score',split,pn,flush=True)
    res=E.bootstrap(fs,comps,np.arange(len(even)))
    res['median_vs_huber']=E.bootstrap(fs,{pn:{'base':d['huber'],'median':d['median']} for pn,d in comps.items()},np.arange(len(even)))['median']
    if split!='opt':
        shares=L.share_array(split); res['OWN_STATE']={a:L.compare(bases,s,shares,pops=('nb',)) for a,s in states.items()}
        old={pn:np.load(ROOT/f'artifacts/lane_s/0426_local/{split}/{pn}.npz')['stats'][:,::2] for pn in meta['panels']}
        parentcheck=L.compare(old,bases,shares,pops=('nb',))
        expected=json.loads((ROOT/f'artifacts/lane_s/0426_local/{split}/ensemble3/result.json').read_text())['occupancy']
        assert parentcheck==expected, ('S0426 parent mismatch',parentcheck,expected)
        res['parent_parity']=parentcheck
    if not seed:
        metric='J' if split=='opt' else 'S_LWR'
        res['seed_sd']=float(np.std([json.loads((dest/f'seed{s}/result.json').read_text())['median']['summary'][metric]['delta'] for s in (4701,4702,4703)],ddof=1))
    res['scope']='Even-date native subset; OWN physics is known-data non-gap proxy only, OPT required. Parent K6+S0424+S0426, speed unchanged.'
    (modeldir/'result.json').write_text(json.dumps(res,indent=2)); print('RESULT',json.dumps(res),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='build': build(sys.argv[2])
    elif sys.argv[1]=='score': score(sys.argv[2],int(sys.argv[3]))
