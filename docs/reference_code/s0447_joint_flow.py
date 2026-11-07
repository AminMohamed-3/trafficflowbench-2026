"""S0447 joint demand-block and same-clock features, on exact S0426."""
import os,sys,json,types
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ.setdefault(k,'4')
import numpy as np
import s0445_density_loss as D
import s0424_canonical as C
import s0418_finalphys as E
ROOT=D.ROOT; OUT=ROOT/'artifacts/lane_s/0447_joint'

def decompose(h,joint):
    import phys4_lsa as LS
    import gen_blk as GB
    st=h.stations.sort_values('idx'); vf=st.free_speed_kmh.to_numpy(float)
    q=np.asarray(h.flow,float); v=np.asarray(h.speed,float); T=np.asarray(h.target,bool); gap=np.asarray(h.gap,bool)
    ob=np.asarray(h.elig,bool)&~T&~gap&np.isfinite(q)&(q>0)&np.isfinite(v)&(v>.9*vf)
    x=np.log(np.maximum(np.where(ob,q,1.),1.)); nday,nt,ns=x.shape
    blk,pos=GB.block_index(nt,st.milepost.to_numpy(float),vf.mean(),.5); bi=blk-blk.min(); nb=bi.max()+1
    clean=(pos>.5)&(pos<5.5); valid=ob&clean[None]
    dc=np.sum(ob,axis=2,keepdims=True).astype(float)
    bc=np.stack([np.bincount(bi.ravel(),weights=valid[d].ravel(),minlength=nb) for d in range(nday)])
    counts=bc[:,bi]; station_n=ob.sum(1,keepdims=True)
    fallback=np.sum(np.where(ob,x,0),axis=(1,2),keepdims=True)/np.maximum(ob.sum((1,2),keepdims=True),1)
    R=np.broadcast_to(np.where(station_n>0,np.sum(np.where(ob,x,0),axis=1,keepdims=True)/np.maximum(station_n,1),fallback),x.shape).copy()
    block=np.zeros_like(x); clock=np.zeros((nday,nt,1)); rden=np.zeros_like(x)
    for _ in range(5):
        if joint:
            resid=np.where(valid,x-R-clock,0.)
            bs=np.stack([np.bincount(bi.ravel(),weights=resid[d].ravel(),minlength=nb) for d in range(nday)])
            block=(bs/np.maximum(bc,1.))[:,bi]
        clock=np.sum(np.where(ob,x-R-block,0.),axis=2,keepdims=True)/np.maximum(dc,1.)
        R,rden=LS.tsmooth(x-block-clock,ob,72)
        R=np.where(rden>0,R,fallback)
    return dict(pred=R+block+clock,clock_count=np.broadcast_to(dc,x.shape),block_count=counts,
        station_count=rden,phase=np.broadcast_to(pos[None]/6,x.shape),clean=np.broadcast_to(clean[None],x.shape))

def features(h,v,q,pf):
    X,ix,names=C.features(h,v,q,pf); I=tuple(ix.T); a=decompose(h,False); b=decompose(h,True)
    logq=np.log(q[I]); extra=dict(clock_residual=a['pred'][I]-logq,
        clock_log_support=np.log1p(a['clock_count'][I]),station_log_support=np.log1p(a['station_count'][I]),
        aligned_phase=a['phase'][I],joint_residual=b['pred'][I]-logq,
        joint_minus_clock=b['pred'][I]-a['pred'][I],joint_log_support=np.log1p(b['block_count'][I]))
    vf=h.stations.sort_values('idx').free_speed_kmh.to_numpy(float)
    gate=(v[I]>.9*vf[ix[:,2]])&a['clean'][I]&(a['clock_count'][I]>=20)&(b['block_count'][I]>=20)&(a['station_count'][I]>1)
    return np.concatenate([X,np.stack(list(extra.values()),1).astype(np.float32)],1),ix,names+list(extra),gate

def build(split):
    meta,_,pf=D.setup(); dest=OUT/split; dest.mkdir(parents=True,exist_ok=True)
    base=ROOT/'artifacts/lane_s/0445_density'
    if not all((base/split/f'{pn}.npz').exists() for pn in meta['panels']):
        D.OUT=OUT/'parent'; D.build(split); base=D.OUT
    for pn in meta['panels']:
        path=dest/f'{pn}.npz'
        if path.exists(): print('reuse',split,pn,flush=True); continue
        z=dict(np.load(base/split/f'{pn}.npz')); h,tv,tq,_,_,_=D.layer(pn,split)
        X,ix,names,gate=features(h,z['v'],z['q'],pf); assert np.array_equal(ix,z['ix'])
        assert np.array_equal(X[:,:21],z['X'],equal_nan=True)
        if pn==meta['panels'][0]:
            hh=types.SimpleNamespace(**vars(h)); hidden=np.asarray(h.target,bool)|np.asarray(h.gap,bool)
            for k in ('speed','flow','occ'):
                val=np.asarray(getattr(h,k),float).copy(); val[hidden]=98765.; setattr(hh,k,val)
            xp,ip,_,gp=features(hh,z['v'],z['q'],pf)
            assert np.array_equal(X,xp,equal_nan=True) and np.array_equal(gate,gp), 'masked observation leak'
            hh=types.SimpleNamespace(**vars(h))
            for k in ('speed','flow','occ'): setattr(hh,k,np.asarray(getattr(h,k),float))
            xp,ip,_,gp=features(hh,z['v'],z['q'],pf)
            assert np.array_equal(X,xp,equal_nan=True) and np.array_equal(gate,gp), 'dtype mismatch'
            (dest/'contract.json').write_text(json.dumps(dict(masked_observation_poison=True,input_precision_invariant=True,canonical21_parity=True)))
        z.update(X=X,ix=ix,names=np.asarray(names),gate=gate,y=np.log(tq[tuple(ix.T)]/z['q'][tuple(ix.T)]))
        np.savez_compressed(dest/f'{pn}.building.npz',**z); (dest/f'{pn}.building.npz').replace(path)
        print('built joint feature',split,pn,int((gate&(ix[:,0]%2==1)).sum()),flush=True)

def score(split,seed):
    import lightgbm as lgb
    meta,fs,_=D.setup(); import state5_lib as L
    dest=OUT/split; modeldir=dest/(f'seed{seed}' if seed else 'ensemble3'); modeldir.mkdir(exist_ok=True)
    Z={pn:dict(np.load(dest/f'{pn}.npz')) for pn in meta['panels']}; names=next(iter(Z.values()))['names']
    sel={pn:z['gate']&(z['ix'][:,0]%2==1) for pn,z in Z.items()}; models={}; uses={}
    for arm in ('control','joint'):
        use=np.asarray([arm=='joint' or not n.startswith('joint_') for n in names]); uses[arm]=use
        if seed:
            X=np.concatenate([z['X'][sel[pn]][:,use] for pn,z in Z.items()]); y=np.concatenate([z['y'][sel[pn]] for pn,z in Z.items()])
            prm=dict(objective='huber',alpha=.03,learning_rate=.05,num_leaves=31,min_data_in_leaf=500,lambda_l2=10.,
                feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,num_threads=4,verbosity=-1,seed=seed)
            m=lgb.train(prm,lgb.Dataset(X,y,feature_name=names[use].tolist()),200); m.save_model(str(modeldir/f'{arm}.txt')); models[arm]=[m]
            print('fit',split,seed,arm,len(y),flush=True); del X,y
        else: models[arm]=[lgb.Booster(model_file=str(dest/f'seed{s}/{arm}.txt')) for s in (4701,4702,4703)]
    comps={}; bases={}; states={a:{} for a in models}
    for pn in meta['panels']:
        z=Z.pop(pn); h,tv,tq,_,_,_=D.layer(pn,split); even=np.arange(len(h.dates))[::2]; he=D.selected(D.proxy_view(h),even)
        ix=z['ix']; I=tuple(ix.T); g=z['gate']; J=tuple(ix[g].T); v=z['v']; q=z['q']
        comps[pn]={'base':fs.day_components(he,tv[even],tq[even],v[even],q[even])}
        if split!='opt': bases[pn]=z['stats'][:,::2]
        for arm,ms in models.items():
            pp=np.mean([m.predict(z['X'][:,uses[arm]],num_threads=4) for m in ms],axis=0); nq=q.copy()
            nq[J]=np.clip(q[J]*np.exp(.5*np.clip(pp[g],-.01,.01)),50.,1.3*z['cap'][ix[g,2]])
            comps[pn][arm]=fs.day_components(he,tv[even],tq[even],v[even],nq[even])
            if split!='opt':
                ss=z['stats'].copy(); lanes=h.stations.sort_values('idx').lanes.to_numpy(float)
                for d,r in enumerate(h.regimes):
                    m=ix[:,0]==d; rr=L.REG.index(str(r)); ss[rr,d,0,2]=np.sum(((nq[I][m]-tq[I][m])/lanes[ix[m,2]])**2)
                states[arm][pn]=ss[:,::2]
        np.savez_compressed(modeldir/f'stats_{pn}.npz',regimes=he.regimes,**{k:x[0] for k,x in comps[pn].items()})
        print('scored',split,pn,flush=True)
    result=E.bootstrap(fs,comps,np.arange(len(even)))
    result['joint_vs_control']=E.bootstrap(fs,{pn:{'base':c['control'],'joint':c['joint']} for pn,c in comps.items()},np.arange(len(even)))['joint']
    if split!='opt':
        shares=L.share_array(split); result['OWN_STATE']={a:L.compare(bases,s,shares,pops=('nb',)) for a,s in states.items()}
        result['OWN_STATE']['joint_vs_control']=L.compare(states['control'],states['joint'],shares,pops=('nb',))
        old={pn:np.load(ROOT/f'artifacts/lane_s/0426_local/{split}/{pn}.npz')['stats'][:,::2] for pn in meta['panels']}
        check=L.compare(old,bases,shares,pops=('nb',)); expected=json.loads((ROOT/f'artifacts/lane_s/0426_local/{split}/ensemble3/result.json').read_text())['occupancy']
        assert check==expected; result['parent_parity']=check
    if not seed:
        points=[json.loads((dest/f'seed{s}/result.json').read_text()) for s in (4701,4702,4703)]
        result['seed_sd']=float(np.std([r['joint']['summary']['J']['delta'] if split=='opt' else r['OWN_STATE']['joint']['nb']['delta'] for r in points],ddof=1))
    result['scope']='Increment over exact K6+S0424+S0426; even dates. OWN physics is known-data non-gap proxy, OPT required. Speed unchanged.'
    (modeldir/'result.json').write_text(json.dumps(result,indent=2))
    print('RESULT',json.dumps({k:v for k,v in result.items() if k in ('OWN_STATE','seed_sd','parent_parity','scope')}),flush=True)
    print('PHYSICS',json.dumps({k:v['summary'] for k,v in result.items() if k in ('joint','control','joint_vs_control')}),flush=True)

def contract(split):
    import lightgbm as lgb
    meta,_,pf=D.setup(); pn=meta['panels'][0]; dest=OUT/split
    z=dict(np.load(dest/f'{pn}.npz')); h,_,_,_,_,_=D.layer(pn,split); results={}
    for label in ('float32','float64','hidden_observation','non_target_parent'):
        hh=types.SimpleNamespace(**vars(h)); vv=z['v'].copy(); qq=z['q'].copy()
        dtype=np.float32 if label=='float32' else np.float64
        for k in ('speed','flow','occ'):
            val=np.asarray(getattr(h,k),dtype=dtype).copy()
            if label=='hidden_observation': val[np.asarray(h.target,bool)|np.asarray(h.gap,bool)]=98765.
            setattr(hh,k,val)
        if label=='non_target_parent': vv[~h.target]=123456.; qq[~h.target]=987654.
        X,ix,names,gate=features(hh,vv,qq,pf)
        assert np.array_equal(X,z['X'],equal_nan=True) and np.array_equal(ix,z['ix']) and np.array_equal(gate,z['gate']), label
        diffs={}
        for arm in ('control','joint'):
            use=np.asarray([arm=='joint' or not n.startswith('joint_') for n in names])
            m=lgb.Booster(model_file=str(dest/f'seed4701/{arm}.txt'))
            diff=float(np.max(np.abs(m.predict(X[:,use],num_threads=4)-m.predict(z['X'][:,use],num_threads=4))))
            assert diff==0.; diffs[arm]=diff
        results[label]=dict(exact_features=True,exact_gate=True,prediction_max_diff=diffs)
    (dest/'serving_contract.json').write_text(json.dumps(dict(panel=pn,rows=len(z['ix']),checks=results,
        scope='Explicit float32 and float64 raw copies; target/gap observation poison; non-target parent poison. Includes canonical21 inherited by S0445.'),indent=2))
    print('CONTRACT',pn,'all four exact',flush=True)

def gate(split):
    r=json.loads((OUT/split/'ensemble3/result.json').read_text())
    if split=='opt':
        assert r['joint']['summary']['J']['ci'][0]>0
        assert r['joint']['summary']['S_phys']['ci'][0]>0
        assert r['seed_sd']<r['joint']['summary']['J']['delta']/2
    else:
        assert r['OWN_STATE']['joint']['nb']['ci'][0]>0
        assert r['OWN_STATE']['joint_vs_control']['nb']['ci'][0]>0
        assert r['seed_sd']<r['OWN_STATE']['joint']['nb']['delta']/2
    print('GATE PASS',split,flush=True)

if __name__=='__main__':
    if sys.argv[1]=='build': build(sys.argv[2])
    elif sys.argv[1]=='score': score(sys.argv[2],int(sys.argv[3]))
    elif sys.argv[1]=='contract': contract(sys.argv[2])
    elif sys.argv[1]=='gate': gate(sys.argv[2])
