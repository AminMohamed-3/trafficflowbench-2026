"""S0418: fixed final projection and TV on K6, only non-gap targets."""
import os,sys,json,time
from pathlib import Path
import numpy as np
import s0416_density as B
ROOT=B.ROOT
OUT=ROOT/'artifacts/lane_s/0418_finalphys'

def bootstrap(fs,comps,held,n=1000):
    panels=list(comps); D=len(next(iter(comps.values()))['base'][1])
    rng=np.random.default_rng(4701); ws=[np.isin(np.arange(D),held).astype(float)]
    tries=0
    while len(ws)<=n:
        tries+=1; w=np.bincount(rng.choice(held,len(held),replace=True),minlength=D).astype(float)
        if all(all(w[np.asarray(c['base'][1])==r].sum()>0 for r in ('R1','R2','R3')) for c in comps.values()): ws.append(w)
        if tries>100000: raise RuntimeError('Insufficient regime support for bootstrap')
    out={}
    for name in next(iter(comps.values())):
        if name=='base': continue
        draws={m:[] for m in ('J','S_state','S_phys','S_LWR','S_FD')}; per={}
        for i,w in enumerate(ws):
            diffs=[]
            for pn,c in comps.items():
                a=fs.aggregate(c['base'][0],c['base'][1],w); b=fs.aggregate(c[name][0],c[name][1],w)
                d={m:b[m]-a[m] for m in draws}; diffs.append(d)
                if i==0: per[pn]=d
            for m in draws: draws[m].append(float(np.mean([d[m] for d in diffs])))
        out[name]={'summary':{m:{'delta':v[0],'ci':np.quantile(v[1:],[.025,.975]).tolist()} for m,v in draws.items()},'per_panel':per}
    return out

def candidates(v,q,T,gap,obs,stations):
    import polish_eval as E, research2_tv as TV
    mask=T&~gap
    pv,pq=E.project_soft(E.Ctx(stations,mask,obs,None,v,q,gap),dict(r=3.,a=.25,g=1.,h=.1))
    tv,tq=TV.smooth(v,q,mask,stations,.002,0.)
    jv,jq=TV.smooth(pv,pq,mask,stations,.002,0.)
    return {'base':(v,q),'project':(pv,pq),'tv':(tv,tq),'joint':(jv,jq),
        'half':(np.sqrt(np.maximum(v*jv,0)),np.sqrt(np.maximum(q*jq,0)))}

def main():
    from tfb.data import load_cache
    from tfb.lane_s.evaluate_baselines import subset_cache,raw_truth
    meta=json.loads((B.INP/'metadata.json').read_text()); fs,pf,ls=B.modules(meta)
    OUT.mkdir(parents=True,exist_ok=True); comps={}; t0=time.time()
    for pn in meta['exported_panels']:
        c=load_cache(pn,'train',root=ROOT); h=subset_cache(c,[c.dates.index(d) for d in meta['OPT_held_dates']]); del c
        tv,tq=raw_truth(h,ROOT); z=np.load(B.INP/f'{pn}.npz')
        assert B.digest(h.target)==meta['panels_info'][pn]['target_sha']
        v,q,sv,sq=B.reconstruct(h,tv,tq,z,pf,ls)
        assert B.digest(sv)==meta['panels_info'][pn]['v_sha'] and B.digest(sq)==meta['panels_info'][pn]['q_sha']
        T=np.asarray(h.target,bool); gap=np.zeros_like(T)
        arr=candidates(v,q,T,gap,h.speed,h.stations)
        comps[pn]={}
        for name,(nv,nq) in arr.items():
            # Every changed speed is below v_cut or unchanged, so generator upper clip remains satisfied.
            assert np.allclose(nv[~T],v[~T],equal_nan=True) and np.allclose(nq[~T],q[~T],equal_nan=True)
            comps[pn][name]=fs.day_components(h,tv,tq,nv,nq)
        np.savez_compressed(OUT/f'stats_{pn}.npz',regimes=np.asarray(comps[pn]['base'][1]),**{k:c[0] for k,c in comps[pn].items()})
        print(pn,'complete',round(time.time()-t0),'seconds',flush=True)
    D=len(meta['OPT_held_dates']); res=bootstrap(fs,comps,np.flatnonzero(np.arange(D)%2==0))
    (OUT/'screen.json').write_text(json.dumps(res,indent=2)); print(json.dumps(res),flush=True)

if __name__=='__main__': main()
