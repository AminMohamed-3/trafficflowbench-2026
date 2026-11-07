"""S0416: portable K6 density-boundary diagnostic. No raw data transfers.
prepare: laptop, two small panels; saves only prediction residuals, models, metadata.
screen: VM; reconstruct observations from local release, score held-date calibration.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '3')
import sys, json, hashlib, shutil, types, time
from pathlib import Path
import numpy as np
ROOT = Path(os.environ.get('TFB_ROOT', str(Path(__file__).resolve().parents[2])))
OUT = ROOT / 'artifacts/lane_s/0416_density'
INP = OUT / 'inputs'
sys.path.insert(0, str(ROOT / 'src'))
PANELS = ['D12_I405_S', 'D7_I210_W']

def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def prepare():
    from tfb.data import load_cache
    p = json.loads((ROOT / 'exp/0354_d_f4_density_score/inputs.json').read_text())
    INP.mkdir(parents=True, exist_ok=True)
    meta = {k:p[k] for k in ('panels', 'families', 'OPT_held_dates')}
    meta['exported_panels'] = PANELS
    meta['panels_info'] = {}
    for pn in PANELS:
        c = load_cache(pn, 'train', root=ROOT)
        ix = [c.dates.index(d) for d in meta['OPT_held_dates']]
        z = np.load(ROOT / f'artifacts/claude_ref/asm/xfit_final_{pn}.npz')
        h = np.load(ROOT / f'artifacts/claude_ref/asm/xfit_s1_{pn}.npz')
        ov, oq = np.asarray(c.speed[ix], float), np.asarray(c.flow[ix], float)
        v, q = z['v'].astype(float), z['q'].astype(float)
        # Residuals at observed cells, pure predictions elsewhere. No observation array is exported.
        dv = v - np.nan_to_num(ov, nan=0.); dq = q - np.nan_to_num(oq, nan=0.)
        np.savez_compressed(INP / f'{pn}.npz', dv=dv, dq=dq, hp=h['hp'])
        meta['panels_info'][pn] = dict(shape=list(v.shape), target_sha=digest(c.target[ix]),
            v_sha=digest(v), q_sha=digest(q), dates=meta['OPT_held_dates'],
            stations=c.stations.sort_values('idx').station_id.astype(str).tolist())
        print('prepared', pn, round((INP/f'{pn}.npz').stat().st_size/2**20, 2), 'MiB', flush=True)
    for f in ('s3_yqL.txt', 's3_yvL.txt', 's3_names.json'):
        shutil.copy2(ROOT / 'artifacts/claude_ref/phys4' / f, INP/f)
    (INP/'metadata.json').write_text(json.dumps(meta, indent=2))

def modules(meta):
    # The old helper's import-time input file has laptop-only paths. Inject metadata only;
    # loaders below use VM-local raw data, and no historical helper source is modified.
    p = types.ModuleType('phys_lwr_diag'); p.ROOT=ROOT; p.P=meta
    sys.modules['phys_lwr_diag']=p
    import flow_screen as fs, phys4_feat as pf, phys4_lsa as ls
    return fs, pf, ls

def reconstruct(held, tv, tq, z, pf, ls, own=False, details=False, features_only=False):
    import lightgbm as lgb
    from tfb import metric
    T = np.asarray(held.target, bool); el=np.asarray(held.elig,bool)
    v=z['dv']+np.nan_to_num(np.asarray(held.speed,float),nan=0.)
    q=z['dq']+np.nan_to_num(np.asarray(held.flow,float),nan=0.)
    st=held.stations.sort_values('idx'); par=metric._parameters(held,ROOT)[0]
    vf=par.free_speed_kmh.to_numpy(float); length=par.length_km.to_numpy(float)
    # Exact asm_inter3 convention: eligible observed cells, train truth, targets excluded.
    if own:
        ok=np.isfinite(held.speed)&np.isfinite(held.flow)
        ov=np.where(ok,held.speed,np.nan); oq=np.where(ok,held.flow,np.nan)
    else:
        ov=np.where(el & ~T & np.isfinite(tv),tv,np.nan)
        oq=np.where(el & ~T & np.isfinite(tq),tq,np.nan)
    ob=np.isfinite(ov)&np.isfinite(oq)&(oq>0)&(ov>.9*vf)
    fq,nq=ls.lsa(np.log(np.maximum(np.nan_to_num(oq,nan=1.),1.)),ob,W=12,K=72)
    fv,nv=ls.lsa(np.log(np.maximum(np.nan_to_num(ov,nan=1.),1.)),ob,W=12,K=72)
    gap=np.asarray(held.gap,bool) if own else np.zeros_like(T)
    m=T & ~gap & np.isfinite(v)&np.isfinite(q)&(v>.9*vf)&(nq>0)&(q>0)
    v14=np.where(m&(nv>0),np.exp(np.log(np.maximum(v,1.))+.3*np.clip(fv-np.log(np.maximum(v,1.)),-.02,.02)),v)
    q14=np.where(m,np.exp(np.log(np.maximum(q,1.))+.5*np.clip(fq-np.log(np.maximum(q,1.)),-.015,.015)),q)
    m=T&~gap&np.isfinite(v14)&np.isfinite(q14)&(v14>0)&(q14>0)
    X,names,ix=pf.build(ov,oq,T,v14,q14,z['hp'],held.stations,length,m)
    I=tuple(ix.T)
    X=np.concatenate([X,np.stack([(fq-np.log(np.maximum(q14,1.)))[I],nq[I],
        (fv-np.log(np.maximum(v14,1.)))[I],nv[I]],1).astype(np.float32)],1)
    assert names+['lsa_dq','lsa_nq','lsa_dv','lsa_nv']==json.loads((INP/'s3_names.json').read_text())
    if features_only:
        import pandas as pd
        topo=pd.read_csv(ROOT/f'data/raw/kaggle_public/corridors/{held.meta["panel"]}/network/lwr_mainline_topology.csv')
        clip=1.15*topo.set_index(topo.link_id.astype(str)).free_flow_speed_kmh.reindex(st.link_id.astype(str)).to_numpy(float)
        return dict(v14=v14,q14=q14,X=X,ix=ix,clip=clip)
    vp=v14.copy(); qp=q14.copy()
    vp[I]*=np.exp(lgb.Booster(model_file=str(INP/'s3_yvL.txt')).predict(X,num_threads=3))
    qp[I]*=np.exp(lgb.Booster(model_file=str(INP/'s3_yqL.txt')).predict(X,num_threads=3))
    import pandas as pd
    topo=pd.read_csv(ROOT/f'data/raw/kaggle_public/corridors/{held.meta["panel"]}/network/lwr_mainline_topology.csv')
    clip=1.15*topo.set_index(topo.link_id.astype(str)).free_flow_speed_kmh.reindex(st.link_id.astype(str)).to_numpy(float)
    vp=np.where(T&(vp>clip+1e-9),clip,vp)
    if details:
        return dict(v=vp,q=qp,v14=v14,q14=q14,X=X,ix=ix,clip=clip)
    return vp,qp,v,q

def bins(T,v,vf):
    before=np.zeros_like(T); after=np.zeros_like(T)
    before[:,1:]=T[:,:-1]; after[:,:-1]=T[:,1:]
    edges=(~before).astype(int)+(~after).astype(int)
    speed_bin=np.digitize(v/vf,[.6,.9])
    return 3*speed_bin+edges,edges,speed_bin

def parity():
    """Small laptop check against unchanged production helper; no fitting or selection."""
    os.environ['PHYS4_S3']='L'
    import phys_lwr_diag as P, phys4_s3 as S3, phys4_lsa as ls, phys4_feat as pf, asm_inter as AI
    from tfb import metric
    from tfb.data import load_cache
    from tfb.lane_s.evaluate_baselines import subset_cache,raw_truth
    meta=json.loads((INP/'metadata.json').read_text()); pn=meta['exported_panels'][0]
    c=load_cache(pn,'train',root=ROOT); h=subset_cache(c,[c.dates.index(d) for d in meta['OPT_held_dates']])
    tv,tq=raw_truth(h,ROOT); z=np.load(INP/f'{pn}.npz')
    v,q,sv,sq=reconstruct(h,tv,tq,z,pf,ls)
    T=np.asarray(h.target,bool); el=np.asarray(h.elig,bool); prm=metric._parameters(h,ROOT)[0]
    (_, _),(rv,rq)=S3.stage3(*S3.models(),sv,sq,T,np.zeros_like(T),np.where(el&~T,tv,np.nan),np.where(el&~T,tq,np.nan),
        z['hp'].astype(float),h.stations,prm.length_km.to_numpy(float),prm.free_speed_kmh.to_numpy(float))
    rv=AI.clipv(rv,T,AI.clipvec(pn,h.stations))
    report=dict(panel=pn,max_speed=float(np.nanmax(np.abs(v[T]-rv[T]))),max_flow=float(np.nanmax(np.abs(q[T]-rq[T]))),
        v_input_sha_ok=digest(sv)==meta['panels_info'][pn]['v_sha'],q_input_sha_ok=digest(sq)==meta['panels_info'][pn]['q_sha'])
    assert report['max_speed']<=1e-6 and report['max_flow']<=1e-6 and report['v_input_sha_ok'] and report['q_input_sha_ok']
    (OUT/'parity.json').write_text(json.dumps(report,indent=2)); print(report,flush=True)

def screen():
    from tfb.data import load_cache
    from tfb.lane_s.evaluate_baselines import subset_cache,raw_truth
    meta=json.loads((INP/'metadata.json').read_text()); fs,pf,ls=modules(meta)
    result={'reference':'K6 non-gap S15c; shipped S3L in-sample on OPT, diagnostic only', 'panels':{}}
    comps={}; start=time.time()
    for pn in meta['exported_panels']:
        cache=load_cache(pn,'train',root=ROOT)
        h=subset_cache(cache,[cache.dates.index(d) for d in meta['OPT_held_dates']]); del cache
        tv,tq=raw_truth(h,ROOT); z=np.load(INP/f'{pn}.npz'); info=meta['panels_info'][pn]
        assert digest(h.target)==info['target_sha'] and list(h.speed.shape)==info['shape']
        assert h.stations.sort_values('idx').station_id.astype(str).tolist()==info['stations']
        v,q,sv,sq=reconstruct(h,tv,tq,z,pf,ls)
        assert digest(sv)==info['v_sha'] and digest(sq)==info['q_sha'],'reconstruction mismatch'
        T=np.asarray(h.target,bool); vf=h.stations.sort_values('idx').free_speed_kmh.to_numpy(float)
        groups,edges,reg=bins(T,v,vf); fit=np.arange(len(h.dates))%2==1
        y=np.log(np.maximum(tq,1.)/np.maximum(tv,1.))-np.log(np.maximum(q,1.)/np.maximum(v,1.))
        corr=np.zeros_like(q); rows=[]
        for b in range(9):
            m=T&(groups==b)&np.isfinite(y); a=m&fit[:,None,None]; test=m&~fit[:,None,None]
            offset=float(np.clip(np.median(y[a]),-.01,.01)) if a.sum()>=300 else 0.
            corr[m]=offset
            rows.append(dict(bin=b,n_fit=int(a.sum()),n_test=int(test.sum()),offset=offset,
                median_test=float(np.median(y[test])) if test.any() else None,
                density_abs_test=float(np.mean(np.abs(q[test]/v[test]-tq[test]/tv[test]))) if test.any() else None))
        cc={'base':fs.day_components(h,tv,tq,v,q)}
        for name,dose in [('full',1.),('half',.5)]:
            cq=np.where(T,np.clip(q*np.exp(dose*corr),50.,1.3*h.stations.sort_values('idx').capacity_vph.to_numpy(float)),q)
            cc[name]=fs.day_components(h,tv,tq,v,cq)
        # Transition attribution uses scorer's total plus within-day breakdown, explicitly separated.
        from tfb import metric
        length=metric._parameters(h,ROOT)[0].length_km.to_numpy(float)
        ne=np.where(T,(q/np.maximum(v,1.)-tq/np.maximum(tv,1.))*length,0.)
        elig=np.asarray(h.elig,bool); ok=elig[:,1:]&elig[:,:-1]&np.isfinite(tv[:,1:])&np.isfinite(tv[:,:-1])
        err=np.abs(np.diff(ne,axis=1)); a,b=T[:,:-1],T[:,1:]
        total=float(err[ok].sum()); breakdown={}
        for name,mask in [('boundary',a^b),('interior',a&b),('congested',np.minimum(tv[:,:-1],tv[:,1:])<.6*vf),('free',np.minimum(tv[:,:-1],tv[:,1:])>.9*vf)]:
            breakdown[name]=float(err[ok&mask].sum()/total)
        r={'bins':rows,'within_day_eligible_LWR_error_shares':breakdown,'scores':{}}
        for name,c in cc.items():
            r['scores'][name]={part:fs.aggregate(c[0],c[1],w.astype(float)) for part,w in [('fit',fit),('held',~fit)]}
        result['panels'][pn]=r; comps[pn]=cc
        np.savez_compressed(OUT/f'stats_{pn}.npz',regimes=np.asarray(cc['base'][1]),**{k:c[0] for k,c in cc.items()})
        print(pn,json.dumps(r['scores']), 'seconds',round(time.time()-start),flush=True)
        (OUT/'screen.json').write_text(json.dumps(result,indent=2))
    rng=np.random.default_rng(4701); D=len(meta['OPT_held_dates']); held=np.flatnonzero(np.arange(D)%2==0)
    result['held_delta']={}
    for name in ('full','half'):
        point=[]; boots=[]
        for pn,c in comps.items():
            w=np.isin(np.arange(D),held).astype(float)
            point.append(fs.aggregate(c[name][0],c[name][1],w)['J']-fs.aggregate(c['base'][0],c['base'][1],w)['J'])
        for _ in range(1000):
            w=np.bincount(rng.choice(held,len(held),replace=True),minlength=D).astype(float)
            boots.append(np.mean([fs.aggregate(c[name][0],c[name][1],w)['J']-fs.aggregate(c['base'][0],c['base'][1],w)['J'] for c in comps.values()]))
        result['held_delta'][name]=dict(J=float(np.mean(point)),ci=np.quantile(boots,[.025,.975]).tolist(),panels=point)
    (OUT/'screen.json').write_text(json.dumps(result,indent=2)); print('RESULT',json.dumps(result['held_delta']),flush=True)

if __name__=='__main__':
    {'prepare':prepare,'screen':screen,'parity':parity}[sys.argv[1]]()
