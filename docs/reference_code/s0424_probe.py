"""Exact K6 state export and canonical occupancy changed-row delta; no submission."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(k,'3')
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
pa.set_cpu_count(3)
pa.set_io_thread_count(1)
import s0424_canonical as C
import s0416_density as B
ROOT=B.ROOT; OUT=ROOT/'artifacts/lane_s/0424_probe'
PARENT=ROOT/'artifacts/claude_ref/asm/asm_k6_full.csv'
PARENT_SHA='f737ad8ceab61f99c312ff7033dc0a060731a56f453f1bfa5d684695d297e1d9'

def sha(p):
    with open(p,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def prepare():
    import pyarrow as pa,pyarrow.parquet as pq
    OUT.mkdir(parents=True,exist_ok=True); assert sha(PARENT)==PARENT_SHA
    dest=OUT/'parent_state.parquet'; writer=None; count=0
    for df in pd.read_csv(PARENT,usecols=['submission_id','task','speed_kmh','flow_vph'],
                         chunksize=200000,float_precision='round_trip'):
        df=df.loc[df.task.eq('state'),['submission_id','speed_kmh','flow_vph']]
        if not len(df): continue
        table=pa.Table.from_pandas(df,preserve_index=False)
        if writer is None: writer=pq.ParquetWriter(dest,table.schema,compression='zstd')
        writer.write_table(table); count+=len(df)
    writer.close(); assert count==6735795
    (OUT/'parent.json').write_text(json.dumps(dict(parent=str(PARENT.relative_to(ROOT)),parent_sha=PARENT_SHA,
        state_rows=count,derived_sha=sha(dest),bytes=dest.stat().st_size),indent=2))
    print('derived parent',count,dest.stat().st_size,flush=True)

def predict(c,parent,pf,models):
    from tfb.data import indices
    st=c.stations.sort_values('idx'); I=indices(parent,dict(zip(c.dates,range(len(c.dates)))),
        dict(zip(st.link_id.astype(str),st.idx)), 'link_id')
    v=np.asarray(c.speed,float).copy(); q=np.asarray(c.flow,float).copy()
    v[I]=parent.speed_kmh.to_numpy(float); q[I]=parent.flow_vph.to_numpy(float)
    T=np.zeros(c.target.shape,bool); T[I]=True; assert np.array_equal(T,c.target)
    X,ix,names=C.features(c,v,q,pf); J=tuple(ix.T)
    assert all(m.feature_name()==names for mm in models.values() for m in mm)
    pred={k:np.mean([m.predict(X,num_threads=3) for m in models[k]],axis=0) for k in ('v','q')}
    nv=v.copy(); nq=q.copy()
    nv[J]*=np.exp(.5*np.clip(pred['v'],-.05,.05)); nq[J]*=np.exp(.5*np.clip(pred['q'],-.05,.05))
    topo=pd.read_csv(ROOT/f'data/raw/kaggle_public/corridors/{c.meta["panel"]}/network/lwr_mainline_topology.csv')
    clip=1.15*topo.set_index(topo.link_id.astype(str)).free_flow_speed_kmh.reindex(st.link_id.astype(str)).to_numpy(float)
    nv[J]=np.clip(nv[J],3.,clip[ix[:,2]]); nq[J]=np.clip(nq[J],50.,1.3*st.capacity_vph.to_numpy(float)[ix[:,2]])
    assert np.array_equal(nv[c.gap],v[c.gap],equal_nan=True) and np.array_equal(nq[c.gap],q[c.gap],equal_nan=True)
    assert np.isfinite(nv[I]).all() and np.isfinite(nq[I]).all()
    changed=(nv[I]!=v[I])|(nq[I]!=q[I]); assert not c.gap[I][changed].any()
    res=parent.loc[changed,['submission_id']].copy(); res['speed_kmh']=nv[I][changed]; res['flow_vph']=nq[I][changed]
    return res,dict(rows=len(parent),changed_rows=int(changed.sum()),features_sha=B.digest(X),
        max_log_speed_change=float(np.max(np.abs(np.log(nv[J]/v[J])))),max_log_flow_change=float(np.max(np.abs(np.log(nq[J]/q[J])))))

def build():
    import lightgbm as lgb
    from tfb.data import load_cache
    meta=json.loads((B.INP/'metadata.json').read_text()); _,pf,_=B.modules(meta)
    pm=json.loads((OUT/'parent.json').read_text()); assert sha(OUT/'parent_state.parquet')==pm['derived_sha']
    one='--reproduce' in sys.argv or '--sample' in sys.argv
    cols=['submission_id','task','split','panel','timestamp','link_id']
    chunks=[]
    for x in pd.read_csv(ROOT/'data/raw/kaggle_public/submission_key.csv',usecols=cols,chunksize=200000,
                         dtype={k:'str' for k in cols if k!='submission_id'}):
        keep=x.task.eq('state')
        if one: keep &= x.split.eq('validation')&x.panel.eq('D12_I405_S')
        chunks.append(x.loc[keep].drop(columns='task'))
    key=pd.concat(chunks,ignore_index=True); del chunks
    parent=pd.read_parquet(OUT/'parent_state.parquet')
    key=key.merge(parent,on='submission_id',validate='one_to_one'); del parent
    if not one: assert len(key)==pm['state_rows']
    files={k:[C.O.BASE_OUT/f'occupancy_{k}.txt']+[C.O.BASE_OUT/f'seed{s}/occupancy_{k}.txt' for s in (4702,4703)] for k in ('v','q')}
    models={k:[lgb.Booster(model_file=str(p)) for p in pp] for k,pp in files.items()}
    tables=[]; stats={}; t0=time.time()
    for (split,pn),g in key.groupby(['split','panel'],sort=True):
        c=load_cache(pn,split,root=ROOT); d,st=predict(c,g,pf,models); tables.append(d); stats[split+'/'+pn]=st
        print(split,pn,st['changed_rows'],'changed',round(time.time()-t0),'s',flush=True)
    delta=pd.concat(tables,ignore_index=True).sort_values('submission_id'); assert not delta.submission_id.duplicated().any()
    if '--sample' in sys.argv:
        delta.to_parquet(OUT/'owner_sample.parquet',index=False)
        (OUT/'owner_sample.json').write_text(json.dumps(stats,indent=2)); return
    if one:
        ref=pd.read_parquet(OUT/'occupancy_on_k6.delta.parquet'); ref=ref[ref.submission_id.isin(delta.submission_id)]
        assert np.array_equal(ref.submission_id.to_numpy(),delta.submission_id.to_numpy())
        err=float(np.max(np.abs(ref[['speed_kmh','flow_vph']].to_numpy()-delta[['speed_kmh','flow_vph']].to_numpy())))
        assert err<1e-9
        (OUT/'owner_reproduction.json').write_text(json.dumps(dict(panel='D12_I405_S',split='validation',rows=len(delta),max_abs_diff=err),indent=2))
        print('OWNER REPRODUCTION',err,flush=True); return
    dest=OUT/'occupancy_on_k6.delta.parquet'; delta.to_parquet(dest,index=False,compression='zstd')
    manifest=dict(parent=pm,recipe='canonical released-neighbor occupancy; seeds4701/2/3 equal; half log correction clipped +/-0.05',
        delta_sha=sha(dest),rows=len(delta),changed_columns=['speed_kmh','flow_vph'],row_list='submission_id column in delta',
        scope='non-gap state targets only; no other task or gap changes',panels=stats,
        models={str(p.relative_to(ROOT)):sha(p) for pp in files.values() for p in pp},
        sources={p.name:sha(p) for p in (Path(__file__),Path(C.__file__),Path(C.O.__file__),Path(B.__file__))})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)); print('DELTA',len(delta),flush=True)

def validate_delta():
    from tfb.submit import validate
    assert sha(PARENT)==PARENT_SHA
    mf=json.loads((OUT/'manifest.json').read_text()); source=OUT/'occupancy_on_k6.delta.parquet'
    assert sha(source)==mf['delta_sha']
    delta=pd.read_parquet(source).set_index('submission_id'); assert delta.index.is_unique
    dest=OUT/'done/validated_expansion.csv'; dest.parent.mkdir(exist_ok=True)
    assert not dest.exists(), 'Preserve existing validated expansion'
    rows=0; applied=0
    for df in pd.read_csv(PARENT,chunksize=200000,float_precision='round_trip'):
        sel=df.submission_id.isin(delta.index); wanted=df.loc[sel,'submission_id']; old=df.copy(deep=True)
        assert df.loc[sel,'task'].eq('state').all()
        vals=delta.loc[wanted,['speed_kmh','flow_vph']].to_numpy()
        assert ((df.loc[sel,['speed_kmh','flow_vph']].to_numpy()!=vals).any(axis=1)).all()
        df.loc[sel,['speed_kmh','flow_vph']]=vals
        assert df.loc[~sel].equals(old.loc[~sel])
        assert df.drop(columns=['speed_kmh','flow_vph']).equals(old.drop(columns=['speed_kmh','flow_vph']))
        df.to_csv(dest,index=False,mode='w' if rows==0 else 'a',header=rows==0)
        rows+=len(df); applied+=int(sel.sum())
    assert applied==len(delta)==mf['rows']
    receipt=validate(dest,root=ROOT)
    receipt.update(parent_sha=PARENT_SHA,delta_sha=mf['delta_sha'],changed_rows=applied,
        isolation='Every non-delta row and every column except speed_kmh/flow_vph equals K6; gap isolation checked during build')
    (OUT/'validator.json').write_text(json.dumps(receipt,indent=2)); print(json.dumps(receipt),flush=True)

if __name__=='__main__': {'prepare':prepare,'build':build,'validate':validate_delta}[sys.argv[1]]()
