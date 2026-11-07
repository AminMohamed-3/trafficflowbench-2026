"""Frozen Q0436 official serving, incremental causality, and owner reproduction."""
import argparse
import copy
import hashlib
import io
import json
import tarfile
from types import MethodType
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import ndtri
import q0436_demand_agreement as A
import q0434_flow_share as FS

ROOT=A.ROOT; OUT=A.OUT


def guard():
    assert A.passed_march()
    tol=json.loads((OUT/'result_tol_all.json').read_text())
    assert all(tol[m]['delta_ci'][0]>=0 for m in ('Jan','Feb'))
    assert json.loads((OUT/'result_pri_all.json').read_text())['delta_ci'][1]>0


def inputs():
    with tarfile.open(OUT/'official_inputs.tar.gz','r:gz') as tar:
        read=lambda name: io.BytesIO(tar.extractfile(name).read())
        q=pd.concat([pd.read_parquet(read(f'q03_{s}.parquet')) for s in ('validation','private')],ignore_index=True)
        feat=pd.read_parquet(read('feat_off.parquet'))
        expected=pd.read_csv(read('s1f_windows.csv'),keep_default_na=False)
    assert len(q)==80 and q.window_id.nunique()==80
    return q,feat,expected


def physical(P,date,t,history,prof,ctrl):
    """Bounded current-day view plus this window's released raw history."""
    import onset4_data as D
    day=P.dates.index(date)
    view=copy.copy(P); view.dates=[date]
    for attr in ('sm','fm','om','elig'):
        old=getattr(P,attr)
        fill=False if old.dtype.kind=='b' else np.nan
        arr=np.full((1,288,P.n),fill,dtype=old.dtype)
        arr[0,:t+1]=old[day,:t+1]
        setattr(view,attr,arr)
    S,F,O=history
    def bounded(self,dd,tt):
        assert dd==0 and tt==t
        return (np.vstack([S,self.sm[0,t:t+1]]),np.vstack([F,self.fm[0,t:t+1]]),np.vstack([O,self.om[0,t:t+1]]))
    view.inputs=MethodType(bounded,view)
    return pd.DataFrame(D.feats(view,0,t,prof,ctrl))


def calibrate(frame):
    import onset5_eval2 as V
    null=FS.raw_stats('null'); null=null[np.isfinite(null.lr)]
    df=frame.copy(); df['u']=.5; df['med']=0.; df['nn']=0
    good=np.isfinite(df.lr)
    if good.any():
        u,med,nn=V.null_quantiles(null,df[good])
        df.loc[good,'u']=u; df.loc[good,'med']=med; df.loc[good,'nn']=nn
    eps=1/(2*(df.nn.to_numpy()+1))
    z=np.clip(ndtri(np.clip(df.u.to_numpy(),eps,1-eps)),-3,3)
    df['correction']=np.where((df.u.to_numpy()<.02)|(df.u.to_numpy()>.98),z,0.)
    return df


def rank_logits(frame):
    cols=A.columns()
    models=[lgb.Booster(model_file=str(A.MODEL_DIR/f'ranker_{s}.txt')) for s in range(3)]
    values=np.empty((3,len(frame)))
    for _,ix in frame.groupby('wkey',sort=False).indices.items():
        block=frame.iloc[ix]; n=len(block)
        x=A.differences(block[cols].to_numpy(np.float32)).reshape(n*n,len(cols))
        for seed,model in enumerate(models):
            pair=A.logit(model.predict(x,num_threads=4)).reshape(n,n)
            z=A.latent(pair); p=np.exp(z-z.max()); p/=p.sum()
            values[seed,ix]=A.logit(p)
    out=frame[['wkey','site_uid']].copy()
    for s in range(3): out[f'logit_rank_{s}']=values[s]
    out['logit_rank']=values.mean(axis=0)
    return out


def build():
    import onset4_data as D
    import onset4_eval as E
    import onset6_common as C
    import onset6_evalF as F
    import onset5_eval2 as V
    guard(); q,cached,expected=inputs()
    # Check the serving calibration copy against the audited evaluator on every validation row.
    cal=calibrate(FS.raw_stats('val'))
    reference=FS.calibrate('val')
    np.testing.assert_array_equal(cal.correction.to_numpy(),reference.correction.to_numpy())
    prof=dict(np.load(ROOT/'artifacts/claude_ref/onset4/profiles.npz'))
    ctrl=json.loads((ROOT/'artifacts/claude_ref/onset4/control_cells.json').read_text())
    features=[]; flow=[]; checks=[]; cols=A.columns()
    for panel in D.PANELS:
        sites=[s['key'] for s in D.site_list(panel) if not s['other']]
        cells=[ctrl[f'{panel}/{s}']['cell'] for s in sites]
        train=D.Panel(panel,'train'); flow_profiles=FS.profile(train,cells); del train
        for split in ('validation','private'):
            P=D.Panel(panel,split)
            directory=ROOT/f'data/raw/kaggle_public/task2/{panel}/{split}'
            wi=pd.read_csv(directory/'window_index.csv')
            wi=wi[wi.condition=='queue_onset'].set_index('window_id')
            hw_all=pd.read_parquet(directory/'window_history.parquet')
            group=q[(q.panel==panel)&(q.split==split)].sort_values('window_id')
            assert set(group.window_id)==set(wi.index)
            links=P.c.stations.sort_values('idx').link_id.astype(str).tolist(); li={l:i for i,l in enumerate(links)}
            for wi_number,r in enumerate(group.itertuples()):
                T=pd.Timestamp(r.origin); t=int(r.t); day=P.dates.index(r.date)
                assert T==pd.Timestamp(wi.loc[r.window_id,'forecast_origin']) and t==T.hour*12+T.minute//5
                hw=hw_all[hw_all.window_id==r.window_id]
                steps=((T-pd.to_datetime(hw.timestamp)).dt.total_seconds()/300).to_numpy()
                assert np.isfinite(steps).all() and (steps==steps.astype(int)).all() and steps.min()==1 and steps.max()==12
                row=12-steps.astype(int); col=hw.link_id.astype(str).map(li).to_numpy()
                S=np.full((12,P.n),np.nan,np.float32); FF=S.copy(); O=S.copy()
                S[row,col]=hw.speed_kmh.to_numpy(float)/P.vf[col]
                FF[row,col]=hw.flow_vph.to_numpy(float)/P.cap[col]
                O[row,col]=hw.occupancy.to_numpy(float)
                history=(S,FF,O)
                feat=physical(P,r.date,t,history,prof,ctrl)
                wanted=cached[cached.wkey==r.window_id].set_index('site_uid')[cols].sort_index()
                actual=feat.set_index('site_uid')[cols].sort_index()
                np.testing.assert_allclose(actual.to_numpy(float),wanted.to_numpy(float),rtol=1e-6,atol=1e-7,equal_nan=True)
                mean,obs,days=FS.online_stat(P,cells,flow_profiles,0,day,t,{})
                if wi_number==0:
                    poison=copy.copy(P)
                    for attr in ('sm','fm','om','elig'):
                        a=np.array(getattr(P,attr),copy=True); fill=False if a.dtype.kind=='b' else np.nan
                        a[day,t+1:]=fill; a[day+1:]=fill; setattr(poison,attr,a)
                    altered=physical(poison,r.date,t,history,prof,ctrl).set_index('site_uid')[cols].sort_index()
                    np.testing.assert_array_equal(actual.to_numpy(float),altered.to_numpy(float))
                    after=FS.online_stat(poison,cells,flow_profiles,0,day,t,{})
                    for a,b in zip((mean,obs,days),after): np.testing.assert_array_equal(a,b)
                    checks.append(dict(window_id=r.window_id,split=split,panel=panel,flow_observations=int(obs.sum()),physical_and_flow_poison='PASS'))
                    del poison
                feat=feat.assign(wkey=r.window_id,panel=panel,split=split,date=r.date,t=t)
                features.append(feat)
                for site,value,n,nd in zip(sites,mean,obs,days):
                    flow.append(dict(wkey=r.window_id,panel=panel,split=split,date=r.date,t=t,j=day,kdays=day,
                                     mon=r.date[:7],site=site,lr=float(value),observations=int(n),days=int(nd)))
            del P,hw_all
            print(f'official {panel}/{split}: source parity and prefix bounds PASS',flush=True)
    feature=pd.concat(features,ignore_index=True)
    flow=calibrate(pd.DataFrame(flow))
    rows=[]
    for r in q.itertuples():
        for site,p in zip(r.site_keys.split(';'),r.p):
            rows.append(dict(wkey=r.window_id,site=site,site_uid=f'{r.panel}/{site}',panel=r.panel,split=r.split,date=r.date,t=int(r.t),pq=float(p)))
    g=pd.DataFrame(rows)
    s1=V.prep('all')['off'][['wkey','site','lr','u','med']]
    ff=F.prepF()['off'][['wkey','site','lr','u','med']].rename(columns={'lr':'lrF','u':'uF','med':'medF'})
    g=g.merge(s1,on=['wkey','site'],validate='one_to_one').merge(ff,on=['wkey','site'],validate='one_to_one')
    g=g.merge(rank_logits(feature),on=['wkey','site_uid'],validate='one_to_one')
    g=g.merge(flow[['wkey','site','correction']],on=['wkey','site'],how='left',validate='one_to_one')
    assert g.loc[g.site!='other','correction'].notna().all()
    baseline=F.adjF(g.assign(pq=C.s1(g)),.02,1.)
    physical_delta=.5*(g.logit_rank.to_numpy()-A.logit(g.pq.to_numpy()))
    delta=A.agreement(physical_delta,g.correction.fillna(0).to_numpy())
    probability=np.where(delta==0,baseline,1/(1+np.exp(-(A.logit(baseline)+delta))))
    g=g.assign(p_s1f=baseline,physical_delta=physical_delta,applied_delta=delta,p_candidate=probability)
    decoded={name:C.decode(g,p) for name,p in [('q03',g.pq.to_numpy()),('s1f',baseline),('candidate',probability)]}
    lookup={p:{i:l for s in E.META['sites'][p] for l,i in zip(s['links'],s['positions'])} for p in D.PANELS}
    qlookup=q.set_index('window_id'); expected=expected.set_index('window_id'); windows=[]
    for w,b in g.groupby('wkey',sort=True):
        panel=b.panel.iloc[0]
        chosen={name:set(lookup[panel][i] for i in d[w]) for name,d in decoded.items()}
        assert chosen['q03']==set(qlookup.loc[w,'cells'].split(';'))
        assert chosen['s1f']==set(expected.loc[w,'new'].split(';'))
        windows.append(dict(window_id=w,split=b.split.iloc[0],panel=panel,date=b.date.iloc[0],t=int(b.t.iloc[0]),
                            **{name:';'.join(sorted(v)) for name,v in chosen.items()},
                            changed_vs_s1f=chosen['candidate']!=chosen['s1f'],changed_vs_q03=chosen['candidate']!=chosen['q03']))
    table=pd.DataFrame(windows)
    assert len(checks)==16 and len(table)==80
    receipt=dict(status='PASS',source_parity_windows=80,q03_and_s1f_decode_parity_windows=80,
                 incremental_future_poison_windows=16,checks=checks,
                 changed_vs_s1f=table.groupby('split').changed_vs_s1f.sum().to_dict(),
                 changed_vs_q03=table.groupby('split').changed_vs_q03.sum().to_dict())
    return table,g,flow,receipt


def robustness():
    import onset4_eval as E
    result={}
    for pop,phases in [('val',['dev','confirm']),('pri',['all'])]:
        score=pd.concat([pd.read_parquet(OUT/f'scores_{pop}_{p}.parquet') for p in phases],ignore_index=True)
        result[pop]={'all':A.summary(score),'observed_only':A.summary(score[score.unobs==0])}
        replay=pd.concat([pd.read_parquet(OUT/f'replay_{pop}_{p}.parquet') for p in phases],ignore_index=True)
        brief=replay[replay.j<=4].groupby('wkey').mean(numeric_only=True).drop(columns='j')
        meta=score[['wkey','panel','date','t']].set_index('wkey')
        brief=meta.join(brief,how='inner').reset_index()
        result[pop]['j_le_4']=A.summary(brief)
    (OUT/'robustness.json').write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['build','reproduce'])
    parser.add_argument('--output-dir',type=__import__('pathlib').Path)
    parser.add_argument('--serving-only',action='store_true',help='skip retrospective robustness tables')
    args=parser.parse_args()
    destination=args.output_dir or OUT
    destination.mkdir(parents=True,exist_ok=True)
    table,g,flow,receipt=build()
    data=table.to_csv(index=False).encode()
    if args.mode=='reproduce':
        assert data==(OUT/'official_windows.csv').read_bytes()
        old=pd.read_parquet(OUT/'official_probabilities.parquet')
        pd.testing.assert_frame_equal(g,old,check_exact=True)
        receipt['reproduction']='bit-exact windows and site probabilities from fresh raw-data rebuild'
        (destination/'official_reproduction.json').write_text(json.dumps(receipt,indent=2))
    else:
        (destination/'official_windows.csv').write_bytes(data)
        g.to_parquet(destination/'official_probabilities.parquet'); flow.to_parquet(destination/'official_flow.parquet')
        receipt['source_sha256']=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()
        (destination/'official_serving_receipt.json').write_text(json.dumps(receipt,indent=2))
        if not args.serving_only: robustness()
    print(json.dumps(receipt,indent=2),flush=True)
    print(table[table.changed_vs_s1f].to_string(index=False),flush=True)


if __name__=='__main__': main()
