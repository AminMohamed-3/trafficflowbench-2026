"""Q0434: train-null-calibrated, online free-flow spatial-demand switch."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtri
import q0425_demand_rank as R

ROOT=R.ROOT
OUT=ROOT/'artifacts/lane_q/0434_flow_share'
R.OUT=OUT


def profile(P,cells):
    """Train-only free-flow expectation; monthly sufficient statistics support LMO."""
    assert P.split=='train'
    dates=np.array(P.dates); months=np.array([d[:7] for d in dates])
    weekend=pd.to_datetime(dates).weekday>=5
    f=P.fr[:,:,cells]; s=P.sr[:,:,cells]
    valid=np.isfinite(f)&np.isfinite(s)&(f>.2)&(s>.9)
    value=np.log(np.maximum(f,1e-8))
    sums,counts={},{}
    for m in sorted(set(months)):
        sums[m]=np.stack([np.where(valid[(months==m)&(weekend==w)],value[(months==m)&(weekend==w)],0).sum(0) for w in (False,True)])
        counts[m]=np.stack([valid[(months==m)&(weekend==w)].sum(0) for w in (False,True)])
    total=sum(sums.values()); count=sum(counts.values())
    result={}
    for m in ['all']+sorted(sums):
        a=total if m=='all' else total-sums[m]
        b=count if m=='all' else count-counts[m]
        a=a+np.roll(a,1,axis=1)+np.roll(a,-1,axis=1)
        b=b+np.roll(b,1,axis=1)+np.roll(b,-1,axis=1)
        result[m]=np.divide(a,b,out=np.full_like(a,np.nan),where=b>0)
    return result


def statistic(values,valid,start,day,t,gaps):
    """Synthetic-array bound fixture; production uses online_stat below."""
    assert 0<=start<=day<len(values) and 0<=t<288
    daily=[]; observations=np.zeros(values.shape[2],int)
    for dd in range(start,day+1):
        lo=max(0,t-36); hi=t if dd==day else min(288,t+37)
        keep=np.ones(hi-lo,bool)
        for origin in gaps.get(dd,[]):
            left=max(lo,origin+1); right=min(hi,origin+19)
            if left<right: keep[left-lo:right-lo]=False
        ok=valid[dd,lo:hi]&keep[:,None]
        n=ok.sum(axis=0); observations+=n
        a=np.where(ok,values[dd,lo:hi],0).sum(axis=0)
        daily.append(np.divide(a,n,out=np.full_like(a,np.nan),where=n>0))
    x=np.array(daily); n=np.isfinite(x).sum(axis=0)
    mean=np.divide(np.nansum(x,axis=0),n,out=np.full(values.shape[2],np.nan),where=n>0)
    return mean,observations,n


def online_stat(P,cells,profiles,start,day,t,gaps):
    """Single bounded feature accessor for (P.panel, P.split, P.dates[day], T=t)."""
    assert 0<=start<=day<len(P.dates) and 0<=t<288
    cutoff=P.dates[day]
    daily=[]; observations=np.zeros(len(cells),int)
    for dd in range(start,day+1):
        lo=max(0,t-36); hi=t if dd==day else min(288,t+37)
        assert P.dates[dd]<cutoff or (P.dates[dd]==cutoff and hi<=t)
        # Slice the permitted date/time range BEFORE accessing or transforming values.
        f=P.fm[dd,lo:hi][:,cells]; s=P.sm[dd,lo:hi][:,cells]; el=P.elig[dd,lo:hi][:,cells]
        key=P.dates[dd][:7] if P.split=='train' else 'all'
        expectation=profiles[key][int(pd.Timestamp(P.dates[dd]).weekday()>=5),lo:hi]
        valid=np.isfinite(f)&np.isfinite(s)&np.isfinite(expectation)&el&(f>.2)&(s>.9)
        residual=np.log(np.maximum(f,1e-8))-expectation
        count=valid.sum(axis=1,keepdims=True)
        common=np.where(valid,residual,0).sum(axis=1,keepdims=True)/np.maximum(count,1)
        valid &= count>=2
        values=np.where(valid,residual-common,0.)
        keep=np.ones(hi-lo,bool)
        for origin in gaps.get(dd,[]):
            left=max(lo,origin+1); right=min(hi,origin+19)
            if left<right: keep[left-lo:right-lo]=False
        ok=valid&keep[:,None]
        n=ok.sum(axis=0); observations+=n
        a=np.where(ok,values,0).sum(axis=0)
        daily.append(np.divide(a,n,out=np.full_like(a,np.nan),where=n>0))
    x=np.array(daily); n=np.isfinite(x).sum(axis=0)
    mean=np.divide(np.nansum(x,axis=0),n,out=np.full(len(cells),np.nan),where=n>0)
    return mean,observations,n


def fixture():
    values=np.ones((3,288,2)); valid=np.ones_like(values,dtype=bool)
    a=statistic(values,valid,0,1,100,{})
    poison=values.copy(); poison[1,100:]=999; poison[2]=999
    b=statistic(poison,valid,0,1,100,{})
    for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
    zero=statistic(values,np.zeros_like(valid),0,1,100,{})
    assert np.isnan(zero[0]).all() and not zero[1].any()
    return dict(status='PASS',checks=['strict no-T/future bound','zero-support abstention'],
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def queries(pop):
    if pop=='null':
        df=pd.read_parquet(ROOT/'artifacts/claude_ref/onset/selected_windows.parquet')
        df=df[(df.split=='train')&(df.layer=='raw')][['panel','date','t']]
        tol=R.source_frame('tol')[['panel','date','t']]
        df=pd.concat([df,tol]).drop_duplicates().copy()
        df['wkey']=df.panel+'|'+df.date+'|'+df.t.astype(str)
        return df
    return R.source_frame('tol' if pop=='tolm' else pop)[['wkey','panel','date','t']].drop_duplicates()


def raw_stats(pop,force=False):
    import onset4_data as D
    import onset5_select as SEL
    import onset5_stat2 as S2
    path=OUT/f'raw_{pop}.parquet'
    if pop in ('tol','tolm'):
        # The reference replays train selector+TOL queries as one ordered union.
        # Replaying TOL alone changes forced-candidate insertion and valid j's.
        # Reuse the already verified full-train table, then subset by query key.
        full=raw_stats('null')
        keys=set(queries('tol').wkey)
        out=full[full.wkey.isin(keys)].copy()
        ref=pd.read_parquet(ROOT/'artifacts/claude_ref/onset5/u2_all_train.parquet')
        ref=ref[ref.wkey.isin(keys)&(ref.site!='other')]
        fields=['wkey','site','j']
        actual=set(out[fields].itertuples(index=False,name=None))
        expected=set(ref[fields].itertuples(index=False,name=None))
        assert actual==expected and len(actual)==len(out), 'Canonical train replay coverage mismatch'
        before=len(pd.read_parquet(path)) if path.exists() else None
        done=OUT/'done'; done.mkdir(exist_ok=True)
        if path.exists() and not (done/f'raw_{pop}_pre_union.parquet').exists():
            pd.read_parquet(path).to_parquet(done/f'raw_{pop}_pre_union.parquet')
        out.to_parquet(path)
        (OUT/f'{pop}_union_repair.json').write_text(json.dumps(dict(status='PASS',
            reference='full train-selector+TOL replay union',previous_rows=before,rows=len(out),
            reference_rows=len(ref),missing=0,extra=0),indent=2))
        return out
    if path.exists() and not force: return pd.read_parquet(path)
    frame=queries(pop)
    split={'null':'train','tol':'train','tolm':'train','val':'validation','pri':'private'}[pop]
    ctrl=json.loads((ROOT/'artifacts/claude_ref/onset4/control_cells.json').read_text())
    rows,checks=[],[]
    for panel,group in frame.groupby('panel',sort=True):
        sites=[s['key'] for s in D.site_list(panel) if not s['other']]
        cells=[ctrl[f'{panel}/{s}']['cell'] for s in sites]
        train=D.Panel(panel,'train'); profiles=profile(train,cells)
        if split=='train': P=train
        else:
            del train
            P=D.Panel(panel,split)
        dates,cands=SEL.split_candidates(panel,split,'raw' if split=='train' else 'masked')
        assert dates==P.dates
        pi=D.PANELS.index(panel); checked=0
        for q in group.itertuples():
            day=dates.index(q.date); t=int(q.t)
            if t not in cands[day][0]:
                cands[day]=(sorted(set(cands[day][0])|{t}),cands[day][1])
            for j in range(9):
                start=day-j
                if start<0 or dates[start][:7]!=q.date[:7]: break
                rank,gaps=S2.emu_gaps(cands,dates,start,day,t,seed=pi*100000+start)
                if rank is None: continue
                mean,observations,days=online_stat(P,cells,profiles,start,day,t,gaps)
                for s,r,n,nd in zip(sites,mean,observations,days):
                    rows.append(dict(wkey=q.wkey,panel=panel,site=s,date=q.date,t=t,j=j,kdays=j,
                                     mon=q.date[:7],lr=float(r),observations=int(n),days=int(nd)))
                if pop in ('val','pri') and j==min(1,day) and checked<3:
                    poisoned=copy.copy(P)
                    for attr in ('sm','fm','elig'):
                        a=np.array(getattr(P,attr),copy=True)
                        fill=False if a.dtype.kind=='b' else np.nan
                        a[day,t:]=fill; a[day+1:]=fill; setattr(poisoned,attr,a)
                    result=online_stat(poisoned,cells,profiles,start,day,t,gaps)
                    for a,b in zip((mean,observations,days),result): np.testing.assert_array_equal(a,b)
                    checks.append(dict(panel=panel,wkey=q.wkey,j=j,observations=int(observations.sum()),future_poison='PASS'))
                    checked+=1; del poisoned
        del P
        print(f'{pop} {panel}: {len(rows)} rows',flush=True)
    out=pd.DataFrame(rows); out.to_parquet(path)
    (OUT/f'causal_{pop}.json').write_text(json.dumps(dict(status='PASS',checks=checks,n=len(checks),
                                                       n_nonzero_support=sum(r['observations']>0 for r in checks)),indent=2))
    return out


def calibrate(pop):
    import onset5_eval2 as V
    null=raw_stats('null')
    null=null[np.isfinite(null.lr)]
    df=raw_stats(pop).copy()
    df['u']=.5; df['med']=0.; df['nn']=0
    good=np.isfinite(df.lr)
    if good.any():
        u,med,nn=V.null_quantiles(null,df[good])
        df.loc[good,'u']=u; df.loc[good,'med']=med; df.loc[good,'nn']=nn
    eps=1/(2*(df.nn.to_numpy()+1))
    u=np.clip(df.u.to_numpy(),eps,1-eps)
    z=np.clip(ndtri(u),-3,3)
    fire=(df.u.to_numpy()<.02)|(df.u.to_numpy()>.98)
    df['correction']=np.where(fire,z,0.)
    df.to_parquet(OUT/f'calibrated_{pop}.parquet')
    return df


def summarize(scores):
    import onset4_eval as E
    return dict(n=len(scores),reference=E.macro(scores,'s1f'),candidate=E.macro(scores,'rank'),
                delta_ci=E.boot(scores,'rank','s1f'),deterministic_statistic=True,
                panels=scores.groupby('panel')[['s1f','rank']].mean().to_dict(orient='index'))


def passed_march():
    return json.loads((OUT/'result_val_confirm.json').read_text())['delta_ci'][1]>0


def evaluate(pop,phase):
    import onset6_common as C
    import onset6_evalF as F
    import onset4_eval as E
    if phase=='confirm': assert json.loads((OUT/'result_val_dev.json').read_text())['delta_ci'][0]>0
    if pop in ('tol','tolm','pri'): assert passed_march()
    if pop=='pri':
        result=json.loads((OUT/'result_tol_all.json').read_text())
        assert all(result[m]['delta_ci'][0]>=0 for m in ('Jan','Feb'))
    flow=calibrate('tol' if pop=='tolm' else pop)
    frame,stats=C.frames()[pop]
    if pop=='val': frame=frame[pd.to_datetime(frame.date).dt.day%2==(1 if phase=='dev' else 0)]
    fp,ren=F.extra_frames()
    fstat=fp[pop][['wkey','site','j','lr','u','med']].rename(columns=ren)
    windows=E.windows(frame.drop_duplicates('wkey')).set_index('wkey')
    rows=[]
    for j in range(9):
        g=frame.merge(stats[stats.j==j][['wkey','site','lr','u','med']],on=['wkey','site'],how='inner')
        g=g.merge(fstat[fstat.j==j].drop(columns='j'),on=['wkey','site'],how='inner',validate='one_to_one')
        g=g.merge(flow[flow.j==j][['wkey','site','correction']],on=['wkey','site'],how='left',validate='one_to_one')
        if g.empty: continue
        # Real sites must have a replay record even when its observation count is zero.
        assert g.loc[g.site!='other','correction'].notna().all(), 'Replay coverage mismatch'
        correction=g.correction.fillna(0).to_numpy()
        baseline=F.adjF(g.assign(pq=C.s1(g)),.02,1.)
        p=np.where(correction==0,baseline,1/(1+np.exp(-(R.logit(baseline)+correction))))
        decoded={'s1f':C.decode(g,baseline),'rank':C.decode(g,p)}
        fired=g.assign(fire=correction!=0).groupby('wkey').fire.any()
        for key in decoded['s1f']:
            truth,valid=windows.loc[key,['truth_l','valid_l']]
            rows.append(dict(wkey=key,j=j,fire=float(fired[key]),unobs=float(not(decoded['s1f'][key]&set(valid)) or not(decoded['rank'][key]&set(valid))),
                             **{name:E.iou(cells[key],truth,valid) for name,cells in decoded.items()}))
    replay=pd.DataFrame(rows); replay.to_parquet(OUT/f'replay_{pop}_{phase}.parquet')
    scores=replay.groupby('wkey').mean(numeric_only=True).drop(columns='j')
    scores=windows[['panel','date','t']].join(scores,how='inner').reset_index()
    assert len(scores)==frame.wkey.nunique()
    scores.to_parquet(OUT/f'scores_{pop}_{phase}.parquet')
    if pop in ('tol','tolm'):
        result={k:summarize(f) for k,f in [('Jan',scores[scores.date<'2031-02-01']),('Feb',scores[scores.date>='2031-02-01'])]}
    else: result={**summarize(scores),'observed_only':summarize(scores[scores.unobs==0]),'activation_rate':float(scores.fire.mean())}
    (OUT/f'result_{pop}_{phase}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['dev','confirm','april','audit'])
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    check=fixture(); (OUT/'fixture.json').write_text(json.dumps(check,indent=2)); print(check,flush=True)
    if args.mode=='dev': evaluate('val','dev')
    elif args.mode=='confirm':
        evaluate('val','confirm')
        if passed_march(): evaluate('tol','all'); evaluate('tolm','all')
    elif args.mode=='april': evaluate('pri','all')
    elif args.mode=='audit':
        done=OUT/'done'; done.mkdir(exist_ok=True)
        old_scores=pd.read_parquet(OUT/'scores_val_dev.parquet')
        for pop in ('null','val'):
            old=pd.read_parquet(OUT/f'raw_{pop}.parquet')
            old.to_parquet(done/f'raw_{pop}_prebounded.parquet')
            new=raw_stats(pop,force=True)
            pd.testing.assert_frame_equal(old,new,check_exact=True)
        check=json.loads((OUT/'causal_val.json').read_text())
        assert check['n']==24 and check['n_nonzero_support']>=12
        evaluate('val','dev')
        pd.testing.assert_frame_equal(old_scores,pd.read_parquet(OUT/'scores_val_dev.parquet'),check_exact=True)
        receipt=dict(status='PASS',raw_null_and_val='bit-exact',development_scores='bit-exact',
                     source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),causal=check)
        (OUT/'bounded_accessor_audit.json').write_text(json.dumps(receipt,indent=2))
        print('Bounded accessor, full-statistic parity, and development-score parity PASS',flush=True)


if __name__=='__main__': main()
