"""Separate273-day full-release F3 refit, never used for ADV evaluation."""
import argparse
from datetime import date
import gc
import json
from pathlib import Path
import time
import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
import yaml
from tfb.data import load_cache,guard
from tfb.paths import project_root,panels
from tfb.lane_q.windows import DayGrid,candidate_windows,TASK2_PANELS
from .build_lgb_data import sha,rng_for,prepare_bases
from .build_gap_data import choose_rows
from .evaluate_baselines import subset_cache
from .instruments import apply_frozen_gaps
from .jobs import owned_job
from .travel_features import day_features
from .travel_model import train_case


def select_new_windows(candidates):
    """Same fixed Q01 selection as OPT, extended to the full273 train dates."""
    if candidates.empty:
        return candidates
    work=candidates.copy()
    work['_t']=pd.to_datetime(work.forecast_origin,utc=True)
    minute=work._t.dt.hour*60+work._t.dt.minute
    work=work[minute.between(60,1345)].copy()
    work['_month']=work.date.str[:7]
    rng=np.random.default_rng(106);selected=[]
    for (_,month),group in work.groupby(['panel','_month'],sort=True):
        chosen=[]
        for _,row in group[group.condition.eq('queue_onset')].sort_values('_t').iterrows():
            if all(abs(row._t-other._t)>=pd.Timedelta(minutes=360) for other in chosen):
                chosen.append(row)
            if len(chosen)==5:break
        ongoing=group[group.condition.eq('queue_ongoing')].sort_values('_t')
        minute=ongoing._t.dt.hour*60+ongoing._t.dt.minute
        ongoing=ongoing[ongoing._t.dt.day.le(12)&minute.between(450,1170)&ongoing.history_last_step_persistence_proxy_iou.le(.9)]
        kept=[]
        for j in rng.permutation(len(ongoing)):
            row=ongoing.iloc[int(j)]
            if all(abs(row._t-other._t)>=pd.Timedelta(minutes=155) for other in chosen+kept):
                kept.append(row)
            if len(kept)==5:break
        selected.extend(chosen+kept)
    result=pd.DataFrame(selected).drop(columns=['_t','_month']).sort_values(
        ['panel','forecast_origin','condition']).reset_index(drop=True)
    origins=pd.to_datetime(result.forecast_origin,utc=True)
    for col,offset in [('gap_start',5),('gap_end',90)]:
        result[col]=(origins+pd.Timedelta(minutes=offset)).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    assert result.date.between('2030-06-01','2031-02-28').all()
    return result


def production_windows(folder,config):
    import tfb.lane_q.windows as qmodule
    root=project_root();path=folder/'windows.csv';stamp=folder/'windows.json'
    original=root/config['training_windows_path']
    original_summary=json.loads((original.parent/'summary.json').read_text())
    assert original_summary['q01_sha256']==sha(Path(qmodule.__file__)),'Production Q01 source differs from frozen training-window generator'
    assert sha(original)==config['training_windows_sha256']
    contract=dict(original_OPT_windows_sha256=sha(original),
                  Q01_source_sha256=sha(Path(qmodule.__file__)),
                  source_sha256=sha(Path(__file__)),selection='same fixed5onset/5ongoing monthly; spacing155;seed106',
                  production_only_train273=True)
    if path.exists() and stamp.exists():
        saved=json.loads(stamp.read_text())
        assert saved['contract']==contract and saved['windows_sha256']==sha(path)
        return pd.read_csv(path)
    chunks=[]
    for panel in TASK2_PANELS:
        raw=load_cache(panel,'train');rows=[]
        for day in raw.dates:
            d=raw.day_index(day);v=raw.speed_true[d];observed=np.isfinite(v)
            grid=DayGrid(panel,day,tuple(raw.stations.link_id),v,observed,
                         raw.elig[d],observed&(v<=raw.stations.v_cut.to_numpy()))
            rows.append(candidate_windows(grid))
        chunks.append(select_new_windows(pd.concat(rows,ignore_index=True)))
        del raw
    result=pd.concat(chunks,ignore_index=True).sort_values(
        ['panel','forecast_origin','condition']).reset_index(drop=True)
    parent=pd.read_csv(original)
    semantic=['panel','date','forecast_origin','condition','gap_start','gap_end']
    assert result.loc[result.date.le('2030-12-31'),semantic].reset_index(drop=True).equals(
        parent.sort_values(['panel','forecast_origin','condition'])[semantic].reset_index(drop=True))
    result.to_csv(path,index=False)
    stamp.write_text(json.dumps(dict(contract=contract,windows_sha256=sha(path),
                                     n_windows=len(result)),indent=2)+'\n')
    return result


def build_panel(panel,code,dates,config,windows,folder):
    raw=load_cache(panel,'train')
    assert raw.dates==dates and raw.meta['split']=='train'
    raw_meta=sha(raw.path/'meta.json')
    cache=subset_cache(raw,list(range(273)));del raw
    apply_frozen_gaps(cache,cache.speed_true,cache.flow_true,windows)
    cache.s_windows=windows
    contract=dict(dates=dates,panel_code=code,cache_meta_sha256=raw_meta,
                  configuration=config,windows_sha256=sha(folder.parents[1]/'windows.csv'),
                  source_sha256={n:sha(Path(__file__).parent/n) for n in
                  ['build_production_travel.py','travel_features.py','gap_features.py',
                   'characteristics.py','features.py','build_lgb_data.py','predict.py','asm.py','instruments.py']},
                  row_population='all gap/nonfree/near-congestion plus10pct otherfree weight10; unchanged OPT recipe',
                  production_only_train273=True)
    folder.mkdir(parents=True,exist_ok=True);path=folder/'manifest.json'
    if path.exists():
        saved=json.loads(path.read_text())
        assert saved['contract']==contract
        for name,h in saved['file_sha256'].items():assert sha(folder/name)==h
        return saved
    counts=[int(choose_rows(cache,d,rng_for(panel,day))[0].sum()) for d,day in enumerate(dates)]
    n=sum(counts)
    guard(project_root(),folder,expected_bytes=n*(177*4+2*4+4*2+4+1)+2*cache.speed.size*4)
    bases=prepare_bases(cache,config['asm_parameters'],folder/'asm.npy')
    example,_=day_features(cache,0,bases,code);columns=list(example)
    assert len(columns)==177
    X=open_memmap(folder/'X.npy',mode='w+',dtype=np.float32,shape=(n,177))
    truth=open_memmap(folder/'truth.npy',mode='w+',dtype=np.float32,shape=(n,2))
    metadata=open_memmap(folder/'metadata.npy',mode='w+',dtype=np.int16,shape=(n,4))
    weight=open_memmap(folder/'weight.npy',mode='w+',dtype=np.float32,shape=(n,))
    override=open_memmap(folder/'override.npy',mode='w+',dtype=np.bool_,shape=(n,))
    cursor=0;lanes=cache.stations.lanes.to_numpy(np.float32)
    for d,day in enumerate(dates):
        features,flag=day_features(cache,d,bases,code)
        assert list(features)==columns
        mask,w=choose_rows(cache,d,rng_for(panel,day));ti,si=np.nonzero(mask);count=len(ti)
        assert count==counts[d]
        sl=slice(cursor,cursor+count)
        X[sl]=np.column_stack([a[mask] for a in features.values()])
        truth[sl,0]=cache.speed_true[d][mask];truth[sl,1]=(cache.flow_true[d]/lanes)[mask]
        metadata[sl]=np.column_stack([np.full(count,d),ti,si,
                         np.full(count,int(date.fromisoformat(day).weekday()>=5))])
        weight[sl]=w[mask];override[sl]=flag[mask];cursor+=count
    assert cursor==n and np.isfinite(truth).all()
    for a in [X,truth,metadata,weight,override]:a.flush()
    del X,truth,metadata,weight,override,bases,cache;gc.collect()
    files=['X.npy','truth.npy','metadata.npy','weight.npy','override.npy','asm.npy','asm.json']
    report=dict(contract=contract,columns=columns,n_rows=n,
                file_sha256={name:sha(folder/name) for name in files})
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(panel=panel,production273_rows=n)),flush=True)
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fit',action='store_true');args=ap.parse_args()
    root=project_root();lane=Path(__file__).parent
    config=yaml.safe_load((lane.parents[2]/'exp/0154_s03_characteristics/config.yaml').read_text())
    candidate=root/'artifacts/lane_s/0154'
    assert (candidate/'adv_read1/result.json').exists(),'Complete the frozen214-fit ADV read first'
    adv=json.loads((candidate/'adv_read1/result.json').read_text())
    assert adv['fit_dates'][-1]=='2030-12-31' and len(adv['fit_dates'])==214
    assert adv['verdict']=='PASS','Production candidate preparation requires the frozen numerical gate'
    destination=candidate/'production273';destination.mkdir(parents=True,exist_ok=True)
    dates=pd.date_range('2030-06-01','2031-02-28').strftime('%Y-%m-%d').tolist()
    assert len(dates)==273
    with owned_job('0154',destination,eta_hours=2,ram_gib=7) as receipt:
        windows=production_windows(destination,config)
        data=destination/'data'
        results={panel:build_panel(panel,code,dates,config,windows,data/panel)
                 for code,panel in enumerate(panels())}
        report=dict(id='0154',phase='production273 separate from ADV',dates=dates,
                    panels={p:a['n_rows'] for p,a in results.items()},adv_access=False,
                    refit_config_sha256=sha(lane.parents[2]/'exp/0154_s03_characteristics/config.yaml'))
        (data/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        if args.fit:
            train_case(destination/'model',data,panels(),dates,[],config,'production273')
            report['model_metadata_sha256']=sha(destination/'model/metadata.json')
        report['peak_rss_gib']=receipt['peak_rss_gib']
        (destination/'result.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
