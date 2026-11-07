"""Split-local S-03 production inference; released observations only."""
from pathlib import Path
import numpy as np,pandas as pd
from tfb.data import load_cache,indices
from tfb.paths import project_root,release_root,panels
from .gap_model import predict_residual_arrays

def released_windows(panel,split):
    path=release_root()/'task2'/panel/split/'window_index.csv'
    if not path.exists():return pd.DataFrame(columns=['panel','date','forecast_origin','condition','gap_start','gap_end'])
    frame=pd.read_csv(path,usecols=['date','forecast_origin','condition'])
    frame['panel']=panel;origin=pd.to_datetime(frame.forecast_origin,utc=True)
    for key,minutes in [('gap_start',5),('gap_end',90)]:
        frame[key]=(origin+pd.Timedelta(minutes=minutes)).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    return frame

def predict_state(panel,split,cfg):
    assert split in ('validation','private'),'Production entry point is split-local'
    folder=Path(cfg['model_dir'])
    if not folder.is_absolute():folder=project_root()/folder
    import json
    metadata=json.loads((folder/'metadata.json').read_text())
    assert metadata['fit_dates'][0]=='2030-06-01' and metadata['fit_dates'][-1]=='2031-02-28','Production must refit all released train, separate from ADV fit214'
    cache=load_cache(panel,split);cache.s_windows=released_windows(panel,split)
    pv,pq=predict_residual_arrays(cache,folder,panels().index(panel))
    template=release_root()/'task1'/panel/split/'sample_submission_state.csv'
    columns=['panel','timestamp','station_id','link_id','mask_regime']
    output=pd.read_csv(template,usecols=columns,dtype=str)
    assert len(output)==int(cache.target.sum()) and not output.duplicated(columns).any()
    key=indices(output,{d:i for i,d in enumerate(cache.dates)},dict(zip(cache.stations.station_id,cache.stations.idx)),'station_id')
    assert cache.target[key].all()
    output['speed_kmh']=pv[key].astype(float);output['flow_vph']=pq[key].astype(float)
    return output

