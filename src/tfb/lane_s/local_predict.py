"""Corrected F4 observed-local production inference on one released split, with a mandatory273-day refit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from tfb.data import load_cache, indices
from tfb.paths import project_root, release_root, panels
from .local_model import predict_residual_arrays
from tfb.split_constants import estimate_constants,__file__ as provider_file
from .build_lgb_data import sha
from .own_instrument import PROVIDER_SHA
from .self_supervision import observation_view
from .bounded_predict import predict_residual_arrays_cpu2
from .predict_lgb import released_windows


def predict_state(panel, split, cfg):
    assert split in ('validation','private')
    folder=Path(cfg['model_dir'])
    if not folder.is_absolute():
        folder=project_root()/folder
    metadata=json.loads((folder/'metadata.json').read_text())
    assert len(metadata['fit_dates'])==273
    assert metadata['fit_dates'][0]=='2030-06-01' and metadata['fit_dates'][-1]=='2031-02-28'
    assert metadata['integrity']['fold']=='production273','Never serve the214-day ADV model'
    assert metadata['columns'] and len(metadata['columns'])==177
    cache=load_cache(panel,split)
    assert cache.meta['split']==split and cache.meta['panel']==panel
    assert sha(provider_file)==sha(project_root()/'src/tfb/split_constants.py')==PROVIDER_SHA
    cache.s_windows=released_windows(panel,split)
    view=observation_view(cache)
    view.s_windows=cache.s_windows
    local=estimate_constants(view,hidden=view.target|view.gap)
    threads=int(cfg.get('prediction_threads',8));assert threads in (2,8)
    if threads==2:pv,pq=predict_residual_arrays_cpu2(view,folder,panels().index(panel),local)
    else:pv,pq=predict_residual_arrays(view,folder,panels().index(panel),local)
    columns=['panel','timestamp','station_id','link_id','mask_regime']
    template=release_root()/'task1'/panel/split/'sample_submission_state.csv'
    output=pd.read_csv(template,usecols=columns,dtype=str)
    assert len(output)==int(cache.target.sum()) and not output.duplicated(columns).any()
    key=indices(output,{d:i for i,d in enumerate(cache.dates)},
                dict(zip(cache.stations.station_id,cache.stations.idx)),'station_id')
    assert cache.target[key].all() and output.panel.eq(panel).all()
    assert np.array_equal(output.link_id,output.station_id.map(
        cache.stations.set_index('station_id').link_id))
    assert np.array_equal(output.mask_regime,cache.regimes[key[0]])
    output['speed_kmh']=pv[key].astype(float)
    output['flow_vph']=pq[key].astype(float)
    assert np.isfinite(output[['speed_kmh','flow_vph']]).all().all()
    assert output.speed_kmh.between(3,120.75).all() and output.flow_vph.ge(50).all()
    cap=output.station_id.map(cache.stations.set_index('station_id').capacity_vph)
    assert (output.flow_vph<=1.3*cap).all()
    return output[columns+['speed_kmh','flow_vph']]
