"""Actual Q-04 fitted-model poisoning on copied released training data."""
from pathlib import Path
import shutil
import tempfile
import numpy as np
import pandas as pd
from .windows import Release,online_view,timestamp
from .ongoing_features import feature_frame


def assert_fitted_poison(model,release,panel,date,origin):
    sites=model.metadata['panels'][panel]['sites']
    with tempfile.TemporaryDirectory(prefix='tfb-q4-poison-') as temp:
        root=Path(temp);network=root/'corridors'/panel/'network';network.mkdir(parents=True)
        for name in ['lwr_mainline_topology.csv','fd_parameters.csv']:
            shutil.copyfile(release.root/'corridors'/panel/'network'/name,network/name)
        dates=[date,str((timestamp(date)+pd.Timedelta(days=1)).date())]
        for source in ['mainline_states','mainline_states_masked','ramp_states']:
            for d in dates:
                paths=release.files(panel,'train',source)
                if d not in paths:continue
                path=root/'corridors'/panel/'train'/source/'days'/paths[d].name
                path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(paths[d],path)
        args=(panel,'train',date,origin)
        view=online_view(*args,release=Release(root))
        before=feature_frame(view,sites)
        probability=model.probabilities(before,panel);prediction=model.forecast(view)
        for value in [-9e8,np.nan]:
            for path in (root/'corridors'/panel/'train').rglob('*.parquet'):
                frame=pd.read_parquet(path)
                future=pd.to_datetime(frame.timestamp,utc=True).gt(timestamp(origin))
                for col in ['speed_kmh','flow_vph','occupancy','pct_observed','is_score_eligible']:
                    if col in frame:
                        frame[col]=frame[col].astype(np.float32);frame.loc[future,col]=value
                frame.to_parquet(path,index=False)
            after_view=online_view(*args,release=Release(root))
            after=feature_frame(after_view,sites)
            pd.testing.assert_frame_equal(before,after,check_exact=True)
            np.testing.assert_array_equal(probability,model.probabilities(after,panel))
            np.testing.assert_array_equal(prediction,model.forecast(after_view))
    return {'panel':panel,'date':date,'origin':origin,'features':'bit-identical','probabilities':'bit-identical',
            'binary_predictions':'bit-identical','poison':['garbage','NaN'],'result':'PASS'}
