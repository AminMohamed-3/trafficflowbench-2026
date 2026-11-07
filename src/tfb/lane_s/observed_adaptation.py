"""Strong S07 released-label masks; all original and OWN omissions remain hidden."""
from hashlib import sha256
import numpy as np
import pandas as pd
from .self_supervision import observation_view,WINDOW_COLUMNS


def random_stream(kind,seed,panel,day):
    token=f"S07 observed adaptation {kind}|{seed}|{panel}|{day}"
    return np.random.default_rng(int.from_bytes(sha256(token.encode()).digest()[:8],"little"))


def make_observed_adaptation(source,mask_seed=8853,block_seed=8854):
    assert not any(name.endswith("_true") for name in vars(source))
    view=observation_view(source)
    original_target=view.target.copy()
    original_history=view.hist.copy()
    original_gap=view.gap.copy()
    known=(view.elig&np.isfinite(view.speed)&np.isfinite(view.flow)&
           ~original_target&~original_history&~original_gap)
    selected=np.zeros_like(known)
    extra_gap=np.zeros_like(known)
    rows=[]
    panel=view.meta["panel"]
    for d,day in enumerate(view.dates):
        rate={"R1":.2,"R2":.3,"R3":.5}[view.regimes[d]]
        rng=random_stream("cell",mask_seed,panel,day)
        selected[d]=known[d]&(rng.random(known[d].shape)<rate)
        month_days=sum(other[:7]==day[:7] for other in view.dates)
        assert month_days>1,"Self-supervised month needs another observed date for levels"
        rng=random_stream("block",block_seed,panel,day)
        if rng.random()>=min(1.,10/month_days):
            continue
        for origin in rng.permutation(np.arange(12,269)):
            horizon=slice(int(origin)+1,int(origin)+19)
            if (original_gap[d,horizon]|original_history[d,horizon]).any():
                continue
            extra_gap[d,horizon]=True
            t=pd.Timestamp(day,tz="UTC")+pd.Timedelta(minutes=5*int(origin))
            rows.append(dict(panel=panel,date=day,
                forecast_origin=t.strftime("%Y-%m-%dT%H:%M:%SZ"),condition="unknown",
                gap_start=(t+pd.Timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                gap_end=(t+pd.Timedelta(minutes=90)).strftime("%Y-%m-%dT%H:%M:%SZ")))
            break
    selected|=known&extra_gap
    assert not (selected&(original_target|original_history|original_gap)).any()
    labels_v=view.speed[selected].copy()
    station=np.nonzero(selected)[2]
    labels_qpl=(view.flow[selected]/view.stations.lanes.to_numpy(np.float32)[station]).astype(np.float32)
    assert np.isfinite(labels_v).all() and np.isfinite(labels_qpl).all()
    view.target=original_target|selected
    view.gap|=extra_gap
    omitted=view.target|view.gap
    for name in ("speed","flow","occ"):
        getattr(view,name)[omitted]=np.nan
    view.pct[omitted]=0
    auxiliary=pd.DataFrame(rows,columns=WINDOW_COLUMNS)
    view.s_windows=pd.concat([view.s_windows.reindex(columns=WINDOW_COLUMNS),auxiliary],ignore_index=True)
    existing=getattr(source,"own_windows",pd.DataFrame(columns=WINDOW_COLUMNS))
    view.own_windows=pd.concat([existing.reindex(columns=WINDOW_COLUMNS),auxiliary],ignore_index=True)
    view.pseudo_target=selected
    assert np.isnan(view.speed[view.target]).all() and np.isnan(view.flow[view.target]).all()
    assert not (view.pseudo_target&(original_target|original_history|original_gap)).any()
    assert not any(name.endswith("_true") for name in vars(view))
    return view,labels_v,labels_qpl,dict(
        n_pseudo_targets=int(selected.sum()),n_extra_blocks=len(rows),
        n_extra_gap_cells=int(extra_gap.sum()),n_extra_gap_labels=int((selected&extra_gap).sum()),
        n_original_targets=int(original_target.sum()),original_target_label_intersection=0,
        original_history_label_intersection=0,original_gap_label_intersection=0,
        label_source="Snapshot of released observations only",hidden_truth_access=False)
