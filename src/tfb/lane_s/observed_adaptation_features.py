"""Unchanged177 F4 fields and canonical gap/near-congestion row sampling for S07."""
from datetime import date
from types import SimpleNamespace
import numpy as np
from tfb.split_constants import estimate_constants
from .build_lgb_data import rng_for
from .features import training_sample
from .travel_features import day_features
from .own_instrument import own_context_for_day
from .local_priors import fill_scored_dynamic


def day_levels(view,day):
    assert not any(k.endswith("_true") for k in vars(view))
    dates=[d for d in view.dates if d[:7]==day[:7] and d!=day]
    assert dates and day not in dates
    return estimate_constants(view,dates=dates,hidden=view.target|view.gap)


def select_rows(source,augmented,d):
    assert not any(k.endswith("_true") for k in vars(source))
    target=augmented.pseudo_target[d]
    assert np.isfinite(source.speed[d][target]).all()
    sampler=SimpleNamespace(stations=source.stations,target=target[None])
    selected,weight=training_sample(sampler,0,source.speed[d],
                                  rng_for(source.meta["panel"],source.dates[d]))
    selected|=target&augmented.gap[d]
    weight=np.where(augmented.gap[d],1.,weight).astype(np.float32)
    assert (selected<=target).all() and not (selected&source.target[d]).any()
    assert (selected[target&augmented.gap[d]]).all()
    return selected,weight


def feature_rows(view,d,bases,columns,panel_code,structural,constants,selected=None,context=None):
    assert not any(k.endswith("_true") for k in vars(view))
    context=own_context_for_day(view,d) if context is None else context
    fields,override=day_features(view,d,bases,panel_code,context)
    assert list(fields)==columns and len(columns)==177
    selected=view.pseudo_target[d] if selected is None else selected
    assert (selected<=view.pseudo_target[d]).all()
    ti,si=np.nonzero(selected)
    x=np.column_stack([value[selected] for value in fields.values()]).astype(np.float32)
    weekend=np.full(len(ti),int(date.fromisoformat(view.dates[d]).weekday()>=5),dtype=np.int16)
    fill_scored_dynamic(x,columns,si,ti,weekend,override[selected],constants,structural)
    assert not np.isinf(x).any()
    return x,np.column_stack([np.full(len(ti),d),ti,si,weekend]).astype(np.int16),override[selected]
