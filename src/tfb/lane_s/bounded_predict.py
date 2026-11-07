"""Additive bounded prediction adapter; frozen numerical/model source unchanged."""
from contextlib import contextmanager
from unittest.mock import patch
from threadpoolctl import threadpool_limits
from . import local_model
from . import own_instrument


@contextmanager
def prediction_threads(limit=2):
    assert isinstance(limit,int) and 1<=limit<=2
    original=local_model.lgb.Booster
    class BoundedBooster:
        def __init__(self,*args,**kwargs):
            self._booster=original(*args,**kwargs)
        def predict(self,*args,**kwargs):
            kwargs["num_threads"]=limit
            return self._booster.predict(*args,**kwargs)
        def __getattr__(self,name):
            return getattr(object.__getattribute__(self,"_booster"),name)
    # Independent process only; do not use this context alongside another
    # training/inference caller in the same Python process.
    with threadpool_limits(limit),patch.object(local_model.lgb,"Booster",BoundedBooster):
        yield


def predict_residual_arrays_cpu2(cache,folder,panel_code,local_constants,asm_values=None):
    with prediction_threads(2):
        return local_model.predict_residual_arrays(cache,folder,panel_code,local_constants,asm_values)


def predict_own_arms_cpu2(view,folder,panel_code,constants_by_arm,bases=None):
    with prediction_threads(2):
        return own_instrument.predict_own_arms(view,folder,panel_code,constants_by_arm,bases)
