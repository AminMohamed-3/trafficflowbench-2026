"""Q07 full-base model using repaired online transport features."""
import lightgbm as lgb
import numpy as np
from .onset_model import OnsetModel, choose_cells, model_frame
from .predict import forecast as baseline_forecast
from .precursor_features import feature_frame

class PrecursorModel(OnsetModel):
    def forecast(self,view):
        if view.condition()=="queue_ongoing":return baseline_forecast(view,self.prior)
        sites=self.metadata["sites"][view.panel]
        frame=feature_frame(view,sites,prior_context=self.context)
        cells=set(choose_cells(sites,self.probabilities(frame),close_ratio=self.metadata.get("close_ratio",.8)))
        result=np.zeros((6,len(view.network.links)),dtype=np.int8)
        result[5]=[int(link in cells) for link in view.network.links]
        return result


def train(frame,features,*,num_threads=3,seed=0):
    """Same Q03 hyperparameters, with explicit seed for the required noise audit."""
    if int(num_threads)<1:raise ValueError("num_threads must be positive")
    categories=sorted(frame.site_uid.unique())
    data=lgb.Dataset(model_frame(frame,features,categories),
        label=np.array(frame.target,dtype=np.float32,copy=True),
        weight=np.array(frame.weight,dtype=np.float32,copy=True),
        feature_name=["site_uid",*features],categorical_feature=[0])
    params={"objective":"binary","num_leaves":15,"learning_rate":.05,
        "min_data_in_leaf":30,"feature_fraction":1.,"bagging_fraction":1.,"bagging_freq":0,
        "deterministic":True,"force_col_wise":True,"num_threads":int(num_threads),"seed":int(seed),"verbosity":-1}
    return lgb.train(params,data,num_boost_round=300),categories
