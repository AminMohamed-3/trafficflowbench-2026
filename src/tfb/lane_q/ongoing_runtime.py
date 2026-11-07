"""Resource-only production wrapper; canonical learner/decision stay frozen."""
import os
import lightgbm as lgb
import numpy as np
from .ongoing_model import OngoingModel
from .onset_model import model_frame

class OngoingPredictor(OngoingModel):
    def __init__(self,boosters,metadata,prior,*,num_threads=None):
        super().__init__(boosters,metadata,prior)
        self.num_threads=int(num_threads if num_threads is not None else os.environ.get('TFB_Q_THREADS','3'))
        if self.num_threads<1:raise ValueError('num_threads must be positive')

    @classmethod
    def load(cls,directory,*,num_threads=None):
        base=OngoingModel.load(directory)
        return cls(base.boosters,base.metadata,base.prior,num_threads=num_threads)

    def probabilities(self,frame,panel):
        categories=self.metadata['panels'][panel]['categories']
        return self.boosters[panel].predict(model_frame(frame,self.metadata['features'],categories),num_threads=self.num_threads)


def train(frame,features,*,num_threads=None):
    """Canonical Q04 raw-row fit with a resource-only thread override.

    Dataset, objective, seeds, all model parameters and300 rounds match the
    retained ongoing_model.train source. Only num_threads can vary.
    """
    threads=int(num_threads if num_threads is not None else os.environ.get('TFB_Q_THREADS','3'))
    if threads<1:raise ValueError('num_threads must be positive')
    categories=sorted(frame.site_uid.unique())
    dataset=lgb.Dataset(model_frame(frame,features,categories),
        label=np.array(frame.target,dtype=np.float32,copy=True),
        weight=np.array(frame.weight,dtype=np.float32,copy=True),
        feature_name=['site_uid',*features],categorical_feature=[0])
    params={'objective':'binary','num_leaves':31,'learning_rate':.05,'min_data_in_leaf':100,
        'feature_fraction':1.,'bagging_fraction':1.,'bagging_freq':0,'deterministic':True,'force_col_wise':True,
        'num_threads':threads,'seed':0,'verbosity':-1}
    return lgb.train(params,dataset,num_boost_round=300),categories
