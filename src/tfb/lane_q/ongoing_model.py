"""Q-04 canonical cell probabilities and exact independent expected-IoU rule."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .runtime import pin_lightgbm
pin_lightgbm()
import lightgbm as lgb
from .onset_model import model_frame
from .ongoing_features import feature_frame,FEATURES
from .predict import forecast as baseline_forecast


def train(frame,features):
    categories=sorted(frame.site_uid.unique())
    x=model_frame(frame,features,categories)
    dataset=lgb.Dataset(x,label=np.array(frame.target,dtype=np.float32,copy=True),
        weight=np.array(frame.weight,dtype=np.float32,copy=True),feature_name=['site_uid',*features],categorical_feature=[0])
    params={'objective':'binary','num_leaves':31,'learning_rate':.05,'min_data_in_leaf':100,
        'feature_fraction':1.,'bagging_fraction':1.,'bagging_freq':0,'deterministic':True,'force_col_wise':True,
        'num_threads':3,'seed':0,'verbosity':-1}
    return lgb.train(params,dataset,num_boost_round=300),categories


def expected_iou_set(probabilities):
    """Top-m maximizes exact E[IoU] under independent Bernoulli cell labels.

    For m>0: E[IoU]=sum(p inside)*E[1/(m+positive cells outside)].
    The suffix Poisson-binomial distribution is accumulated in O(N^2) time,
    O(N) memory. Onset/ongoing selection guarantees a nonempty future queue;
    conditioning on this divides all positive-m scores by the same constant.
    """
    p=np.asarray(probabilities,dtype=float).reshape(-1)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Finite binary probabilities required')
    if not len(p):return np.zeros(0,dtype=np.int8)
    order=np.argsort(-p,kind='stable')
    sorted_p=p[order];prefix=np.cumsum(sorted_p)
    distribution=np.array([1.])
    best=-1.;best_m=1
    for m in range(len(p),0,-1):
        score=prefix[m-1]*np.sum(distribution/(m+np.arange(len(distribution))))
        if score>=best:
            best,best_m=score,m
        probability=sorted_p[m-1]
        extended=np.zeros(len(distribution)+1)
        extended[:-1]+=distribution*(1-probability)
        extended[1:]+=distribution*probability
        distribution=extended
    result=np.zeros(len(p),dtype=np.int8);result[order[:best_m]]=1
    return result


class OngoingModel:
    def __init__(self,boosters,metadata,prior):
        self.boosters,self.metadata,self.prior=boosters,metadata,prior

    @classmethod
    def load(cls,directory):
        directory=Path(directory);metadata=json.loads((directory/'metadata.json').read_text())
        boosters={p:lgb.Booster(model_file=str(directory/f'{p}.txt')) for p in metadata['panels']}
        prior=json.loads((directory/'prior.json').read_text())
        return cls(boosters,metadata,prior)

    def probabilities(self,frame,panel):
        meta=self.metadata['panels'][panel]
        return self.boosters[panel].predict(model_frame(frame,self.metadata['features'],meta['categories']),num_threads=3)

    def forecast(self,view):
        if view.condition()!='queue_ongoing':return baseline_forecast(view,self.prior)
        frame=feature_frame(view,self.metadata['panels'][view.panel]['sites'])
        return expected_iou_set(self.probabilities(frame,view.panel)).reshape(6,len(view.network.links))

    def save(self,directory):
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        for panel,booster in self.boosters.items():booster.save_model(str(directory/f'{panel}.txt'))
        (directory/'metadata.json').write_text(json.dumps(self.metadata,indent=2),encoding='utf-8')
        (directory/'prior.json').write_text(json.dumps(self.prior,indent=2),encoding='utf-8')
