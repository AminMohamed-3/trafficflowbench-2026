"""Train-only Q-03 binary site model and expected-IoU cell-set decision."""
from __future__ import annotations
import json
from pathlib import Path

import os
import lightgbm as lgb
import numpy as np
import pandas as pd

from .onset_features import feature_frame, PriorDayContext, STATIC_FEATURES, LOCAL_FEATURES, NUMERIC_FEATURES
from .predict import forecast as baseline_forecast

DOSES = {"F0":STATIC_FEATURES, "Fhalf":STATIC_FEATURES+LOCAL_FEATURES, "Ffull":NUMERIC_FEATURES}


def model_frame(frame, features, categories):
    category = pd.Categorical(frame.site_uid,categories=categories).codes.astype(np.float32)
    category[category<0]=np.nan
    numeric = frame[features].to_numpy(dtype=np.float32,copy=True)
    return np.ascontiguousarray(np.column_stack([category,numeric]),dtype=np.float32)


def train(frame, features, *, num_threads=None):
    threads=int(num_threads if num_threads is not None else os.environ.get("TFB_Q_THREADS","4"))
    if threads<1:raise ValueError("num_threads must be positive")
    categories = sorted(frame.site_uid.unique())
    x = model_frame(frame,features,categories)
    data=lgb.Dataset(x,label=np.array(frame.target,dtype=np.float32,copy=True),
        weight=np.array(frame.weight,dtype=np.float32,copy=True),
        feature_name=["site_uid",*features],categorical_feature=[0])
    params={"objective":"binary","num_leaves":15,"learning_rate":.05,
        "min_data_in_leaf":30,"feature_fraction":1.,"bagging_fraction":1.,"bagging_freq":0,
        "deterministic":True,"force_col_wise":True,"num_threads":threads,"seed":0,"verbosity":-1}
    return lgb.train(params,data,num_boost_round=300),categories


def choose_cells(sites, probabilities, *, close_ratio=.8):
    """Expected IoU under independent one/two-site scenarios conditioned on a queue.

    Truth cell sets follow each site's train-only empirical distribution. Candidate
    sets are its observed cell sets, plus the union of the top two modal sets when
    their probabilities are close. Three-or-more-site scenarios are excluded.
    """
    available = [i for i,s in enumerate(sites) if s["cell_sets"]]
    if not available:
        return ()
    p = np.clip(np.asarray(probabilities,dtype=float),1e-6,1-1e-6)
    distributions = {}
    candidates = set()
    for i in available:
        entries = sites[i]["cell_sets"]
        total = sum(e["count"] for e in entries)
        distributions[i] = [(frozenset(e["links"]),e["count"]/total) for e in entries]
        candidates.update(cells for cells,_ in distributions[i])
    ranking = sorted(available,key=lambda i:(-p[i],min(sites[i]["positions"])))
    if len(ranking)>=2 and p[ranking[1]]/p[ranking[0]]>=close_ratio:
        candidates.add(distributions[ranking[0]][0][0] | distributions[ranking[1]][0][0])
    # Common product of (1-p) cancels from the argmax. Odds give exactly
    # independent singleton and pair masses, without numerical underflow.
    odds = p/(1-p)
    scenarios = []
    for i in available:
        scenarios.extend((cells,odds[i]*weight) for cells,weight in distributions[i])
    for a,i in enumerate(available):
        for j in available[a+1:]:
            for ci,wi in distributions[i]:
                for cj,wj in distributions[j]:
                    scenarios.append((ci|cj,odds[i]*odds[j]*wi*wj))
    positions = {link:pos for s in sites for link,pos in zip(s["links"],s["positions"])}
    def value(cells):
        return sum(weight*len(cells&truth)/len(cells|truth) for truth,weight in scenarios)
    chosen = min(candidates,key=lambda cells:(-value(cells),tuple(sorted(positions[x] for x in cells))))
    return tuple(sorted(chosen,key=positions.get))


class OnsetModel:
    def __init__(self, booster, metadata, prior, *, num_threads=None):
        self.booster,self.metadata,self.prior=booster,metadata,prior
        self.num_threads=int(num_threads if num_threads is not None else os.environ.get("TFB_Q_THREADS","4"))
        if self.num_threads<1:raise ValueError("num_threads must be positive")
        self.context=PriorDayContext()

    @classmethod
    def load(cls,directory, *, num_threads=None):
        directory=Path(directory)
        metadata=json.loads((directory/"metadata.json").read_text())
        prior=json.loads((directory/"prior.json").read_text())
        return cls(lgb.Booster(model_file=str(directory/"model.txt")),metadata,prior,num_threads=num_threads)

    def probabilities(self, frame):
        x=model_frame(frame,self.metadata["features"],self.metadata["categories"])
        return self.booster.predict(x,num_threads=self.num_threads)

    def forecast(self,view):
        if view.condition()=="queue_ongoing":
            return baseline_forecast(view,self.prior)
        sites=self.metadata["sites"][view.panel]
        frame=feature_frame(view,sites,prior_context=self.context)
        cells=set(choose_cells(sites,self.probabilities(frame),close_ratio=self.metadata.get("close_ratio",.8)))
        result=np.zeros((6,len(view.network.links)),dtype=np.int8)
        result[5]=[int(link in cells) for link in view.network.links]
        return result

    def save(self,directory):
        directory=Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        self.booster.save_model(str(directory/"model.txt"))
        (directory/"metadata.json").write_text(json.dumps(self.metadata,indent=2),encoding="utf-8")
        (directory/"prior.json").write_text(json.dumps(self.prior,indent=2),encoding="utf-8")
