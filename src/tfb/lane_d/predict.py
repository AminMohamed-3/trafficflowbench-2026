"""Task 4 prediction contract for the shared pipeline.

Default: organizer NNLS, lambda 0.05. cfg supports method in nnls, maxent,
lognormal, geometric; method parameters lambda, sigma/tau, alpha; release_root.
Raw estimates are never calibrated by unobservable local ODME truth terms.
"""
import numpy as np
import pandas as pd
from .data import load_problem
from .solvers import estimate


def predict_odme(panel, split, cfg=None) -> pd.DataFrame:
    cfg = dict(cfg or {})
    problem = load_problem(panel, split, cfg)
    flow, _ = estimate(problem, cfg)
    if not np.isfinite(flow).all() or (flow < 0).any():
        raise ValueError('ODME estimator returned invalid flow')
    by_path = pd.Series(flow, index=problem.paths.path_id)
    output = problem.template[['panel', 'departure_time', 'path_id']].copy()
    output['path_flow'] = output.path_id.map(by_path)
    if output.path_flow.isna().any():
        raise ValueError('Missing ODME template paths')
    return output
