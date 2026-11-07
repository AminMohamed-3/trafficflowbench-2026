"""Released Task 4 data and independent count-fit diagnostics."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def main_root():
    here = Path(__file__).resolve()
    for directory in here.parents:
        marker = directory / '.git'
        if marker.is_dir():
            return directory
        if marker.is_file():
            pointer = marker.read_text().strip().partition(':')[2].strip()
            target = Path(pointer)
            if not target.is_absolute():
                target = (directory / target).resolve()
            return target.parent.parent.parent
    raise FileNotFoundError('Cannot locate main worktree; set release_root in cfg')


@dataclass
class Problem:
    panel: str
    split: str
    template: pd.DataFrame
    paths: pd.DataFrame
    A: csr_matrix
    counts: np.ndarray
    prior: np.ndarray
    links: list[str]
    C: np.ndarray
    means: np.ndarray
    weights: np.ndarray
    groups: np.ndarray


def load_problem(panel, split, cfg=None):
    cfg = cfg or {}
    if split not in {'train', 'validation', 'private'}:
        raise ValueError(f'Unknown split: {split}')
    release = Path(cfg['release_root']) if 'release_root' in cfg else main_root() / 'data/raw/kaggle_public'
    network = release / 'corridors' / panel / 'network'
    folder = release / 'task4' / panel / split
    paths = pd.read_csv(network / 'path_set.csv', dtype=str)
    incidence = pd.read_csv(network / 'path_link_incidence.csv', dtype=str)
    template = pd.read_csv(folder / 'sample_submission_path_flow.csv', dtype={'path_id': str, 'departure_time': str})
    prior_frame = pd.read_csv(folder / 'synthetic_weak_prior.csv', dtype={'path_id': str})
    counts_frame = pd.read_csv(folder / 'synthetic_link_counts.csv', dtype={'link_id': str})
    ids = paths.path_id.tolist()
    if len(set(ids)) != len(ids) or template.path_id.duplicated().any() or prior_frame.path_id.duplicated().any():
        raise ValueError('Duplicate path IDs')
    if set(template.path_id) != set(ids) or len(template) != len(paths):
        raise ValueError('Template must cover the released path set exactly')
    if counts_frame.link_id.duplicated().any():
        raise ValueError('Duplicate counted links')
    path_index = {value: i for i, value in enumerate(ids)}
    all_links = set(incidence.link_id)
    # Match organizer scorer: only counted links present in the operator.
    counts_frame = counts_frame[counts_frame.link_id.isin(all_links)].copy()
    links = counts_frame.link_id.tolist()
    link_index = {value: i for i, value in enumerate(links)}
    inc = incidence[incidence.link_id.isin(link_index)].drop_duplicates(['link_id', 'path_id'])
    rr = inc.link_id.map(link_index).to_numpy()
    cc = inc.path_id.map(path_index).to_numpy()
    if pd.isna(cc).any():
        raise ValueError('Incidence references an unknown path')
    A = csr_matrix((np.ones(len(rr), dtype=np.float32), (rr, cc)), shape=(len(links), len(ids)))
    counts = pd.to_numeric(counts_frame['count'], errors='raise').to_numpy(dtype=np.float64)
    prior = prior_frame.set_index('path_id').reindex(ids).path_flow.to_numpy(dtype=np.float64)
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.isfinite(prior).all() or (prior < 0).any():
        raise ValueError('Counts and priors must be finite and nonnegative')
    # Identical observed rows are repeated measurements, not extra free cuts.
    # Float32 holds the binary incidence exactly; float64 is used for small solves
    # and accumulations to meet the organizer's 1e-9 parity requirement.
    C, groups, weights = np.unique(A.toarray(), axis=0, return_inverse=True, return_counts=True)
    means = np.bincount(groups, weights=counts) / weights
    return Problem(panel, split, template, paths, A, counts, prior, links, C, means, weights, groups)


def diagnostics(problem, flow):
    flow = np.asarray(flow, dtype=np.float64)
    if flow.shape != problem.prior.shape or not np.isfinite(flow).all() or (flow < 0).any():
        raise ValueError('Invalid path flows')
    loaded = problem.A @ flow
    denom = max(float(problem.counts.sum()), 1e-9)
    dest = problem.paths.destination_zone.to_numpy()
    _, destination_index = np.unique(dest, return_inverse=True)
    total_f, total_b = float(flow.sum()), float(problem.prior.sum())
    share_f = np.bincount(destination_index, weights=flow) / max(total_f, 1e-300)
    share_b = np.bincount(destination_index, weights=problem.prior) / max(total_b, 1e-300)
    positive = flow > 0
    if np.any(positive & (problem.prior <= 0)):
        kl = float('inf')
    else:
        kl = float(np.sum(flow[positive] * np.log(flow[positive] / problem.prior[positive])) - total_f + total_b)
    return {'S_link': max(0., 1. - float(np.abs(loaded - problem.counts).sum()) / denom),
            'Dhat_over_sum_b': float(np.abs(flow - problem.prior).sum()) / max(total_b, 1e-300),
            'share_L1_vs_prior': float(np.abs(share_f - share_b).sum()), 'kl': kl,
            'n_paths': len(flow), 'n_counted_links': len(problem.counts), 'n_cuts': len(problem.C),
            'max_count_residual': float(np.abs(loaded - problem.counts).max()),
            'total_flow': total_f}
