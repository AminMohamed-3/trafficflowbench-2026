"""Paired date bootstrap from additive state/physics score statistics.

Each transition belongs to its interval-start date. Include only scorer-valid
transitions (including valid midnight transitions) in lwr_error/lwr_rhs.
The caller supplies the organizer's explicit panel-to-family map.
"""
import numpy as np
import pandas as pd

REGIMES = ('R1', 'R2', 'R3')
COLUMNS = ('n_target', 'sq_v', 'sq_q_lane', 'fd_error', 'fd_flow',
           'n_fd_rows', 'lwr_error', 'lwr_rhs', 'n_transition', 'n_low_flow', 'n_submitted')
KEYS = ['panel', 'date', 'regime']


def _validate(rows, families):
    missing = set(KEYS + list(COLUMNS)) - set(rows.columns)
    if missing:
        raise ValueError(f'Missing day statistics: {sorted(missing)}')
    if rows.duplicated(KEYS).any():
        raise ValueError('Duplicate panel/day/regime statistics')
    if set(rows.panel) - set(families):
        raise ValueError('A panel is absent from the explicit family map')
    if set(rows.regime) - set(REGIMES):
        raise ValueError('Unknown regime')
    values = rows[list(COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Score statistics must be finite and nonnegative')
    if (rows.n_low_flow > rows.n_submitted).any():
        raise ValueError('Low-flow count exceeds submitted count')


def _regime_scores(values):
    n, sv, sq, fde, fdq, fdn, lwre, lwrrhs, lwrn, low, submitted = values
    state = (0.54 * max(0., 1 - np.sqrt(sv / n) / 25)
             + 0.46 * max(0., 1 - np.sqrt(sq / n) / 600)) if n else 0.
    fd = max(0., 1 - fde / (fdq + 1e-9)) if fdn else 0.
    if submitted and low / submitted > 0.2:
        fd = 0.
    lwr = max(0., 1 - min(1., lwre / (lwrrhs + 1e-9))) if lwrn else 0.
    physics = (fd + 2 * lwr) / 3
    return state, physics, 0.35 * state + 0.15 * physics


def _encode(rows, families, days):
    rows = rows.sort_values(KEYS)
    panels = sorted(rows.panel.unique())
    day_index = {d: i for i, d in enumerate(days)}
    array = np.zeros((len(panels), 3, len(days), len(COLUMNS)), dtype=float)
    panel_index = {p: i for i, p in enumerate(panels)}
    regime_index = {r: i for i, r in enumerate(REGIMES)}
    for row in rows.itertuples(index=False):
        array[panel_index[row.panel], regime_index[row.regime], day_index[row.date]] = [getattr(row, c) for c in COLUMNS]
    family_indices = [[i for i, p in enumerate(panels) if families[p] == family]
                      for family in sorted({families[p] for p in panels})]
    return array, family_indices


def _score(array, weights, family_indices):
    totals = np.einsum('prdc,d->prc', array, weights)
    scores = np.asarray([[_regime_scores(r) for r in panel] for panel in totals]).mean(axis=1)
    return np.asarray([scores[indices].mean(axis=0) for indices in family_indices]).mean(axis=0)


def aggregate(rows, families):
    _validate(rows, families)
    days = sorted(rows.date.unique())
    array, indices = _encode(rows, families, days)
    scores = _score(array, np.ones(len(days)), indices)
    return dict(zip(('S_state', 'S_physics', 'J_S'), (float(v) for v in scores)))


def paired_ci(incumbent, candidate, families, bootstrap_samples=1000, seed=4701):
    """Resample dates together across panels; preserve regime/family macro means."""
    _validate(incumbent, families)
    _validate(candidate, families)
    a = incumbent.sort_values(KEYS).reset_index(drop=True)
    b = candidate.sort_values(KEYS).reset_index(drop=True)
    if not a[KEYS].equals(b[KEYS]):
        raise ValueError('Paired arms have different panel/day/regime populations')
    for count in ('n_target', 'n_fd_rows', 'n_transition', 'n_submitted'):
        if not np.array_equal(a[count], b[count]):
            raise ValueError(f'Paired arms have different {count}')
    days = sorted(a.date.unique())
    if len(days) < 2 or bootstrap_samples < 2:
        raise ValueError('Bootstrap needs at least two dates and two samples')
    aa, indices = _encode(a, families, days)
    bb, _ = _encode(b, families, days)
    weights = np.ones(len(days))
    point_a, point_b = _score(aa, weights, indices), _score(bb, weights, indices)
    rng = np.random.default_rng(seed)
    deltas = np.empty((bootstrap_samples, 3))
    incumbent_js = np.empty(bootstrap_samples)
    for i in range(bootstrap_samples):
        weights = np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days))
        sa, sb = _score(aa, weights, indices), _score(bb, weights, indices)
        deltas[i], incumbent_js[i] = sb - sa, sa[2]
    lower, upper = np.quantile(deltas, [0.025, 0.975], axis=0)
    inc_ci = np.quantile(incumbent_js, [0.025, 0.975])
    delta_js = float(point_b[2] - point_a[2])
    paired_half_width = float((upper[2] - lower[2]) / 2)
    incumbent_half_width = float((inc_ci[1] - inc_ci[0]) / 2)
    return {'n_dates': len(days), 'bootstrap_samples': bootstrap_samples, 'seed': seed,
            'incumbent': dict(zip(('S_state', 'S_physics', 'J_S'), map(float, point_a))),
            'candidate': dict(zip(('S_state', 'S_physics', 'J_S'), map(float, point_b))),
            'delta_J_S': delta_js, 'ci95': [float(lower[2]), float(upper[2])],
            'paired_ci_half_width': paired_half_width,
            'incumbent_J_S_ci95': [float(inc_ci[0]), float(inc_ci[1])],
            'incumbent_instrument_ci_half_width': incumbent_half_width}


def gate(result, seed_sd=0.):
    """Apply CI exclusion and the pre-comparison B5.2 noise floor."""
    floor = max(float(seed_sd), result['incumbent_instrument_ci_half_width'])
    delta = result['delta_J_S']
    lower, upper = result['ci95']
    if lower <= 0 <= upper or abs(delta) < floor:
        return 'NULL'
    return 'PASS' if delta > 0 else 'FAIL'
