"""Convex weighted Euclidean ODME with the unchanged NNLS count term."""
import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits


def prior_penalty_weights(prior, p=-1., floor_percentile=5., normalization='none'):
    if normalization not in {'none', 'mean_positive_prior'}:
        raise ValueError('Unknown weighted prior normalization')
    if not np.isfinite(p) or not np.isfinite(floor_percentile) or not 0 <= floor_percentile <= 100:
        raise ValueError('Invalid prior power or percentile')
    positive = prior[prior > 0]
    if not len(positive):
        raise ValueError('Positive-prior percentile floor undefined')
    floor = float(np.percentile(positive, floor_percentile))
    scale = float(positive.mean()) if normalization == 'mean_positive_prior' else 1.
    weights = (np.maximum(prior, floor) / scale) ** (-p)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Weighted prior penalty must be finite and positive')
    return weights, floor, scale


def weighted_nnls(problem, lam=.05, p=-1., floor_percentile=5., normalization='none'):
    if not np.isfinite(lam) or lam <= 0 or not np.isfinite(p):
        raise ValueError('Weighted NNLS requires positive lambda and finite p')
    if not np.isfinite(floor_percentile) or not 0 <= floor_percentile <= 100:
        raise ValueError('Prior floor percentile outside [0,100]')
    b = problem.prior
    weights, floor, scale = prior_penalty_weights(b, p, floor_percentile, normalization)
    inverse = 1. / weights
    C = problem.C.astype(np.float64); c = problem.means
    reg = lam / problem.weights; diagonal = np.diag(reg)
    with threadpool_limits(limits=2):
        u = solve((C * inverse) @ C.T + diagonal, C @ b - c, assume_a='pos')
        def value(x):
            f = np.maximum(0., b - (C.T @ x) * inverse)
            return .5 * np.dot(weights * f, f) + np.dot(c, x) + .5 * np.dot(reg * x, x)
        for iteration in range(300):
            f = np.maximum(0., b - (C.T @ u) * inverse)
            residual = C @ f - c - reg * u
            if np.max(np.abs(residual)) < 2e-8:
                gradient = C.T @ (problem.weights * (C @ f - c)) + lam * weights * (f - b)
                violation = float(np.abs(gradient[f > 0]).max(initial=0.))
                violation = max(violation, float(np.maximum(-gradient[f == 0], 0.).max(initial=0.)))
                if violation > 2e-5: raise RuntimeError(f'Weighted NNLS KKT residual {violation}')
                count_residual = problem.A @ f - problem.counts
                return f, {'iterations': iteration, 'kkt_max_abs': violation,
                           'dual_residual_max_abs': float(np.abs(residual).max()),
                           'p': p, 'lambda': lam, 'floor_percentile': floor_percentile,
                           'b_floor': floor, 'weight_min': float(weights.min()),
                           'weight_max': float(weights.max()), 'weight_mean': float(weights.mean()),
                           'weight_normalization': normalization, 'weight_scale': scale,
                           'objective': float(np.dot(count_residual, count_residual) + lam * np.dot(weights * (f - b), f - b))}
            active = f > 0; rows = C[:, active]
            step = solve((rows * inverse[active]) @ rows.T + diagonal, residual, assume_a='pos')
            old = value(u); descent = -np.dot(residual, step); rate = 1.
            # Numerical tolerance only; the returned primal must still meet KKT.
            allowance = 1e-12 * max(1., abs(old))
            for _ in range(60):
                candidate = u + rate * step
                if value(candidate) <= old + 1e-4 * rate * descent + allowance:
                    u = candidate; break
                rate *= .5
            else: raise RuntimeError('Weighted NNLS dual line search failed')
    raise RuntimeError('Weighted NNLS dual did not converge')
