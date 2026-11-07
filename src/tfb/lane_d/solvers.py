"""Memory-safe solvers for the released count constraints."""
import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits


def ridge_nnls(problem, lam=0.05):
    """Exactly the organizer objective, solved in count-space via its dual.

    For repeated cut rows, sum ||Af-c||^2 = sum w*(Cf-mean(c))^2
    plus a constant. The unique minimizer is f=max(0,b-C.T@u), where
    C f - mean(c) - lambda*u/w = 0. No paths-by-paths matrix is allocated.
    """
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError('NNLS lambda must be positive')
    C = problem.C.astype(np.float64)
    b, c = problem.prior, problem.means
    reg = lam / problem.weights
    diagonal = np.diag(reg)
    with threadpool_limits(limits=2):
        u = solve(C @ C.T + diagonal, C @ b - c, assume_a='pos')
        def value(x):
            f = np.maximum(0., b - C.T @ x)
            return 0.5 * np.dot(f, f) + np.dot(c, x) + 0.5 * np.dot(reg*x, x)
        for iteration in range(200):
            f = np.maximum(0., b - C.T @ u)
            residual = C @ f - c - reg * u
            if np.max(np.abs(residual)) < 2e-8:
                # Check the primal KKT certificate with original multiplicities.
                gradient = C.T @ (problem.weights * (C @ f - c)) + lam*(f-b)
                violation = np.max(np.abs(gradient[f > 0])) if np.any(f > 0) else 0.
                violation = max(violation, float(np.maximum(-gradient[f == 0], 0.).max(initial=0.)))
                if violation > 2e-5:
                    raise RuntimeError(f'NNLS KKT residual {violation}')
                return f, {'iterations': iteration, 'kkt_max_abs': float(violation)}
            active = C[:, f > 0]
            step = solve(active @ active.T + diagonal, residual, assume_a='pos')
            old = value(u)
            descent = -np.dot(residual, step)
            rate = 1.
            for _ in range(50):
                candidate = u + rate*step
                if value(candidate) <= old + 1e-4*rate*descent + 1e-8:
                    u = candidate
                    break
                rate *= 0.5
            else:
                raise RuntimeError('NNLS dual line search failed')
    raise RuntimeError('NNLS dual did not converge')


def max_entropy(problem):
    """KL projection onto the unique counted cuts (exact when rows agree)."""
    if (problem.prior <= 0).any():
        raise ValueError('Canonical KL projection requires a positive prior')
    C = problem.C.astype(np.float64)
    b, c = problem.prior, problem.means
    with threadpool_limits(limits=2):
        u = np.zeros(len(c))
        def value(x):
            exponent = C.T @ x
            if exponent.max(initial=0.) > 700:
                return float('inf')
            return np.sum(b * np.exp(exponent)) - np.dot(c, x)
        for iteration in range(100):
            flow = b * np.exp(C.T @ u)
            gradient = C @ flow - c
            if np.max(np.abs(gradient)) < 1e-7:
                return flow, {'iterations':iteration, 'cut_max_abs_residual':float(np.abs(gradient).max()),
                              'multipliers':u.tolist()}
            hessian = (C * flow) @ C.T
            step = solve(hessian, gradient, assume_a='pos')
            old, rate = value(u), 1.
            directional = np.dot(gradient, step)
            for _ in range(50):
                candidate = u - rate*step
                if value(candidate) <= old - 1e-4*rate*directional + 1e-8:
                    u = candidate
                    break
                rate *= .5
            else:
                raise RuntimeError('Max-entropy dual line search failed')
    raise RuntimeError('Max-entropy dual did not converge')


def lognormal_map(problem, sigma=1., tau=10., initial=None, maxiter=500):
    """Soft-count log-normal MAP in log-flow coordinates, damped Newton.

    Hessian = diagonal + U.T@U. Woodbury reduces Newton solves to the
    counted-cut dimension. This objective is nonconvex; the caller compares
    two declared initializations by the same objective, never by hidden truth.
    """
    if sigma <= 0 or tau <= 0 or (problem.prior <= 0).any():
        raise ValueError('Log-normal MAP requires positive sigma, tau and prior')
    C = problem.C.astype(np.float64)
    b, c, w = problem.prior, problem.means, problem.weights
    logb = np.log(b)
    x = logb.copy() if initial is None else np.log(np.maximum(initial, 1e-12))
    with threadpool_limits(limits=2):
        def evaluate(z):
            if z.max(initial=0.) > 700:
                return float('inf'), None, None, None
            f = np.exp(z)
            residual = C@f-c
            count_gradient = C.T @ (w*residual / tau**2)
            delta = z-logb
            objective = .5*np.dot(delta,delta)/sigma**2 + .5*np.dot(w*residual,residual)/tau**2
            gradient = delta/sigma**2 + f*count_gradient
            return objective,gradient,f,count_gradient
        for iteration in range(maxiter):
            objective, gradient, flow, count_gradient = evaluate(x)
            gradmax = float(np.abs(gradient).max())
            if gradmax < 2e-5:
                return flow, {'iterations':iteration,'objective':float(objective),'gradient_max_abs':gradmax,
                              'sigma':sigma,'tau':tau,'converged':True}
            diagonal = 1./sigma**2 + flow*count_gradient
            damping = max(0., 1e-3-float(diagonal.min()))
            diagonal = diagonal + damping
            U = (C*flow)*np.sqrt(w[:,None])/tau
            dinvg = gradient/diagonal
            inner = np.eye(len(c)) + (U/diagonal)@U.T
            step = dinvg - (U.T@solve(inner,U@dinvg,assume_a='pos'))/diagonal
            descent = float(np.dot(gradient,step))
            if not np.isfinite(descent) or descent <= 0:
                raise RuntimeError('Log-normal MAP lost a descent direction')
            old = objective
            rate=1.
            for _ in range(50):
                candidate=x-rate*step
                value=evaluate(candidate)[0]
                if value <= old-1e-4*rate*descent + 1e-8:
                    x=candidate
                    break
                rate*=.5
            else:
                raise RuntimeError('Log-normal MAP line search failed')
        # Negative diagonal curvature can make the safe diagonal damping slow.
        # A matrix-free trust-region solve handles the same full Hessian directly.
        from scipy.optimize import minimize
        newton_gradient = float(np.abs(evaluate(x)[1]).max())
        def hessp(z, direction):
            _, _, f, count_gradient = evaluate(z)
            return (1./sigma**2 + f*count_gradient)*direction + f*(C.T@(w*(C@(f*direction))/tau**2))
        trust = minimize(lambda z:evaluate(z)[:2], x, method='trust-krylov', jac=True, hessp=hessp,
                         options={'gtol':2e-5,'maxiter':500})
        objective,gradient,flow,_=evaluate(trust.x)
        gradmax=float(np.abs(gradient).max())
        return flow, {'iterations':maxiter+int(trust.nit),'objective':float(objective),
                      'gradient_max_abs':gradmax,'sigma':sigma,'tau':tau,'converged':gradmax<2e-5,
                      'numeric_repair':'matrix-free trust-krylov', 'newton_gradient_max_abs':newton_gradient,
                      'trust_message':str(trust.message)}


def estimate(problem, cfg=None):
    cfg=dict(cfg or {})
    method=cfg.get('method','nnls')
    if method=='nnls':
        scale=float(cfg.get('prior_scale',1.))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('NNLS prior scale must be finite and positive')
        if scale == 1.:
            return ridge_nnls(problem,float(cfg.get('lambda',.05)))
        from dataclasses import replace
        flow,info=ridge_nnls(replace(problem,prior=problem.prior*scale),float(cfg.get('lambda',.05)))
        info['prior_scale']=scale
        return flow,info
    if method=='weighted_nnls':
        from .weighted import weighted_nnls
        return weighted_nnls(problem, float(cfg.get('lambda',.05)), float(cfg.get('p',-1.)),
                             float(cfg.get('floor_percentile',5.)), cfg.get('weight_normalization','none'))
    if method=='maxent':
        from dataclasses import replace
        scale=float(cfg.get('prior_scale',1.))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Max-entropy prior scale must be finite and positive')
        flow,info=max_entropy(problem if scale==1. else replace(problem,prior=problem.prior*scale))
        info['prior_scale']=scale
        return flow,info
    if method=='lognormal':
        e2,_=max_entropy(problem)
        arms=[lognormal_map(problem,float(cfg.get('sigma',1.)),float(cfg.get('tau',10.)),initial=initial,
                            maxiter=int(cfg.get('maxiter',500))) for initial in [e2,problem.prior]]
        flow,info=min(arms,key=lambda arm:arm[1]['objective'])
        if not info['converged']:
            raise RuntimeError(f'Log-normal MAP did not converge: {info}')
        info['initialization_objectives']=[arm[1]['objective'] for arm in arms]
        info['initialization_converged']=[arm[1]['converged'] for arm in arms]
        return flow,info
    if method=='geometric':
        alpha=float(cfg.get('alpha',1.))
        if not np.isfinite(alpha):
            raise ValueError('Geometric alpha must be finite')
        e2,info=max_entropy(problem)
        return problem.prior*np.exp(alpha*np.log(e2/problem.prior)), {'alpha':alpha,'projection':info}
    raise ValueError(f'Unknown ODME method: {method}')
