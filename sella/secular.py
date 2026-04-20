import numpy as np
from scipy.linalg import eigh as scipy_eigh
import numba


@numba.jit(nopython=True, cache=True)
def _solve_secular_roots(d, z_sq, sigma, lo, hi, max_iter):
    """Pure bisection for all secular equation roots."""
    n = len(d)
    for _ in range(max_iter):
        for k in range(n):
            mu = 0.5 * (lo[k] + hi[k])
            f = 1.0
            for i in range(n):
                diff = d[i] - mu
                if abs(diff) < 1e-300:
                    diff = 1e-300 if diff >= 0 else -1e-300
                f += sigma * z_sq[i] / diff
            if f * sigma > 0:
                hi[k] = mu
            else:
                lo[k] = mu
    mu_out = np.empty(n)
    for k in range(n):
        mu_out[k] = 0.5 * (lo[k] + hi[k])
    return mu_out


@numba.jit(nopython=True, cache=True)
def _compute_eigenvectors(d, z, mu, n):
    """Compute eigenvector matrix Q[:, k] = z / (d - mu[k]), normalized."""
    Q = np.empty((n, n))
    for k in range(n):
        norm_sq = 0.0
        has_inf = False
        inf_idx = 0
        for i in range(n):
            diff = d[i] - mu[k]
            if abs(diff) < 1e-300:
                has_inf = True
                inf_idx = i
                break
            Q[i, k] = z[i] / diff
            norm_sq += Q[i, k] * Q[i, k]

        if has_inf:
            for i in range(n):
                Q[i, k] = 0.0
            Q[inf_idx, k] = 1.0
        else:
            norm = np.sqrt(norm_sq)
            if norm > 1e-300:
                for i in range(n):
                    Q[i, k] /= norm
            else:
                for i in range(n):
                    Q[i, k] = 0.0
                Q[k if k < n else 0, k] = 1.0
    return Q


def rank1_secular_update(d, z, sigma, tol=1e-14):
    """Eigendecomposition of diag(d) + sigma * outer(z, z).

    Parameters
    ----------
    d : (n,) sorted ascending eigenvalues
    z : (n,) update vector
    sigma : scalar

    Returns
    -------
    mu : (n,) new eigenvalues, sorted ascending
    Q : (n, n) rotation matrix (new eigvecs in old eigenbasis)
    """
    n = len(d)
    mu = d.copy()
    Q = np.eye(n)

    if abs(sigma) < 1e-300 or n == 0:
        return mu, Q

    z_sq = z * z
    z_norm_sq = np.sum(z_sq)

    if z_norm_sq < 1e-300:
        return mu, Q

    deflation_tol = tol * z_norm_sq
    active = z_sq > deflation_tol
    n_active = int(np.sum(active))

    if n_active == 0:
        return mu, Q

    active_idx = np.where(active)[0]
    d_a = np.ascontiguousarray(d[active_idx])
    z_a = np.ascontiguousarray(z[active_idx])
    z_sq_a = np.ascontiguousarray(z_sq[active_idx])
    na = len(d_a)

    lo = np.empty(na)
    hi = np.empty(na)
    if sigma > 0:
        lo[:na - 1] = np.nextafter(d_a[:na - 1], np.inf)
        hi[:na - 1] = np.nextafter(d_a[1:na], -np.inf)
        lo[na - 1] = np.nextafter(d_a[-1], np.inf)
        hi[na - 1] = d_a[-1] + sigma * np.sum(z_sq_a) + abs(d_a[-1]) + 1.0
    else:
        lo[1:] = np.nextafter(d_a[:na - 1], np.inf)
        hi[1:] = np.nextafter(d_a[1:], -np.inf)
        hi[0] = np.nextafter(d_a[0], -np.inf)
        lo[0] = d_a[0] + sigma * np.sum(z_sq_a) - abs(d_a[0]) - 1.0

    invalid = lo >= hi
    if np.any(invalid):
        for k in np.where(invalid)[0]:
            if 0 < k < na:
                gap = d_a[k] - d_a[k - 1]
                lo[k] = d_a[k - 1] + gap * 0.01
                hi[k] = d_a[k] - gap * 0.01

    mu_a = _solve_secular_roots(d_a, z_sq_a, sigma, lo, hi, 30)
    Q_a = _compute_eigenvectors(d_a, z_a, mu_a, na)

    mu_full = d.copy()
    mu_full[active_idx] = mu_a
    Q_full = np.eye(n)
    Q_full[np.ix_(active_idx, active_idx)] = Q_a

    sort_idx = np.argsort(mu_full)
    return mu_full[sort_idx], Q_full[:, sort_idx]


def rank2_secular_update(d, V_old, W, M, tol=1e-14):
    """Eigendecomposition after rank-2 update:
    B_new = V_old @ diag(d) @ V_old.T + W @ M @ W.T"""
    n = len(d)

    Z = V_old.T @ W
    sigmas, P = scipy_eigh(M)
    Z_prime = Z @ P

    mu1, Q1 = rank1_secular_update(d, Z_prime[:, 0], sigmas[0], tol)
    z_double_prime = Q1.T @ Z_prime[:, 1]
    mu2, Q2 = rank1_secular_update(mu1, z_double_prime, sigmas[1], tol)

    Q_combined = Q1 @ Q2

    # Newton-Schulz orthogonalization (1 iteration)
    QtQ = Q_combined.T @ Q_combined
    Q_combined = Q_combined @ (1.5 * np.eye(n) - 0.5 * QtQ)

    # Rayleigh quotient refinement
    ZTQ = Z.T @ Q_combined
    evals_new = np.sum(d[:, np.newaxis] * Q_combined**2, axis=0) + \
        np.sum(ZTQ * (M @ ZTQ), axis=0)

    sort_idx = np.argsort(evals_new)
    evals_new = evals_new[sort_idx]
    Q_combined = Q_combined[:, sort_idx]

    evecs_new = V_old @ Q_combined
    return evals_new, evecs_new
