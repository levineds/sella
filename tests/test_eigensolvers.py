import pytest

import numpy as np

from sella.eigensolvers import exact, rayleigh_ritz
from sella.linalg import NumericalHessian

from test_utils import poly_factory, get_matrix


@pytest.mark.parametrize("dim,order,eta,threepoint",
                         [(10, 4, 1e-6, True),
                          (10, 4, 1e-6, False)])
def test_exact(dim, order, eta, threepoint):
    rng = np.random.RandomState(1)

    tol = dict(atol=1e-4, rtol=eta**2)

    poly = poly_factory(dim, order, rng=rng)
    x = rng.normal(size=dim)

    _, g, h = poly(x)

    H = NumericalHessian(lambda x: poly(x)[:2], g0=g, x0=x,
                         eta=eta, threepoint=threepoint)

    l1, V1, AV1 = exact(h)
    l2, V2, AV2 = exact(H)

    np.testing.assert_allclose(l1, l2, **tol)
    np.testing.assert_allclose(np.abs(V1.T @ V2), np.eye(dim), **tol)
    np.testing.assert_allclose(h @ V1, AV1, **tol)
    np.testing.assert_allclose(h @ V2, AV2, **tol)

    P = h + get_matrix(dim, dim, rng=rng) * 1e-3
    l3, V3, AV3 = exact(H, P=P)

    np.testing.assert_allclose(l1, l3, **tol)
    np.testing.assert_allclose(np.abs(V1.T @ V2), np.eye(dim), **tol)


@pytest.mark.parametrize("dim,order,eta,threepoint,gamma,method,maxiter",
                         [(10, 4, 1e-6, False, 0., 'jd0', None),
                          (10, 4, 1e-6, False, 1e-32, 'jd0', 3),
                          (10, 4, 1e-6, True, 1e-1, 'jd0', None),
                          (10, 4, 1e-6, False, 1e-1, 'jd0', None),
                          (10, 4, 1e-6, False, 1e-1, 'lanczos', None),
                          (10, 4, 1e-6, False, 1e-1, 'gd', None),
                          (10, 4, 1e-6, False, 1e-1, 'jd0_alt', None),
                          (10, 4, 1e-6, False, 1e-1, 'mjd0_alt', None),
                          (10, 4, 1e-6, False, 1e-1, 'mjd0', None),
                          ])
def test_rayleigh_ritz(dim, order, eta, threepoint, gamma, method, maxiter):
    rng = np.random.RandomState(1)

    tol = dict(atol=1e-4, rtol=eta**2)

    poly = poly_factory(dim, order, rng=rng)
    x = rng.normal(size=dim)

    _, g, h = poly(x)
    H = NumericalHessian(lambda x: poly(x)[:2], g0=g, x0=x,
                         eta=eta, threepoint=threepoint)

    l1, V1, AV1 = rayleigh_ritz(H, gamma, np.eye(dim), method=method,
                                maxiter=maxiter)
    np.testing.assert_allclose(l1, np.linalg.eigh(V1.T @ AV1)[0], **tol)

    v0 = rng.normal(size=dim)
    rayleigh_ritz(H, gamma, np.eye(dim), method=method, v0=v0,
                  maxiter=maxiter, vref=np.linalg.eigh(h)[1][:, 0])


@pytest.mark.parametrize(
    "diagonal",
    [np.array([0.0, 1.0, 2.0]), np.zeros(3), np.array([-1.0, 0.0, 2.0])],
)
def test_rayleigh_ritz_accepts_exact_zero_residual(diagonal):
    A = np.diag(diagonal)
    lams, vecs, avecs = rayleigh_ritz(A, 0.1, np.eye(3), method='jd0')

    assert np.all(np.isfinite(lams))
    assert np.all(np.isfinite(vecs))
    np.testing.assert_allclose(A @ vecs, avecs, atol=1e-14)
    np.testing.assert_allclose(
        avecs - vecs * lams[np.newaxis, :], 0.0, atol=1e-14
    )
