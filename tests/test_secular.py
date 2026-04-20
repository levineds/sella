import pytest
import numpy as np
from scipy.linalg import eigh

from sella.secular import rank1_secular_update, rank2_secular_update
from sella.hessian_update import (
    _MS_TS_BFGS, _MS_TS_BFGS_rank2_factors, update_H
)
from test_utils import get_matrix


class TestRank1Secular:
    @pytest.mark.parametrize("n", [3, 5, 10, 50, 200])
    def test_correctness_positive_sigma(self, n):
        rng = np.random.RandomState(42)
        d = np.sort(rng.normal(size=n))
        z = rng.normal(size=n)
        sigma = abs(rng.normal()) + 0.1

        A = np.diag(d) + sigma * np.outer(z, z)
        evals_ref, evecs_ref = eigh(A)

        mu, Q = rank1_secular_update(d, z, sigma)

        np.testing.assert_allclose(mu, evals_ref, atol=1e-10)
        for k in range(n):
            dot = abs(Q[:, k] @ evecs_ref[:, k])
            assert dot > 1 - 1e-8, f"Eigenvector {k} mismatch: dot={dot}"

    @pytest.mark.parametrize("n", [3, 5, 10, 50, 200])
    def test_correctness_negative_sigma(self, n):
        rng = np.random.RandomState(123)
        d = np.sort(rng.normal(size=n))
        z = rng.normal(size=n)
        sigma = -(abs(rng.normal()) + 0.1)

        A = np.diag(d) + sigma * np.outer(z, z)
        evals_ref, _ = eigh(A)

        mu, Q = rank1_secular_update(d, z, sigma)
        np.testing.assert_allclose(mu, evals_ref, atol=1e-10)

    def test_orthogonality(self):
        rng = np.random.RandomState(7)
        n = 50
        d = np.sort(rng.normal(size=n))
        z = rng.normal(size=n)
        sigma = 1.5

        mu, Q = rank1_secular_update(d, z, sigma)
        np.testing.assert_allclose(Q.T @ Q, np.eye(n), atol=1e-7)

    def test_sorted_output(self):
        rng = np.random.RandomState(99)
        n = 30
        d = np.sort(rng.normal(size=n))
        z = rng.normal(size=n)
        sigma = 2.0

        mu, Q = rank1_secular_update(d, z, sigma)
        assert np.all(mu[1:] >= mu[:-1])

    def test_deflation_zero_components(self):
        rng = np.random.RandomState(55)
        n = 10
        d = np.sort(rng.normal(size=n))
        z = rng.normal(size=n)
        z[0] = 0.0
        z[3] = 0.0
        z[7] = 0.0
        sigma = 1.0

        A = np.diag(d) + sigma * np.outer(z, z)
        evals_ref, _ = eigh(A)

        mu, Q = rank1_secular_update(d, z, sigma)
        np.testing.assert_allclose(mu, evals_ref, atol=1e-10)

    def test_sigma_zero(self):
        n = 5
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = np.ones(n)

        mu, Q = rank1_secular_update(d, z, 0.0)
        np.testing.assert_allclose(mu, d, atol=1e-14)
        np.testing.assert_allclose(Q, np.eye(n), atol=1e-14)

    def test_z_zero(self):
        n = 5
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = np.zeros(n)

        mu, Q = rank1_secular_update(d, z, 1.0)
        np.testing.assert_allclose(mu, d, atol=1e-14)
        np.testing.assert_allclose(Q, np.eye(n), atol=1e-14)

    def test_n_equals_1(self):
        d = np.array([3.0])
        z = np.array([2.0])
        sigma = 0.5

        mu, Q = rank1_secular_update(d, z, sigma)
        np.testing.assert_allclose(mu, [5.0], atol=1e-14)

    def test_n_equals_2(self):
        rng = np.random.RandomState(11)
        d = np.sort(rng.normal(size=2))
        z = rng.normal(size=2)
        sigma = 1.0

        A = np.diag(d) + sigma * np.outer(z, z)
        evals_ref, _ = eigh(A)

        mu, Q = rank1_secular_update(d, z, sigma)
        np.testing.assert_allclose(mu, evals_ref, atol=1e-12)


class TestRank2Secular:
    @pytest.mark.parametrize("n", [5, 10, 50, 200])
    def test_correctness(self, n):
        rng = np.random.RandomState(42)
        A = get_matrix(n, n, symm=True, rng=rng)
        d, V = eigh(A)

        W = rng.normal(size=(n, 2))
        M_raw = rng.normal(size=(2, 2))
        M = 0.5 * (M_raw + M_raw.T)

        B_new = V @ np.diag(d) @ V.T + W @ M @ W.T
        evals_ref, evecs_ref = eigh(B_new)

        evals_new, evecs_new = rank2_secular_update(d, V, W, M)

        np.testing.assert_allclose(evals_new, evals_ref, atol=1e-8)

    def test_reconstruction(self):
        rng = np.random.RandomState(77)
        n = 20
        A = get_matrix(n, n, symm=True, rng=rng)
        d, V = eigh(A)

        W = rng.normal(size=(n, 2))
        M = np.array([[0.5, 0.2], [0.2, -0.3]])

        B_new = V @ np.diag(d) @ V.T + W @ M @ W.T
        evals_new, evecs_new = rank2_secular_update(d, V, W, M)

        B_recon = (evecs_new * evals_new[np.newaxis, :]) @ evecs_new.T
        np.testing.assert_allclose(B_recon, B_new, atol=1e-8)

    def test_orthogonality(self):
        rng = np.random.RandomState(88)
        n = 50
        A = get_matrix(n, n, symm=True, rng=rng)
        d, V = eigh(A)
        W = rng.normal(size=(n, 2))
        M = np.array([[1.0, 0.3], [0.3, -0.5]])

        evals_new, evecs_new = rank2_secular_update(d, V, W, M)
        np.testing.assert_allclose(evecs_new.T @ evecs_new, np.eye(n), atol=1e-8)

    def test_sorted_output(self):
        rng = np.random.RandomState(33)
        n = 30
        A = get_matrix(n, n, symm=True, rng=rng)
        d, V = eigh(A)
        W = rng.normal(size=(n, 2))
        M = np.array([[0.5, 0.1], [0.1, 0.5]])

        evals_new, _ = rank2_secular_update(d, V, W, M)
        assert np.all(evals_new[1:] >= evals_new[:-1])


class TestFactorExtraction:
    @pytest.mark.parametrize("dim", [5, 10, 50])
    def test_factors_match_full_update(self, dim):
        rng = np.random.RandomState(42)
        B = get_matrix(dim, dim, pd=False, symm=True, rng=rng)
        lams, vecs = eigh(B)
        s = rng.normal(size=dim)
        y = rng.normal(size=dim)

        deltaB = _MS_TS_BFGS(B, s[:, None], y[:, None], lams, vecs)
        W, M = _MS_TS_BFGS_rank2_factors(B, s, y, lams, vecs)

        np.testing.assert_allclose(W @ M @ W.T, deltaB, atol=1e-10)


class TestEndToEnd:
    @pytest.mark.parametrize("dim", [10, 50])
    def test_secular_matches_eigh_after_bfgs(self, dim):
        rng = np.random.RandomState(42)
        B = get_matrix(dim, dim, pd=False, symm=True, rng=rng)
        lams, vecs = eigh(B)

        s = rng.normal(size=dim)
        s /= np.linalg.norm(s)
        y = rng.normal(size=dim)

        B_new = update_H(B, s, y, method='TS-BFGS', symm=2,
                         lams=lams, vecs=vecs)
        evals_ref, _ = eigh(B_new)

        W, M = _MS_TS_BFGS_rank2_factors(B, s, y, lams, vecs)
        evals_sec, _ = rank2_secular_update(lams, vecs, W, M)

        np.testing.assert_allclose(evals_sec, evals_ref, atol=1e-8)
