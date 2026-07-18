import numpy as np

from sella.linalg import ApproximateHessian
from sella.optimize.restricted_step import TrustRegion
from sella.optimize.stepper import (
    PartitionedRationalFunctionOptimization,
    QuasiNewton,
    RationalFunctionOptimization,
)


class _StationaryQuadraticPES:
    dim = 3

    def __init__(self):
        self.H = ApproximateHessian(3, 3, np.eye(3))

    def get_g(self):
        return np.zeros(3)

    def get_scons(self):
        return np.zeros(3)

    def get_H(self):
        return self.H

    def get_fast_restricted_step_data(self, *args):
        return None

    def get_Ufree(self):
        return np.eye(3)

    def get_HL_projected(self, basis):
        return self.H.project(basis)


class _QuadraticPES(_StationaryQuadraticPES):
    def __init__(self, gradient, hessian):
        self.g = np.asarray(gradient)
        self.H = ApproximateHessian(3, 3, np.asarray(hessian))

    def get_g(self):
        return self.g


def test_trust_region_accepts_zero_step_at_stationary_point():
    step, magnitude = TrustRegion(
        _StationaryQuadraticPES(), order=0, delta=0.1
    ).get_s()

    np.testing.assert_array_equal(step, np.zeros(3))
    assert magnitude == 0.0


def _assert_step_derivative(stepper, alpha):
    delta = 1e-6
    _, analytic = stepper.get_s(alpha)
    plus = stepper.get_s(alpha + delta)[0]
    minus = stepper.get_s(alpha - delta)[0]
    numeric = (plus - minus) / (2 * delta)
    np.testing.assert_allclose(analytic, numeric, atol=1e-8, rtol=1e-7)


def test_quasi_newton_ascent_step_derivative():
    H = ApproximateHessian(4, 4, np.diag([-2.0, -0.7, 1.3, 2.8]))
    g = np.array([0.3, -0.2, 0.4, 0.1])
    _assert_step_derivative(QuasiNewton(g, H, order=2), alpha=0.4)


def test_rfo_step_derivative():
    H = ApproximateHessian(4, 4, np.diag([-2.0, -0.7, 1.3, 2.8]))
    g = np.array([0.3, -0.2, 0.4, 0.1])
    _assert_step_derivative(
        RationalFunctionOptimization(g, H, order=1), alpha=0.4
    )


def test_partitioned_rfo_step_derivative():
    H = ApproximateHessian(4, 4, np.diag([-2.0, -0.7, 1.3, 2.8]))
    g = np.array([0.3, -0.2, 0.4, 0.1])
    _assert_step_derivative(
        PartitionedRationalFunctionOptimization(g, H, order=1), alpha=0.4
    )


def test_interior_rfo_reaches_trust_region_boundary():
    pes = _QuadraticPES(
        [0.3, -0.2, 0.4], np.diag([-2.0, -0.7, 1.3])
    )
    delta = 0.01
    step, reported = TrustRegion(
        pes, order=1, delta=delta, method=RationalFunctionOptimization
    ).get_s()

    np.testing.assert_allclose(np.linalg.norm(step), delta, atol=1e-10)
    np.testing.assert_allclose(reported, delta, atol=1e-10)
