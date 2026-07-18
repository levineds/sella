from typing import Optional, Tuple, Type, List
import logging

import numpy as np
from scipy.linalg import eigh

logger = logging.getLogger(__name__)

from sella.linalg import ApproximateHessian


def _eigh_symmetric(
    A: np.ndarray,
    sym_rtol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetric eigensolver with LAPACK-driver fallback."""
    asym = np.linalg.norm(A - A.T, ord='fro')
    scale = max(np.linalg.norm(A, ord='fro'), 1.0)
    if not np.isfinite(asym) or asym > sym_rtol * scale:
        raise ValueError(
            "Symmetric eigensolver received a non-symmetric matrix: "
            f"||A - A.T||_F={asym:.3e}, ||A||_F={scale:.3e}"
        )
    try:
        return eigh(A)
    except np.linalg.LinAlgError:
        for driver in ('evd', 'evx', 'ev'):
            try:
                return eigh(A, driver=driver)
            except np.linalg.LinAlgError:
                continue
        return np.linalg.eigh(A)


# Classes for optimization algorithms (e.g. MMF, Newton, RFO)
class BaseStepper:
    alpha0: Optional[float] = None
    alphamin: Optional[float] = None
    alphamax: Optional[float] = None
    # Whether the step size increases or decreases with increasing alpha
    slope: Optional[float] = None
    # Whether get_s is smooth enough for Newton to converge reliably
    newton_safe: bool = True
    synonyms: List[str] = []

    def __init__(
        self,
        g: np.ndarray,
        H: ApproximateHessian,
        order: int = 0,
        d1: Optional[np.ndarray] = None,
    ) -> None:
        self.g = g
        self.H = H
        self.order = order
        self.d1 = d1
        self._stepper_init()

    @classmethod
    def match(cls, name: str) -> bool:
        return name in cls.synonyms

    def _stepper_init(self) -> None:
        raise NotImplementedError  # pragma: no cover

    def get_s(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError  # pragma: no cover


class NaiveStepper(BaseStepper):
    synonyms = []  # No synonyms, we don't want someone using this accidentally
    alpha0 = 0.5
    alphamin = 0.
    alphamax = 1.
    slope = 1.

    def __init__(self, dx: np.ndarray) -> None:
        self.dx = dx

    def get_s(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        return alpha * self.dx, self.dx


class QuasiNewton(BaseStepper):
    alpha0 = 0.
    alphamin = 0.
    alphamax = np.inf
    slope = -1
    synonyms = [
        'qn',
        'quasi-newton',
        'quasi newton',
        'newton',
        'mmf',
        'minimum mode following',
        'minimum-mode following',
        'dimer',
    ]

    def _stepper_init(self) -> None:
        # Get eigenvalues and eigenvectors from the Hessian
        # If not already computed, compute them now
        if self.H.evals is None:
            H_array = self.H.asarray()
            self.H.evals, self.H.evecs = _eigh_symmetric(H_array)

        self.L = np.abs(self.H.evals)
        self.L[:self.order] *= -1

        self.V = self.H.evecs
        self.Vg = self.V.T @ self.g

        self.ones = np.ones_like(self.L)
        self.ones[:self.order] = -1

    def get_s(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        denom = self.L + alpha * self.ones
        sproj = self.Vg / denom
        s = -self.V @ sproj
        dsda = self.V @ (self.ones * sproj / denom)
        return s, dsda


class QuasiNewtonIRC(QuasiNewton):
    synonyms = []

    def _stepper_init(self) -> None:
        QuasiNewton._stepper_init(self)
        self.Vd1 = self.V.T @ self.d1

    def get_s(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        denom = np.abs(self.L) + alpha
        sproj = -(self.Vg + alpha * self.Vd1) / denom
        s = self.V @ sproj
        dsda = -self.V @ ((sproj + self.Vd1) / denom)
        return s, dsda


class RationalFunctionOptimization(BaseStepper):
    alpha0 = 1.
    alphamin = 0.
    alphamax = 1.
    slope = 1.
    newton_safe = False
    synonyms = ['rfo', 'rational function optimization']

    def _stepper_init(self) -> None:
        self.A = np.block([
            [self.H.asarray(), self.g[:, np.newaxis]],
            [self.g, 0]
        ])
        self._interior_full_step = None

    def _get_eigenstep(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        A = self.A * alpha
        A[:-1, :-1] *= alpha
        L, V = _eigh_symmetric(A)

        idx = self.order
        # For saddle-point search (order > 0): when eigenvalues near the
        # target index are nearly degenerate, select the eigenvector with
        # the largest |V[-1, j]| (denominator in the RFO step formula).
        # This gives the most well-defined step and avoids noise-driven
        # eigenvector swaps that would change the ascent direction.
        if self.order > 0:
            L_scale = max(np.abs(L).max(), 1.0)
            degen_tol = 1e-8 * L_scale
            degen_mask = np.abs(L - L[idx]) < degen_tol
            if np.sum(degen_mask) > 1:
                candidates = np.where(degen_mask)[0]
                idx = candidates[np.argmax(np.abs(V[-1, candidates]))]
                logger.debug(
                    "RFO degenerate eigenvalues at order=%d: "
                    "candidates=%s, selected idx=%d", self.order, candidates, idx
                )

        denom = V[-1, idx]
        if abs(denom) < 1e-12:
            denom = np.sign(denom) * 1e-12 if denom != 0 else 1e-12
        s = V[:-1, idx] * alpha / denom

        dAda = self.A.copy()
        dAda[:-1, :-1] *= 2 * alpha

        V1 = np.delete(V, idx, 1)
        L1 = np.delete(L, idx)

        # Regularize eigenvalue differences: clamp small values while preserving sign
        L_diff = L[idx] - L1
        L_diff = np.where(L_diff >= 0,
                         np.maximum(L_diff, 1e-12),
                         np.minimum(L_diff, -1e-12))
        # Reassociate to do two matvecs (V1.T @ dAda is otherwise a (k-1, k)
        # matmul that costs ~25× more for the same final vector result).
        dVda = V1 @ ((V1.T @ (dAda @ V[:, idx])) / L_diff)

        dsda = (V[:-1, idx] / denom
                + (alpha / denom) * dVda[:-1]
                - (V[:-1, idx] * alpha / denom**2) * dVda[-1])
        return s, dsda

    def get_s(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        # At an interior order the eigenvalue branch selected from the
        # alpha=0 degenerate augmented matrix need not approach a zero step.
        # Use the full RFO direction and a continuous linear restriction;
        # the extremal orders used by minimum RFO and PRFO retain the usual
        # RFO homotopy below.
        if 0 < self.order < len(self.g):
            if self._interior_full_step is None:
                self._interior_full_step = self._get_eigenstep(1.0)[0]
            return alpha * self._interior_full_step, self._interior_full_step
        return self._get_eigenstep(alpha)


class PartitionedRationalFunctionOptimization(RationalFunctionOptimization):
    synonyms = ['prfo', 'p-rfo', 'partitioned rational function optimization']

    def _stepper_init(self) -> None:
        self.Vmax = self.H.evecs[:, :self.order]
        self.Vmin = self.H.evecs[:, self.order:]

        self.max = RationalFunctionOptimization(
            self.Vmax.T @ self.g,
            self.H.project(self.Vmax),
            order=self.Vmax.shape[1],
        )

        self.min = RationalFunctionOptimization(
            self.Vmin.T @ self.g,
            self.H.project(self.Vmin),
            order=0,
        )

    def get_s(self, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
        smax, dsmaxda = self.max.get_s(alpha)
        smin, dsminda = self.min.get_s(alpha)

        s = self.Vmax @ smax + self.Vmin @ smin
        dsda = self.Vmax @ dsmaxda + self.Vmin @ dsminda
        return s, dsda


_all_steppers = [
    QuasiNewton,
    RationalFunctionOptimization,
    PartitionedRationalFunctionOptimization,
]


def get_stepper(name: str) -> Type[BaseStepper]:
    for stepper in _all_steppers:
        if stepper.match(name):
            return stepper
    raise ValueError("Unknown stepper name: {}".format(name))
