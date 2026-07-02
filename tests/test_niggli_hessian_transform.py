import numpy as np
from scipy.linalg import expm

from sella.peswrapper import _cell_deformation_jacobian


def test_cell_deformation_jacobian_matches_fd():
    """d(cell)/d(L) must match finite differences of the parameterization
    cell(L) = expm(L / factor) @ orig_cell at a *deformed* reference (F != I).

    Regression for the Niggli Hessian transform: the Frechet base point was
    wrongly scaled by exp_cell_factor (logm(F)/factor instead of logm(F)),
    which is only harmless when F is near the identity. With a real cell
    deformation the analytic Jacobian diverged from finite differences.
    """
    orig_cell = np.array([[4.0, 0.0, 0.0],
                          [0.1, 4.2, 0.0],
                          [0.2, 0.15, 4.1]])
    factor = 3.0  # a realistic exp_cell_factor (defaults to ~natoms, never 1)

    # A nonzero (deformed) log-deformation parameter matrix.
    L = np.array([[0.10, 0.05, 0.00],
                  [0.05, -0.08, 0.02],
                  [0.00, 0.02, 0.12]])
    F = expm(L / factor)

    J = _cell_deformation_jacobian(F, orig_cell, factor)

    delta = 1e-6
    J_num = np.zeros((9, 9))
    for idx in range(9):
        i, j = divmod(idx, 3)
        Lp = L.copy(); Lp[i, j] += delta
        Lm = L.copy(); Lm[i, j] -= delta
        cp = expm(Lp / factor) @ orig_cell
        cm = expm(Lm / factor) @ orig_cell
        J_num[:, idx] = ((cp - cm) / (2 * delta)).ravel()

    np.testing.assert_allclose(J, J_num, atol=1e-6, rtol=1e-5)
