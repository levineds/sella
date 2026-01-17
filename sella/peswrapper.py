from typing import Union, Callable

import numpy as np
from scipy.linalg import eigh, expm, logm
from scipy.integrate import LSODA
from ase import Atoms
from ase.utils import basestring
from ase.visualize import view
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.trajectory import Trajectory

from sella.utilities.math import modified_gram_schmidt
from sella.hessian_update import symmetrize_Y
from sella.linalg import NumericalHessian, ApproximateHessian
from sella.eigensolvers import rayleigh_ritz
from sella.internal import Internals, Constraints, DuplicateInternalError


class PES:
    def __init__(
        self,
        atoms: Atoms,
        H0: np.ndarray = None,
        constraints: Constraints = None,
        eigensolver: str = 'jd0',
        trajectory: Union[str, Trajectory] = None,
        eta: float = 1e-4,
        v0: np.ndarray = None,
        proj_trans: bool = None,
        proj_rot: bool = None,
        hessian_function: Callable[[Atoms], np.ndarray] = None,
    ) -> None:
        self.atoms = atoms
        if constraints is None:
            constraints = Constraints(self.atoms)
        if proj_trans is None:
            if constraints.internals['translations']:
                proj_trans = False
            else:
                proj_trans = True
        if proj_trans:
            try:
                constraints.fix_translation()
            except DuplicateInternalError:
                pass

        if proj_rot is None:
            if np.any(atoms.pbc):
                proj_rot = False
            else:
                proj_rot = True
        if proj_rot:
            try:
                constraints.fix_rotation()
            except DuplicateInternalError:
                pass
        self.cons = constraints
        self.eigensolver = eigensolver

        if trajectory is not None:
            if isinstance(trajectory, basestring):
                self.traj = Trajectory(trajectory, 'w', self.atoms)
            else:
                self.traj = trajectory
        else:
            self.traj = None

        self.eta = eta
        self.v0 = v0

        self.neval = 0
        self.curr = dict(
            x=None,
            f=None,
            g=None,
        )
        self.last = self.curr.copy()

        # Internal coordinate specific things
        self.int = None
        self.dummies = None

        self.dim = 3 * len(atoms)
        self.ncart = self.dim
        if H0 is None:
            self.set_H(None, initialized=False)
        else:
            self.set_H(H0, initialized=True)

        self.savepoint = dict(apos=None, dpos=None)
        self.first_diag = True

        self.hessian_function = hessian_function

        # Cache for _calc_basis to avoid redundant SVD computations
        self._basis_cache = dict(pos_hash=None, result=None)

    apos = property(lambda self: self.atoms.positions.copy())
    dpos = property(lambda self: None)

    def save(self):
        self.savepoint = dict(apos=self.apos, dpos=self.dpos)

    def restore(self):
        apos = self.savepoint['apos']
        dpos = self.savepoint['dpos']
        assert apos is not None
        self.atoms.positions = apos
        if dpos is not None:
            self.dummies.positions = dpos

    def close(self):
        """Close any open file handles (e.g., trajectory file)."""
        if self.traj is not None:
            self.traj.close()
            self.traj = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures trajectory is closed."""
        self.close()
        return False

    # Position getter/setter
    def set_x(self, target):
        diff = target - self.get_x()
        self.atoms.positions = target.reshape((-1, 3))
        return diff, diff, self.curr.get('g', np.zeros_like(diff))

    def get_x(self):
        return self.apos.ravel().copy()

    # Hessian getter/setter
    def get_H(self):
        return self.H

    def set_H(self, target, *args, **kwargs):
        self.H = ApproximateHessian(
            self.dim, self.ncart, target, *args, **kwargs
        )

    # Hessian of the constraints
    def get_Hc(self):
        return self.cons.hessian().ldot(self.curr['L'])

    # Hessian of the Lagrangian
    def get_HL(self):
        return self.get_H() - self.get_Hc()

    # Getters for constraints and their derivatives
    def get_res(self):
        return self.cons.residual()

    def get_drdx(self):
        return self.cons.jacobian()

    def _calc_basis(self):
        # Check if cached result is valid
        pos_hash = self.atoms.positions.tobytes()
        if self._basis_cache['pos_hash'] == pos_hash:
            return self._basis_cache['result']

        drdx = self.get_drdx()
        U, S, VT = np.linalg.svd(drdx)
        ncons = np.sum(S > 1e-6)
        Ucons = VT[:ncons].T
        Ufree = VT[ncons:].T
        Unred = np.eye(self.dim)
        result = (drdx, Ucons, Unred, Ufree)

        # Cache the result
        self._basis_cache['pos_hash'] = pos_hash
        self._basis_cache['result'] = result
        return result

    def write_traj(self):
        if self.traj is not None:
            self.traj.write()

    def eval(self):
        self.neval += 1
        f = self.atoms.get_potential_energy()
        g = -self.atoms.get_forces().ravel()
        self.write_traj()
        return f, g

    def _calc_eg(self, x):
        self.save()
        self.set_x(x)

        f, g = self.eval()

        self.restore()
        return f, g

    def get_scons(self):
        """Returns displacement vector for linear constraint correction."""
        Ucons = self.get_Ucons()

        scons = -Ucons @ np.linalg.lstsq(
            self.get_drdx() @ Ucons,
            self.get_res(),
            rcond=None,
        )[0]
        return scons

    def _update(self, feval=True):
        x = self.get_x()
        new_point = True
        if self.curr['x'] is not None and np.all(x == self.curr['x']):
            if feval and self.curr['f'] is None:
                new_point = False
            else:
                return False
        basis = self._calc_basis()

        if feval:
            f, g = self.eval()
        else:
            f = None
            g = None

        if new_point:
            self.last = self.curr.copy()

        self.curr['x'] = x
        self.curr['f'] = f
        self.curr['g'] = g
        self._update_basis(basis)
        return True

    def _update_basis(self, basis=None):
        if basis is None:
            basis = self._calc_basis()
        drdx, Ucons, Unred, Ufree = basis
        self.curr['drdx'] = drdx
        self.curr['Ucons'] = Ucons
        self.curr['Unred'] = Unred
        self.curr['Ufree'] = Ufree

        if self.curr['g'] is None:
            L = None
        else:
            L = np.linalg.lstsq(drdx.T, self.curr['g'], rcond=None)[0]
        self.curr['L'] = L

    def _update_H(self, dx, dg):
        if self.last['x'] is None or self.last['g'] is None:
            return
        self.H.update(dx, dg)

    def get_f(self):
        self._update()
        return self.curr['f']

    def get_g(self):
        self._update()
        return self.curr['g'].copy()

    def get_Unred(self):
        self._update(False)
        return self.curr['Unred']

    def get_Ufree(self):
        self._update(False)
        return self.curr['Ufree']

    def get_Ucons(self):
        self._update(False)
        return self.curr['Ucons']

    def diag(self, gamma=0.1, threepoint=False, maxiter=None):
        if self.curr['f'] is None:
            self._update(feval=True)

        Ufree = self.get_Ufree()
        nfree = Ufree.shape[1]

        # If there are no free DOF, there's nothing to diagonalize
        if nfree == 0:
            return

        P = self.get_HL().project(Ufree)
        P_is_none = P.B is None

        # Determine initial guess vector
        if P_is_none or self.first_diag:
            v0 = self.v0 if self.v0 is not None else self.get_g() @ Ufree
            # If v0 is near-zero, let rayleigh_ritz choose its own initial guess
            if v0 is not None and np.linalg.norm(v0) < 1e-12:
                v0 = None
        else:
            v0 = None

        # Convert P to array
        P = np.eye(nfree) if P_is_none else P.asarray()

        Hproj = NumericalHessian(self._calc_eg, self.get_x(), self.get_g(),
                                 self.eta, threepoint, Ufree)
        Hc = self.get_Hc()
        rayleigh_ritz(Hproj - Ufree.T @ Hc @ Ufree, gamma, P, v0=v0,
                      method=self.eigensolver,
                      maxiter=maxiter)

        # Extract eigensolver iterates
        Vs = Hproj.Vs
        AVs = Hproj.AVs

        # Re-calculate Ritz vectors
        Atilde = Vs.T @ symmetrize_Y(Vs, AVs, symm=2) - Vs.T @ Hc @ Vs
        _, X = eigh(Atilde)

        # Rotate Vs and AVs into X
        Vs = Vs @ X
        AVs = AVs @ X

        # Update the approximate Hessian
        self.H.update(Vs, AVs)

        self.first_diag = False

    # FIXME: temporary functions for backwards compatibility
    def get_projected_forces(self):
        """Returns Nx3 array of atomic forces orthogonal to constraints."""
        g = self.get_g()
        Ufree = self.get_Ufree()
        return -((Ufree @ Ufree.T) @ g).reshape((-1, 3))

    def converged(self, fmax, cmax=1e-5):
        fmax1 = np.linalg.norm(self.get_projected_forces(), axis=1).max()
        cmax1 = np.linalg.norm(self.get_res())
        conv = (fmax1 < fmax) and (cmax1 < cmax)
        return conv, fmax1, cmax1

    def wrap_dx(self, dx):
        return dx

    def get_df_pred(self, dx, g, H):
        if H is None:
            return None
        return g.T @ dx + (dx.T @ H @ dx) / 2.

    def kick(self, dx, diag=False, **diag_kwargs):
        x0 = self.get_x()
        f0 = self.get_f()
        g0 = self.get_g()
        B0 = self.H.asarray()

        dx_initial, dx_final, g_par = self.set_x(x0 + dx)

        df_pred = self.get_df_pred(dx_initial, g0, B0)
        dg_actual = self.get_g() - g_par
        df_actual = self.get_f() - f0
        if df_pred is None or abs(df_pred) < 1e-14:
            ratio = None
        else:
            ratio = df_actual / df_pred

        # Only update Hessian if the step quality (ratio = actual/predicted energy change)
        # is reasonable. Skip update when ratio indicates a poor step:
        # - ratio <= 0.1: step was much worse than predicted (or went uphill)
        # - ratio >= 3.0: prediction was very inaccurate
        # Skipping bad updates prevents Hessian oscillation when optimizer alternates
        # between overshooting in opposite directions.
        if ratio is None or 0.1 < ratio < 3.0:
            self._update_H(dx_final, dg_actual)

        if diag:
            if self.hessian_function is not None:
                self.calculate_hessian()
            else:
                self.diag(**diag_kwargs)

        return ratio

    def calculate_hessian(self):
        assert self.hessian_function is not None
        self.H.set_B(self.hessian_function(self.atoms))


class InternalPES(PES):
    def __init__(
        self,
        atoms: Atoms,
        internals: Internals,
        *args,
        H0: np.ndarray = None,
        iterative_stepper: int = 0,
        auto_find_internals: bool = True,
        **kwargs
    ):
        self.int_orig = internals
        new_int = internals.copy()
        if auto_find_internals:
            new_int.find_all_bonds()
            new_int.find_all_angles()
            new_int.find_all_dihedrals()
        new_int.validate_basis()

        PES.__init__(
            self,
            atoms,
            *args,
            constraints=new_int.cons,
            H0=None,
            proj_trans=False,
            proj_rot=False,
            **kwargs
        )

        self.int = new_int
        self.dummies = self.int.dummies
        self.dim = len(self.get_x())
        self.ncart = self.int.ndof
        if H0 is None:
            # Construct guess hessian and zero out components in
            # infeasible subspace
            B = self.int.jacobian()
            P = B @ np.linalg.pinv(B)
            H0 = P @ self.int.guess_hessian() @ P
            self.set_H(H0, initialized=False)
        else:
            self.set_H(H0, initialized=True)

        # Flag used to indicate that new internal coordinates are required
        self.bad_int = None
        self.iterative_stepper = iterative_stepper

        # Cache for Jacobian pseudo-inverse
        self._pinv_cache = dict(version=None, pinv=None)

    dpos = property(lambda self: self.dummies.positions.copy())

    # =========================================================================
    # Cache optimization: Store and reuse Jacobian pseudo-inverse
    # =========================================================================
    # Computing pinv(B) is expensive. Cache it and use the internal coordinate
    # system's _cache_version counter to detect when positions have changed.
    # =========================================================================

    def _get_Binv(self):
        """Get cached pseudo-inverse of internal Jacobian."""
        B = self.int.jacobian()
        # Use cache version counter to detect changes
        version = self.int._cache_version
        if (self._pinv_cache.get('version') == version and
                self._pinv_cache.get('pinv') is not None):
            return self._pinv_cache['pinv']

        Binv = np.linalg.pinv(B)

        self._pinv_cache['version'] = version
        self._pinv_cache['pinv'] = Binv
        return Binv

    # =========================================================================
    # Iterative stepper with improved convergence checking
    # =========================================================================
    # Uses Newton-Raphson iteration with robust convergence detection:
    # - Strict absolute tolerance (1e-8) for convergence
    # - Divergence detection (2x initial error)
    # - Stagnation detection (3 consecutive iterations without progress)
    # - Final verification pass before accepting solution
    # Falls back to ODE integrator on failure.
    # =========================================================================

    def _set_x_iterative(self, target, max_iter=20):
        """Fast iterative stepper for internal coordinate updates.

        Uses Newton-Raphson iteration to update Cartesian positions to match
        target internal coordinates. Returns None if convergence fails.
        """
        pos0 = self.atoms.positions.copy()
        dpos0 = self.dummies.positions.copy()
        x0 = self.get_x()
        dx_initial = target - x0

        # Get initial gradient in Cartesian space
        g0 = np.linalg.lstsq(
            self.int.jacobian(),
            self.curr.get('g', np.zeros_like(dx_initial)),
            rcond=None,
        )[0]

        rms_prev = np.inf
        initial_rms = None
        pos_first = None
        dpos_first = None
        stagnation_count = 0

        for iteration in range(max_iter):
            residual = self.wrap_dx(target - self.get_x())
            rms = np.linalg.norm(residual) / np.sqrt(len(residual))

            if initial_rms is None:
                initial_rms = rms

            # Converged
            if rms < 1e-8:
                break

            # Check for divergence (getting significantly worse)
            if rms > initial_rms * 2.0:
                # Diverging, restore and fall back
                self.atoms.positions = pos0
                self.dummies.positions = dpos0
                return None

            # Check for stagnation (after first few iterations)
            if iteration > 3:
                if rms > rms_prev * 0.95:
                    stagnation_count += 1
                    if stagnation_count >= 3:
                        # Stagnating, give up if we haven't made progress
                        if rms > initial_rms * 0.5:
                            self.atoms.positions = pos0
                            self.dummies.positions = dpos0
                            return None
                        break  # Accept partial convergence
                else:
                    stagnation_count = 0

            rms_prev = rms

            # Newton step
            dx = np.linalg.lstsq(
                self.int.jacobian(),
                residual,
                rcond=None,
            )[0].reshape((-1, 3))

            # Update positions
            self.atoms.positions += dx[:len(self.atoms)]
            self.dummies.positions += dx[len(self.atoms):]

            # Save first iteration result as fallback
            if pos_first is None:
                pos_first = self.atoms.positions.copy()
                dpos_first = self.dummies.positions.copy()

            # Check for bad internals during iteration
            self.bad_int = self.int.check_for_bad_internals()
            if self.bad_int is not None:
                # Restore and return None to trigger ODE fallback
                self.atoms.positions = pos0
                self.dummies.positions = dpos0
                self.bad_int = None
                return None

        # After loop, verify we actually converged well enough
        final_residual = self.wrap_dx(target - self.get_x())
        final_rms = np.linalg.norm(final_residual) / np.sqrt(len(dx_initial))
        if final_rms > 1e-6:
            # Didn't converge well enough, fall back to ODE
            self.atoms.positions = pos0
            self.dummies.positions = dpos0
            return None

        dx_final = self.get_x() - x0
        g_final = self.int.jacobian() @ g0
        return dx_initial, dx_final, g_final

    def _set_x_ode(self, target):
        """ODE-based stepper for internal coordinate updates.

        Uses LSODA to integrate the geodesic equation for reliable convergence
        on large or ill-conditioned steps.
        """
        dx = target - self.get_x()
        t0 = 0.
        Binv = self._get_Binv()
        # Store Binv for reuse in _q_ode to avoid repeated SVD computations
        self._ode_Binv = Binv
        y0 = np.hstack((self.apos.ravel(), self.dpos.ravel(),
                        Binv @ dx,
                        Binv @ self.curr.get('g', np.zeros_like(dx))))
        ode = LSODA(self._q_ode, t0, y0, t_bound=1., atol=1e-6)

        while ode.status == 'running':
            ode.step()
            y = ode.y
            t0 = ode.t
            self.bad_int = self.int.check_for_bad_internals()
            if self.bad_int is not None:
                break
            if ode.nfev > 1000:
                view(self.atoms + self.dummies)
                raise RuntimeError("Geometry update ODE is taking too long "
                                   "to converge!")

        if ode.status == 'failed':
            raise RuntimeError("Geometry update ODE failed to converge!")

        nxa = 3 * len(self.atoms)
        nxd = 3 * len(self.dummies)
        y = y.reshape((3, nxa + nxd))
        self.atoms.positions = y[0, :nxa].reshape((-1, 3))
        self.dummies.positions = y[0, nxa:].reshape((-1, 3))
        B = self.int.jacobian()
        dx_final = t0 * B @ y[1]
        g_final = B @ y[2]
        dx_initial = t0 * dx
        return dx_initial, dx_final, g_final

    # Position getter/setter
    def set_x(self, target):
        """Update internal coordinates to target values.

        Uses fast iterative stepper by default, with ODE fallback for robustness.
        """
        if self.iterative_stepper:
            res = self._set_x_iterative(target)
            if res is not None:
                return res
        # Fall back to ODE solver
        return self._set_x_ode(target)

    def get_x(self):
        return self.int.calc()

    # Hessian of the constraints
    def get_Hc(self):
        D_cons = self.cons.hessian().ldot(self.curr['L'])
        Binv_int = self._get_Binv()
        B_cons = self.cons.jacobian()
        L_int = self.curr['L'] @ B_cons @ Binv_int
        D_int = self.int.hessian().ldot(L_int)
        Hc = Binv_int.T @ (D_cons - D_int) @ Binv_int
        return Hc

    def get_drdx(self):
        # dr/dq = dr/dx dx/dq
        return PES.get_drdx(self) @ self._get_Binv()

    def _calc_basis(self, internal=None, cons=None):
        # If custom internal/cons provided, bypass cache
        if internal is not None or cons is not None:
            if internal is None:
                internal = self.int
            if cons is None:
                cons = self.cons
            B = internal.jacobian()
            Ui, Si, VTi = np.linalg.svd(B)
            nnred = np.sum(Si > 1e-6)
            Unred = Ui[:, :nnred]
            Vnred = VTi[:nnred].T
            Siinv = np.diag(1 / Si[:nnred])
            drdxnred = cons.jacobian() @ Vnred @ Siinv
            drdx = drdxnred @ Unred.T
            Uc, Sc, VTc = np.linalg.svd(drdxnred)
            ncons = np.sum(Sc > 1e-6)
            Ucons = Unred @ VTc[:ncons].T
            Ufree = Unred @ VTc[ncons:].T
            return drdx, Ucons, Unred, Ufree

        # Check if cached result is valid
        pos_hash = (self.atoms.positions.tobytes() +
                    self.dummies.positions.tobytes())
        if self._basis_cache['pos_hash'] == pos_hash:
            return self._basis_cache['result']

        internal = self.int
        cons = self.cons
        B = internal.jacobian()
        Ui, Si, VTi = np.linalg.svd(B)
        nnred = np.sum(Si > 1e-6)
        Unred = Ui[:, :nnred]
        Vnred = VTi[:nnred].T
        Siinv = np.diag(1 / Si[:nnred])

        # Compute pinv from SVD components and cache it for _get_Binv
        # pinv(B) = V @ diag(1/S) @ U.T
        Binv = Vnred @ Siinv @ Unred.T
        self._pinv_cache['version'] = internal._cache_version
        self._pinv_cache['pinv'] = Binv

        drdxnred = cons.jacobian() @ Vnred @ Siinv
        drdx = drdxnred @ Unred.T
        Uc, Sc, VTc = np.linalg.svd(drdxnred)
        ncons = np.sum(Sc > 1e-6)
        Ucons = Unred @ VTc[:ncons].T
        Ufree = Unred @ VTc[ncons:].T
        result = (drdx, Ucons, Unred, Ufree)

        # Cache the result
        self._basis_cache['pos_hash'] = pos_hash
        self._basis_cache['result'] = result
        return result

    def eval(self):
        f, g_cart = PES.eval(self)
        Binv = self._get_Binv()
        return f, g_cart @ Binv[:len(g_cart)]

    def update_internals(self, dx):
        self._update(True)

        nold = 3 * (len(self.atoms) + len(self.dummies))

        # FIXME: Testing to see if disabling this works
        #if self.bad_int is not None:
        #    for bond in self.bad_int['bonds']:
        #        self.int_orig.forbid_bond(bond)
        #    for angle in self.bad_int['angles']:
        #        self.int_orig.forbid_angle(angle)

        # Find new internals, constraints, and dummies
        new_int = self.int_orig.copy()
        new_int.find_all_bonds()
        new_int.find_all_angles()
        new_int.find_all_dihedrals()
        new_int.validate_basis()
        new_cons = new_int.cons

        # Calculate B matrix and its inverse for new and old internals
        Blast = self.int.jacobian()
        B = new_int.jacobian()
        Binv = np.linalg.pinv(B)
        Dlast = self.int.hessian()
        D = new_int.hessian()

        # # Projection matrices
        # P2 = B[:, nold:] @ Binv[nold:, :]

        # Update the info in self.curr
        x = new_int.calc()
        g = -self.atoms.get_forces().ravel() @ Binv[:3*len(self.atoms)]
        drdx, Ucons, Unred, Ufree = self._calc_basis(
            internal=new_int,
            cons=new_cons,
        )
        L = np.linalg.lstsq(drdx.T, g, rcond=None)[0]

        # Update H using old data where possible. For new (dummy) atoms,
        # use the guess hessian info.
        H = self.get_H().asarray()
        Hcart = Blast.T @ H @ Blast
        Hcart += Dlast.ldot(self.curr['g'])
        Hnew = Binv.T[:, :nold] @ (Hcart - D.ldot(g)) @ Binv
        self.dim = len(x)
        self.set_H(Hnew)

        self.int = new_int
        self.cons = new_cons

        self.curr.update(x=x, g=g, drdx=drdx, Ufree=Ufree,
                         Unred=Unred, Ucons=Ucons, L=L, B=B, Binv=Binv)

    def get_df_pred(self, dx, g, H):
        if H is None:
            return None
        Unred = self.get_Unred()
        dx_r = dx @ Unred
        # dx_r = self.wrap_dx(dx) @ Unred
        g_r = g @ Unred
        H_r = Unred.T @ H @ Unred
        return g_r.T @ dx_r + (dx_r.T @ H_r @ dx_r) / 2.

    # FIXME: temporary functions for backwards compatibility
    def get_projected_forces(self):
        """Returns Nx3 array of atomic forces orthogonal to constraints."""
        g = self.get_g()
        Ufree = self.get_Ufree()
        # Use cached jacobian from curr if available
        if 'B' in self.curr and self.curr['B'] is not None:
            B = self.curr['B']
        else:
            B = self.int.jacobian()
        return -((Ufree @ Ufree.T) @ g @ B).reshape((-1, 3))

    def wrap_dx(self, dx):
        return self.int.wrap(dx)

    # x setter aux functions
    def _q_ode(self, t, y):
        nxa = 3 * len(self.atoms)
        nxd = 3 * len(self.dummies)
        x, dxdt, g = y.reshape((3, nxa + nxd))

        dydt = np.zeros((3, nxa + nxd))
        dydt[0] = dxdt

        self.atoms.positions = x[:nxa].reshape((-1, 3)).copy()
        self.dummies.positions = x[nxa:].reshape((-1, 3)).copy()

        # Use direct HVP computation instead of forming full Hessians
        D_rdot = self.int.hessian_rdot(dxdt)
        # Reuse Binv from ODE initialization to avoid repeated SVD
        Binv = self._ode_Binv
        D_tmp = -Binv @ D_rdot
        dydt[1] = D_tmp @ dxdt
        dydt[2] = D_tmp @ g

        return dydt.ravel()

    def kick(self, dx, diag=False, **diag_kwargs):
        ratio = PES.kick(self, dx, diag=diag, **diag_kwargs)

        # FIXME: Testing to see if this works
        #if self.bad_int is not None:
        #    self.update_internals(dx)
        #    self.bad_int = None

        return ratio

    def write_traj(self):
        if self.traj is not None:
            energy = self.atoms.calc.results['energy']
            forces = np.zeros((len(self.atoms) + len(self.dummies), 3))
            forces[:len(self.atoms)] = self.atoms.calc.results['forces']
            atoms_tmp = self.atoms + self.dummies
            atoms_tmp.calc = SinglePointCalculator(atoms_tmp, energy=energy,
                                                   forces=forces)
            self.traj.write(atoms_tmp)

    def _update(self, feval=True):
        if not PES._update(self, feval=feval):
            return

        B = self.int.jacobian()
        Binv = self._get_Binv()  # Use cached version instead of recomputing
        self.curr.update(B=B, Binv=Binv)
        return True

    def _convert_cartesian_hessian_to_internal(
        self,
        Hcart: np.ndarray,
    ) -> np.ndarray:
        ncart = 3 * len(self.atoms)
        # Get Jacobian and calculate redundant and non-redundant spaces
        B = self.int.jacobian()[:, :ncart]
        Ui, Si, VTi = np.linalg.svd(B)
        nnred = np.sum(Si > 1e-6)
        Unred = Ui[:, :nnred]
        Ured = Ui[:, nnred:]

        # Calculate inverse Jacobian in non-redundant space
        Bnred_inv = VTi[:nnred].T @ np.diag(1 / Si[:nnred])

        # Convert Cartesian Hessian to non-redundant internal Hessian
        Hcart_coupled = self.int.hessian().ldot(self.get_g())[:ncart, :ncart]
        Hcart_corr = Hcart - Hcart_coupled
        Hnred = Bnred_inv.T @ Hcart_corr @ Bnred_inv

        # Find eigenvalues of non-redundant internal Hessian
        lnred, _ = np.linalg.eigh(Hnred)

        # The redundant part of the Hessian will be initialized to the
        # geometric mean of the non-redundant eigenvalues
        lnred_mean = np.exp(np.log(np.abs(lnred)).mean())

        # finish reconstructing redundant internal Hessian
        return Unred @ Hnred @ Unred.T + lnred_mean * Ured @ Ured.T

    def _convert_internal_hessian_to_cartesian(
        self,
        Hint: np.ndarray,
    ) -> np.ndarray:
        B = self.int.jacobian()
        return B.T @ Hint @ B + self.int.hessian().ldot(self.get_g())

    def calculate_hessian(self):
        assert self.hessian_function is not None
        self.H.set_B(self._convert_cartesian_hessian_to_internal(
            self.hessian_function(self.atoms)
        ))


# =============================================================================
# Utility functions for cell optimization
# =============================================================================

def voigt_6_to_full_3x3_stress(stress_voigt: np.ndarray) -> np.ndarray:
    """Convert 6-component Voigt stress to full 3x3 stress tensor.

    ASE uses the convention: [xx, yy, zz, yz, xz, xy]
    """
    xx, yy, zz, yz, xz, xy = stress_voigt
    return np.array([
        [xx, xy, xz],
        [xy, yy, yz],
        [xz, yz, zz]
    ])


def full_3x3_to_voigt_6_stress(stress_3x3: np.ndarray) -> np.ndarray:
    """Convert 3x3 stress tensor to 6-component Voigt notation."""
    return np.array([
        stress_3x3[0, 0],  # xx
        stress_3x3[1, 1],  # yy
        stress_3x3[2, 2],  # zz
        stress_3x3[1, 2],  # yz
        stress_3x3[0, 2],  # xz
        stress_3x3[0, 1],  # xy
    ])


class CellInternalPES(InternalPES):
    """Internal coordinate PES with unit cell optimization.

    This class extends InternalPES to simultaneously optimize both internal
    coordinates (bonds, angles, dihedrals) and the unit cell parameters.

    The cell is parameterized using the log of the deformation gradient:
        F = cell @ inv(orig_cell)
        cell_params = logm(F) * exp_cell_factor

    This parameterization ensures that:
    1. The identity corresponds to zero cell parameters
    2. Small deformations are approximately linear in the parameters
    3. Large deformations are handled smoothly

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object with periodic boundary conditions.
    internals : Internals
        Internal coordinate system definition.
    exp_cell_factor : float, optional
        Scaling factor for cell parameterization. Default is number of atoms.
    cell_mask : ndarray, optional
        Boolean mask of shape (3, 3) indicating which cell DOF are free.
        Default is all True (full cell optimization).
    scalar_pressure : float, optional
        External pressure in eV/Å³. Default is 0.
    refine_initial_hessian : bool, optional
        If True, compute cell-coordinate coupling and cell-cell Hessian blocks
        via finite differences. This requires additional force evaluations
        (2 * n_cell_dof) but can improve convergence for coupled systems.
        Default is False.
    hessian_delta : float, optional
        Finite difference step size for Hessian refinement. Default is 1e-5.
    """

    def __init__(
        self,
        atoms: Atoms,
        internals: Internals,
        *args,
        exp_cell_factor: float = None,
        cell_mask: np.ndarray = None,
        scalar_pressure: float = 0.0,
        refine_initial_hessian: Union[bool, int] = False,
        hessian_delta: float = 1e-5,
        save_hessian: str = None,
        H0: np.ndarray = None,
        **kwargs
    ):
        """Initialize CellInternalPES.

        Parameters
        ----------
        refine_initial_hessian : bool or int
            Level of Hessian refinement via finite differences:
            - False or 0: No refinement (default)
            - True or 1: Refine cell-related blocks only (2 * n_cell_dof evals)
            - 2: Also refine translation/rotation blocks (adds 2 * n_tric evals)
            - 3: Refine full internal Hessian (2 * n_internal evals, expensive!)
        save_hessian : str, optional
            Path to save the initial Hessian as .npy file for analysis.
        """
        # Store original cell as reference before any optimization
        self.orig_cell = atoms.get_cell().array.copy()

        # Cell parameterization scaling (like ASE's FrechetCellFilter)
        if exp_cell_factor is None:
            exp_cell_factor = float(len(atoms))
        self.exp_cell_factor = exp_cell_factor

        # Cell mask: which of the 9 cell matrix elements are free
        if cell_mask is None:
            cell_mask = np.ones((3, 3), dtype=bool)
        self.cell_mask = np.asarray(cell_mask, dtype=bool).reshape((3, 3))
        self.n_cell_dof = int(self.cell_mask.sum())

        # External pressure
        self.scalar_pressure = scalar_pressure

        # Flag to control get_x behavior during parent initialization
        # When True, get_x returns only internal coords (for parent __init__)
        self._initializing = True
        self.n_internal = None  # Will be set by parent

        # Initialize parent class - this will set up internal coords
        InternalPES.__init__(self, atoms, internals, *args, H0=H0, **kwargs)

        # Now parent is initialized. Store internal-only dimension.
        self.n_internal = self.dim  # Parent set dim to internal coords count

        # Update dimension to include cell DOF
        self.dim = self.n_internal + self.n_cell_dof

        # Done initializing - now get_x returns full vector
        self._initializing = False

        # Create proper Hessian with correct dimensions
        # Use block-diagonal structure: internal Hessian + cell Hessian
        H_old = self.H.B if self.H is not None and self.H.B is not None else None

        # Pad internal Hessian and add cell block
        H0_full = np.zeros((self.dim, self.dim))
        if H_old is not None:
            H0_full[:self.n_internal, :self.n_internal] = H_old
        else:
            B = self.int.jacobian()
            P = B @ np.linalg.pinv(B)
            H_internal = P @ self.int.guess_hessian() @ P
            H0_full[:self.n_internal, :self.n_internal] = H_internal

        # Convert bool to int for refinement level
        if refine_initial_hessian is True:
            refine_level = 1
        elif refine_initial_hessian is False:
            refine_level = 0
        else:
            refine_level = int(refine_initial_hessian)

        if refine_level >= 1:
            # Level 1: Refine cell-related blocks
            H_cell_cols = self._compute_cell_hessian_columns(hessian_delta)
            # Set internal-cell coupling (and its transpose for symmetry)
            H0_full[:self.n_internal, self.n_internal:] = H_cell_cols[:self.n_internal, :]
            H0_full[self.n_internal:, :self.n_internal] = H_cell_cols[:self.n_internal, :].T
            # Set cell-cell block with explicit symmetrization
            H_cell_cell = H_cell_cols[self.n_internal:, :]
            H0_full[self.n_internal:, self.n_internal:] = (H_cell_cell + H_cell_cell.T) / 2

        if refine_level >= 2:
            # Level 2: Also refine translation and rotation blocks
            H_tric_cols = self._compute_tric_hessian_columns(hessian_delta)
            tric_indices = self._get_tric_indices()
            for i, idx in enumerate(tric_indices):
                H0_full[:, idx] = H_tric_cols[:, i]
                H0_full[idx, :] = H_tric_cols[:, i]

        if refine_level >= 3:
            # Level 3: Refine full internal Hessian (expensive!)
            H_int_cols = self._compute_internal_hessian_columns(hessian_delta)
            # Symmetrize and set the internal-internal block
            H0_full[:self.n_internal, :self.n_internal] = (H_int_cols + H_int_cols.T) / 2

        if refine_level == 0:
            # No refinement: use diagonal guess for cell block
            h0_cell = 70.0
            H0_full[self.n_internal:, self.n_internal:] = h0_cell * np.eye(self.n_cell_dof)

        # Save Hessian if requested
        if save_hessian is not None:
            np.save(save_hessian, H0_full)
            print(f"Initial Hessian saved to {save_hessian}")

        self.set_H(H0_full, initialized=False)

    def save(self):
        """Save current state including cell."""
        InternalPES.save(self)
        self.savepoint['cell'] = self.atoms.get_cell().array.copy()

    def restore(self):
        """Restore saved state including cell."""
        InternalPES.restore(self)
        if 'cell' in self.savepoint:
            self.atoms.set_cell(self.savepoint['cell'], scale_atoms=False)

    def _compute_cell_hessian_columns(self, delta: float) -> np.ndarray:
        """Compute Hessian columns for cell DOF via finite differences.

        This computes d(gradient)/d(cell_param) for all cell parameters,
        giving us both the internal-cell coupling block and the cell-cell block.

        Parameters
        ----------
        delta : float
            Finite difference step size.

        Returns
        -------
        H_cols : ndarray
            Array of shape (dim, n_cell_dof) containing Hessian columns.
        """
        H_cols = np.zeros((self.dim, self.n_cell_dof))

        # Save current state
        x0 = self.get_x()
        cell0 = self.atoms.get_cell().array.copy()
        pos0 = self.atoms.positions.copy()

        n_evals = 2 * self.n_cell_dof
        print(f"Refining initial Hessian: 0/{n_evals} force calls", end="", flush=True)

        for i in range(self.n_cell_dof):
            # Displace cell parameter +delta
            x_plus = x0.copy()
            x_plus[self.n_internal + i] += delta
            self.set_x(x_plus)
            _, g_plus = self.eval()
            print(f"\rRefining initial Hessian: {2*i + 1}/{n_evals} force calls", end="", flush=True)

            # Displace cell parameter -delta
            x_minus = x0.copy()
            x_minus[self.n_internal + i] -= delta
            self.set_x(x_minus)
            _, g_minus = self.eval()
            print(f"\rRefining initial Hessian: {2*i + 2}/{n_evals} force calls", end="", flush=True)

            # Central difference
            H_cols[:, i] = (g_plus - g_minus) / (2 * delta)

        print()  # Newline after progress

        # Restore original state
        self.atoms.positions = pos0
        self.atoms.set_cell(cell0, scale_atoms=False)
        # Clear cached values to force recomputation
        self.curr['x'] = None
        self.curr['f'] = None
        self.curr['g'] = None

        return H_cols

    def _get_tric_indices(self) -> np.ndarray:
        """Get indices of translation and rotation coordinates in internal space."""
        n_trans = len(self.int.internals['translations'])
        n_bonds = len(self.int.internals['bonds'])
        n_angles = len(self.int.internals['angles'])
        n_dihedrals = len(self.int.internals['dihedrals'])
        n_rot = len(self.int.internals['rotations'])

        # Internal coord order: translations, bonds, angles, dihedrals, other, rotations
        trans_indices = list(range(n_trans))
        rot_start = n_trans + n_bonds + n_angles + n_dihedrals + len(self.int.internals['other'])
        rot_indices = list(range(rot_start, rot_start + n_rot))

        return np.array(trans_indices + rot_indices)

    def _compute_tric_hessian_columns(self, delta: float) -> np.ndarray:
        """Compute Hessian columns for translation/rotation DOF via finite differences.

        This refines the coupling between TRICs and all other coordinates,
        which is important for molecular crystals where fragment motions are coupled.

        Parameters
        ----------
        delta : float
            Finite difference step size.

        Returns
        -------
        H_cols : ndarray
            Array of shape (dim, n_tric) containing Hessian columns.
        """
        tric_indices = self._get_tric_indices()
        n_tric = len(tric_indices)
        H_cols = np.zeros((self.dim, n_tric))

        # Save current state
        x0 = self.get_x()
        cell0 = self.atoms.get_cell().array.copy()
        pos0 = self.atoms.positions.copy()

        n_evals = 2 * n_tric
        print(f"Refining TRIC Hessian: 0/{n_evals} force calls", end="", flush=True)

        for i, idx in enumerate(tric_indices):
            # Displace TRIC parameter +delta
            x_plus = x0.copy()
            x_plus[idx] += delta
            self.set_x(x_plus)
            _, g_plus = self.eval()
            print(f"\rRefining TRIC Hessian: {2*i + 1}/{n_evals} force calls", end="", flush=True)

            # Displace TRIC parameter -delta
            x_minus = x0.copy()
            x_minus[idx] -= delta
            self.set_x(x_minus)
            _, g_minus = self.eval()
            print(f"\rRefining TRIC Hessian: {2*i + 2}/{n_evals} force calls", end="", flush=True)

            # Central difference
            H_cols[:, i] = (g_plus - g_minus) / (2 * delta)

        print()  # Newline after progress

        # Restore original state
        self.atoms.positions = pos0
        self.atoms.set_cell(cell0, scale_atoms=False)
        # Clear cached values to force recomputation
        self.curr['x'] = None
        self.curr['f'] = None
        self.curr['g'] = None

        return H_cols

    def _compute_internal_hessian_columns(self, delta: float) -> np.ndarray:
        """Compute full internal-internal Hessian block via finite differences.

        This is expensive: requires 2 * n_internal force evaluations.
        Only use when a highly accurate initial Hessian is needed.

        Parameters
        ----------
        delta : float
            Finite difference step size.

        Returns
        -------
        H_int : ndarray
            Array of shape (n_internal, n_internal) containing the internal Hessian.
        """
        H_int = np.zeros((self.n_internal, self.n_internal))

        # Save current state
        x0 = self.get_x()
        cell0 = self.atoms.get_cell().array.copy()
        pos0 = self.atoms.positions.copy()

        n_evals = 2 * self.n_internal
        print(f"Refining internal Hessian: 0/{n_evals} force calls", end="", flush=True)

        for i in range(self.n_internal):
            # Displace internal coordinate +delta
            x_plus = x0.copy()
            x_plus[i] += delta
            self.set_x(x_plus)
            _, g_plus = self.eval()

            # Displace internal coordinate -delta
            x_minus = x0.copy()
            x_minus[i] -= delta
            self.set_x(x_minus)
            _, g_minus = self.eval()

            # Central difference - only internal part
            H_int[:, i] = (g_plus[:self.n_internal] - g_minus[:self.n_internal]) / (2 * delta)

            # Progress update every 10 columns or at the end
            if (i + 1) % 10 == 0 or i == self.n_internal - 1:
                print(f"\rRefining internal Hessian: {2*(i+1)}/{n_evals} force calls", end="", flush=True)

        print()  # Newline after progress

        # Restore original state
        self.atoms.positions = pos0
        self.atoms.set_cell(cell0, scale_atoms=False)
        # Clear cached values to force recomputation
        self.curr['x'] = None
        self.curr['f'] = None
        self.curr['g'] = None

        return H_int

    def get_x(self) -> np.ndarray:
        """Return combined internal coordinates + cell parameters.

        During initialization (_initializing=True), returns only internal coords
        to be compatible with parent class initialization.
        """
        q = self.int.calc()  # Internal coordinates

        # During parent initialization, return only internal coords
        if getattr(self, '_initializing', True):
            return q

        cell_params = self._masked_cell_params()  # Cell DOF
        return np.concatenate([q, cell_params])

    def _get_deformation_gradient(self) -> np.ndarray:
        """Get current deformation gradient F = cell @ inv(orig_cell)."""
        return self.atoms.get_cell().array @ np.linalg.inv(self.orig_cell)

    def _get_log_deform(self) -> np.ndarray:
        """Get log of deformation gradient, scaled by exp_cell_factor."""
        F = self._get_deformation_gradient()
        return logm(F).real * self.exp_cell_factor

    def _set_cell_from_log_deform(self, log_deform_scaled: np.ndarray) -> None:
        """Set cell from scaled log-deformation gradient."""
        log_deform = log_deform_scaled / self.exp_cell_factor
        F = expm(log_deform.real)
        new_cell = self.orig_cell @ F.T
        self.atoms.set_cell(new_cell, scale_atoms=False)

    def _masked_cell_params(self) -> np.ndarray:
        """Get cell parameters as flat array (only free DOF)."""
        log_deform = self._get_log_deform()
        return log_deform[self.cell_mask]

    def _set_masked_cell_params(self, params: np.ndarray) -> None:
        """Set cell from flat array of free DOF."""
        log_deform = self._get_log_deform()
        log_deform[self.cell_mask] = params
        self._set_cell_from_log_deform(log_deform)

    def set_x(self, target: np.ndarray):
        """Set internal coordinates and cell parameters.

        This is more complex than InternalPES.set_x because:
        1. We first update the cell (which changes internal coord values)
        2. Then update atomic positions to match target internal coords

        Returns
        -------
        dx_initial, dx_final, g_par : tuple of np.ndarray
            Displacement information for Hessian update.
        """
        x0 = self.get_x()
        dx_initial = target - x0

        # Split target into internal and cell parts
        q_target = target[:self.n_internal]
        cell_target = target[self.n_internal:]

        # Get initial cell params
        cell_params0 = self._masked_cell_params()

        # Update cell
        self._set_masked_cell_params(cell_target)

        # If there are no internal coordinates, we're done
        if self.n_internal == 0:
            # Cell-only case: dx_final equals the cell displacement
            dx_cell = cell_target - cell_params0
            dx_final = dx_cell.copy()
            # Return actual cell gradient at starting position for proper Hessian update
            # (dg_actual = get_g() - g_par needs g_par to be the old gradient)
            g_old = self.curr.get('g', None)
            if g_old is not None:
                g_final = g_old[-self.n_cell_dof:].copy()
            else:
                g_final = np.zeros(self.n_cell_dof)
            return dx_initial, dx_final, g_final

        # Get initial gradient in Cartesian for internal coord update
        g0 = np.linalg.lstsq(
            self.int.jacobian(),
            self.curr.get('g', np.zeros(self.n_internal))[:self.n_internal],
            rcond=None,
        )[0] if 'g' in self.curr and self.curr['g'] is not None else np.zeros(3 * len(self.atoms))

        # Now update atomic positions to match internal coordinate target
        # This is tricky: the internal coord values changed when cell changed
        # We use the iterative stepper from parent class
        if self.iterative_stepper:
            res = self._set_x_iterative_internal(q_target)
            if res is None:
                res = self._set_x_ode_internal(q_target)
        else:
            res = self._set_x_ode_internal(q_target)

        if res is None:
            # Fallback: just do parent set_x ignoring cell
            dx_int, _, g_int = InternalPES.set_x(self, q_target)
            dx_final = np.concatenate([dx_int, cell_target - cell_params0])
            g_final = np.concatenate([g_int, np.zeros(self.n_cell_dof)])
        else:
            dx_int_initial, dx_int_final, g_int = res
            dx_final = np.concatenate([dx_int_final, cell_target - cell_params0])
            g_final = np.concatenate([g_int, np.zeros(self.n_cell_dof)])

        return dx_initial, dx_final, g_final

    def _set_x_iterative_internal(self, q_target: np.ndarray, max_iter: int = 20):
        """Iterative stepper for internal coords only (cell already updated)."""
        pos0 = self.atoms.positions.copy()
        dpos0 = self.dummies.positions.copy()
        x0 = self.int.calc()
        dx_initial = q_target - x0

        g0 = np.linalg.lstsq(
            self.int.jacobian(),
            self.curr.get('g', np.zeros_like(dx_initial))[:self.n_internal]
            if 'g' in self.curr and self.curr['g'] is not None
            else np.zeros_like(dx_initial),
            rcond=None,
        )[0]

        rms_prev = np.inf
        initial_rms = None

        for iteration in range(max_iter):
            residual = self.int.wrap(q_target - self.int.calc())
            rms = np.linalg.norm(residual) / np.sqrt(len(residual))

            if initial_rms is None:
                initial_rms = rms

            if rms < 1e-8:
                break

            if rms > initial_rms * 2.0:
                self.atoms.positions = pos0
                self.dummies.positions = dpos0
                return None

            rms_prev = rms

            dx = np.linalg.lstsq(
                self.int.jacobian(),
                residual,
                rcond=None,
            )[0].reshape((-1, 3))

            self.atoms.positions += dx[:len(self.atoms)]
            self.dummies.positions += dx[len(self.atoms):]

            self.bad_int = self.int.check_for_bad_internals()
            if self.bad_int is not None:
                self.atoms.positions = pos0
                self.dummies.positions = dpos0
                self.bad_int = None
                return None

        final_residual = self.int.wrap(q_target - self.int.calc())
        final_rms = np.linalg.norm(final_residual) / np.sqrt(len(dx_initial))
        if final_rms > 1e-6:
            self.atoms.positions = pos0
            self.dummies.positions = dpos0
            return None

        dx_final = self.int.calc() - x0
        g_final = self.int.jacobian() @ g0
        return dx_initial, dx_final, g_final

    def _set_x_ode_internal(self, q_target: np.ndarray):
        """ODE-based stepper for internal coords only (cell already updated)."""
        x0 = self.int.calc()
        dx = q_target - x0
        t0 = 0.
        Binv = self._get_Binv()
        self._ode_Binv = Binv

        y0 = np.hstack((
            self.apos.ravel(),
            self.dpos.ravel(),
            Binv @ dx,
            Binv @ self.curr.get('g', np.zeros_like(dx))[:self.n_internal]
            if 'g' in self.curr and self.curr['g'] is not None
            else np.zeros(3 * (len(self.atoms) + len(self.dummies)))
        ))
        ode = LSODA(self._q_ode, t0, y0, t_bound=1., atol=1e-6)

        while ode.status == 'running':
            ode.step()
            y = ode.y
            t0 = ode.t
            self.bad_int = self.int.check_for_bad_internals()
            if self.bad_int is not None:
                break
            if ode.nfev > 1000:
                raise RuntimeError("Geometry update ODE is taking too long!")

        if ode.status == 'failed':
            raise RuntimeError("Geometry update ODE failed to converge!")

        nxa = 3 * len(self.atoms)
        nxd = 3 * len(self.dummies)
        y = y.reshape((3, nxa + nxd))
        self.atoms.positions = y[0, :nxa].reshape((-1, 3))
        self.dummies.positions = y[0, nxa:].reshape((-1, 3))
        B = self.int.jacobian()
        dx_final = t0 * B @ y[1]
        g_final = B @ y[2]
        dx_initial = t0 * dx
        return dx_initial, dx_final, g_final

    def eval(self) -> tuple:
        """Evaluate energy and combined gradient (internal + cell)."""
        self.neval += 1
        f = self.atoms.get_potential_energy()

        # Add pressure contribution: H = E + P*V
        if self.scalar_pressure != 0.0:
            f += self.scalar_pressure * self.atoms.get_volume()

        # Atomic forces -> internal coordinate gradient
        forces = self.atoms.get_forces()
        g_cart = -forces.ravel()
        Binv = self._get_Binv()
        g_internal = g_cart @ Binv[:len(g_cart)]

        # Stress tensor -> cell gradient
        stress = self.atoms.get_stress()  # 6-component Voigt, eV/Å³
        g_cell = self._stress_to_cell_gradient(stress)

        self.write_traj()
        return f, np.concatenate([g_internal, g_cell])

    def _stress_to_cell_gradient(self, stress_voigt: np.ndarray) -> np.ndarray:
        """Convert stress tensor to gradient w.r.t. cell parameters.

        Following ASE's FrechetCellFilter formulation.

        ASE cell force = virial / cell_factor where virial = -V * stress
        Since Sella uses gradients = -forces, we have:
        gradient = -cell_force = -virial / cell_factor = V * stress / cell_factor
        """
        volume = self.atoms.get_volume()
        stress_3x3 = voigt_6_to_full_3x3_stress(stress_voigt)

        # Add external pressure contribution
        if self.scalar_pressure != 0.0:
            stress_3x3 += self.scalar_pressure * np.eye(3)

        # Gradient w.r.t. cell = V * stress (positive because gradient = -force)
        # This is the opposite sign from ASE's "force" = -V * stress
        g_cell_3x3 = volume * stress_3x3

        # Apply cell mask
        g_cell_masked = g_cell_3x3 * self.cell_mask

        # Scale by exp_cell_factor (inverse of the scaling in get_x)
        g_cell_full = g_cell_masked / self.exp_cell_factor

        # Return only free DOF
        return g_cell_full[self.cell_mask]

    def get_g(self) -> np.ndarray:
        """Get combined gradient (internal + cell)."""
        self._update()
        return self.curr['g'].copy()

    def _update(self, feval: bool = True):
        """Update current state including cell gradient."""
        x = self.get_x()
        new_point = True
        if self.curr['x'] is not None and np.all(x == self.curr['x']):
            if feval and self.curr['f'] is None:
                new_point = False
            else:
                return False

        if feval:
            f, g = self.eval()
        else:
            f = None
            g = None

        if new_point:
            self.last = self.curr.copy()

        self.curr['x'] = x
        self.curr['f'] = f
        self.curr['g'] = g

        # Update basis for internal coordinates
        basis = self._calc_basis()
        self._update_basis(basis)
        return True

    def _calc_basis(self, internal=None, cons=None):
        """Calculate basis including cell DOF.

        The cell DOF are treated as unconstrained additional coordinates.
        """
        # Get internal coordinate basis from parent
        result = InternalPES._calc_basis(self, internal=internal, cons=cons)
        drdx_int, Ucons_int, Unred_int, Ufree_int = result

        # Extend to include cell DOF
        n_int = drdx_int.shape[1]
        n_total = n_int + self.n_cell_dof

        # Cell DOF are not constrained, so they're all in Ufree
        # drdx extended with zeros for cell columns
        drdx = np.zeros((drdx_int.shape[0], n_total))
        drdx[:, :n_int] = drdx_int

        # Ucons stays the same (no cell constraints)
        Ucons = np.zeros((n_total, Ucons_int.shape[1]))
        Ucons[:n_int, :] = Ucons_int

        # Unred extended with identity for cell DOF
        Unred = np.zeros((n_total, Unred_int.shape[1] + self.n_cell_dof))
        Unred[:n_int, :Unred_int.shape[1]] = Unred_int
        Unred[n_int:, Unred_int.shape[1]:] = np.eye(self.n_cell_dof)

        # Ufree extended with identity for cell DOF
        Ufree = np.zeros((n_total, Ufree_int.shape[1] + self.n_cell_dof))
        Ufree[:n_int, :Ufree_int.shape[1]] = Ufree_int
        Ufree[n_int:, Ufree_int.shape[1]:] = np.eye(self.n_cell_dof)

        return drdx, Ucons, Unred, Ufree

    def converged(self, fmax: float, smax: float = None, cmax: float = 1e-5):
        """Check convergence of forces and stress.

        Parameters
        ----------
        fmax : float
            Maximum force tolerance (eV/Å).
        smax : float, optional
            Maximum stress tolerance. If None, uses fmax.
        cmax : float, optional
            Constraint residual tolerance.

        Returns
        -------
        conv : bool
            True if converged.
        fmax_actual : float
            Maximum force.
        cmax_actual : float
            Constraint residual norm.
        smax_actual : float
            Maximum stress gradient.
        """
        if smax is None:
            smax = fmax

        # Force convergence (project out constraints)
        g = self.get_g()
        g_internal = g[:self.n_internal]
        Ufree_int = self.curr['Ufree'][:self.n_internal, :self.curr['Ufree'].shape[1] - self.n_cell_dof]
        g_proj = Ufree_int @ Ufree_int.T @ g_internal

        # Convert to Cartesian for force norm
        B = self.int.jacobian()
        g_cart = (g_proj @ B).reshape((-1, 3))
        fmax_actual = np.linalg.norm(g_cart, axis=1).max()

        # Stress convergence
        g_cell = g[self.n_internal:]
        smax_actual = np.abs(g_cell).max() if len(g_cell) > 0 else 0.0

        # Constraint residual
        cmax_actual = np.linalg.norm(self.get_res())

        conv = (fmax_actual < fmax) and (smax_actual < smax) and (cmax_actual < cmax)
        return conv, fmax_actual, cmax_actual, smax_actual

    def get_projected_forces(self) -> np.ndarray:
        """Returns Nx3 array of atomic forces orthogonal to constraints."""
        g = self.get_g()
        g_internal = g[:self.n_internal]
        Ufree = self.get_Ufree()
        Ufree_int = Ufree[:self.n_internal, :]
        B = self.int.jacobian()
        return -((Ufree_int @ Ufree_int.T) @ g_internal @ B).reshape((-1, 3))

    def get_drdx(self):
        """Get constraint Jacobian extended for cell DOF.

        The constraint Jacobian from the parent class only has columns for
        internal coordinates. We extend it with zero columns for cell DOF
        since there are no constraints on the cell.
        """
        # Get internal constraint Jacobian from parent
        drdx_int = InternalPES.get_drdx(self)

        # Extend with zeros for cell DOF
        n_cons = drdx_int.shape[0]
        drdx = np.zeros((n_cons, self.dim))
        drdx[:, :self.n_internal] = drdx_int

        return drdx

    def get_Hc(self):
        """Get constraint Hessian extended for cell DOF.

        The constraint Hessian from InternalPES has shape (n_internal, n_internal).
        We extend it with zeros to (dim, dim) since there are no constraints
        on cell DOF.
        """
        # Get internal constraint Hessian from parent
        Hc_int = InternalPES.get_Hc(self)

        # Extend to full dimension
        Hc = np.zeros((self.dim, self.dim))
        n_int = self.n_internal
        if Hc_int.size > 0:
            Hc[:n_int, :n_int] = Hc_int

        return Hc


class CellCartesianPES(PES):
    """Cartesian PES with unit cell optimization.

    This class extends PES to simultaneously optimize both atomic Cartesian
    positions and the unit cell parameters.

    The cell is parameterized using the log of the deformation gradient:
        F = cell @ inv(orig_cell)
        cell_params = logm(F) * exp_cell_factor

    This parameterization ensures that:
    1. The identity corresponds to zero cell parameters
    2. Small deformations are approximately linear in the parameters
    3. Large deformations are handled smoothly

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object with periodic boundary conditions.
    exp_cell_factor : float, optional
        Scaling factor for cell parameterization. Default is number of atoms.
    cell_mask : ndarray, optional
        Boolean mask of shape (3, 3) indicating which cell DOF are free.
        Default is all True (full cell optimization).
    scalar_pressure : float, optional
        External pressure in eV/Å³. Default is 0.
    refine_initial_hessian : bool or int, optional
        Level of Hessian refinement via finite differences:
        - False or 0: No refinement (default)
        - True or 1: Refine cell-related blocks only (2 * n_cell_dof force calls)
        Note: Level 2 (TRICs) is not applicable for Cartesian coordinates.
    hessian_delta : float, optional
        Finite difference step size for Hessian refinement. Default is 1e-5.
    save_hessian : str, optional
        Path to save the initial Hessian as .npy file for analysis.
    """

    def __init__(
        self,
        atoms: Atoms,
        *args,
        exp_cell_factor: float = None,
        cell_mask: np.ndarray = None,
        scalar_pressure: float = 0.0,
        refine_initial_hessian: Union[bool, int] = False,
        hessian_delta: float = 1e-5,
        save_hessian: str = None,
        H0: np.ndarray = None,
        **kwargs
    ):
        """Initialize CellCartesianPES.

        Parameters
        ----------
        refine_initial_hessian : bool or int
            Level of Hessian refinement via finite differences:
            - False or 0: No refinement (default)
            - True or 1: Refine cell-related blocks only
            Note: Level 2 (TRICs) is not applicable for Cartesian coordinates.
        save_hessian : str, optional
            Path to save the initial Hessian as .npy file for analysis.
        """
        # Store original cell as reference before any optimization
        self.orig_cell = atoms.get_cell().array.copy()

        # Cell parameterization scaling (like ASE's FrechetCellFilter)
        if exp_cell_factor is None:
            exp_cell_factor = float(len(atoms))
        self.exp_cell_factor = exp_cell_factor

        # Cell mask: which of the 9 cell matrix elements are free
        if cell_mask is None:
            cell_mask = np.ones((3, 3), dtype=bool)
        self.cell_mask = np.asarray(cell_mask, dtype=bool).reshape((3, 3))
        self.n_cell_dof = int(self.cell_mask.sum())

        # External pressure
        self.scalar_pressure = scalar_pressure

        # Flag to control get_x behavior during parent initialization
        self._initializing = True

        # Initialize parent class - PES uses 3*natoms as dimension
        PES.__init__(self, atoms, *args, H0=H0, **kwargs)

        # Store Cartesian dimension (set by parent)
        self.n_cart = self.dim  # 3 * natoms

        # Update dimension to include cell DOF
        self.dim = self.n_cart + self.n_cell_dof

        # Done initializing - now get_x returns full vector
        self._initializing = False

        # Create proper Hessian with correct dimensions
        # Use block-diagonal structure: Cartesian Hessian + cell Hessian
        H_old = self.H.B if self.H is not None and self.H.B is not None else None

        H0_full = np.zeros((self.dim, self.dim))
        if H_old is not None:
            H0_full[:self.n_cart, :self.n_cart] = H_old
        else:
            # Default: 70 eV/Å² is reasonable for stiff materials
            H0_full[:self.n_cart, :self.n_cart] = 70.0 * np.eye(self.n_cart)

        # Convert bool to int for refinement level
        if refine_initial_hessian is True:
            refine_level = 1
        elif refine_initial_hessian is False:
            refine_level = 0
        else:
            refine_level = int(refine_initial_hessian)

        if refine_level >= 1:
            # Level 1: Refine cell-related blocks
            H_cell_cols = self._compute_cell_hessian_columns(hessian_delta)
            # Set Cartesian-cell coupling (and its transpose for symmetry)
            H0_full[:self.n_cart, self.n_cart:] = H_cell_cols[:self.n_cart, :]
            H0_full[self.n_cart:, :self.n_cart] = H_cell_cols[:self.n_cart, :].T
            # Set cell-cell block with explicit symmetrization
            H_cell_cell = H_cell_cols[self.n_cart:, :]
            H0_full[self.n_cart:, self.n_cart:] = (H_cell_cell + H_cell_cell.T) / 2

        if refine_level == 0:
            # No refinement: use diagonal guess for cell block
            h0_cell = 70.0
            H0_full[self.n_cart:, self.n_cart:] = h0_cell * np.eye(self.n_cell_dof)

        # Save Hessian if requested
        if save_hessian is not None:
            np.save(save_hessian, H0_full)
            print(f"Initial Hessian saved to {save_hessian}")

        self.set_H(H0_full, initialized=False)

    def save(self):
        """Save current state including cell."""
        PES.save(self)
        self.savepoint['cell'] = self.atoms.get_cell().array.copy()

    def restore(self):
        """Restore saved state including cell."""
        PES.restore(self)
        if 'cell' in self.savepoint:
            self.atoms.set_cell(self.savepoint['cell'], scale_atoms=False)

    def _compute_cell_hessian_columns(self, delta: float) -> np.ndarray:
        """Compute Hessian columns for cell DOF via finite differences.

        This computes d(gradient)/d(cell_param) for all cell parameters,
        giving us both the Cartesian-cell coupling block and the cell-cell block.

        Parameters
        ----------
        delta : float
            Finite difference step size.

        Returns
        -------
        H_cols : ndarray
            Array of shape (dim, n_cell_dof) containing Hessian columns.
        """
        H_cols = np.zeros((self.dim, self.n_cell_dof))

        # Save current state
        x0 = self.get_x()
        cell0 = self.atoms.get_cell().array.copy()
        pos0 = self.atoms.positions.copy()

        n_evals = 2 * self.n_cell_dof
        print(f"Refining initial Hessian: 0/{n_evals} force calls", end="", flush=True)

        for i in range(self.n_cell_dof):
            # Displace cell parameter +delta
            x_plus = x0.copy()
            x_plus[self.n_cart + i] += delta
            self.set_x(x_plus)
            _, g_plus = self.eval()
            print(f"\rRefining initial Hessian: {2*i + 1}/{n_evals} force calls", end="", flush=True)

            # Displace cell parameter -delta
            x_minus = x0.copy()
            x_minus[self.n_cart + i] -= delta
            self.set_x(x_minus)
            _, g_minus = self.eval()
            print(f"\rRefining initial Hessian: {2*i + 2}/{n_evals} force calls", end="", flush=True)

            # Central difference
            H_cols[:, i] = (g_plus - g_minus) / (2 * delta)

        print()  # Newline after progress

        # Restore original state
        self.atoms.positions = pos0
        self.atoms.set_cell(cell0, scale_atoms=False)
        # Clear cached values to force recomputation
        self.curr['x'] = None
        self.curr['f'] = None
        self.curr['g'] = None

        return H_cols

    def get_x(self) -> np.ndarray:
        """Return Cartesian positions + cell parameters.

        During initialization (_initializing=True), returns only Cartesian coords
        to be compatible with parent class initialization.
        """
        x_cart = self.apos.ravel().copy()

        # During parent initialization, return only Cartesian coords
        if getattr(self, '_initializing', True):
            return x_cart

        cell_params = self._masked_cell_params()
        return np.concatenate([x_cart, cell_params])

    def _get_deformation_gradient(self) -> np.ndarray:
        """Get current deformation gradient F = cell @ inv(orig_cell)."""
        return self.atoms.get_cell().array @ np.linalg.inv(self.orig_cell)

    def _get_log_deform(self) -> np.ndarray:
        """Get log of deformation gradient, scaled by exp_cell_factor."""
        F = self._get_deformation_gradient()
        return logm(F).real * self.exp_cell_factor

    def _set_cell_from_log_deform(self, log_deform_scaled: np.ndarray) -> None:
        """Set cell from scaled log-deformation gradient."""
        log_deform = log_deform_scaled / self.exp_cell_factor
        F = expm(log_deform.real)
        new_cell = self.orig_cell @ F.T
        self.atoms.set_cell(new_cell, scale_atoms=False)

    def _masked_cell_params(self) -> np.ndarray:
        """Get cell parameters as flat array (only free DOF)."""
        log_deform = self._get_log_deform()
        return log_deform[self.cell_mask]

    def _set_masked_cell_params(self, params: np.ndarray) -> None:
        """Set cell from flat array of free DOF."""
        log_deform = self._get_log_deform()
        log_deform[self.cell_mask] = params
        self._set_cell_from_log_deform(log_deform)

    def set_x(self, target: np.ndarray):
        """Set Cartesian positions and cell parameters.

        Much simpler than CellInternalPES since Cartesian positions can be
        set directly without iterative or ODE-based solvers.

        Returns
        -------
        dx_initial, dx_final, g_par : tuple of np.ndarray
            Displacement information for Hessian update.
        """
        x0 = self.get_x()
        dx_initial = target - x0

        # Split target into Cartesian and cell parts
        x_cart_target = target[:self.n_cart]
        cell_target = target[self.n_cart:]

        # Get initial cell params
        cell_params0 = self._masked_cell_params()

        # Update cell first
        self._set_masked_cell_params(cell_target)

        # Update positions directly (simple for Cartesian!)
        x_cart0 = self.apos.ravel()
        diff = x_cart_target - x_cart0
        self.atoms.positions = x_cart_target.reshape((-1, 3))

        dx_final = np.concatenate([diff, cell_target - cell_params0])

        # Return parallel gradient for Hessian update
        g_old = self.curr.get('g', None)
        if g_old is not None:
            g_par = g_old.copy()
        else:
            g_par = np.zeros(self.dim)

        return dx_initial, dx_final, g_par

    def eval(self) -> tuple:
        """Evaluate energy and combined gradient (Cartesian + cell)."""
        self.neval += 1
        f = self.atoms.get_potential_energy()

        # Add pressure contribution: H = E + P*V
        if self.scalar_pressure != 0.0:
            f += self.scalar_pressure * self.atoms.get_volume()

        # Cartesian gradient (no Jacobian needed)
        forces = self.atoms.get_forces()
        g_cart = -forces.ravel()

        # Stress tensor -> cell gradient
        stress = self.atoms.get_stress()  # 6-component Voigt, eV/Å³
        g_cell = self._stress_to_cell_gradient(stress)

        self.write_traj()
        return f, np.concatenate([g_cart, g_cell])

    def _stress_to_cell_gradient(self, stress_voigt: np.ndarray) -> np.ndarray:
        """Convert stress tensor to gradient w.r.t. cell parameters.

        Following ASE's FrechetCellFilter formulation.
        """
        volume = self.atoms.get_volume()
        stress_3x3 = voigt_6_to_full_3x3_stress(stress_voigt)

        # Add external pressure contribution
        if self.scalar_pressure != 0.0:
            stress_3x3 += self.scalar_pressure * np.eye(3)

        # Gradient w.r.t. cell = V * stress
        g_cell_3x3 = volume * stress_3x3

        # Apply cell mask
        g_cell_masked = g_cell_3x3 * self.cell_mask

        # Scale by exp_cell_factor (inverse of the scaling in get_x)
        g_cell_full = g_cell_masked / self.exp_cell_factor

        # Return only free DOF
        return g_cell_full[self.cell_mask]

    def get_g(self) -> np.ndarray:
        """Get combined gradient (Cartesian + cell)."""
        self._update()
        return self.curr['g'].copy()

    def _update(self, feval: bool = True):
        """Update current state including cell gradient."""
        x = self.get_x()
        new_point = True
        if self.curr['x'] is not None and np.all(x == self.curr['x']):
            if feval and self.curr['f'] is None:
                new_point = False
            else:
                return False

        if feval:
            f, g = self.eval()
        else:
            f = None
            g = None

        if new_point:
            self.last = self.curr.copy()

        self.curr['x'] = x
        self.curr['f'] = f
        self.curr['g'] = g
        self._update_basis()
        return True

    def _calc_basis(self):
        """Calculate basis including cell DOF.

        The cell DOF are treated as unconstrained additional coordinates.
        """
        # Compute Cartesian basis directly (not via parent, since parent uses self.dim)
        # This mirrors PES._calc_basis but uses n_cart instead of self.dim
        pos_hash = self.atoms.positions.tobytes()
        if self._basis_cache['pos_hash'] == pos_hash:
            return self._basis_cache['result']

        drdx_cart = self.cons.jacobian()  # Constraint Jacobian for Cartesian coords
        U, S, VT = np.linalg.svd(drdx_cart)
        ncons = np.sum(S > 1e-6)
        Ucons_cart = VT[:ncons].T
        Ufree_cart = VT[ncons:].T
        Unred_cart = np.eye(self.n_cart)

        # Extend to include cell DOF
        n_total = self.n_cart + self.n_cell_dof

        # drdx extended with zeros for cell columns
        drdx = np.zeros((drdx_cart.shape[0], n_total))
        drdx[:, :self.n_cart] = drdx_cart

        # Ucons stays the same (no cell constraints)
        Ucons = np.zeros((n_total, Ucons_cart.shape[1]))
        Ucons[:self.n_cart, :] = Ucons_cart

        # Unred extended with identity for cell DOF
        Unred = np.zeros((n_total, Unred_cart.shape[1] + self.n_cell_dof))
        Unred[:self.n_cart, :Unred_cart.shape[1]] = Unred_cart
        Unred[self.n_cart:, Unred_cart.shape[1]:] = np.eye(self.n_cell_dof)

        # Ufree extended with identity for cell DOF
        Ufree = np.zeros((n_total, Ufree_cart.shape[1] + self.n_cell_dof))
        Ufree[:self.n_cart, :Ufree_cart.shape[1]] = Ufree_cart
        Ufree[self.n_cart:, Ufree_cart.shape[1]:] = np.eye(self.n_cell_dof)

        result = drdx, Ucons, Unred, Ufree

        # Cache the result
        self._basis_cache['pos_hash'] = pos_hash
        self._basis_cache['result'] = result
        return result

    def converged(self, fmax: float, smax: float = None, cmax: float = 1e-5):
        """Check convergence of forces and stress.

        Parameters
        ----------
        fmax : float
            Maximum force tolerance (eV/Å).
        smax : float, optional
            Maximum stress tolerance. If None, uses fmax.
        cmax : float, optional
            Constraint residual tolerance.

        Returns
        -------
        conv : bool
            True if converged.
        fmax_actual : float
            Maximum force.
        cmax_actual : float
            Constraint residual norm.
        smax_actual : float
            Maximum stress gradient.
        """
        if smax is None:
            smax = fmax

        # Force convergence (project out constraints)
        g = self.get_g()
        g_cart = g[:self.n_cart]
        Ufree = self.get_Ufree()
        Ufree_cart = Ufree[:self.n_cart, :Ufree.shape[1] - self.n_cell_dof]
        g_proj = (Ufree_cart @ Ufree_cart.T @ g_cart).reshape((-1, 3))

        fmax_actual = np.linalg.norm(g_proj, axis=1).max()

        # Stress convergence
        g_cell = g[self.n_cart:]
        smax_actual = np.abs(g_cell).max() if len(g_cell) > 0 else 0.0

        # Constraint residual
        cmax_actual = np.linalg.norm(self.get_res())

        conv = (fmax_actual < fmax) and (smax_actual < smax) and (cmax_actual < cmax)
        return conv, fmax_actual, cmax_actual, smax_actual

    def get_projected_forces(self) -> np.ndarray:
        """Returns Nx3 array of atomic forces orthogonal to constraints."""
        g = self.get_g()
        g_cart = g[:self.n_cart]
        Ufree = self.get_Ufree()
        Ufree_cart = Ufree[:self.n_cart, :]
        return -((Ufree_cart @ Ufree_cart.T) @ g_cart).reshape((-1, 3))

    def get_drdx(self):
        """Get constraint Jacobian extended for cell DOF."""
        drdx_cart = PES.get_drdx(self)
        n_cons = drdx_cart.shape[0]
        drdx = np.zeros((n_cons, self.dim))
        drdx[:, :self.n_cart] = drdx_cart
        return drdx

    def get_Hc(self):
        """Get constraint Hessian extended for cell DOF."""
        Hc_cart = PES.get_Hc(self)
        Hc = np.zeros((self.dim, self.dim))
        Hc[:self.n_cart, :self.n_cart] = Hc_cart
        return Hc
