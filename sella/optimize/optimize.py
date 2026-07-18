#!/usr/bin/env python3

import logging
import warnings
from time import localtime, strftime
from typing import Union, Callable, Optional

import numpy as np
from ase import Atoms
from ase.optimize.optimize import Optimizer
from ase.utils import basestring
from ase.io.trajectory import Trajectory

from .restricted_step import get_restricted_step, MaxInternalStep, \
    RestrictedAtomicStep
from sella.peswrapper import PES, InternalPES, CellInternalPES, CellCartesianPES
from sella.internal import Internals, Constraints

logger = logging.getLogger(__name__)

_default_kwargs = dict(
    minimum=dict(
        delta0=1e-1,
        sigma_inc=1.15,
        sigma_dec=0.90,
        rho_inc=1.035,
        rho_dec=100,
        method='qn',
        eig=False
    ),
    saddle=dict(
        delta0=0.1,
        sigma_inc=1.15,
        sigma_dec=0.65,
        rho_inc=1.035,
        rho_dec=5.0,
        method='prfo',
        eig=True
    )
)


class Sella(Optimizer):
    def __init__(
        self,
        atoms: Atoms,
        restart: bool = None,
        logfile: str = '-',
        trajectory: Union[str, Trajectory] = None,
        master: bool = None,
        delta0: float = None,
        sigma_inc: float = None,
        sigma_dec: float = None,
        rho_dec: float = None,
        rho_inc: float = None,
        order: int = 1,
        eig: bool = None,
        eta: float = 1e-4,
        method: str = None,
        gamma: float = 0.1,
        threepoint: bool = False,
        constraints: Constraints = None,
        constraints_tol: float = 1e-5,
        v0: np.ndarray = None,
        internal: Union[bool, Internals] = False,
        append_trajectory: bool = False,
        rs: str = None,
        nsteps_per_diag: int = 3,
        diag_every_n: Optional[int] = None,
        hessian_function: Optional[Callable[[Atoms], np.ndarray]] = None,
        optimize_cell: bool = False,
        cell_mask: np.ndarray = None,
        exp_cell_factor: float = None,
        scalar_pressure: float = 0.0,
        smax: float = None,
        allow_fragments: bool = False,
        niggli: bool = False,
        refine_initial_hessian: Union[bool, int] = False,
        save_hessian: str = None,
        exact_geodesic: bool = None,
        **kwargs
    ):
        """Initialize Sella optimizer.

        Parameters
        ----------
        atoms : Atoms
            ASE Atoms object to optimize.
        optimize_cell : bool, optional
            If True, optimize unit cell parameters along with atomic positions.
            Requires order=0. Default is False.
        cell_mask : ndarray, optional
            Boolean mask of shape (3, 3) indicating which cell DOF are free.
            Default is all True (full cell optimization).
        exp_cell_factor : float, optional
            Scaling factor for cell parameterization. Default is sqrt(n_atoms)
            for rigid-fragment internal cell optimization, otherwise n_atoms.
        scalar_pressure : float, optional
            External pressure in eV/Å³ for cell optimization. Default is 0.
        smax : float, optional
            Maximum effective stress tolerance for convergence when
            optimize_cell=True, derived from Sella's optimized cell-gradient
            path. If None, uses fmax as the tolerance on Sella's ASE-like
            scaled log-cell gradient.
        allow_fragments : bool, optional
            If True, allow disconnected molecular fragments when using internal
            coordinates. Adds translation and rotation coordinates (TRICs) for
            each fragment. Useful for molecular crystals. Default is False.
        niggli : bool, optional
            If True, apply Niggli reduction during cell optimization when cell
            angles deviate more than 30 deg from 90 deg. This remaps to the
            most compact unit cell and resets the Hessian cell block.
            Default is False.
        refine_initial_hessian : bool or int, optional
            Level of Hessian refinement via finite differences:
            - False or 0: No refinement (default)
            - -1: Forward differences for cell blocks (n_cell_dof force calls)
            - True or 1: Central differences for cell blocks (2 * n_cell_dof force calls)
            - 2: Also refine translation/rotation blocks for molecular crystals
              (adds 2 * n_tric force calls, where n_tric = n_fragments * 6)
            - 3: Refine full internal Hessian (2 * n_internal force calls, expensive!)
        save_hessian : str, optional
            Path to save the initial Hessian as .npy file for analysis.
        """
        if order == 0:
            default = _default_kwargs['minimum']
        else:
            default = _default_kwargs['saddle']

        # Validate cell optimization parameters
        self.optimize_cell = optimize_cell
        self.allow_fragments = allow_fragments
        self.niggli = niggli
        self.exact_geodesic = exact_geodesic if exact_geodesic is not None else True
        self.smax = smax
        if optimize_cell:
            if order != 0:
                raise ValueError(
                    "Cell optimization is only supported for minima (order=0), "
                    f"got order={order}."
                )
            if not np.any(atoms.pbc):
                raise ValueError(
                    "Cell optimization requires periodic boundary conditions. "
                    "Set atoms.pbc = True for periodic systems."
                )

        if trajectory is not None:
            if isinstance(trajectory, basestring):
                mode = "a" if append_trajectory else "w"
                trajectory = Trajectory(trajectory, mode=mode,
                                        atoms=atoms, master=master)
            # Register trajectory for cleanup when close() is called
            self.closelater(trajectory)

        self.peskwargs = kwargs.copy()
        self.user_internal = internal
        self.user_constraints = (
            constraints.copy() if constraints is not None else None
        )
        self.initialize_pes(
            atoms,
            trajectory,
            order,
            eta,
            constraints,
            v0,
            internal,
            hessian_function,
            optimize_cell=optimize_cell,
            cell_mask=cell_mask,
            exp_cell_factor=exp_cell_factor,
            scalar_pressure=scalar_pressure,
            allow_fragments=allow_fragments,
            refine_initial_hessian=refine_initial_hessian,
            save_hessian=save_hessian,
            exact_geodesic=self.exact_geodesic,
            **kwargs
        )

        if rs is None:
            rs = 'mis' if internal else 'ras'
        self.rs = get_restricted_step(rs)
        Optimizer.__init__(self, atoms, restart=restart,
                           logfile=logfile, trajectory=None,
                           master=master)

        if delta0 is None:
            delta0 = default['delta0']
        if rs in ['mis', 'ras']:
            self.delta = delta0
        else:
            self.delta = delta0 * self.pes.get_Ufree().shape[1]
        self.delta_cell = delta0

        self.sigma_inc = sigma_inc if sigma_inc is not None else default['sigma_inc']
        self.sigma_dec = sigma_dec if sigma_dec is not None else default['sigma_dec']
        self.rho_inc = rho_inc if rho_inc is not None else default['rho_inc']
        self.rho_dec = rho_dec if rho_dec is not None else default['rho_dec']
        self.method = method if method is not None else default['method']
        self.eig = eig if eig is not None else default['eig']

        self.ord = order
        self.eta = eta
        self.delta_min = self.eta
        self.constraints_tol = constraints_tol
        self.diagkwargs = dict(gamma=gamma, threepoint=threepoint)
        self.rho = 1.

        if self.ord != 0 and not self.eig:
            warnings.warn("Saddle point optimizations with eig=False will "
                          "most likely fail!\n Proceeding anyway, but you "
                          "shouldn't be optimistic.")

        self.initialized = False
        self.xi = 1.
        self.nsteps_per_diag = nsteps_per_diag

        # Set by run() / first converged() call.
        self.fmax = None
        self._last_converged = None
        self.nsteps_since_diag = 0
        self.diag_every_n = np.inf if diag_every_n is None else diag_every_n

    def initialize_pes(
        self,
        atoms: Atoms,
        trajectory: str = None,
        order: int = 1,
        eta: float = 1e-4,
        constraints: Constraints = None,
        v0: np.ndarray = None,
        internal: Union[bool, Internals] = False,
        hessian_function: Optional[Callable[[Atoms], np.ndarray]] = None,
        optimize_cell: bool = False,
        cell_mask: np.ndarray = None,
        exp_cell_factor: float = None,
        scalar_pressure: float = 0.0,
        allow_fragments: bool = False,
        refine_initial_hessian: Union[bool, int] = False,
        save_hessian: str = None,
        exact_geodesic: bool = None,
        **kwargs
    ):
        if internal:
            if isinstance(internal, Internals):
                auto_find_internals = False
                if constraints is not None:
                    raise ValueError(
                        "Internals object and Constraint object cannot both "
                        "be provided to Sella. Instead, you must pass the "
                        "Constraints object to the constructor of the "
                        "Internals object."
                    )
            else:
                auto_find_internals = True
                internal = Internals(
                    atoms, cons=constraints, allow_fragments=allow_fragments,
                )
            self.internal = internal.copy()
            self.constraints = None

            if optimize_cell:
                # Use CellInternalPES for combined internal + cell optimization
                self.pes = CellInternalPES(
                    atoms,
                    internals=internal,
                    trajectory=trajectory,
                    eta=eta,
                    v0=v0,
                    auto_find_internals=auto_find_internals,
                    hessian_function=hessian_function,
                    exp_cell_factor=exp_cell_factor,
                    cell_mask=cell_mask,
                    scalar_pressure=scalar_pressure,
                    refine_initial_hessian=refine_initial_hessian,
                    save_hessian=save_hessian,
                    exact_geodesic=exact_geodesic,
                    **kwargs
                )
            else:
                self.pes = InternalPES(
                    atoms,
                    internals=internal,
                    trajectory=trajectory,
                    eta=eta,
                    v0=v0,
                    auto_find_internals=auto_find_internals,
                    hessian_function=hessian_function,
                    exact_geodesic=exact_geodesic,
                    **kwargs
                )
        else:
            self.internal = None
            if constraints is None:
                constraints = Constraints(atoms)
            self.constraints = constraints
            if optimize_cell:
                # Use CellCartesianPES for Cartesian + cell optimization
                self.pes = CellCartesianPES(
                    atoms,
                    constraints=constraints,
                    trajectory=trajectory,
                    eta=eta,
                    v0=v0,
                    hessian_function=hessian_function,
                    exp_cell_factor=exp_cell_factor,
                    cell_mask=cell_mask,
                    scalar_pressure=scalar_pressure,
                    refine_initial_hessian=refine_initial_hessian,
                    save_hessian=save_hessian,
                    **kwargs
                )
            else:
                self.pes = PES(
                    atoms,
                    constraints=constraints,
                    trajectory=trajectory,
                    eta=eta,
                    v0=v0,
                    hessian_function=hessian_function,
                    **kwargs
                )
        self.trajectory = self.pes.traj

    def _initialize_eigensolver(self):
        """Build or diagonalize the initial Hessian when eig mode is active."""
        if self.pes.hessian_function is not None:
            self.pes.calculate_hessian()
        else:
            self.pes.diag(**self.diagkwargs)
        self.nsteps_since_diag = -1

    def _ensure_initialized(self):
        if not self.initialized:
            self.pes.get_g()
            if self.eig:
                self._initialize_eigensolver()
            self.initialized = True

    def _cell_rs_kwargs(self):
        """Restricted-step kwargs needed when atom and cell radii differ."""
        if self.optimize_cell and isinstance(self.rs, type) and issubclass(
            self.rs, (MaxInternalStep, RestrictedAtomicStep)
        ):
            return {'wc': self.delta / self.delta_cell}
        return {}

    def _restricted_step(self, rs_kwargs):
        return self.rs(
            self.pes, self.ord, self.delta, method=self.method, **rs_kwargs
        ).get_s()

    def _valid_inequality_step(self, x0, rs_kwargs):
        """Retry restricted steps until inactive inequality constraints stay valid."""
        while True:
            s, smag = self._restricted_step(rs_kwargs)
            self.pes.set_x(x0 + s)
            all_valid = self.pes.cons.validate_inequalities()
            self.pes._update_basis()
            self.pes.restore()
            if all_valid:
                self.pes._update_basis()
                return s, smag

    def _predict_step(self):
        self._ensure_initialized()
        self.pes.cons.disable_satisfied_inequalities()
        self.pes._update_basis()
        self.pes.save()
        x0 = self.pes.get_x()

        rs_kwargs = self._cell_rs_kwargs()
        if self.pes.cons.has_inequalities():
            return self._valid_inequality_step(x0, rs_kwargs)
        return self._restricted_step(rs_kwargs)

    def _should_diagonalize(self):
        """Return whether this step should run the eigensolver."""
        if self.nsteps_since_diag >= self.diag_every_n:
            return True
        if not (self.eig and self.nsteps_since_diag >= self.nsteps_per_diag):
            return False
        if self.pes.H.evals is None:
            return True

        Unred = self.pes.get_Unred()
        evals = self.pes.get_HL_projected(Unred).evals[:self.ord]
        return bool((evals > 0).any())

    def _record_diagonalization(self, ev):
        if ev:
            self.nsteps_since_diag = 0
        else:
            self.nsteps_since_diag += 1

    def _cell_trust_components(self, s, smag):
        if not (
            self.optimize_cell
            and isinstance(self.pes, (CellInternalPES, CellCartesianPES))
        ):
            return smag, 0

        # Split the step into coordinate (internal or Cartesian) and cell blocks
        # so each trust radius adapts independently. The boundary is n_coords
        # (n_internal for CellInternalPES, n_cart for CellCartesianPES).
        n_coord = self.pes.n_coords
        smag_int = np.max(np.abs(s[:n_coord])) if n_coord > 0 else 0
        smag_cell = np.max(np.abs(s[n_coord:])) if len(s) > n_coord else 0
        return smag_int, smag_cell

    def _update_trust_radius(self, rho, s, smag):
        if rho is None:
            self.rho = 1.
            return

        smag_int, smag_cell = self._cell_trust_components(s, smag)
        if rho < 1. / self.rho_dec or rho > self.rho_dec:
            component_tol = (
                32 * np.finfo(float).eps * max(1.0, abs(float(smag)))
            )
            if smag_int > component_tol:
                self.delta = max(
                    smag_int * self.sigma_dec, self.delta_min
                )
            if smag_cell > component_tol:
                self.delta_cell = max(
                    self.delta_cell * self.sigma_dec, self.delta_min
                )
        elif 1. / self.rho_inc < rho < self.rho_inc:
            self.delta = max(self.sigma_inc * smag_int, self.delta)
            if smag_cell > 0:
                self.delta_cell = max(
                    self.sigma_inc * smag_cell, self.delta_cell
                )
        self.rho = rho

    def _bad_internal_cell_kwargs(self):
        if isinstance(self.pes, CellInternalPES):
            return (
                self.pes.cell_mask,
                self.pes.exp_cell_factor,
                self.pes.scalar_pressure,
            )
        return None, None, 0.0

    def _rebuild_after_bad_internals(self):
        """Recreate the PES after internals become singular or ill-defined."""
        cell_mask, exp_cell_factor, scalar_pressure = (
            self._bad_internal_cell_kwargs()
        )
        constraints = (
            None if self.user_constraints is None
            else self.user_constraints.copy()
        )
        self.initialize_pes(
            atoms=self.pes.atoms,
            trajectory=self.pes.traj,
            order=self.ord,
            eta=self.pes.eta,
            constraints=constraints,
            v0=None,  # TODO: use leftmost eigenvector from old H
            internal=self.user_internal,
            hessian_function=self.pes.hessian_function,
            optimize_cell=self.optimize_cell,
            cell_mask=cell_mask,
            exp_cell_factor=exp_cell_factor,
            scalar_pressure=scalar_pressure,
            allow_fragments=self.allow_fragments,
            exact_geodesic=self.exact_geodesic,
            # Forward the original PES kwargs (e.g. an explicit
            # rigid_fragments) so a user override is not lost on rebuild and
            # silently replaced by auto-detection.
            **self.peskwargs,
        )
        self.initialized = False
        self.rho = 1

    def _maybe_rebuild_bad_internals(self):
        # Bad internal-coordinate rebuilds skip the trust-radius update because
        # the PES coordinate system has just been regenerated.
        if self.internal and self.pes.int.check_for_bad_internals():
            self._rebuild_after_bad_internals()
            return True
        return False

    def _maybe_apply_niggli(self):
        if self.optimize_cell and self.niggli and self.pes.maybe_niggli_reduce():
            logger.info("Applied Niggli reduction to reduce cell skewness")
            self.initialized = False
            self.rho = 1.

    def step(self):
        s, smag = self._predict_step()
        ev = self._should_diagonalize()
        self._record_diagonalization(ev)
        rho = self.pes.kick(s, ev, **self.diagkwargs)

        if self._maybe_rebuild_bad_internals():
            return

        self._update_trust_radius(rho, s, smag)
        self._maybe_apply_niggli()

    def gradient_converged(self, gradient=None):
        return self.converged()

    def converged(self, forces=None):
        # fmax may still be None if converged() is called before run()
        fmax = self.fmax if self.fmax is not None else 0.05  # Default threshold
        if self.optimize_cell:
            result = self.pes.converged(fmax, smax=self.smax)
            self._last_converged = result
            return result[0]
        result = self.pes.converged(fmax)
        self._last_converged = result
        return result[0]

    def log(self, forces=None):
        if self.logfile is None:
            return
        if self.optimize_cell:
            result = self._last_converged
            if result is None or len(result) != 4:
                fmax = self.fmax if self.fmax is not None else 0.05
                result = self.pes.converged(fmax, smax=self.smax)
            _, fmax, cmax, smax_actual = result
            e = self.pes.get_f()
            T = strftime("%H:%M:%S", localtime())
            name = self.__class__.__name__
            buf = " " * len(name)
            if self.nsteps == 0:
                self.logfile.write(buf + "{:>4s} {:>8s} {:>15s} {:>12s} {:>12s} "
                                   "{:>12s} {:>12s} {:>12s} {:>12s}\n"
                                   .format("Step", "Time", "Energy", "fmax",
                                           "smax", "cmax", "rtrust",
                                           "strust", "rho"))
            self.logfile.write("{} {:>3d} {:>8s} {:>15.6f} {:>12.4f} {:>12.4f} "
                               "{:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}\n"
                               .format(name, self.nsteps, T, e, fmax, smax_actual,
                                       cmax, self.delta, self.delta_cell,
                                       self.rho))
        else:
            result = self._last_converged
            if result is None or len(result) != 3:
                result = self.pes.converged(self.fmax)
            _, fmax, cmax = result
            e = self.pes.get_f()
            T = strftime("%H:%M:%S", localtime())
            name = self.__class__.__name__
            buf = " " * len(name)
            if self.nsteps == 0:
                self.logfile.write(buf + "{:>4s} {:>8s} {:>15s} {:>12s} {:>12s} "
                                   "{:>12s} {:>12s}\n"
                                   .format("Step", "Time", "Energy", "fmax",
                                           "cmax", "rtrust", "rho"))
            self.logfile.write("{} {:>3d} {:>8s} {:>15.6f} {:>12.4f} {:>12.4f} "
                               "{:>12.4f} {:>12.4f}\n"
                               .format(name, self.nsteps, T, e, fmax, cmax,
                                       self.delta, self.rho))
        try:
            self.logfile.flush()
        except (AttributeError, TypeError):
            pass
