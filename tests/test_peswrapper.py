import pytest

import numpy as np

from ase.build import molecule
from ase.calculators.emt import EMT

from sella.peswrapper import PES, InternalPES, CellInternalPES, \
    _range_space_projector
from sella.internal import Internals

@pytest.mark.parametrize("name,traj,cons",
                         [("CH4", "CH4.traj", None),
                          ("CH4", None, dict(bonds=((0, 1)))),
                          ("C6H6", None, None),
                          ])
def test_PES(name, traj, cons):
    tol = dict(atol=1e-6, rtol=1e-6)

    atoms = molecule(name)

    # EMT is *not* appropriate for molecules like this, but this is one
    # of the few calculators that is guaranteed to be available to all
    # users, and we don't need to use a physical PES to test this.
    atoms.calc = EMT()
    for MyPES in [PES, InternalPES]:
        pes = PES(atoms, trajectory=traj)

        pes.kick(0., diag=True, gamma=0.1)

        for i in range(2):
            pes.kick(-pes.get_g() * 0.01)

        assert pes.H is not None
        assert not pes.converged(0.)[0]
        assert pes.converged(1e100)
        A = pes.get_Ufree().T @ pes.get_Ucons()
        np.testing.assert_allclose(A, 0, **tol)

        pes.kick(-pes.get_g() * 0.001, diag=True, gamma=0.1)


# NH3/CH4/C2H6/C6H6 are rank-deficient (redundant internals + rigid-body
# null space at machine epsilon); H2O is full rank and exercises the
# non-deficient branch.
@pytest.mark.parametrize("name", ['H2O', 'NH3', 'CH4', 'C2H6', 'C6H6'])
def test_range_space_projector_matches_svd_truth(name):
    """The guess-Hessian projector must equal the SVD projector onto range(B).

    The internal-coordinate Jacobian of a free (non-periodic) molecule is
    typically rank-deficient: redundant internals plus the molecule's
    rigid-body modes produce singular values at machine epsilon. A
    non-pivoting economic QR keeps those near-null directions, leaking that
    subspace into the projected Hessian; _range_space_projector uses pivoting
    QR with rank truncation and must match the SVD-truth projector to machine
    precision.
    """
    atoms = molecule(name)
    internals = Internals(atoms)
    internals.find_all_bonds()
    internals.find_all_angles()
    internals.find_all_dihedrals()
    B = internals.jacobian()

    S = np.linalg.svd(B, compute_uv=False)
    rank = int(np.sum(S > 1e-6 * S[0]))
    full_rank = min(B.shape)

    # SVD-truth projector onto range(B).
    U = np.linalg.svd(B, full_matrices=False)[0]
    P_true = U[:, :rank] @ U[:, :rank].T

    P = _range_space_projector(B)
    np.testing.assert_allclose(P, P_true, atol=1e-10)

    if rank < full_rank:
        # The old non-pivoting economic-QR projector retains the null space
        # and is materially wrong; this guards against a regression to it.
        Q, _ = np.linalg.qr(B, mode='reduced')
        P_old = Q @ Q.T
        assert np.linalg.norm(P_old - P_true) > 1e-3


def test_range_space_projector_exercises_rank_deficient_input():
    """At least one parametrized molecule must be genuinely rank-deficient,
    otherwise the regression guard above never runs."""
    atoms = molecule('C2H6')
    internals = Internals(atoms)
    internals.find_all_bonds()
    internals.find_all_angles()
    internals.find_all_dihedrals()
    B = internals.jacobian()
    S = np.linalg.svd(B, compute_uv=False)
    rank = int(np.sum(S > 1e-6 * S[0]))
    assert rank < min(B.shape)


def test_range_space_projector_periodic_rank_deficient():
    """Periodic systems are also column-rank deficient when redundant internals
    (bonds/angles/dihedrals) are used: the intramolecular coordinates fail to
    span the inter-fragment rigid-body motions, and the cell does not rescue
    that. (Overcompleteness alone does not -- a fully bonded network like a
    metal is overcomplete but full column rank.) The projector must still match
    SVD truth here, exactly as in the non-periodic case.
    """
    from ase.calculators.lj import LennardJones

    atoms = None
    for pos in [(1, 1, 1), (4, 1, 1), (1, 4, 1), (1, 1, 4)]:
        w = molecule('H2O')
        w.positions += pos
        atoms = w if atoms is None else atoms + w
    atoms.set_cell([8, 8, 8])
    atoms.pbc = True
    atoms.calc = LennardJones()

    # allow_fragments=False forces redundant bond/angle/dihedral internals
    # (the minimal TRIC fragment set would instead be square and full rank).
    internals = Internals(atoms, allow_fragments=False)
    pes = CellInternalPES(atoms, internals, auto_find_internals=True)
    B = pes.int.jacobian()

    S = np.linalg.svd(B, compute_uv=False)
    rank = int(np.sum(S > 1e-6 * S[0]))
    full_rank = min(B.shape)
    assert rank < full_rank, "expected a column-rank-deficient periodic Jacobian"

    U = np.linalg.svd(B, full_matrices=False)[0]
    P_true = U[:, :rank] @ U[:, :rank].T

    P = _range_space_projector(B)
    np.testing.assert_allclose(P, P_true, atol=1e-10)

    # Guard against regression to the non-pivoting economic-QR projector.
    Q, _ = np.linalg.qr(B, mode='reduced')
    P_old = Q @ Q.T
    assert np.linalg.norm(P_old - P_true) > 1e-3


def test_fixatoms_does_not_enable_rigid_fragments():
    """A FixAtoms constraint must not trigger rigid-fragment cell mode.

    FixAtoms becomes Translation coordinates, but those are constraints, not
    rigid fragments. rigid_fragments auto-detection must key on genuine
    fragment_atom_groups (set only by find_all_bonds with allow_fragments),
    not on the mere presence of any translation coordinate.
    """
    from ase import Atoms
    from ase.constraints import FixAtoms
    from ase.calculators.lj import LennardJones
    from sella import Sella

    atoms = Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0]],
                  cell=np.eye(3) * 8.0, pbc=True)
    atoms.calc = LennardJones()
    atoms.set_constraint(FixAtoms(indices=[0]))
    pes = Sella(atoms, order=0, internal=True, optimize_cell=True).pes
    assert pes.rigid_fragments is False


def test_genuine_fragments_enable_rigid_fragments():
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from sella import Sella

    atoms = Atoms('OH2OH2',
                  positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0],
                             [6, 0, 0], [6.96, 0, 0], [6, 0.96, 0]],
                  cell=np.eye(3) * 12.0, pbc=True)
    atoms.calc = LennardJones()
    pes = Sella(atoms, order=0, internal=True, optimize_cell=True,
                allow_fragments=True).pes
    assert pes.rigid_fragments is True


def test_copy_preserves_fragment_atom_groups():
    from ase import Atoms
    atoms = Atoms('OH2OH2',
                  positions=[[0, 0, 0], [0.96, 0, 0], [0, 0.96, 0],
                             [6, 0, 0], [6.96, 0, 0], [6, 0.96, 0]],
                  cell=np.eye(3) * 12.0, pbc=True)
    ic = Internals(atoms, allow_fragments=True)
    ic.find_all_bonds()
    cp = ic.copy()
    assert cp.fragment_atom_groups is not None
    assert len(cp.fragment_atom_groups) == len(ic.fragment_atom_groups)


def test_inequality_toggle_invalidates_basis_cache():
    """Disabling a satisfied inequality must refresh the cached basis.

    _calc_basis / QR / Binv are cached by _state_hash, which keyed on geometry
    only. disable_satisfied_inequalities flips the constraint active mask
    without moving atoms, so drdx shrinks but Ucons stayed stale until the
    geometry changed. The mask is now folded into _state_hash.
    """
    from ase import Atoms
    from ase.calculators.lj import LennardJones
    from sella import Sella
    from sella.internal import Constraints

    atoms = Atoms('H3', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    atoms.calc = LennardJones()
    cons = Constraints(atoms)
    # bond(0,1) currently 1.0, constrained < 2.0 -> satisfied inequality.
    cons.fix_bond((0, 1), target=2.0, comparator='lt')
    pes = Sella(atoms, order=0, constraints=cons, internal=False).pes

    ncols_before = pes.get_Ucons().shape[1]
    nrows_before = pes.get_drdx().shape[0]

    cons.disable_satisfied_inequalities()
    nrows_after = pes.get_drdx().shape[0]
    ncols_after = pes.get_Ucons().shape[1]

    # One inequality row dropped; the cached constraint basis must follow.
    assert nrows_after == nrows_before - 1
    assert ncols_after == ncols_before - 1
