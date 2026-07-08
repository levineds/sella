import pytest

import numpy as np
from ase import Atoms
from ase.build import molecule

from sella.internal import (
    Bond, Angle, Dihedral, Displacement, Constraints, Internals
)


def res(pos: np.ndarray, internal: Internals) -> np.ndarray:
    internal.atoms.positions = pos.reshape((-1, 3))
    return internal.calc()


def jacobian(pos: np.ndarray, internal: Internals) -> np.ndarray:
    internal.atoms.positions = pos.reshape((-1, 3))
    return internal.jacobian()


def hessian(pos: np.ndarray, internal: Internals) -> np.ndarray:
    internal.atoms.positions = pos.reshape((-1, 3))
    return internal.hessian()


@pytest.mark.parametrize("name", ['CH4', 'C6H6', 'C2H6'])
def test_get_internal(name: str) -> None:
    atoms = molecule(name)
    internal = Internals(atoms)
    internal.find_all_bonds()
    internal.find_all_angles()
    internal.find_all_dihedrals()
    jac = internal.jacobian()
    hess = internal.hessian()

    x0 = atoms.positions.ravel().copy()
    x = x0.copy()
    dx = 1e-4

    jac_numer = np.zeros_like(jac)
    hess_numer = np.zeros_like(hess)
    for i in range(len(x)):
        x[i] += dx
        atoms.positions = x.reshape((-1, 3))
        res_plus = internal.calc()
        jac_plus = internal.jacobian()
        x[i] = x0[i] - dx
        atoms.positions = x.reshape((-1, 3))
        res_minus = internal.calc()
        jac_minus = internal.jacobian()
        x[i] = x0[i]
        atoms.positions = x.reshape((-1, 3))
        jac_numer[:, i] = (internal.wrap(res_plus - res_minus)) / (2 * dx)
        hess_numer[:, i, :] = (jac_plus - jac_minus) / (2 * dx)
    np.testing.assert_allclose(jac, jac_numer, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(hess, hess_numer, rtol=1e-7, atol=1e-7)


class TestTRICs:
    """Tests for Translation-Rotation Internal Coordinates (TRICs)."""

    def test_tric_single_atom_fragment(self):
        """Test TRICs with a single-atom fragment (should not raise assertion).

        This tests the bug fix for the line ordering issue in find_all_bonds()
        where single atoms would incorrectly get rotation ICs added.
        """
        # Bi(NO3)3 cluster from the bug report - Bi is a single atom, NO3 are fragments
        atoms = Atoms(
            'BiN3O9',
            positions=[
                [-0.168754, 0.103309, -0.601068],   # Bi
                [-1.452579, 0.996969, 1.671974],    # N
                [-1.906613, 1.312382, 2.719561],    # O
                [-0.390479, 0.236458, 1.599985],    # O
                [-1.916359, 1.339852, 0.548706],    # O
                [2.088604, 1.559729, 0.184556],     # N
                [3.081561, 2.106988, 0.537575],     # O
                [0.991304, 2.160371, -0.042657],    # O
                [2.046745, 0.279049, -0.004926],    # O
                [-0.824031, -2.516641, 0.135921],   # N
                [-1.024602, -3.638619, 0.469313],   # O
                [0.376482, -2.057305, -0.023988],   # O
                [-1.745220, -1.672049, -0.097571],  # O
            ]
        )
        # Use scale=1.0 to ensure fragments are detected (not bonded via 1.25 scale)
        ints = Internals(atoms, allow_fragments=True)
        # This should not raise an assertion error even though Bi is a single atom
        ints.find_all_bonds(scale=1.0)
        ints.find_all_angles()
        ints.find_all_dihedrals()

        # Should have translations (including for the single Bi atom)
        assert len(ints.internals['translations']) > 0

        # Rotations should only be for multi-atom fragments (NO3 groups)
        # Bi should NOT have rotation ICs
        for rot in ints.internals['rotations']:
            assert len(rot.indices) >= 2, "Rotation IC added to single atom!"

    def test_tric_scale_parameter(self):
        """Test that scale parameter affects bond detection."""
        atoms = Atoms(
            'BiN3O9',
            positions=[
                [-0.168754, 0.103309, -0.601068],   # Bi
                [-1.452579, 0.996969, 1.671974],    # N
                [-1.906613, 1.312382, 2.719561],    # O
                [-0.390479, 0.236458, 1.599985],    # O
                [-1.916359, 1.339852, 0.548706],    # O
                [2.088604, 1.559729, 0.184556],     # N
                [3.081561, 2.106988, 0.537575],     # O
                [0.991304, 2.160371, -0.042657],    # O
                [2.046745, 0.279049, -0.004926],    # O
                [-0.824031, -2.516641, 0.135921],   # N
                [-1.024602, -3.638619, 0.469313],   # O
                [0.376482, -2.057305, -0.023988],   # O
                [-1.745220, -1.672049, -0.097571],  # O
            ]
        )

        # With small scale, should have fragments (TRICs added)
        ints_small = Internals(atoms, allow_fragments=True)
        ints_small.find_all_bonds(scale=1.0)
        n_trans_small = len(ints_small.internals['translations'])
        n_rot_small = len(ints_small.internals['rotations'])

        # With large scale, might connect everything (no TRICs)
        ints_large = Internals(atoms, allow_fragments=True)
        ints_large.find_all_bonds(scale=1.5)
        n_trans_large = len(ints_large.internals['translations'])
        n_rot_large = len(ints_large.internals['rotations'])

        # Smaller scale should result in more fragments (more TRICs)
        assert n_trans_small >= n_trans_large
        assert n_rot_small >= n_rot_large

    def test_tric_two_separate_molecules(self):
        """Test TRICs with two well-separated molecules."""
        # Two water molecules far apart - use explicit element list for clarity
        atoms = Atoms(
            symbols=['O', 'H', 'H', 'O', 'H', 'H'],
            positions=[
                [0.0, 0.0, 0.0],     # O (first molecule)
                [0.96, 0.0, 0.0],    # H
                [0.0, 0.96, 0.0],    # H
                [10.0, 0.0, 0.0],    # O (second molecule, far away)
                [10.96, 0.0, 0.0],   # H
                [10.0, 0.96, 0.0],   # H
            ]
        )

        ints = Internals(atoms, allow_fragments=True)
        ints.find_all_bonds()
        ints.find_all_angles()

        # Should have 2 fragments, so 2 translation sets (6 coords) and 2 rotation sets (6 coords)
        assert len(ints.internals['translations']) == 6  # 3 per fragment × 2 fragments
        assert len(ints.internals['rotations']) == 6     # 3 per fragment × 2 fragments

    def test_validate_basis_with_trics(self):
        """Test that validate_basis correctly calculates DOF with TRICs."""
        # Two water molecules far apart - use explicit element list for clarity
        atoms = Atoms(
            symbols=['O', 'H', 'H', 'O', 'H', 'H'],
            positions=[
                [0.0, 0.0, 0.0],     # O (first molecule)
                [0.96, 0.0, 0.0],    # H
                [0.0, 0.96, 0.0],    # H
                [10.0, 0.0, 0.0],    # O (second molecule, far away)
                [10.96, 0.0, 0.0],   # H
                [10.0, 0.96, 0.0],   # H
            ]
        )

        ints = Internals(atoms, allow_fragments=True)
        ints.find_all_bonds()
        ints.find_all_angles()

        # With TRICs, expect 3N = 18 DOF (translations+rotations span full space)
        # validate_basis should not raise warnings for TRICs
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ints.validate_basis()
            # Should not warn if TRIC DOF calculation is correct. Ignore
            # ResourceWarnings, which can leak in from garbage collection of
            # unclosed files opened by unrelated tests (GC timing is not
            # deterministic across the suite).
            relevant = [x for x in w
                        if not issubclass(x.category, ResourceWarning)]
            assert len(relevant) == 0, (
                f"Unexpected warning: "
                f"{relevant[0].message if relevant else 'none'}"
            )

    def test_tric_optimization_convergence(self):
        """Test that optimization with TRICs converges (ODE doesn't fail).

        This tests the fix for the ODE convergence issue with ill-conditioned
        Jacobians that arise from TRICs.
        """
        from ase.calculators.lj import LennardJones
        from sella import Sella

        # Bi(NO3)3 cluster - a real-world TRIC test case
        atoms = Atoms(
            'BiN3O9',
            positions=[
                [-0.168754, 0.103309, -0.601068],   # Bi
                [-1.452579, 0.996969, 1.671974],    # N
                [-1.906613, 1.312382, 2.719561],    # O
                [-0.390479, 0.236458, 1.599985],    # O
                [-1.916359, 1.339852, 0.548706],    # O
                [2.088604, 1.559729, 0.184556],     # N
                [3.081561, 2.106988, 0.537575],     # O
                [0.991304, 2.160371, -0.042657],    # O
                [2.046745, 0.279049, -0.004926],    # O
                [-0.824031, -2.516641, 0.135921],   # N
                [-1.024602, -3.638619, 0.469313],   # O
                [0.376482, -2.057305, -0.023988],   # O
                [-1.745220, -1.672049, -0.097571],  # O
            ]
        )
        atoms.calc = LennardJones()

        # Use TRICs with small scale to ensure fragments are detected
        ints = Internals(atoms, allow_fragments=True)
        ints.find_all_bonds(scale=1.0)
        ints.find_all_angles()
        ints.find_all_dihedrals()

        # This should not raise RuntimeError about ODE convergence
        opt = Sella(atoms, internal=ints)
        # Just run a few steps to verify ODE works
        opt.run(fmax=1.0, steps=5)

    def test_periodic_multi_image_bonds(self):
        """Angle construction with two bonds connecting the same atom pair
        through different periodic images.

        Regression test for a bug where Internal.__add__ picked the wrong
        orientation, producing degenerate angles like [4, 16, 4].
        """
        atoms = Atoms(
            symbols=['F', 'C', 'C', 'C', 'Br', 'C', 'Cl', 'C', 'Br', 'C',
                     'H', 'H', 'F', 'C', 'C', 'C', 'Br', 'C', 'Cl', 'C',
                     'Br', 'C', 'H', 'H'],
            positions=[
                [0.0000, 7.4288, 8.1335],
                [0.0000, 7.4288, 0.9168],
                [0.1080, 6.2248, 1.5800],
                [0.1073, 6.2349, 2.9650],
                [0.2550, 4.5837, 3.8585],
                [0.0000, 7.4288, 3.6780],
                [0.0000, 7.4288, 5.3959],
                [4.4338, 8.6227, 2.9650],
                [4.2861, 10.2738, 3.8585],
                [4.4331, 8.6328, 1.5800],
                [0.1903, 5.3016, 1.0250],
                [4.3508, 9.5560, 1.0250],
                [2.2705, 14.8576, 0.4253],
                [2.2705, 14.8576, 7.6420],
                [2.3785, 1.2040, 6.9788],
                [2.3778, 1.1939, 5.5938],
                [2.5255, 2.8451, 4.7003],
                [2.2705, 14.8576, 4.8809],
                [2.2705, 14.8576, 3.1629],
                [2.1633, 13.6637, 5.5938],
                [2.0155, 12.0125, 4.7003],
                [2.1626, 13.6536, 6.9788],
                [2.4608, 2.1272, 7.5338],
                [2.0802, 12.7304, 7.5338],
            ],
            cell=[[4.541073, 0.0, 0.0], [0.0, 14.857576, 0.0],
                  [0.0, 0.0, 8.55882]],
            pbc=True,
        )

        ints = Internals(atoms, allow_fragments=True)
        ints.find_all_bonds()
        ints.find_all_angles()

        for angle in ints.internals['angles']:
            if angle.indices[0] == angle.indices[2]:
                assert not np.array_equal(
                    angle.kwargs['ncvecs'][0], angle.kwargs['ncvecs'][1]
                ), (
                    f"Degenerate angle {angle.indices} with identical "
                    f"ncvecs {angle.kwargs['ncvecs']}"
                )


class TestInternalEquality:
    """Regression tests for Internal.__eq__.

    A refactor once restructured the equality logic into
    ``forward_indices AND reverse_indices AND (ncvecs...)``, which only ever
    matched palindromic index sequences -- so ``Bond((0,1)) == Bond((0,1))``
    returned False. That leaked into constraint dedup (duplicate constraints
    instead of replaced targets) and degenerate constructed coordinates.
    """

    def test_identity_equality(self):
        assert Bond((0, 1)) == Bond((0, 1))
        assert Angle((0, 1, 2)) == Angle((0, 1, 2))
        assert Dihedral((0, 1, 2, 3)) == Dihedral((0, 1, 2, 3))

    def test_reversed_equality(self):
        # Internals are direction-agnostic: a coordinate equals its reverse.
        assert Bond((0, 1)) == Bond((1, 0))
        assert Angle((0, 1, 2)) == Angle((2, 1, 0))
        assert Dihedral((0, 1, 2, 3)) == Dihedral((3, 2, 1, 0))

    def test_distinct_coordinates_not_equal(self):
        assert Bond((0, 1)) != Bond((0, 2))
        assert Angle((0, 1, 2)) != Angle((0, 2, 1))
        assert Bond((0, 1)) != Angle((0, 1, 2))

    def test_ncvecs_distinguish_periodic_images(self):
        # Same atom pair through different periodic images must not compare
        # equal, or __eq__ would over-merge distinct periodic coordinates.
        b0 = Bond((0, 1), ncvecs=[[0, 0, 0]])
        b1 = Bond((0, 1), ncvecs=[[1, 0, 0]])
        assert b0 != b1
        # Reversing an image bond negates its ncvecs, so the reverse is equal.
        assert b1 == b1.reverse()

    def test_fix_bond_replaces_target(self):
        # Constraint dedup relies on __eq__: refixing the same bond must
        # replace the target, not append a second constraint.
        atoms = Atoms('H3', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        cons = Constraints(atoms)
        cons.fix_bond((0, 1), target=1.0)
        cons.fix_bond((0, 1), target=2.0)
        assert len(cons.internals['bonds']) == 1
        assert cons._targets['bonds'] == [2.0]

    def test_fix_bond_replaces_target_reversed(self):
        # Refixing via the reversed index order must also dedup.
        atoms = Atoms('H3', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        cons = Constraints(atoms)
        cons.fix_bond((0, 1), target=1.0)
        cons.fix_bond((1, 0), target=2.0)
        assert len(cons.internals['bonds']) == 1


class TestForbid:
    """Regression tests for forbid_* keeping parallel lists in sync.

    forbid_translation removed from internals['translations'] but left
    _active['translations'] untouched, desyncing the parallel lists (calc
    and jacobian then disagreed on shape). _forbid_internal removed from
    forbidden[name] (a no-op) instead of internals[name], so forbidding an
    already-added bond left it active while also marking it forbidden.
    """

    def test_forbid_translation_syncs_active(self):
        atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
        ic = Internals(atoms)
        ic.add_translation(0)
        ic.forbid_translation(0)
        assert len(ic.internals['translations']) == 0
        assert ic._active['translations'] == []
        assert ic.nint == 0
        # calc and jacobian must agree on the active-coordinate count.
        assert ic.calc().shape[0] == ic.jacobian().shape[0]

    def test_forbid_bond_removes_active_internal(self):
        atoms = Atoms('H3', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        ic = Internals(atoms)
        ic.add_bond((0, 1))
        ic.add_bond((1, 2))
        ic.forbid_bond((0, 1))
        assert len(ic.internals['bonds']) == 1
        assert ic._active['bonds'] == [True]
        assert ic.internals['bonds'][0] == Bond((1, 2))
        # Forbidding should also purge the dedup key and register the ban.
        assert Bond((0, 1)) in ic.forbidden['bonds']
        assert ic.calc().shape[0] == ic.jacobian().shape[0]

    def test_forbid_bond_reversed_index_order(self):
        # forbid is direction-agnostic: (1,0) must remove an added (0,1).
        atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
        ic = Internals(atoms)
        ic.add_bond((0, 1))
        ic.forbid_bond((1, 0))
        assert len(ic.internals['bonds']) == 0
        assert ic._active['bonds'] == []

    def test_forbid_then_add_raises(self):
        # A forbidden coordinate cannot subsequently be added.
        atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
        ic = Internals(atoms)
        ic.forbid_bond((0, 1))
        with pytest.raises(Exception):
            ic.add_bond((0, 1))


class TestDisplacementEquality:
    """Displacement.__eq__ raised KeyError('refpos') against other classes.

    Coordinate.__eq__ returns NotImplemented (truthy) for a different class,
    so the guard fell through to other.kwargs['refpos'], which doesn't exist
    on non-Displacement coordinates.
    """

    def test_displacement_vs_other_class(self):
        d = Displacement(np.array([0, 1]), np.zeros((2, 3)), np.eye(6))
        assert (d == Bond((0, 1))) is False

    def test_displacement_identity(self):
        d0 = Displacement(np.array([0, 1]), np.zeros((2, 3)), np.eye(6))
        d1 = Displacement(np.array([0, 1]), np.zeros((2, 3)), np.eye(6))
        assert d0 == d1
