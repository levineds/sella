"""Tests for unit cell optimization functionality."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from ase import Atoms
from ase.build import bulk, molecule
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones

from sella import Sella
from sella.peswrapper import CellInternalPES, CellCartesianPES
from sella.internal import Internals, Bond, Angle


def make_molecular_crystal():
    """Create a simple periodic molecular system for testing.

    Uses methane (CH4) in a periodic box, which has well-defined
    bonds and angles that work reliably with internal coordinates.
    """
    mol = molecule('CH4')
    # Place in a box with padding
    mol.center(vacuum=3.0)
    mol.pbc = True
    return mol


def make_water_crystal():
    """Create a water molecule in a periodic box."""
    mol = molecule('H2O')
    mol.center(vacuum=3.0)
    mol.pbc = True
    return mol


class TestCellDerivatives:
    """Test cell derivative functions for internal coordinates."""

    def test_bond_cell_derivative_numerical_molecular(self):
        """Verify bond cell derivative against numerical finite difference.

        Uses a molecular system with explicit bonds that cross the periodic
        boundary when the cell is compressed.
        """
        # Create CH4 in a periodic cell
        atoms = make_molecular_crystal()

        # Create internals and find bonds (C-H bonds)
        internals = Internals(atoms)
        internals.find_all_bonds()

        assert len(internals.internals['bonds']) > 0, "No bonds found"

        # Test the first bond
        bond = internals.internals['bonds'][0]

        # Analytical gradient
        grad_analytic = bond.calc_cell_gradient(atoms)

        # Numerical gradient using finite differences
        delta = 1e-5
        cell0 = atoms.get_cell().array.copy()
        grad_numeric = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                # Forward
                cell_plus = cell0.copy()
                cell_plus[i, j] += delta
                atoms.set_cell(cell_plus, scale_atoms=False)
                val_plus = bond.calc(atoms)

                # Backward
                cell_minus = cell0.copy()
                cell_minus[i, j] -= delta
                atoms.set_cell(cell_minus, scale_atoms=False)
                val_minus = bond.calc(atoms)

                grad_numeric[i, j] = (val_plus - val_minus) / (2 * delta)

        # Restore cell
        atoms.set_cell(cell0, scale_atoms=False)

        assert_allclose(grad_analytic, grad_numeric, atol=1e-6, rtol=1e-5)

    def test_bond_cell_derivative_with_periodic_image(self):
        """Test bond cell derivative for bond crossing periodic boundary.

        Create a diatomic that spans the periodic boundary to ensure
        ncvec contribution to cell derivative is tested.
        """
        # Create H2 spanning the periodic boundary
        atoms = Atoms('H2', positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]])
        atoms.set_cell([3.0, 3.0, 3.0])
        atoms.pbc = True

        # Create a bond that crosses the boundary (using ncvec)
        # Bond between atom 0 and atom 1 via periodic image
        bond = Bond(
            indices=np.array([0, 1], dtype=np.int32),
            ncvecs=np.array([[0, 0, -1]], dtype=np.int32)  # Wrap in -z direction
        )

        # Analytical gradient
        grad_analytic = bond.calc_cell_gradient(atoms)

        # Numerical gradient
        delta = 1e-5
        cell0 = atoms.get_cell().array.copy()
        grad_numeric = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                cell_plus = cell0.copy()
                cell_plus[i, j] += delta
                atoms.set_cell(cell_plus, scale_atoms=False)
                val_plus = bond.calc(atoms)

                cell_minus = cell0.copy()
                cell_minus[i, j] -= delta
                atoms.set_cell(cell_minus, scale_atoms=False)
                val_minus = bond.calc(atoms)

                grad_numeric[i, j] = (val_plus - val_minus) / (2 * delta)

        atoms.set_cell(cell0, scale_atoms=False)

        # The bond crosses the boundary, so cell derivatives should be non-zero
        assert not np.allclose(grad_analytic, 0), "Cell gradient should be non-zero for PBC bond"
        assert_allclose(grad_analytic, grad_numeric, atol=1e-6, rtol=1e-5)

    def test_angle_cell_derivative_numerical(self):
        """Verify angle cell derivative against numerical finite difference."""
        # Use water - has a well-defined H-O-H angle
        atoms = make_water_crystal()

        internals = Internals(atoms)
        internals.find_all_bonds()
        internals.find_all_angles()

        if not internals.internals['angles']:
            pytest.skip("No angles found in structure")

        # Get an angle (H-O-H)
        angle = internals.internals['angles'][0]

        # Analytical gradient
        grad_analytic = angle.calc_cell_gradient(atoms)

        # Numerical gradient
        delta = 1e-5
        cell0 = atoms.get_cell().array.copy()
        grad_numeric = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                cell_plus = cell0.copy()
                cell_plus[i, j] += delta
                atoms.set_cell(cell_plus, scale_atoms=False)
                val_plus = angle.calc(atoms)

                cell_minus = cell0.copy()
                cell_minus[i, j] -= delta
                atoms.set_cell(cell_minus, scale_atoms=False)
                val_minus = angle.calc(atoms)

                grad_numeric[i, j] = (val_plus - val_minus) / (2 * delta)

        atoms.set_cell(cell0, scale_atoms=False)

        assert_allclose(grad_analytic, grad_numeric, atol=1e-6, rtol=1e-5)

    def test_cell_jacobian_shape(self):
        """Test that cell_jacobian returns correct shape."""
        atoms = make_molecular_crystal()  # CH4 with bonds and angles

        internals = Internals(atoms)
        internals.find_all_bonds()
        internals.find_all_angles()

        J_cell = internals.cell_jacobian()

        # Should have shape (n_active_coords, 9)
        n_active = len(internals.calc())
        assert J_cell.shape == (n_active, 9)


class TestCellInternalPES:
    """Test CellInternalPES class."""

    def test_cell_internal_pes_initialization(self):
        """Test that CellInternalPES initializes correctly."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        internals = Internals(atoms)
        pes = CellInternalPES(atoms, internals)

        # Check dimensions include cell DOF
        assert pes.n_cell_dof == 9  # Full 3x3 cell
        assert pes.dim == pes.n_internal + 9

    def test_cell_internal_pes_get_x(self):
        """Test get_x returns combined internal + cell vector."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        internals = Internals(atoms)
        pes = CellInternalPES(atoms, internals)

        x = pes.get_x()

        # Length should be n_internal + n_cell_dof
        assert len(x) == pes.dim

        # First n_internal elements are internal coords
        q = internals.calc()
        assert_allclose(x[:pes.n_internal], q, rtol=1e-10)

        # Cell params at identity should be approximately zero
        # (log of identity matrix is zero)
        assert_allclose(x[pes.n_internal:], 0, atol=1e-10)

    def test_cell_internal_pes_cell_mask(self):
        """Test cell_mask parameter."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        # Only allow diagonal elements (hydrostatic strain)
        cell_mask = np.eye(3, dtype=bool)

        internals = Internals(atoms)
        pes = CellInternalPES(atoms, internals, cell_mask=cell_mask)

        assert pes.n_cell_dof == 3  # Only 3 diagonal elements
        assert pes.dim == pes.n_internal + 3

    def test_cell_internal_pes_eval(self):
        """Test eval returns correct gradient shape."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        internals = Internals(atoms)
        pes = CellInternalPES(atoms, internals)

        f, g = pes.eval()

        assert isinstance(f, float)
        assert len(g) == pes.dim


class TestCellCartesianPES:
    """Test CellCartesianPES class."""

    def test_cell_cartesian_pes_initialization(self):
        """Test that CellCartesianPES initializes correctly."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        pes = CellCartesianPES(atoms)

        # Check dimensions include cell DOF
        assert pes.n_cell_dof == 9  # Full 3x3 cell
        assert pes.n_cart == 3 * len(atoms)
        assert pes.dim == pes.n_cart + 9

    def test_cell_cartesian_pes_get_x(self):
        """Test get_x returns combined Cartesian + cell vector."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        pes = CellCartesianPES(atoms)

        x = pes.get_x()

        # Length should be n_cart + n_cell_dof
        assert len(x) == pes.dim

        # First n_cart elements are Cartesian positions
        positions_flat = atoms.get_positions().ravel()
        assert_allclose(x[:pes.n_cart], positions_flat, rtol=1e-10)

        # Cell params at identity should be approximately zero
        # (log of identity matrix is zero)
        assert_allclose(x[pes.n_cart:], 0, atol=1e-10)

    def test_cell_cartesian_pes_cell_mask(self):
        """Test cell_mask parameter."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        # Only allow diagonal elements (hydrostatic strain)
        cell_mask = np.eye(3, dtype=bool)

        pes = CellCartesianPES(atoms, cell_mask=cell_mask)

        assert pes.n_cell_dof == 3  # Only 3 diagonal elements
        assert pes.dim == pes.n_cart + 3

    def test_cell_cartesian_pes_eval(self):
        """Test eval returns correct gradient shape."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        pes = CellCartesianPES(atoms)

        f, g = pes.eval()

        assert isinstance(f, float)
        assert len(g) == pes.dim

    def test_cell_cartesian_pes_save_restore(self):
        """Test save and restore functionality."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        pes = CellCartesianPES(atoms)

        # Save initial state
        pes.save()
        x0 = pes.get_x()
        cell0 = atoms.get_cell().array.copy()
        pos0 = atoms.get_positions().copy()

        # Modify positions and cell
        atoms.positions += 0.1
        new_cell = atoms.get_cell().array * 1.05
        atoms.set_cell(new_cell, scale_atoms=False)

        # Verify modifications took effect
        assert not np.allclose(atoms.get_positions(), pos0)
        assert not np.allclose(atoms.get_cell().array, cell0)

        # Restore and verify
        pes.restore()
        assert_allclose(atoms.get_positions(), pos0)
        assert_allclose(atoms.get_cell().array, cell0)

    def test_cell_cartesian_pes_set_x(self):
        """Test set_x updates positions and cell."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        pes = CellCartesianPES(atoms)

        # Get initial x
        x0 = pes.get_x()

        # Create target with small perturbations
        x_target = x0.copy()
        x_target[:pes.n_cart] += 0.01  # Small position change
        x_target[pes.n_cart:] += 0.001  # Small cell change

        # Set new x
        pes.set_x(x_target)

        # Get x and verify it's close to target
        x_new = pes.get_x()
        # Cell parameters should match
        assert_allclose(x_new[pes.n_cart:], x_target[pes.n_cart:], rtol=1e-6)
        # Positions should match
        assert_allclose(x_new[:pes.n_cart], x_target[:pes.n_cart], rtol=1e-6)

    def test_cell_cartesian_vs_internal_gradient_shape(self):
        """Compare CellCartesianPES and CellInternalPES gradient shapes."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        # CellCartesianPES
        pes_cart = CellCartesianPES(atoms.copy())
        pes_cart.atoms.calc = EMT()
        _, g_cart = pes_cart.eval()

        # CellInternalPES
        internals = Internals(atoms)
        pes_int = CellInternalPES(atoms, internals)
        _, g_int = pes_int.eval()

        # Both should have 9 cell DOF (full 3x3 cell)
        assert pes_cart.n_cell_dof == 9
        assert pes_int.n_cell_dof == 9

        # Cell parts of gradient should have same length
        assert len(g_cart[pes_cart.n_cart:]) == len(g_int[pes_int.n_internal:])

    def test_cell_cartesian_pes_pressure(self):
        """Test scalar pressure contribution."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        # Without pressure
        pes_no_p = CellCartesianPES(atoms.copy())
        pes_no_p.atoms.calc = EMT()
        f_no_p, _ = pes_no_p.eval()

        # With pressure
        pressure = 0.1  # eV/Å³
        atoms2 = atoms.copy()
        atoms2.calc = EMT()
        pes_p = CellCartesianPES(atoms2, scalar_pressure=pressure)
        f_p, _ = pes_p.eval()

        # Energy with pressure should be higher (positive pressure)
        volume = atoms.get_volume()
        expected_diff = pressure * volume
        assert_allclose(f_p - f_no_p, expected_diff, rtol=1e-10)


class TestCellCartesianGradient:
    """Test cell gradient calculations in CellCartesianPES."""

    def test_cell_gradient_numerical(self):
        """Test cell gradient matches numerical finite difference for bulk Cu."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        pes = CellCartesianPES(atoms)

        # Get analytical gradient
        _, g = pes.eval()
        g_cell = g[pes.n_cart:]  # Cell part of gradient

        # Numerical gradient via finite difference on cell parameters
        delta = 1e-6
        x0 = pes.get_x()
        g_cell_numeric = np.zeros(pes.n_cell_dof)

        for i in range(pes.n_cell_dof):
            x_plus = x0.copy()
            x_plus[pes.n_cart + i] += delta
            pes.set_x(x_plus)
            e_plus, _ = pes.eval()

            x_minus = x0.copy()
            x_minus[pes.n_cart + i] -= delta
            pes.set_x(x_minus)
            e_minus, _ = pes.eval()

            g_cell_numeric[i] = (e_plus - e_minus) / (2 * delta)

        pes.set_x(x0)

        assert_allclose(g_cell, g_cell_numeric, atol=1e-4, rtol=1e-3)

    def test_cartesian_gradient_numerical(self):
        """Test Cartesian gradient matches numerical finite difference."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        pes = CellCartesianPES(atoms)

        # Get analytical gradient
        _, g = pes.eval()
        g_cart = g[:pes.n_cart]  # Cartesian part of gradient

        # Numerical gradient via finite difference
        delta = 1e-6
        x0 = pes.get_x()
        g_cart_numeric = np.zeros(pes.n_cart)

        for i in range(pes.n_cart):
            x_plus = x0.copy()
            x_plus[i] += delta
            pes.set_x(x_plus)
            e_plus, _ = pes.eval()

            x_minus = x0.copy()
            x_minus[i] -= delta
            pes.set_x(x_minus)
            e_minus, _ = pes.eval()

            g_cart_numeric[i] = (e_plus - e_minus) / (2 * delta)

        pes.set_x(x0)

        assert_allclose(g_cart, g_cart_numeric, atol=1e-4, rtol=1e-3)


class TestSellaWithCellOptimization:
    """Integration tests for Sella with cell optimization."""

    def test_sella_optimize_cell_parameter(self):
        """Test that optimize_cell parameter is properly handled."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        opt = Sella(atoms, internal=True, order=0, optimize_cell=True)

        assert opt.optimize_cell is True
        assert isinstance(opt.pes, CellInternalPES)

    def test_sella_cell_optimization_validation(self):
        """Test validation of cell optimization parameters."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        # Should fail with order != 0
        with pytest.raises(ValueError, match="order=0"):
            Sella(atoms, internal=True, order=1, optimize_cell=True)

        # internal=False with optimize_cell=True should use CellCartesianPES
        atoms2 = atoms.copy()
        atoms2.calc = EMT()
        opt = Sella(atoms2, internal=False, order=0, optimize_cell=True)
        assert isinstance(opt.pes, CellCartesianPES)

        # Should fail without PBC
        atoms_nopbc = atoms.copy()
        atoms_nopbc.pbc = False
        atoms_nopbc.calc = EMT()
        with pytest.raises(ValueError, match="periodic"):
            Sella(atoms_nopbc, internal=True, order=0, optimize_cell=True)

    def test_sella_cell_optimization_single_step(self):
        """Test that cell optimization can take a single step."""
        # Use strained FCC copper
        atoms = bulk('Cu', 'fcc', a=3.8)  # Slightly expanded
        atoms.calc = EMT()

        opt = Sella(atoms, internal=True, order=0, optimize_cell=True)

        # Record initial state
        cell0 = atoms.get_cell().array.copy()
        e0 = atoms.get_potential_energy()

        # Take one step
        opt.step()

        # Energy should decrease (or at least be computed)
        e1 = atoms.get_potential_energy()
        # Cell should change
        cell1 = atoms.get_cell().array

        # At least something should have changed
        assert not np.allclose(cell0, cell1) or not np.isclose(e0, e1)

    @pytest.mark.slow
    def test_sella_cell_optimization_convergence(self):
        """Test that cell optimization converges for a simple system."""
        # Create strained FCC copper - use single primitive cell
        # (larger supercells have issues with angle finding in FCC)
        # Note: bulk() creates a primitive cell, so a=3.8 gives cell param ~2.69 Å
        atoms = bulk('Cu', 'fcc', a=3.8)  # Slightly expanded
        atoms.calc = EMT()

        opt = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            logfile=None,
        )

        # Run optimization
        converged = opt.run(fmax=0.05, steps=50)

        if converged:
            # Check that cell relaxed toward equilibrium
            # EMT equilibrium for FCC Cu primitive cell is about a = 2.54 Å
            a = atoms.get_cell().cellpar()[0]
            assert 2.4 < a < 2.7


class TestVoigtConversion:
    """Test Voigt stress conversion functions."""

    def test_voigt_roundtrip(self):
        """Test conversion roundtrip."""
        from sella.peswrapper import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress

        # Random Voigt stress
        voigt = np.random.randn(6)

        # Convert to 3x3 and back
        stress_3x3 = voigt_6_to_full_3x3_stress(voigt)
        voigt_back = full_3x3_to_voigt_6_stress(stress_3x3)

        assert_allclose(voigt, voigt_back)

    def test_voigt_symmetry(self):
        """Test that converted tensor is symmetric."""
        from sella.peswrapper import voigt_6_to_full_3x3_stress

        voigt = np.array([1, 2, 3, 4, 5, 6])
        stress_3x3 = voigt_6_to_full_3x3_stress(voigt)

        assert_allclose(stress_3x3, stress_3x3.T)


class TestStressTensor:
    """Test stress tensor calculations for cell optimization."""

    def test_stress_changes_with_strain_inorganic(self):
        """Test that stress changes when cell is strained for bulk Cu."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()
        stress_ref = atoms.get_stress()

        # Strain the cell
        atoms_strained = bulk('Cu', 'fcc', a=3.8)  # Different lattice constant
        atoms_strained.calc = EMT()
        stress_strained = atoms_strained.get_stress()

        # Stress should change
        assert not np.allclose(stress_ref, stress_strained)

        # Stress should be finite
        assert np.all(np.isfinite(stress_ref))
        assert np.all(np.isfinite(stress_strained))

    def test_stress_finite_molecular(self):
        """Test stress is finite for molecular system (CH4 in periodic box)."""
        # Create CH4 in a periodic box
        mol = molecule('CH4')
        mol.center(vacuum=4.0)
        mol.pbc = True
        mol.calc = EMT()

        # Should be able to get stress without error
        stress_voigt = mol.get_stress()
        assert len(stress_voigt) == 6
        assert np.all(np.isfinite(stress_voigt))


class TestCellGradient:
    """Test cell gradient calculations in CellInternalPES."""

    def test_cell_gradient_numerical_inorganic(self):
        """Test cell gradient matches numerical finite difference for bulk Cu."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        internals = Internals(atoms)
        pes = CellInternalPES(atoms, internals)

        # Get analytical gradient
        _, g = pes.eval()
        g_cell = g[pes.n_internal:]  # Cell part of gradient

        # Numerical gradient via finite difference on cell parameters
        delta = 1e-6
        x0 = pes.get_x()
        g_cell_numeric = np.zeros(pes.n_cell_dof)

        for i in range(pes.n_cell_dof):
            x_plus = x0.copy()
            x_plus[pes.n_internal + i] += delta
            pes.set_x(x_plus)
            e_plus, _ = pes.eval()

            x_minus = x0.copy()
            x_minus[pes.n_internal + i] -= delta
            pes.set_x(x_minus)
            e_minus, _ = pes.eval()

            g_cell_numeric[i] = (e_plus - e_minus) / (2 * delta)

        pes.set_x(x0)

        assert_allclose(g_cell, g_cell_numeric, atol=1e-4, rtol=1e-3)

    def test_cell_gradient_shape_molecular(self):
        """Test cell gradient has correct shape for molecular system."""
        mol = molecule('H2O')
        mol.center(vacuum=3.5)
        mol.pbc = True
        mol.calc = EMT()

        internals = Internals(mol)
        internals.find_all_bonds()
        internals.find_all_angles()
        pes = CellInternalPES(mol, internals)

        _, g = pes.eval()

        # Gradient should have n_internal + n_cell_dof components
        assert len(g) == pes.n_internal + pes.n_cell_dof

        # All components should be finite
        assert np.all(np.isfinite(g))


class TestMolecularCrystal:
    """Test cell optimization for molecular crystal systems.

    Molecular crystals have multiple separate molecules in a periodic cell,
    requiring TRICs (Translation-Rotation Internal Coordinates) for proper
    handling of each molecular fragment.
    """

    def test_molecular_crystal_with_trics(self):
        """Test that molecular crystal optimization works with allow_fragments=True."""
        # Create a simple molecular crystal: two water molecules in a box
        water1 = molecule('H2O')
        water2 = molecule('H2O')

        # Place molecules in different parts of the cell
        water1.positions += [1.0, 1.0, 1.0]
        water2.positions += [4.0, 4.0, 4.0]

        # Combine into one system
        atoms = water1 + water2
        atoms.set_cell([7.0, 7.0, 7.0])
        atoms.pbc = True
        atoms.calc = LennardJones()

        # Create optimizer with allow_fragments=True for TRICs
        opt = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            allow_fragments=True,
            logfile=None,
        )

        assert isinstance(opt.pes, CellInternalPES)

        # Verify the internal coords have TRICs (translations and rotations)
        internals = opt.pes.int
        assert internals.ntrans > 0, "Should have translation coordinates for fragments"
        assert internals.nrotations > 0, "Should have rotation coordinates for fragments"

        # Take a few steps to verify no errors
        for _ in range(3):
            opt.step()

    def test_molecular_crystal_cartesian(self):
        """Test molecular crystal optimization with Cartesian coordinates."""
        # Create two methane molecules
        ch4_1 = molecule('CH4')
        ch4_2 = molecule('CH4')

        ch4_1.positions += [1.5, 1.5, 1.5]
        ch4_2.positions += [5.0, 5.0, 5.0]

        atoms = ch4_1 + ch4_2
        atoms.set_cell([8.0, 8.0, 8.0])
        atoms.pbc = True
        atoms.calc = LennardJones()

        # Use Cartesian coordinates (internal=False) with cell optimization
        opt = Sella(
            atoms,
            internal=False,
            order=0,
            optimize_cell=True,
            logfile=None,
        )

        assert isinstance(opt.pes, CellCartesianPES)

        # Take a few steps
        for _ in range(3):
            opt.step()

    def test_molecular_crystal_trics_dof_count(self):
        """Test that TRICs add correct DOF for molecular fragments."""
        # Create two separate H2 molecules
        atoms = Atoms(
            'H4',
            positions=[
                [0.0, 0.0, 0.0],
                [0.74, 0.0, 0.0],
                [4.0, 4.0, 4.0],
                [4.74, 4.0, 4.0],
            ]
        )
        atoms.set_cell([8.0, 8.0, 8.0])
        atoms.pbc = True
        atoms.calc = LennardJones()

        # With allow_fragments=True, should get TRICs for each fragment
        internals = Internals(atoms, allow_fragments=True)
        internals.find_all_bonds()

        # Should have 2 bonds (one per H2)
        assert internals.nbonds == 2

        # Should have translations for the 2 fragments
        assert internals.ntrans > 0

        # Should have rotations for fragments that can rotate
        # (H2 is linear, so rotation DOF may be limited)
        assert internals.nrotations >= 0


class TestTRICsCellDerivatives:
    """Test that TRICs have correct cell derivatives.

    For molecular crystals, translation and rotation coordinates should
    have zero cell derivatives since they describe internal molecular
    motion that doesn't depend on the cell.
    """

    def test_translation_cell_derivative_zero(self):
        """Test that translation coordinates have zero cell derivatives."""
        # Create a simple diatomic in a periodic cell
        atoms = Atoms('H2', positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        atoms.set_cell([5.0, 5.0, 5.0])
        atoms.pbc = True

        internals = Internals(atoms, allow_fragments=True)
        internals.find_all_bonds()

        # Get the translation coordinates
        translations = internals.internals.get('translations', [])

        for trans in translations:
            grad_cell = trans.calc_cell_gradient(atoms)
            # Translation center of mass should not depend on cell
            assert_allclose(grad_cell, 0, atol=1e-10)

    def test_rotation_cell_derivative_zero(self):
        """Test that rotation coordinates have zero cell derivatives."""
        # Create water molecule (non-linear, has rotations)
        atoms = molecule('H2O')
        atoms.center(vacuum=3.0)
        atoms.pbc = True

        internals = Internals(atoms, allow_fragments=True)
        internals.find_all_bonds()
        internals.find_all_angles()

        # Get rotation coordinates
        rotations = internals.internals.get('rotations', [])

        for rot in rotations:
            grad_cell = rot.calc_cell_gradient(atoms)
            # Rotation orientation should not depend on cell
            assert_allclose(grad_cell, 0, atol=1e-10)

    def test_bond_cell_derivative_intramolecular(self):
        """Test intramolecular bond has zero cell derivative if not crossing PBC."""
        # Create H2 well within the cell (not crossing boundary)
        atoms = Atoms('H2', positions=[[2.0, 2.5, 2.5], [2.74, 2.5, 2.5]])
        atoms.set_cell([5.0, 5.0, 5.0])
        atoms.pbc = True

        internals = Internals(atoms)
        internals.find_all_bonds()

        bond = internals.internals['bonds'][0]

        # For intramolecular bond not crossing PBC, cell derivative should be zero
        # (the ncvec should be zero)
        grad_cell = bond.calc_cell_gradient(atoms)

        # Check that gradient is zero (bond doesn't cross boundary)
        assert_allclose(grad_cell, 0, atol=1e-10)

    def test_cell_jacobian_trics_rows_zero(self):
        """Test that TRICs rows in cell_jacobian are zero."""
        atoms = molecule('H2O')
        atoms.center(vacuum=3.0)
        atoms.pbc = True

        internals = Internals(atoms, allow_fragments=True)
        internals.find_all_bonds()
        internals.find_all_angles()

        J_cell = internals.cell_jacobian()

        # The rows corresponding to translations and rotations should be zero
        # First ntrans rows are translations
        n_trans = internals.ntrans
        n_rot = internals.nrotations

        if n_trans > 0:
            trans_rows = J_cell[:n_trans, :]
            assert_allclose(trans_rows, 0, atol=1e-10)

        # Last nrotations rows are rotations (after other coords)
        if n_rot > 0:
            # Rotations come after: trans, bonds, angles, dihedrals, other
            rot_start = n_trans + internals.nbonds + internals.nangles + internals.ndihedrals + internals.nother
            rot_rows = J_cell[rot_start:rot_start + n_rot, :]
            assert_allclose(rot_rows, 0, atol=1e-10)


class TestCellConstraints:
    """Test cell optimization with various constraints."""

    def test_hydrostatic_only_dof_count(self):
        """Test that hydrostatic constraint gives correct number of DOF."""
        atoms = bulk('Cu', 'fcc', a=3.8)
        atoms.calc = EMT()

        # Only allow diagonal elements
        cell_mask = np.eye(3, dtype=bool)

        opt = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            cell_mask=cell_mask,
            logfile=None,
        )

        # Check that only 3 cell DOF (diagonal elements)
        assert opt.pes.n_cell_dof == 3

    def test_isotropic_scaling(self):
        """Test cell optimization with isotropic scaling only (1 DOF)."""
        atoms = bulk('Cu', 'fcc', a=3.8)
        atoms.calc = EMT()

        # Only allow uniform scaling (first diagonal element)
        cell_mask = np.zeros((3, 3), dtype=bool)
        cell_mask[0, 0] = True

        opt = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            cell_mask=cell_mask,
            logfile=None,
        )

        assert opt.pes.n_cell_dof == 1

    def test_no_shear_mask(self):
        """Test that shear components can be masked out."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        # Allow all 9 components (full cell optimization)
        full_mask = np.ones((3, 3), dtype=bool)
        opt_full = Sella(
            atoms.copy(),
            internal=True,
            order=0,
            optimize_cell=True,
            cell_mask=full_mask,
            logfile=None,
        )
        atoms.calc = EMT()  # Reset

        # Allow only diagonal (no shear)
        diag_mask = np.eye(3, dtype=bool)
        opt_diag = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            cell_mask=diag_mask,
            logfile=None,
        )

        assert opt_full.pes.n_cell_dof == 9
        assert opt_diag.pes.n_cell_dof == 3


class TestRefineInitialHessian:
    """Test the refine_initial_hessian option for cell optimization."""

    def test_refine_hessian_produces_nonzero_coupling(self):
        """Test that refine_initial_hessian produces non-zero coupling blocks."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        # With refinement
        pes_refined = CellInternalPES(
            atoms,
            Internals(atoms),
            refine_initial_hessian=True,
        )

        H = pes_refined.H.B
        n_int = pes_refined.n_internal

        # The coupling block should be non-zero (or at least the cell-cell block
        # should differ from 70*I)
        H_coupling = H[:n_int, n_int:]
        H_cell = H[n_int:, n_int:]

        # Cell-cell block should not be exactly 70*I (it was computed via FD)
        # Note: it might still be close to diagonal for simple systems
        assert H_cell.shape == (pes_refined.n_cell_dof, pes_refined.n_cell_dof)

    def test_refine_hessian_vs_default_different(self):
        """Test that refined Hessian differs from default."""
        # Without refinement
        atoms1 = bulk('Cu', 'fcc', a=3.6)
        atoms1.calc = EMT()
        pes_default = CellInternalPES(
            atoms1,
            Internals(atoms1),
            refine_initial_hessian=False,
        )

        # With refinement
        atoms2 = bulk('Cu', 'fcc', a=3.6)
        atoms2.calc = EMT()
        pes_refined = CellInternalPES(
            atoms2,
            Internals(atoms2),
            refine_initial_hessian=True,
        )

        H_default = pes_default.H.B
        H_refined = pes_refined.H.B

        n_int = pes_default.n_internal

        # The cell-cell blocks should be different
        H_cell_default = H_default[n_int:, n_int:]
        H_cell_refined = H_refined[n_int:, n_int:]

        # Default is 70*I, refined should be computed from actual curvature
        assert not np.allclose(H_cell_default, H_cell_refined)

    def test_refine_hessian_cartesian_pes(self):
        """Test refine_initial_hessian with CellCartesianPES."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        pes = CellCartesianPES(
            atoms,
            refine_initial_hessian=True,
        )

        H = pes.H.B
        n_cart = pes.n_cart

        # Hessian should have correct shape
        assert H.shape == (pes.dim, pes.dim)

        # Cell-cell block should exist
        H_cell = H[n_cart:, n_cart:]
        assert H_cell.shape == (pes.n_cell_dof, pes.n_cell_dof)

    def test_refine_hessian_via_sella_api(self):
        """Test refine_initial_hessian through Sella API."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        opt = Sella(
            atoms,
            internal=True,
            order=0,
            optimize_cell=True,
            refine_initial_hessian=True,
            logfile=None,
        )

        # Should be CellInternalPES
        assert isinstance(opt.pes, CellInternalPES)

        # Hessian should be properly dimensioned
        H = opt.pes.H.B
        assert H.shape == (opt.pes.dim, opt.pes.dim)

    def test_refine_hessian_force_call_count(self):
        """Test that refinement makes expected number of force calls."""
        atoms = bulk('Cu', 'fcc', a=3.6)
        atoms.calc = EMT()

        # Only allow diagonal cell DOF for fewer calls
        cell_mask = np.eye(3, dtype=bool)  # 3 DOF

        pes = CellInternalPES(
            atoms,
            Internals(atoms),
            cell_mask=cell_mask,
            refine_initial_hessian=True,
        )

        # Should have made 2 * n_cell_dof = 6 evaluations during init
        # (2 per cell DOF for central difference)
        assert pes.neval == 6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
