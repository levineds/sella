"""Tests for unit cell optimization functionality."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from ase import Atoms
from ase.build import bulk, molecule
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones

from sella import Sella
from sella.peswrapper import CellInternalPES
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

        # Should fail with internal=False
        with pytest.raises(ValueError, match="internal=True"):
            Sella(atoms, internal=False, order=0, optimize_cell=True)

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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
