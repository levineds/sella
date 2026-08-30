import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from ase import Atoms
from scipy import sparse
import torch

import sella.internal as internal_module
from sella.internal import Internals


def _mixed_internals():
    atoms = Atoms(
        'H5',
        positions=[
            [0.1, 0.2, 0.3],
            [1.0, 0.2, 0.3],
            [1.2, 1.1, 0.4],
            [2.0, 1.3, 1.2],
            [2.8, 1.8, 1.0],
        ],
        cell=[[4.0, 0.2, 0.1], [0.1, 4.3, 0.2], [0.2, 0.1, 4.2]],
        pbc=True,
    )
    internals = Internals(atoms)
    internals.add_bond((0, 1), ncvecs=[[1, 0, 0]])
    internals.add_bond((3, 4))
    internals.add_angle(
        (0, 1, 2), ncvecs=[[1, 0, 0], [0, 0, 0]]
    )
    internals.add_angle((1, 2, 3))
    internals.add_dihedral(
        (0, 1, 2, 3),
        ncvecs=[[1, 0, 0], [0, 0, 0], [0, 0, 0]],
    )
    return internals


def _coordinate_rows(internals):
    return [
        coord
        for key in ('bonds', 'angles', 'dihedrals')
        for coord in internals.internals[key]
    ]


def test_consolidated_batches_match_eager_coordinate_math():
    internals = _mixed_internals()
    atoms = internals.atoms
    coords = _coordinate_rows(internals)
    natoms = len(atoms)

    expected_values = np.array([coord.calc(atoms) for coord in coords])
    expected_jacobian = np.zeros((len(coords), 3 * natoms))
    expected_hessians = np.zeros((len(coords), 3 * natoms, 3 * natoms))
    expected_cell = np.array([
        coord.calc_cell_gradient(atoms) for coord in coords
    ]).reshape((len(coords), 9))

    for row, coord in enumerate(coords):
        expected_jacobian[row].reshape((natoms, 3))[coord.indices] = (
            coord.calc_gradient(atoms)
        )
        local_hessian = coord.calc_hessian(atoms)
        for i, atom_i in enumerate(coord.indices):
            for j, atom_j in enumerate(coord.indices):
                expected_hessians[
                    row,
                    3 * atom_i:3 * atom_i + 3,
                    3 * atom_j:3 * atom_j + 3,
                ] = local_hessian[i, :, j, :]

    np.testing.assert_allclose(internals.calc(), expected_values, atol=1e-12)
    np.testing.assert_allclose(
        internals.jacobian(), expected_jacobian, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(internals.hessian()), expected_hessians, atol=1e-12
    )
    np.testing.assert_allclose(
        internals.cell_jacobian(), expected_cell, atol=1e-12
    )

    tangent = np.linspace(-0.4, 0.7, internals.ndof)
    expected_hvp = np.einsum('aij,j->ai', expected_hessians, tangent)
    actual_hvp = internals.hessian_rdot(tangent)
    if sparse.issparse(actual_hvp):
        actual_hvp = actual_hvp.toarray()
    np.testing.assert_allclose(actual_hvp, expected_hvp, atol=1e-12)


def test_consolidated_entry_points_accept_zero_batches():
    empty_pos = [
        torch.empty((0, n_atoms, 3), dtype=torch.float64)
        for n_atoms in (2, 3, 4)
    ]
    empty_tvec = [
        torch.empty((0, n_tvecs, 3), dtype=torch.float64)
        for n_tvecs in (1, 2, 3)
    ]
    value_args = [
        value
        for pair in zip(empty_pos, empty_tvec)
        for value in pair
    ]
    hvp_args = [
        value
        for pos, tvec in zip(empty_pos, empty_tvec)
        for value in (pos, tvec, torch.empty_like(pos))
    ]
    cell = torch.eye(3, dtype=torch.float64)

    values = internal_module._batched_values(*value_args)
    gradients = internal_module._batched_gradients(*value_args)
    hvps = internal_module._batched_hvps(*hvp_args)
    cell_gradients = internal_module._batched_cell_gradients(
        *value_args, cell
    )

    assert [tuple(value.shape) for value in values] == [(0,), (0,), (0,)]
    assert [tuple(value.shape) for value in gradients] == [
        (0, 2, 3), (0, 3, 3), (0, 4, 3)
    ]
    assert [tuple(value.shape) for value in hvps] == [
        (0, 2, 3), (0, 3, 3), (0, 4, 3)
    ]
    assert [tuple(value.shape) for value in cell_gradients] == [
        (0, 3, 3), (0, 3, 3), (0, 3, 3)
    ]


def test_single_periodic_family_handles_other_empty_families():
    atoms = Atoms(
        'H3',
        positions=[[0.1, 0.2, 0.3], [1.0, 0.2, 0.3], [1.2, 1.1, 0.4]],
        cell=[[4.0, 0.2, 0.1], [0.1, 4.3, 0.2], [0.2, 0.1, 4.2]],
        pbc=True,
    )
    internals = Internals(atoms)
    internals.add_angle(
        (0, 1, 2), ncvecs=[[1, 0, 0], [0, 0, 0]]
    )
    angle = internals.internals['angles'][0]

    np.testing.assert_allclose(internals.calc(), [angle.calc(atoms)])

    expected_jacobian = np.zeros((1, internals.ndof))
    expected_jacobian.reshape((1, len(atoms), 3))[0, angle.indices] = (
        angle.calc_gradient(atoms)
    )
    np.testing.assert_allclose(internals.jacobian(), expected_jacobian)
    np.testing.assert_allclose(
        internals.cell_jacobian(),
        angle.calc_cell_gradient(atoms).reshape((1, 9)),
    )

    tangent = np.linspace(-0.3, 0.5, internals.ndof)
    expected_hvp = np.asarray(internals.hessian()).reshape(
        (1, internals.ndof, internals.ndof)
    )[0] @ tangent
    actual_hvp = internals.hessian_rdot(tangent)
    if sparse.issparse(actual_hvp):
        actual_hvp = actual_hvp.toarray()
    np.testing.assert_allclose(actual_hvp, expected_hvp[None, :], atol=1e-12)


@pytest.mark.parametrize('angle_active', [[False, False], [True, False]])
def test_consolidated_hvp_respects_inactive_masks(angle_active):
    internals = _mixed_internals()
    internals._active['angles'][:] = angle_active
    tangent = np.linspace(-0.4, 0.7, internals.ndof)
    matrix = np.arange(internals.ndof * 2, dtype=float).reshape(
        (internals.ndof, 2)
    ) / 13.0

    active_hessians = np.asarray(internals.hessian())
    expected = np.einsum('aij,j->ai', active_hessians, tangent)
    actual = internals.hessian_rdot(tangent)
    if sparse.issparse(actual):
        actual = actual.toarray()

    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(
        internals.hessian_rdot_mat(tangent, matrix),
        expected @ matrix,
        atol=1e-12,
    )


def test_bad_angle_check_uses_consolidated_value_batch():
    atoms = Atoms(
        'H3',
        positions=[[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )
    internals = Internals(atoms)
    internals.add_angle((0, 1, 2))

    assert internals._bad_angles() == internals.internals['angles']

    atoms.positions[2] = [0.0, 1.0, 0.0]
    assert internals._bad_angles() == []


def test_aot_compilation_caches_four_coordinate_regions(tmp_path):
    """The four consolidated roots each persist exactly one AOT artifact.

    Compilation is lazy (per input signature, on first call), so this drives
    each root once with a zero-batch signature and checks the four expected
    cache-name prefixes appear in an isolated cache directory.
    """
    root = Path(__file__).resolve().parents[2]
    code = """
import os

os.environ['SELLA_TORCH_COMPILE'] = 'all'
import torch
import sella.internal as m

empty_pos = [torch.empty((0, n, 3), dtype=torch.float64) for n in (2, 3, 4)]
empty_tvec = [torch.empty((0, n, 3), dtype=torch.float64) for n in (1, 2, 3)]
value_args = [v for pair in zip(empty_pos, empty_tvec) for v in pair]
hvp_args = [v for pos, tvec in zip(empty_pos, empty_tvec)
            for v in (pos, tvec, torch.empty_like(pos))]
cell = torch.eye(3, dtype=torch.float64)
m._batched_values(*value_args)
m._batched_gradients(*value_args)
m._batched_hvps(*hvp_args)
m._batched_cell_gradients(*value_args, cell)
"""
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        [str(root), env.get('PYTHONPATH', '')]
    ).rstrip(os.pathsep)
    env['SELLA_TORCH_AOT_CACHE_DIR'] = str(tmp_path)
    subprocess.run(
        [sys.executable, '-c', code],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    prefixes = sorted(
        {path.name.rsplit('-', 1)[0] for path in tmp_path.glob('*.pt')}
    )
    assert prefixes == [
        'batched-cell-gradients',
        'batched-gradients',
        'batched-hvps',
        'batched-values',
    ]


def test_aot_failure_falls_back_to_torch_compile(tmp_path, monkeypatch):
    monkeypatch.setattr(
        internal_module, '_TORCH_AOT_CACHE_DIR', str(tmp_path)
    )
    monkeypatch.setattr(internal_module, '_AOT_DISABLED_REASON', None)
    monkeypatch.setattr(internal_module, '_AOT_WARNED', False)
    compile_calls = []

    def fake_compile(func, **kwargs):
        compile_calls.append(kwargs)
        return func

    monkeypatch.setattr(torch, 'compile', fake_compile)
    wrapped = internal_module._TorchAOTFunction(
        lambda value: value + 1.0,
        enabled=True,
        cache_name='fallback-test',
    )

    def fail_aot(*args, **kwargs):
        raise RuntimeError('unsupported AOT artifact')

    monkeypatch.setattr(wrapped, '_compile_and_save', fail_aot)
    with pytest.warns(RuntimeWarning, match='falling back to torch.compile'):
        result = wrapped(torch.tensor([2.0], dtype=torch.float64))

    torch.testing.assert_close(result, torch.tensor([3.0], dtype=torch.float64))
    assert wrapped._aot_disabled is True
    assert compile_calls == [{'fullgraph': True, 'dynamic': False}]
    assert len(list(tmp_path.glob('*.failed'))) == 1

    # Simulate a fresh process: the persistent failure marker must bypass AOT
    # instead of paying for the same incompatible compile/load again.
    monkeypatch.setattr(internal_module, '_AOT_DISABLED_REASON', None)
    monkeypatch.setattr(internal_module, '_AOT_WARNED', False)
    wrapped_again = internal_module._TorchAOTFunction(
        lambda value: value + 1.0,
        enabled=True,
        cache_name='fallback-test',
    )

    def unexpected_aot(*args, **kwargs):
        raise AssertionError('failure marker should skip AOT')

    monkeypatch.setattr(wrapped_again, '_compile_and_save', unexpected_aot)
    with pytest.warns(RuntimeWarning, match='previous AOT failure'):
        result = wrapped_again(torch.tensor([4.0], dtype=torch.float64))

    torch.testing.assert_close(result, torch.tensor([5.0], dtype=torch.float64))
    assert len(compile_calls) == 2


def test_torch_compile_is_enabled_by_default():
    root = Path(__file__).resolve().parents[2]
    code = """
import sella.internal as internal
print(internal._TORCH_COMPILE_ALL)
"""
    env = os.environ.copy()
    env.pop('SELLA_TORCH_COMPILE', None)
    env['PYTHONPATH'] = os.pathsep.join(
        [str(root), env.get('PYTHONPATH', '')]
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, '-c', code],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == 'True'


def test_default_aot_cache_is_created_under_home(tmp_path):
    root = Path(__file__).resolve().parents[2]
    code = """
import os
import sella.internal as internal

print(internal._TORCH_AOT_CACHE_DIR)
print(os.path.isdir(internal._TORCH_AOT_CACHE_DIR))
"""
    env = os.environ.copy()
    env['HOME'] = str(tmp_path)
    env.pop('SELLA_TORCH_AOT_CACHE_DIR', None)
    env['PYTHONPATH'] = os.pathsep.join(
        [str(root), env.get('PYTHONPATH', '')]
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, '-c', code],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    cache_dir, exists = completed.stdout.splitlines()
    assert cache_dir == str(tmp_path / '.cache' / 'sella' / 'torch-aot')
    assert exists == 'True'


def test_torch_compile_respects_coordinate_thread_cap():
    root = Path(__file__).resolve().parents[2]
    code = """
import os

os.environ['SELLA_TORCH_COMPILE'] = 'all'
os.environ['SELLA_TORCH_COORD_THREADS'] = '7'
import sella.internal  # noqa: F401
import torch._inductor.config as config

print(config.cpp.threads)
"""
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        [str(root), env.get('PYTHONPATH', '')]
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, '-c', code],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == '7'
