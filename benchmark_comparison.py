#!/usr/bin/env python
"""Benchmark comparison between JAX and PyTorch backends."""

import time
import numpy as np
from ase.build import molecule
from ase.io import read
from ase.calculators.emt import EMT

def benchmark_internals():
    """Benchmark internal coordinate calculations."""
    from sella.internal import Internals, Bond, Angle, Dihedral

    # Test molecules of different sizes
    test_cases = []

    # Small molecules from ASE
    for mol_name in ['H2O', 'CH4', 'C2H6', 'C6H6']:
        test_cases.append((mol_name, molecule(mol_name)))

    # Larger molecules from xyz files
    try:
        orca = read('orca.xyz')
        test_cases.append(('orca.xyz', orca))
    except:
        pass

    try:
        orca2 = read('orca2.xyz')
        test_cases.append(('orca2.xyz', orca2))
    except:
        pass

    results = {}

    for name, atoms in test_cases:
        # Create internals
        ints = Internals(atoms)
        ints.find_all_bonds()
        ints.find_all_angles()
        ints.find_all_dihedrals()

        # Warmup
        for _ in range(3):
            ints.calc()
            ints.jacobian()
            ints.hessian()

        # Benchmark calc
        n_iters = 50 if len(atoms) > 20 else 100
        start = time.perf_counter()
        for _ in range(n_iters):
            ints.calc()
        calc_time = (time.perf_counter() - start) / n_iters * 1000

        # Benchmark jacobian
        start = time.perf_counter()
        for _ in range(n_iters):
            ints.jacobian()
        jac_time = (time.perf_counter() - start) / n_iters * 1000

        # Benchmark hessian
        n_hess = 10 if len(atoms) > 20 else 20
        start = time.perf_counter()
        for _ in range(n_hess):
            ints.hessian()
        hess_time = (time.perf_counter() - start) / n_hess * 1000

        results[name] = {
            'natoms': len(atoms),
            'nints': ints.nbonds + ints.nangles + ints.ndihedrals,
            'calc_ms': calc_time,
            'jacobian_ms': jac_time,
            'hessian_ms': hess_time,
        }

    return results


def benchmark_optimization():
    """Benchmark a full optimization."""
    from sella import Sella

    atoms = molecule('C2H6')
    np.random.seed(42)
    atoms.positions += np.random.normal(0, 0.1, atoms.positions.shape)
    atoms.calc = EMT()

    # Warmup
    atoms_warmup = atoms.copy()
    atoms_warmup.calc = EMT()
    opt = Sella(atoms_warmup, order=0, internal=True, logfile=None)
    opt.run(fmax=0.1, steps=5)

    # Benchmark
    atoms_bench = atoms.copy()
    atoms_bench.calc = EMT()

    start = time.perf_counter()
    opt = Sella(atoms_bench, order=0, internal=True, logfile=None)
    opt.run(fmax=0.05, steps=50)
    total_time = time.perf_counter() - start

    return {
        'total_time_s': total_time,
        'steps': opt.nsteps,
        'time_per_step_ms': total_time / opt.nsteps * 1000,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("Internal Coordinate Benchmarks")
    print("=" * 70)

    int_results = benchmark_internals()

    print(f"{'Molecule':<12} {'Atoms':<6} {'Ints':<6} {'calc':<12} {'jacobian':<12} {'hessian':<12}")
    print("-" * 70)
    for mol, r in int_results.items():
        print(f"{mol:<12} {r['natoms']:<6} {r['nints']:<6} {r['calc_ms']:.3f} ms    {r['jacobian_ms']:.3f} ms    {r['hessian_ms']:.3f} ms")

    print("\n" + "=" * 70)
    print("Optimization Benchmark (C2H6)")
    print("=" * 70)

    opt_results = benchmark_optimization()
    print(f"Total time: {opt_results['total_time_s']:.3f} s")
    print(f"Steps: {opt_results['steps']}")
    print(f"Time per step: {opt_results['time_per_step_ms']:.1f} ms")
