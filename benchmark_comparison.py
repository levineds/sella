#!/usr/bin/env python
"""
Benchmark comparison script for Sella performance testing.

Compares three versions:
1. origin/master
2. consolidated-optimizations
3. sella_ugrad:vectorized

Uses cProfile to isolate Sella time from MLIP time.
"""

import cProfile
import pstats
import time
import sys
import subprocess
import argparse
import json

# Test configuration
NUM_TRIALS = 5
FMAX = 0.01
STEPS = 10


def run_benchmark_subprocess(sella_path: str, xyz_path: str, fmax: float = FMAX, steps: int = STEPS):
    """Run benchmark in subprocess for clean import state."""

    benchmark_code = f'''
import sys
import time
import cProfile
import pstats
import json
import warnings
warnings.filterwarnings('ignore')

# Set up path for sella
sys.path.insert(0, "{sella_path}")

from ase.io import read
from sella import Sella
from fairchem.core import FAIRChemCalculator, pretrained_mlip

def get_sella_time_from_profile(profiler):
    stats = pstats.Stats(profiler)
    mlip_time = 0.0
    sella_time = 0.0
    other_time = 0.0

    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, lineno, funcname = func
        if any(x in filename for x in ['fairchem', 'torch', 'ase/calculators']):
            mlip_time += tt
        elif 'sella' in filename:
            sella_time += tt
        else:
            other_time += tt
    return sella_time, mlip_time, other_time

# Setup
setup_start = time.time()
atoms = read("{xyz_path}", index=0)
atoms.info.update({{"charge": 2, "spin": 5}})
predictor = pretrained_mlip.get_predict_unit("esen-sm-conserving-all-omol", device="cpu")
calc = FAIRChemCalculator(predictor, task_name="omol")
atoms.calc = calc
dyn = Sella(atoms, order=0, internal=True)
setup_time = time.time() - setup_start

# Profile
profiler = cProfile.Profile()
profiler.enable()
run_start = time.time()
dyn.run(fmax={fmax}, steps={steps})
run_time = time.time() - run_start
profiler.disable()

sella_time, mlip_time, other_time = get_sella_time_from_profile(profiler)

print(json.dumps({{
    "setup_time": setup_time,
    "run_time": run_time,
    "sella_time": sella_time,
    "mlip_time": mlip_time,
    "other_time": other_time,
}}))
'''

    result = subprocess.run(
        [sys.executable, '-c', benchmark_code],
        capture_output=True,
        text=True,
        timeout=600
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        return None

    # Find and parse JSON output
    for line in result.stdout.strip().split('\n'):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def run_trials(name: str, sella_path: str, xyz_path: str, num_trials: int = NUM_TRIALS):
    """Run multiple benchmark trials."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"Path: {sella_path}")
    print(f"Running {num_trials} trials...")
    print('='*60)

    results = []
    for i in range(num_trials):
        print(f"  Trial {i+1}/{num_trials}...", end=' ', flush=True)
        result = run_benchmark_subprocess(sella_path, xyz_path)
        if result:
            results.append(result)
            print(f"Sella: {result['sella_time']:.3f}s, MLIP: {result['mlip_time']:.3f}s, Total: {result['run_time']:.3f}s")
        else:
            print("FAILED")

    if not results:
        return None

    def stats(key):
        values = [r[key] for r in results]
        return {
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
        }

    return {
        'name': name,
        'path': sella_path,
        'trials': len(results),
        'sella_time': stats('sella_time'),
        'mlip_time': stats('mlip_time'),
        'run_time': stats('run_time'),
        'other_time': stats('other_time'),
    }


def print_results(all_results: list):
    """Print comparison table."""
    print("\n" + "="*80)
    print("RESULTS COMPARISON (excluding MLIP time)")
    print("="*80)

    # Sort by avg sella time
    sorted_results = sorted(all_results, key=lambda x: x['sella_time']['avg'])
    baseline = max(r['sella_time']['avg'] for r in sorted_results)

    print(f"\n{'Version':<35} {'Avg':>10} {'Min':>10} {'Max':>10} {'Speedup':>10}")
    print("-"*80)

    for r in sorted_results:
        s = r['sella_time']
        speedup = baseline / s['avg'] if s['avg'] > 0 else 0
        print(f"{r['name']:<35} {s['avg']:>9.3f}s {s['min']:>9.3f}s {s['max']:>9.3f}s {speedup:>9.2f}x")

    print("\n" + "-"*80)
    print("Note: Speedup is relative to slowest version")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Sella versions')
    parser.add_argument('--xyz', default='/Users/levineds/sella_refactoring/orca.xyz')
    parser.add_argument('--trials', type=int, default=NUM_TRIALS)
    parser.add_argument('--versions', nargs='+',
                        default=['master', 'consolidated', 'vectorized'],
                        choices=['master', 'consolidated', 'vectorized'])
    args = parser.parse_args()

    configs = {
        'master': ('origin/master', '/Users/levineds/packages/sella', 'master'),
        'consolidated': ('consolidated-optimizations', '/Users/levineds/packages/sella', 'consolidated-optimizations'),
        'vectorized': ('sella_ugrad:vectorized', '/Users/levineds/packages/sella_ugrad', 'vectorized'),
    }

    all_results = []
    original_branch = None

    # Get original branch
    result = subprocess.run(['git', 'branch', '--show-current'],
                          cwd='/Users/levineds/packages/sella', capture_output=True, text=True)
    original_branch = result.stdout.strip()

    for version in args.versions:
        name, path, branch = configs[version]

        # Switch branches if needed (only for main sella repo)
        if 'sella_ugrad' not in path:
            print(f"\nSwitching to branch: {branch}")
            subprocess.run(['git', 'checkout', branch], cwd=path, capture_output=True)

        result = run_trials(name, path, args.xyz, args.trials)
        if result:
            all_results.append(result)

    # Switch back to original branch
    if original_branch:
        subprocess.run(['git', 'checkout', original_branch],
                      cwd='/Users/levineds/packages/sella', capture_output=True)

    if all_results:
        print_results(all_results)


if __name__ == '__main__':
    main()
