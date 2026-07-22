#!/usr/bin/env python3
"""Compare Sella JAX and PyTorch backend startup and per-step costs.

This script is intentionally subprocess-based: each measured variant/system
runs in a fresh Python process. FairChem model loading and input parsing happen
before the Sella timer starts. The reported per-step startup cost includes only:

  import sella + Sella(...) construction + untimed warmup optimizer steps

It also records a fresh-process complete optimization wall time to fmax=0.01
by default, so backend startup/compile overhead can be viewed against typical
production job durations rather than only against a 15-step microbenchmark.

The optional prewarm phase runs first to populate one-time per-machine caches
such as JAX's persistent compilation cache and TorchInductor's disk cache.

Typical use:

    source ~/fc_env/bin/activate
    python /home/levineds/sella/benchmarks/backend_startup_perstep.py \
        --jax-path /tmp/sella_jax_head_0a259aa \
        --torch-path /home/levineds/sella

If --jax-path is omitted, a detached worktree is created from --jax-ref
(default: HEAD) under /tmp and built in-place.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


DEFAULT_PARQUET_DIR = "/checkpoint/ocp/levineds/csp_parquets/tier2_parquet"
DEFAULT_SNAPSHOT = (
    "/home/levineds/.cache/fairchem/models--facebook--UMA/snapshots/"
    "38529caa2c51a9a8a0d71f0b56b79ac33bc9eceb"
)
DEFAULT_CHECKPOINT = f"{DEFAULT_SNAPSHOT}/checkpoints/uma-s-1p1.pt"
DEFAULT_SYSTEMS = [
    ("QQQAUG", 23),
    ("BISMEV", 40),
    ("BEDMIG", 60),
    ("BISMEV", 80),
    ("SUTHAZ", 100),
    ("AJEYAQ", 120),
    ("HXACAN", 160),
    ("YEXGEQ", 192),
    ("PAPTUX", 248),
    ("BEQGIN", 400),
]

VARIANT_ENVS = {
    "jax": {},
    "torch_eager": {"SELLA_TORCH_COMPILE": "0"},
    "torch_compile": {"SELLA_TORCH_COMPILE": "all"},
}
DEFAULT_VARIANTS = ["jax", "torch_eager", "torch_compile"]
PREWARM_VARIANTS = {"jax", "torch_compile"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, cwd: Path | None = None,
        env: dict[str, str] | None = None, check: bool = True
        ) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def parse_systems(text: str) -> list[tuple[str, int]]:
    if text == "all":
        return DEFAULT_SYSTEMS
    systems = []
    for item in text.split(","):
        mol, n_atoms = item.split(":", 1)
        systems.append((mol.strip(), int(n_atoms)))
    return systems


def parse_variants(text: str) -> list[str]:
    variants = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [variant for variant in variants if variant not in VARIANT_ENVS]
    if unknown:
        raise SystemExit(f"Unknown variants: {', '.join(unknown)}")
    return variants


def pythonpath_for(path: Path, env: dict[str, str]) -> str:
    existing = env.get("PYTHONPATH", "")
    return str(path) if not existing else f"{path}{os.pathsep}{existing}"


def import_check(path: Path, env: dict[str, str]) -> bool:
    check_env = env.copy()
    check_env["PYTHONPATH"] = pythonpath_for(path, check_env)
    result = run(
        [
            sys.executable,
            "-c",
            (
                "import sella; import sella.force_match; "
                "import sella.utilities.blas; import sella.utilities.math"
            ),
        ],
        env=check_env,
        check=False,
    )
    return result.returncode == 0


def ensure_extensions(path: Path, env: dict[str, str]) -> None:
    if import_check(path, env):
        return
    setup_py = path / "setup.py"
    if not setup_py.exists():
        raise RuntimeError(f"{path} cannot import extensions and has no setup.py")
    print(f"building Cython extensions in {path}", flush=True)
    result = run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=path,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"extension build failed in {path}")
    if not import_check(path, env):
        raise RuntimeError(f"{path} still cannot import extensions after build")


def short_ref(ref: str) -> str:
    result = run(["git", "rev-parse", "--short", ref], cwd=repo_root())
    return result.stdout.strip()


def ensure_jax_worktree(args: argparse.Namespace, env: dict[str, str]) -> Path:
    if args.jax_path:
        path = Path(args.jax_path).expanduser().resolve()
        ensure_extensions(path, env)
        return path

    ref = args.jax_ref
    path = Path(args.worktree_dir).expanduser() / f"sella_jax_{short_ref(ref)}"
    if not path.exists():
        print(f"creating JAX baseline worktree {path} from {ref}", flush=True)
        result = run(
            ["git", "worktree", "add", "--detach", str(path), ref],
            cwd=repo_root(),
            check=False,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise RuntimeError("failed to create JAX baseline worktree")
    ensure_extensions(path, env)
    return path.resolve()


def variant_path(variant: str, jax_path: Path, torch_path: Path) -> Path:
    return jax_path if variant == "jax" else torch_path


def run_worker(args: argparse.Namespace, *, variant: str, path: Path,
               mol: str, n_atoms: int, mode: str) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(VARIANT_ENVS[variant])
    env["PYTHONPATH"] = pythonpath_for(path, env)
    if args.cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-mode", mode,
        "--variant", variant,
        "--mol", mol,
        "--n-atoms", str(n_atoms),
        "--parquet-dir", args.parquet_dir,
        "--checkpoint", args.checkpoint,
        "--warmup-steps", str(args.warmup_steps),
        "--timed-steps", str(args.timed_steps),
        "--prewarm-steps", str(args.prewarm_steps),
        "--fmax", str(args.fmax),
        "--complete-fmax", str(args.complete_fmax),
        "--complete-max-steps", str(args.complete_max_steps),
    ]
    result = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError(
            f"worker failed for {variant} {mol}:{n_atoms} ({mode})"
        )
    prefix = "BENCH_JSON "
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(prefix):
            return json.loads(line[len(prefix):])
    raise RuntimeError(f"worker did not emit {prefix.strip()}")


def prewarm(args: argparse.Namespace, variants: list[str],
            systems: list[tuple[str, int]], jax_path: Path,
            torch_path: Path) -> None:
    if args.no_prewarm:
        return
    print("\nprewarm: populating one-time per-machine caches", flush=True)
    for variant in variants:
        if variant not in PREWARM_VARIANTS:
            continue
        path = variant_path(variant, jax_path, torch_path)
        for mol, n_atoms in systems:
            print(f"prewarm {variant:<13s} {mol}:{n_atoms}", flush=True)
            run_worker(
                args,
                variant=variant,
                path=path,
                mol=mol,
                n_atoms=n_atoms,
                mode="prewarm",
            )


def summarize(results: list[dict[str, Any]]) -> None:
    perstep = [row for row in results if row.get("record_type") == "perstep"]
    complete = [row for row in results if row.get("record_type") == "complete"]

    if not perstep and not complete:
        return

    if perstep:
        print("\nper-step results")
        print(
            f"{'variant':<13s} {'mol':<8s} {'n':>4s} "
            f"{'steady':>10s} {'amort':>10s} {'startup':>10s} "
            f"{'init':>8s} {'warmup':>8s}"
        )
        print("-" * 82)
        for row in perstep:
            print(
                f"{row['variant']:<13s} {row['mol']:<8s} "
                f"{row['n_atoms']:4d} "
                f"{row['steady_s_per_step']:10.6f} "
                f"{row['amortized_s_per_step']:10.6f} "
                f"{row['startup_s']:10.3f} "
                f"{row['init_s']:8.3f} {row['warmup_s']:8.3f}"
            )

        by_key = {
            (row["mol"], row["n_atoms"], row["variant"]): row
            for row in perstep
        }
        print("\nper-step ratios vs jax")
        print(
            f"{'variant':<13s} {'mol':<8s} {'n':>4s} "
            f"{'steady_x':>10s} {'amort_x':>10s}"
        )
        print("-" * 52)
        for row in perstep:
            if row["variant"] == "jax":
                continue
            base = by_key.get((row["mol"], row["n_atoms"], "jax"))
            if base is None:
                continue
            print(
                f"{row['variant']:<13s} {row['mol']:<8s} "
                f"{row['n_atoms']:4d} "
                f"{row['steady_s_per_step'] / base['steady_s_per_step']:10.3f} "
                f"{row['amortized_s_per_step'] / base['amortized_s_per_step']:10.3f}"
            )

    if complete:
        print("\ncomplete optimization results")
        print(
            f"{'variant':<13s} {'mol':<8s} {'n':>4s} "
            f"{'wall':>9s} {'run':>9s} {'init':>8s} "
            f"{'steps':>6s} {'conv':>5s}"
        )
        print("-" * 74)
        for row in complete:
            print(
                f"{row['variant']:<13s} {row['mol']:<8s} "
                f"{row['n_atoms']:4d} "
                f"{row['complete_total_s']:9.3f} "
                f"{row['complete_run_s']:9.3f} "
                f"{row['complete_init_s']:8.3f} "
                f"{row['complete_steps']:6d} "
                f"{str(row['complete_converged']):>5s}"
            )

        by_key = {
            (row["mol"], row["n_atoms"], row["variant"]): row
            for row in complete
        }
        print("\ncomplete optimization ratios vs jax")
        print(
            f"{'variant':<13s} {'mol':<8s} {'n':>4s} {'wall_x':>10s}"
        )
        print("-" * 42)
        for row in complete:
            if row["variant"] == "jax":
                continue
            base = by_key.get((row["mol"], row["n_atoms"], "jax"))
            if base is None:
                continue
            print(
                f"{row['variant']:<13s} {row['mol']:<8s} "
                f"{row['n_atoms']:4d} "
                f"{row['complete_total_s'] / base['complete_total_s']:10.3f}"
            )


def write_outputs(results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if not args.output_json and not args.output_csv:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_json = f"/tmp/sella_backend_bench_{stamp}.json"
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.output_json}")
    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            fieldnames = sorted({key for row in results for key in row})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"wrote {args.output_csv}")


def worker_main(args: argparse.Namespace) -> None:
    import gc
    import importlib
    from io import StringIO
    import warnings

    warnings.filterwarnings("ignore")

    import pandas as pd
    import torch
    from ase.io import read
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.calculate.pretrained_mlip import load_predict_unit

    predictor = load_predict_unit(
        args.checkpoint, "default", None, "cuda", None, None
    )
    df = pd.read_parquet(f"{args.parquet_dir}/{args.mol}.parquet")
    conv = df[(df["converged"] == True) & (df["n_atoms"] == args.n_atoms)]
    if len(conv) == 0:
        raise RuntimeError(f"no rows for {args.mol}:{args.n_atoms}")
    row = conv.sample(n=1, random_state=42).iloc[0]
    atoms = read(StringIO(row["cif"]), format="cif")
    calc = FAIRChemCalculator(predictor, task_name="omc")
    atoms.calc = calc
    atoms.get_potential_energy()

    if args.worker_mode == "prewarm":
        sella_mod = importlib.import_module("sella")
        dyn = sella_mod.Sella(
            atoms, order=0, internal=True, optimize_cell=True,
            allow_fragments=True, refine_initial_hessian=0,
            exact_geodesic=False, hessian_delta=1e-4, logfile=None,
        )
        dyn.run(fmax=args.fmax, steps=args.prewarm_steps)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out = {
            "record_type": "prewarm",
            "variant": args.variant,
            "mol": args.mol,
            "n_atoms": args.n_atoms,
            "mode": "prewarm",
            "sella_file": sella_mod.__file__,
        }
        print("BENCH_JSON " + json.dumps(out, sort_keys=True), flush=True)
        return

    if args.worker_mode == "complete":
        t0 = time.perf_counter()
        sella_mod = importlib.import_module("sella")
        Sella = sella_mod.Sella
        t_import = time.perf_counter()
        dyn = Sella(
            atoms, order=0, internal=True, optimize_cell=True,
            allow_fragments=True, refine_initial_hessian=0,
            exact_geodesic=False, hessian_delta=1e-4, logfile=None,
        )
        t_init = time.perf_counter()
        dyn.run(fmax=args.complete_fmax, steps=args.complete_max_steps)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_done = time.perf_counter()

        out = {
            "record_type": "complete",
            "variant": args.variant,
            "mol": args.mol,
            "n_atoms": args.n_atoms,
            "complete_fmax": args.complete_fmax,
            "complete_max_steps": args.complete_max_steps,
            "complete_converged": bool(dyn.converged()),
            "complete_steps": int(dyn.nsteps),
            "complete_total_s": t_done - t0,
            "complete_import_s": t_import - t0,
            "complete_init_s": t_init - t_import,
            "complete_run_s": t_done - t_init,
            "sella_file": sella_mod.__file__,
        }
        print("BENCH_JSON " + json.dumps(out, sort_keys=True), flush=True)

        del calc, dyn, predictor, atoms, df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    t0 = time.perf_counter()
    sella_mod = importlib.import_module("sella")
    Sella = sella_mod.Sella
    t_import = time.perf_counter()
    dyn = Sella(
        atoms, order=0, internal=True, optimize_cell=True,
        allow_fragments=True, refine_initial_hessian=0,
        exact_geodesic=False, hessian_delta=1e-4, logfile=None,
    )
    t_init = time.perf_counter()
    dyn.run(fmax=args.fmax, steps=args.warmup_steps)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_warm = time.perf_counter()
    dyn.run(fmax=args.fmax, steps=args.timed_steps)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_done = time.perf_counter()

    startup_s = t_warm - t0
    timed_s = t_done - t_warm
    total_s = t_done - t0
    out = {
        "record_type": "perstep",
        "variant": args.variant,
        "mol": args.mol,
        "n_atoms": args.n_atoms,
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "steady_s_per_step": timed_s / args.timed_steps,
        "amortized_s_per_step": total_s / (
            args.warmup_steps + args.timed_steps
        ),
        "startup_s": startup_s,
        "total_s": total_s,
        "import_s": t_import - t0,
        "init_s": t_init - t_import,
        "warmup_s": t_warm - t_init,
        "timed_s": timed_s,
        "steps_completed": dyn.nsteps,
        "sella_file": sella_mod.__file__,
    }
    print("BENCH_JSON " + json.dumps(out, sort_keys=True), flush=True)

    del calc, dyn, predictor, atoms, df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark JAX vs PyTorch Sella backend steady-state and "
            "startup-amortized per-step costs."
        )
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=["prewarm", "measure", "complete"],
                        default="measure", help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=sorted(VARIANT_ENVS),
                        help=argparse.SUPPRESS)
    parser.add_argument("--mol", help=argparse.SUPPRESS)
    parser.add_argument("--n-atoms", type=int, help=argparse.SUPPRESS)

    parser.add_argument("--jax-path", default=os.environ.get("SELLA_JAX_PATH"))
    parser.add_argument("--jax-ref", default="HEAD")
    parser.add_argument("--torch-path", default=str(repo_root()))
    parser.add_argument("--worktree-dir", default="/tmp")
    parser.add_argument("--parquet-dir", default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--systems", default="all",
                        help="all, or comma-separated MOL:N_ATOMS entries")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--prewarm-steps", type=int, default=5)
    parser.add_argument("--fmax", type=float, default=0.001)
    parser.add_argument("--complete-fmax", type=float, default=0.01)
    parser.add_argument("--complete-max-steps", type=int, default=300)
    parser.add_argument("--no-complete", action="store_true")
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--no-prewarm", action="store_true")
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")

    args = parser.parse_args()
    if args.worker:
        worker_main(args)
        return

    env = os.environ.copy()
    torch_path = Path(args.torch_path).expanduser().resolve()
    jax_path = ensure_jax_worktree(args, env)
    ensure_extensions(torch_path, env)

    systems = parse_systems(args.systems)
    variants = parse_variants(args.variants)

    print(f"jax path:   {jax_path}")
    print(f"torch path: {torch_path}")
    print(f"systems:    {', '.join(f'{m}:{n}' for m, n in systems)}")
    print(f"variants:   {', '.join(variants)}")
    print(
        "timing:     "
        f"{args.warmup_steps} warmup + {args.timed_steps} timed steps"
    )

    prewarm(args, variants, systems, jax_path, torch_path)

    results = []
    print("\nper-step measurements", flush=True)
    for variant in variants:
        path = variant_path(variant, jax_path, torch_path)
        for mol, n_atoms in systems:
            print(f"measure {variant:<13s} {mol}:{n_atoms}", flush=True)
            row = run_worker(
                args,
                variant=variant,
                path=path,
                mol=mol,
                n_atoms=n_atoms,
                mode="measure",
            )
            results.append(row)

    if not args.no_complete:
        print(
            "\ncomplete optimization measurements "
            f"(fmax={args.complete_fmax}, max_steps={args.complete_max_steps})",
            flush=True,
        )
        for variant in variants:
            path = variant_path(variant, jax_path, torch_path)
            for mol, n_atoms in systems:
                print(
                    f"complete {variant:<13s} {mol}:{n_atoms}",
                    flush=True,
                )
                row = run_worker(
                    args,
                    variant=variant,
                    path=path,
                    mol=mol,
                    n_atoms=n_atoms,
                    mode="complete",
                )
                results.append(row)

    summarize(results)
    write_outputs(results, args)


if __name__ == "__main__":
    main()
