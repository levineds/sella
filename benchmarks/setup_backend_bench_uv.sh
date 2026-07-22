#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Set up a uv environment and two Sella checkouts for backend benchmarking.

Typical setup:

  bash benchmarks/setup_backend_bench_uv.sh

Then run the printed benchmark command. The packaged benchmark systems and
FairChem model name uma-s-1p1 are used by default. To run the benchmark
immediately after setup:

  bash benchmarks/setup_backend_bench_uv.sh --run

Options:
  --workdir DIR          Parent directory for venv and checkouts.
                         Default: ~/sella-backend-bench
  --repo-url URL         Sella repository URL.
                         Default: https://github.com/levineds/sella.git
  --jax-branch BRANCH    Branch/ref used as the JAX baseline. Default: master
  --torch-branch BRANCH  Branch/ref used as the PyTorch port. Default: torch
  --python VERSION       Python version for uv venv. Default: 3.12
  --venv DIR             Virtualenv path. Default: WORKDIR/.venv
  --jax-dir DIR          JAX checkout path. Default: WORKDIR/sella-master
  --torch-dir DIR        PyTorch checkout path. Default: WORKDIR/sella-torch
  --torch-index-url URL  Optional PyTorch wheel index, for example
                         https://download.pytorch.org/whl/cu128
  --data-dir DIR         Directory containing packaged MOL_N.cif inputs.
                         Default: TORCH_DIR/benchmarks/data/systems
  --parquet-dir DIR      Optional directory containing MOL.parquet inputs.
                         Overrides --data-dir.
  --model NAME           FairChem pretrained model name. Default: uma-s-1p1
  --checkpoint FILE      Optional local FairChem checkpoint file.
                         Overrides --model.
  --systems SPEC         Benchmark systems. Default: all
  --variants SPEC        Benchmark variants. Default: jax,torch_eager,torch_compile
  --output-prefix PATH   Output path prefix. Default: WORKDIR/results/sella_backend_bench
  --run                  Run the benchmark after setup.
  --no-install           Skip dependency installation.
  --no-prefetch-model    Skip the upfront HuggingFace model download/cache step.
  --no-build             Skip Sella extension builds.
  --no-pull              Do not fetch/pull existing checkouts.
  -h, --help             Show this help.

The same settings can be supplied as environment variables:
WORKDIR, REPO_URL, JAX_BRANCH, TORCH_BRANCH, PYTHON_VERSION, VENV, JAX_DIR,
TORCH_DIR, TORCH_INDEX_URL, DATA_DIR, PARQUET_DIR, MODEL, CHECKPOINT,
SYSTEMS, VARIANTS, OUTPUT_PREFIX.
EOF
}

WORKDIR="${WORKDIR:-"$HOME/sella-backend-bench"}"
REPO_URL="${REPO_URL:-https://github.com/levineds/sella.git}"
JAX_BRANCH="${JAX_BRANCH:-master}"
TORCH_BRANCH="${TORCH_BRANCH:-torch}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV="${VENV:-}"
JAX_DIR="${JAX_DIR:-}"
TORCH_DIR="${TORCH_DIR:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
DATA_DIR="${DATA_DIR:-}"
PARQUET_DIR="${PARQUET_DIR:-}"
MODEL="${MODEL:-uma-s-1p1}"
CHECKPOINT="${CHECKPOINT:-}"
SYSTEMS="${SYSTEMS:-all}"
VARIANTS="${VARIANTS:-jax,torch_eager,torch_compile}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-}"
RUN_BENCH=0
INSTALL_DEPS=1
BUILD_EXTENSIONS=1
PULL_CHECKOUTS=1
PREFETCH_MODEL=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workdir)
            WORKDIR="$2"
            shift 2
            ;;
        --repo-url)
            REPO_URL="$2"
            shift 2
            ;;
        --jax-branch)
            JAX_BRANCH="$2"
            shift 2
            ;;
        --torch-branch)
            TORCH_BRANCH="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --venv)
            VENV="$2"
            shift 2
            ;;
        --jax-dir)
            JAX_DIR="$2"
            shift 2
            ;;
        --torch-dir)
            TORCH_DIR="$2"
            shift 2
            ;;
        --torch-index-url)
            TORCH_INDEX_URL="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --parquet-dir)
            PARQUET_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --systems)
            SYSTEMS="$2"
            shift 2
            ;;
        --variants)
            VARIANTS="$2"
            shift 2
            ;;
        --output-prefix)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        --run)
            RUN_BENCH=1
            shift
            ;;
        --no-install)
            INSTALL_DEPS=0
            shift
            ;;
        --no-prefetch-model)
            PREFETCH_MODEL=0
            shift
            ;;
        --no-build)
            BUILD_EXTENSIONS=0
            shift
            ;;
        --no-pull)
            PULL_CHECKOUTS=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

VENV="${VENV:-"$WORKDIR/.venv"}"
JAX_DIR="${JAX_DIR:-"$WORKDIR/sella-master"}"
TORCH_DIR="${TORCH_DIR:-"$WORKDIR/sella-torch"}"
DATA_DIR="${DATA_DIR:-"$TORCH_DIR/benchmarks/data/systems"}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-"$WORKDIR/results/sella_backend_bench"}"
UV_BIN="${UV:-uv}"

require_command() {
    local cmd="$1"
    local install_hint="$2"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "missing required command: $cmd" >&2
        echo "$install_hint" >&2
        exit 1
    fi
}

sync_checkout() {
    local label="$1"
    local branch="$2"
    local dir="$3"

    if [[ -d "$dir/.git" ]]; then
        echo "updating $label checkout: $dir"
        if [[ "$PULL_CHECKOUTS" -eq 1 ]]; then
            git -C "$dir" fetch --prune origin
            git -C "$dir" checkout "$branch"
            git -C "$dir" pull --ff-only origin "$branch"
        else
            git -C "$dir" checkout "$branch"
        fi
    elif [[ -e "$dir" ]]; then
        echo "$dir exists but is not a git checkout" >&2
        exit 1
    else
        echo "cloning $label checkout: $REPO_URL ($branch) -> $dir"
        git clone --branch "$branch" "$REPO_URL" "$dir"
    fi
}

prepare_checkout() {
    local label="$1"
    local dir="$2"
    local python_bin="$3"

    if PYTHONPATH="$dir" "$python_bin" -c 'import sella.eigensolvers'; then
        echo "verified $label: $dir"
        return
    fi
    if [[ ! -f "$dir/setup.py" ]]; then
        echo "$label cannot be imported and has no setup.py: $dir" >&2
        exit 1
    fi

    echo "building extensions required by $label"
    (
        cd "$dir"
        "$python_bin" setup.py build_ext --inplace
    )
    PYTHONPATH="$dir" "$python_bin" - <<'PY'
import sella
import sella.eigensolvers
print(f"verified {sella.__file__}")
PY
}

prefetch_model() {
    local python_bin="$1"

    if [[ "$PREFETCH_MODEL" -eq 0 || -n "$CHECKPOINT" ]]; then
        return
    fi

    echo "prefetching FairChem model from HuggingFace: $MODEL"
    "$python_bin" - "$MODEL" <<'PY'
import gc
import sys

from fairchem.core.calculate.pretrained_mlip import get_predict_unit

model_name = sys.argv[1]
predictor = get_predict_unit(model_name, "default", None, "cpu")
print(f"cached FairChem model: {model_name}")
del predictor
gc.collect()
PY
}

print_benchmark_command() {
    local python_bin="$1"
    local bench_script="$TORCH_DIR/benchmarks/backend_startup_perstep.py"
    local stamp
    stamp="$(date +%Y%m%d_%H%M%S)"
    local json_output="${OUTPUT_PREFIX}_${stamp}.json"
    local csv_output="${OUTPUT_PREFIX}_${stamp}.csv"

    cat <<EOF

Benchmark command:

  source "$VENV/bin/activate"
  "$python_bin" "$bench_script" \\
    --jax-path "$JAX_DIR" \\
    --torch-path "$TORCH_DIR" \\
    --data-dir "$DATA_DIR" \\
    --model "$MODEL" \\
    --systems "$SYSTEMS" \\
    --variants "$VARIANTS" \\
EOF
    if [[ -n "$CHECKPOINT" ]]; then
        cat <<EOF
    --checkpoint "$CHECKPOINT" \\
EOF
    fi
    if [[ -n "$PARQUET_DIR" ]]; then
        cat <<EOF
    --parquet-dir "$PARQUET_DIR" \\
EOF
    fi
    cat <<EOF
    --output-json "$json_output" \\
    --output-csv "$csv_output"
EOF
}

run_benchmark() {
    local python_bin="$1"
    local bench_script="$TORCH_DIR/benchmarks/backend_startup_perstep.py"
    local stamp
    stamp="$(date +%Y%m%d_%H%M%S)"
    local json_output="${OUTPUT_PREFIX}_${stamp}.json"
    local csv_output="${OUTPUT_PREFIX}_${stamp}.csv"

    mkdir -p "$(dirname "$OUTPUT_PREFIX")"
    local cmd=(
        "$python_bin" "$bench_script"
        --jax-path "$JAX_DIR"
        --torch-path "$TORCH_DIR"
        --data-dir "$DATA_DIR"
        --model "$MODEL"
        --systems "$SYSTEMS"
        --variants "$VARIANTS"
        --output-json "$json_output"
        --output-csv "$csv_output"
    )
    if [[ -n "$CHECKPOINT" ]]; then
        cmd+=(--checkpoint "$CHECKPOINT")
    fi
    if [[ -n "$PARQUET_DIR" ]]; then
        cmd+=(--parquet-dir "$PARQUET_DIR")
    fi
    "${cmd[@]}"
}

require_command git "Install git with your system package manager."
require_command "$UV_BIN" "Install uv: https://docs.astral.sh/uv/getting-started/installation/"

mkdir -p "$WORKDIR"
sync_checkout "JAX baseline" "$JAX_BRANCH" "$JAX_DIR"
sync_checkout "PyTorch port" "$TORCH_BRANCH" "$TORCH_DIR"

echo "creating/updating uv environment: $VENV"
"$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV"
PYTHON_BIN="$VENV/bin/python"

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
    echo "installing benchmark dependencies"
    if [[ -n "$TORCH_INDEX_URL" ]]; then
        "$UV_BIN" pip install --python "$PYTHON_BIN" --upgrade \
            --index-url "$TORCH_INDEX_URL" torch
    else
        "$UV_BIN" pip install --python "$PYTHON_BIN" --upgrade torch
    fi
    "$UV_BIN" pip install --python "$PYTHON_BIN" --upgrade \
        setuptools wheel cython \
        "numpy<3" scipy ase pandas pyarrow \
        jax jaxlib fairchem-core
fi

prefetch_model "$PYTHON_BIN"

if [[ "$BUILD_EXTENSIONS" -eq 1 ]]; then
    prepare_checkout "JAX baseline" "$JAX_DIR" "$PYTHON_BIN"
    prepare_checkout "PyTorch port" "$TORCH_DIR" "$PYTHON_BIN"
fi

print_benchmark_command "$PYTHON_BIN"

if [[ "$RUN_BENCH" -eq 1 ]]; then
    run_benchmark "$PYTHON_BIN"
fi
