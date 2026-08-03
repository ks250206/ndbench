#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}/results"
PYTHON_ENTRYPOINT="${SCRIPT_DIR}/ndbench.py"

UV_BIN="${UV_BIN:-uv}"
HYPERFINE_RUNS="${HYPERFINE_RUNS:-10}"
HYPERFINE_WARMUP="${HYPERFINE_WARMUP:-2}"
VECTOR_ITERATIONS="${VECTOR_ITERATIONS:-1000000}"
POINTS="${POINTS:-100000}"
AFFINE_ITERATIONS="${AFFINE_ITERATIONS:-5}"
MATVEC_SIZE="${MATVEC_SIZE:-512}"
MATVEC_ITERATIONS="${MATVEC_ITERATIONS:-10}"
MATMUL_SIZE="${MATMUL_SIZE:-256}"
MATMUL_ITERATIONS="${MATMUL_ITERATIONS:-5}"
COSINE_ITERATIONS="${COSINE_ITERATIONS:-1000}"
EIGH_SIZE="${EIGH_SIZE:-128}"
EIGH_ITERATIONS="${EIGH_ITERATIONS:-5}"

command -v "${UV_BIN}" >/dev/null || {
    echo "uv is required: https://docs.astral.sh/uv/" >&2
    exit 1
}
command -v hyperfine >/dev/null || {
    echo "hyperfine is required: https://github.com/sharkdp/hyperfine" >&2
    exit 1
}

# NumPy uses the BLAS available in the uv environment. Keep it serial so the
# comparison matches the Rust scripts' single-thread CPU setting. PyTorch is
# fixed to one intra-op/inter-op thread by ndbench.py before tensor work.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

"${UV_BIN}" sync --project "${SCRIPT_DIR}"
mkdir -p "${RESULT_DIR}"

run_group() {
    local operation="$1"
    local size="$2"
    local iterations="$3"
    local common="--op ${operation} --size ${size} --iterations ${iterations}"
    local numpy_command="${UV_BIN} run --project ${SCRIPT_DIR} python ${PYTHON_ENTRYPOINT} --backend numpy ${common}"
    local pytorch_command="${UV_BIN} run --project ${SCRIPT_DIR} python ${PYTHON_ENTRYPOINT} --backend pytorch ${common}"
    local raw_command="${UV_BIN} run --project ${SCRIPT_DIR} python ${PYTHON_ENTRYPOINT} --backend raw ${common}"

    # The measured command intentionally includes uv run, Python startup, and
    # deterministic input construction, just as the Rust executable includes
    # its own process startup and input construction.
    hyperfine \
        --shell=none \
        --warmup "${HYPERFINE_WARMUP}" \
        --runs "${HYPERFINE_RUNS}" \
        --export-markdown "${RESULT_DIR}/${operation}.md" \
        --export-json "${RESULT_DIR}/${operation}.json" \
        --command-name numpy "${numpy_command}" \
        --command-name pytorch "${pytorch_command}" \
        --command-name raw "${raw_command}"
}

run_group vector2 1 "${VECTOR_ITERATIONS}"
run_group vector3 1 "${VECTOR_ITERATIONS}"
run_group affine2 "${POINTS}" "${AFFINE_ITERATIONS}"
run_group affine3 "${POINTS}" "${AFFINE_ITERATIONS}"
run_group matvec "${MATVEC_SIZE}" "${MATVEC_ITERATIONS}"
run_group matmul "${MATMUL_SIZE}" "${MATMUL_ITERATIONS}"
run_group cosine1024 1024 "${COSINE_ITERATIONS}"
run_group eigh "${EIGH_SIZE}" "${EIGH_ITERATIONS}"

echo "Benchmark results written to ${RESULT_DIR}"
