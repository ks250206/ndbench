#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENTRYPOINT="${SCRIPT_DIR}/ndbench.jl"
RESULT_DIR="${SCRIPT_DIR}/results"

JULIA_BIN="${JULIA_BIN:-julia}"
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

command -v "${JULIA_BIN}" >/dev/null || {
    echo "Julia is required: https://julialang.org/downloads/" >&2
    exit 1
}
command -v hyperfine >/dev/null || {
    echo "hyperfine is required: https://github.com/sharkdp/hyperfine" >&2
    exit 1
}

# Keep both Julia's task layer and the BLAS/LAPACK layer single-threaded.
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

mkdir -p "${RESULT_DIR}"

run_group() {
    local operation="$1"
    local size="$2"
    local iterations="$3"
    local qjulia qentry qproject
    printf -v qjulia '%q' "${JULIA_BIN}"
    printf -v qentry '%q' "${ENTRYPOINT}"
    printf -v qproject '%q' "${SCRIPT_DIR}"
    local command="${qjulia} --startup-file=no --history-file=no --threads=1 --project=${qproject} ${qentry} --op ${operation} --size ${size} --iterations ${iterations}"

    hyperfine \
        --shell=none \
        --warmup "${HYPERFINE_WARMUP}" \
        --runs "${HYPERFINE_RUNS}" \
        --export-markdown "${RESULT_DIR}/${operation}.md" \
        --export-json "${RESULT_DIR}/${operation}.json" \
        --command-name julia "${command}"
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
