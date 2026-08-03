#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/target"
BINARY="${BUILD_DIR}/ndbench-mojo"

UV_BIN="${UV_BIN:-uv}"
HYPERFINE_RUNS="${HYPERFINE_RUNS:-10}"
HYPERFINE_WARMUP="${HYPERFINE_WARMUP:-3}"
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

mkdir -p "${BUILD_DIR}" "${SCRIPT_DIR}/results"
cd "${SCRIPT_DIR}"
"${UV_BIN}" run --project . mojo build --optimization-level 3 -D ASSERT=none main.mojo -o "${BINARY}"

run_bench() {
    local operation="$1"
    local size="$2"
    local iterations="$3"
    hyperfine \
        --shell=none \
        --warmup "${HYPERFINE_WARMUP}" \
        --runs "${HYPERFINE_RUNS}" \
        --export-json "${SCRIPT_DIR}/results/${operation}.json" \
        --export-markdown "${SCRIPT_DIR}/results/${operation}.md" \
        --command-name mojo \
        "${BINARY} --op ${operation} --size ${size} --iterations ${iterations}"
}

run_bench vector2 8 "${VECTOR_ITERATIONS}"
run_bench vector3 8 "${VECTOR_ITERATIONS}"
run_bench affine2 "${POINTS}" "${AFFINE_ITERATIONS}"
run_bench affine3 "${POINTS}" "${AFFINE_ITERATIONS}"
run_bench matvec "${MATVEC_SIZE}" "${MATVEC_ITERATIONS}"
run_bench matmul "${MATMUL_SIZE}" "${MATMUL_ITERATIONS}"
run_bench cosine1024 1024 "${COSINE_ITERATIONS}"
run_bench eigh "${EIGH_SIZE}" "${EIGH_ITERATIONS}"
