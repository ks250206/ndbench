#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BIN="${REPO_DIR}/target/release/ndbench"
RESULT_DIR="${REPO_DIR}/results"

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

command -v hyperfine >/dev/null || {
    echo "hyperfine is required: https://github.com/sharkdp/hyperfine" >&2
    exit 1
}

cargo build --release --features ndarray-eigh-openblas-static --manifest-path "${REPO_DIR}/Cargo.toml"
mkdir -p "${RESULT_DIR}"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export CANDLE_NUM_THREADS="${CANDLE_NUM_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"

run_group() {
    local operation="$1"
    local size="$2"
    local iterations="$3"
    local qbin
    printf -v qbin '%q' "${BIN}"
    local common="--op ${operation} --size ${size} --iterations ${iterations}"

    local -a commands=(
        --command-name ndarray "${qbin} --backend ndarray ${common}"
        --command-name faer "${qbin} --backend faer ${common}"
        --command-name nalgebra "${qbin} --backend nalgebra ${common}"
    )
    if [[ "${operation}" != eigh ]]; then
        commands+=(--command-name candle "${qbin} --backend candle ${common}")
        commands+=(--command-name burn "${qbin} --backend burn ${common}")
    fi

    hyperfine \
        --shell=none \
        --warmup "${HYPERFINE_WARMUP}" \
        --runs "${HYPERFINE_RUNS}" \
        --export-markdown "${RESULT_DIR}/${operation}.md" \
        --export-json "${RESULT_DIR}/${operation}.json" \
        "${commands[@]}"
}

run_group vector2 1 "${VECTOR_ITERATIONS}"
run_group vector3 1 "${VECTOR_ITERATIONS}"
run_group affine2 "${POINTS}" "${AFFINE_ITERATIONS}"
run_group affine3 "${POINTS}" "${AFFINE_ITERATIONS}"
run_group matvec "${MATVEC_SIZE}" "${MATVEC_ITERATIONS}"
run_group matmul "${MATMUL_SIZE}" "${MATMUL_ITERATIONS}"
run_group cosine1024 1024 "${COSINE_ITERATIONS}"

# Candle and Burn currently have no general symmetric eigendecomposition API.
run_group_without_tensor_eigh() {
    local operation="$1"
    local size="$2"
    local iterations="$3"
    local qbin
    printf -v qbin '%q' "${BIN}"
    local common="--op ${operation} --size ${size} --iterations ${iterations}"

    hyperfine \
        --shell=none \
        --warmup "${HYPERFINE_WARMUP}" \
        --runs "${HYPERFINE_RUNS}" \
        --export-markdown "${RESULT_DIR}/${operation}.md" \
        --export-json "${RESULT_DIR}/${operation}.json" \
        --command-name ndarray "${qbin} --backend ndarray ${common}" \
        --command-name faer "${qbin} --backend faer ${common}" \
        --command-name nalgebra "${qbin} --backend nalgebra ${common}"
}

run_group_without_tensor_eigh eigh "${EIGH_SIZE}" "${EIGH_ITERATIONS}"

echo "Benchmark results written to ${RESULT_DIR}"
