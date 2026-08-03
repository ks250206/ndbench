#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}/results"
ENTRYPOINT="${SCRIPT_DIR}/dist/ndbench.js"

PNPM_BIN="${PNPM_BIN:-pnpm}"
NODE_BIN="${NODE_BIN:-node}"
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

command -v "${PNPM_BIN}" >/dev/null || {
    echo "pnpm is required: https://pnpm.io/" >&2
    exit 1
}
command -v "${NODE_BIN}" >/dev/null || {
    echo "node is required: https://nodejs.org/" >&2
    exit 1
}
command -v hyperfine >/dev/null || {
    echo "hyperfine is required: https://github.com/sharkdp/hyperfine" >&2
    exit 1
}

cd "${SCRIPT_DIR}"
"${PNPM_BIN}" install --frozen-lockfile
"${PNPM_BIN}" build
mkdir -p "${RESULT_DIR}"

run_group() {
    local operation="$1"
    local size="$2"
    local iterations="$3"
    local qnode qentry
    printf -v qnode '%q' "${NODE_BIN}"
    printf -v qentry '%q' "${ENTRYPOINT}"
    local common="--op ${operation} --size ${size} --iterations ${iterations}"

    # This intentionally times the compiled CLI, including Node/V8 startup,
    # module loading, input creation, and the operation itself.
    hyperfine \
        --shell=none \
        --warmup "${HYPERFINE_WARMUP}" \
        --runs "${HYPERFINE_RUNS}" \
        --export-markdown "${RESULT_DIR}/${operation}.md" \
        --export-json "${RESULT_DIR}/${operation}.json" \
        --command-name raw "${qnode} ${qentry} --backend raw ${common}" \
        --command-name ml-matrix "${qnode} ${qentry} --backend ml-matrix ${common}"
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
