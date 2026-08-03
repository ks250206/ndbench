#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BIN="${REPO_DIR}/target/release/ndbench"
RESULT_DIR="${REPO_DIR}/results"
OUTPUT="${RESULT_DIR}/memory.tsv"

POINTS="${POINTS:-100000}"
MATVEC_SIZE="${MATVEC_SIZE:-512}"
MATMUL_SIZE="${MATMUL_SIZE:-256}"
COSINE_ITERATIONS="${COSINE_ITERATIONS:-1000}"
EIGH_SIZE="${EIGH_SIZE:-128}"
VECTOR_ITERATIONS="${VECTOR_ITERATIONS:-1000000}"

cargo build --release --features ndarray-eigh-openblas-static --manifest-path "${REPO_DIR}/Cargo.toml"
mkdir -p "${RESULT_DIR}"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export CANDLE_NUM_THREADS="${CANDLE_NUM_THREADS:-1}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"

printf 'backend\toperation\tpeak_rss_bytes\tpeak_rss_mib\n' > "${OUTPUT}"

measure() {
    local backend="$1"
    local operation="$2"
    shift 2
    local stats_file
    stats_file="$(mktemp "${TMPDIR:-/tmp}/ndbench-memory.XXXXXX")"

    if [[ "$(uname -s)" == "Darwin" ]]; then
        /usr/bin/time -l "${BIN}" --backend "${backend}" --op "${operation}" "$@" \
            >/dev/null 2>"${stats_file}"
        local rss_bytes
        rss_bytes="$(awk '/maximum resident set size/ { print $1; exit }' "${stats_file}")"
    else
        /usr/bin/time -v "${BIN}" --backend "${backend}" --op "${operation}" "$@" \
            >/dev/null 2>"${stats_file}"
        local rss_kib
        rss_kib="$(awk -F: '/Maximum resident set size/ { gsub(/^[[:space:]]+/, "", $2); print $2; exit }' "${stats_file}")"
        local rss_bytes=$((rss_kib * 1024))
    fi

    local rss_mib
    rss_mib="$(awk -v bytes="${rss_bytes}" 'BEGIN { printf "%.3f", bytes / 1048576 }')"
    printf '%s\t%s\t%s\t%s\n' "${backend}" "${operation}" "${rss_bytes}" "${rss_mib}" | tee -a "${OUTPUT}"
    rm -f "${stats_file}"
}

for backend in ndarray faer nalgebra candle burn raw; do
    measure "${backend}" vector2 --size 1 --iterations "${VECTOR_ITERATIONS}"
    measure "${backend}" vector3 --size 1 --iterations "${VECTOR_ITERATIONS}"
    measure "${backend}" affine2 --size "${POINTS}" --iterations 1
    measure "${backend}" affine3 --size "${POINTS}" --iterations 1
    measure "${backend}" matvec --size "${MATVEC_SIZE}" --iterations 1
    measure "${backend}" matmul --size "${MATMUL_SIZE}" --iterations 1
    measure "${backend}" cosine1024 --size 1024 --iterations "${COSINE_ITERATIONS}"
    if [[ "${backend}" != candle && "${backend}" != burn ]]; then
        measure "${backend}" eigh --size "${EIGH_SIZE}" --iterations 1
    fi
done

echo "Memory results written to ${OUTPUT}"
