#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BIN="${SCRIPT_DIR}/.build/release/ndbench"
RESULT_DIR="${SCRIPT_DIR}/results"
OUTPUT="${RESULT_DIR}/memory.tsv"

POINTS="${POINTS:-100000}"
MATVEC_SIZE="${MATVEC_SIZE:-512}"
MATMUL_SIZE="${MATMUL_SIZE:-256}"
COSINE_ITERATIONS="${COSINE_ITERATIONS:-1000}"
EIGH_SIZE="${EIGH_SIZE:-128}"
VECTOR_ITERATIONS="${VECTOR_ITERATIONS:-1000000}"

swift build -c release --package-path "${SCRIPT_DIR}"
mkdir -p "${RESULT_DIR}"

export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export VECLIB_MAXIMUM_NUMBER_OF_THREADS="${VECLIB_MAXIMUM_NUMBER_OF_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

printf 'backend\toperation\tpeak_rss_bytes\tpeak_rss_mib\n' > "${OUTPUT}"

measure() {
    local operation="$1"
    shift
    local stats_file
    stats_file="$(mktemp "${TMPDIR:-/tmp}/ndbench-swift-memory.XXXXXX")"
    trap 'rm -f "${stats_file}"' RETURN

    if [[ "$(uname -s)" == "Darwin" ]]; then
        /usr/bin/time -l "${BIN}" --backend swift --op "${operation}" "$@" \
            >/dev/null 2>"${stats_file}"
        local rss_bytes
        rss_bytes="$(awk '/maximum resident set size/ { print $1; exit }' "${stats_file}")"
    else
        /usr/bin/time -v "${BIN}" --backend swift --op "${operation}" "$@" \
            >/dev/null 2>"${stats_file}"
        local rss_kib
        rss_kib="$(awk -F: '/Maximum resident set size/ { gsub(/^[[:space:]]+/, "", $2); print $2; exit }' "${stats_file}")"
        local rss_bytes=$((rss_kib * 1024))
    fi

    local rss_mib
    rss_mib="$(awk -v bytes="${rss_bytes}" 'BEGIN { printf "%.3f", bytes / 1048576 }')"
    printf 'swift\t%s\t%s\t%s\n' "${operation}" "${rss_bytes}" "${rss_mib}" | tee -a "${OUTPUT}"
}

measure vector2 --size 1 --iterations "${VECTOR_ITERATIONS}"
measure vector3 --size 1 --iterations "${VECTOR_ITERATIONS}"
measure affine2 --size "${POINTS}" --iterations 1
measure affine3 --size "${POINTS}" --iterations 1
measure matvec --size "${MATVEC_SIZE}" --iterations 1
measure matmul --size "${MATMUL_SIZE}" --iterations 1
measure cosine1024 --size 1024 --iterations "${COSINE_ITERATIONS}"
measure eigh --size "${EIGH_SIZE}" --iterations 1

echo "Swift memory results written to ${OUTPUT}"
