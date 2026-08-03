#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/target"
BINARY="${BUILD_DIR}/ndbench-mojo"

UV_BIN="${UV_BIN:-uv}"
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
RESULT_DIR="${SCRIPT_DIR}/results"
OUTPUT="${RESULT_DIR}/memory.tsv"

command -v "${UV_BIN}" >/dev/null || {
    echo "uv is required: https://docs.astral.sh/uv/" >&2
    exit 1
}

mkdir -p "${BUILD_DIR}" "${RESULT_DIR}"
cd "${SCRIPT_DIR}"
"${UV_BIN}" run --project . mojo build --optimization-level 3 -D ASSERT=none main.mojo -o "${BINARY}"

printf 'backend\toperation\tpeak_rss_bytes\tpeak_rss_mib\n' > "${OUTPUT}"

run_memory() {
    local operation="$1"
    local size="$2"
    local iterations="$3"
    local stats_file
    stats_file="$(mktemp "${TMPDIR:-/tmp}/ndbench-mojo-memory.XXXXXX")"
    local rss_bytes=""

    if [[ "$(uname -s)" == "Darwin" ]]; then
        /usr/bin/time -l "${BINARY}" --op "${operation}" --size "${size}" --iterations "${iterations}" \
            >/dev/null 2>"${stats_file}"
        rss_bytes="$(awk '/maximum resident set size/ { print $1; exit }' "${stats_file}")"
    elif [[ -x /usr/bin/time ]]; then
        /usr/bin/time -v "${BINARY}" --op "${operation}" --size "${size}" --iterations "${iterations}" \
            >/dev/null 2>"${stats_file}"
        local rss_kib
        rss_kib="$(awk -F: '/Maximum resident set size/ { gsub(/^[[:space:]]+/, "", $2); print $2; exit }' "${stats_file}")"
        if [[ -n "${rss_kib}" ]]; then
            rss_bytes=$((rss_kib * 1024))
        fi
    else
        echo "peak RSS measurement requires /usr/bin/time" >&2
        rm -f "${stats_file}"
        return 1
    fi

    if [[ -z "${rss_bytes}" || "${rss_bytes}" == "0" ]]; then
        echo "could not parse peak RSS for mojo/${operation}" >&2
        sed -n '1,40p' "${stats_file}" >&2
        rm -f "${stats_file}"
        return 1
    fi

    local rss_mib
    rss_mib="$(awk -v bytes="${rss_bytes}" 'BEGIN { printf "%.3f", bytes / 1048576 }')"
    printf 'mojo\t%s\t%s\t%s\n' "${operation}" "${rss_bytes}" "${rss_mib}" | tee -a "${OUTPUT}"
    rm -f "${stats_file}"
}

run_memory vector2 8 "${VECTOR_ITERATIONS}"
run_memory vector3 8 "${VECTOR_ITERATIONS}"
run_memory affine2 "${POINTS}" 1
run_memory affine3 "${POINTS}" 1
run_memory matvec "${MATVEC_SIZE}" 1
run_memory matmul "${MATMUL_SIZE}" 1
run_memory cosine1024 1024 "${COSINE_ITERATIONS}"
run_memory eigh "${EIGH_SIZE}" 1

echo "Memory results written to ${OUTPUT}"
