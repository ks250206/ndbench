#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${SCRIPT_DIR}/results"
ENTRYPOINT="${SCRIPT_DIR}/dist/ndbench.js"

PNPM_BIN="${PNPM_BIN:-pnpm}"
NODE_BIN="${NODE_BIN:-node}"
POINTS="${POINTS:-100000}"
MATVEC_SIZE="${MATVEC_SIZE:-512}"
MATMUL_SIZE="${MATMUL_SIZE:-256}"
COSINE_ITERATIONS="${COSINE_ITERATIONS:-1000}"
EIGH_SIZE="${EIGH_SIZE:-128}"
VECTOR_ITERATIONS="${VECTOR_ITERATIONS:-1000000}"

command -v "${PNPM_BIN}" >/dev/null || {
    echo "pnpm is required: https://pnpm.io/" >&2
    exit 1
}
command -v "${NODE_BIN}" >/dev/null || {
    echo "node is required: https://nodejs.org/" >&2
    exit 1
}

cd "${SCRIPT_DIR}"
"${PNPM_BIN}" install --frozen-lockfile
"${PNPM_BIN}" build
mkdir -p "${RESULT_DIR}"
OUTPUT="${RESULT_DIR}/memory.tsv"
printf 'backend\toperation\tpeak_rss_bytes\tpeak_rss_mib\n' > "${OUTPUT}"

measure() {
    local backend="$1"
    local operation="$2"
    shift 2
    local stats_file
    stats_file="$(mktemp "${TMPDIR:-/tmp}/ndbench-js-memory.XXXXXX")"

    if [[ "$(uname -s)" == "Darwin" ]]; then
        /usr/bin/time -l \
            "${NODE_BIN}" "${ENTRYPOINT}" --backend "${backend}" --op "${operation}" "$@" \
            >/dev/null 2>"${stats_file}"
        local rss_bytes
        rss_bytes="$(awk '/maximum resident set size/ { print $1; exit }' "${stats_file}")"
    else
        /usr/bin/time -v \
            "${NODE_BIN}" "${ENTRYPOINT}" --backend "${backend}" --op "${operation}" "$@" \
            >/dev/null 2>"${stats_file}"
        local rss_kib
        rss_kib="$(awk -F: '/Maximum resident set size/ { gsub(/^[[:space:]]+/, "", $2); print $2; exit }' "${stats_file}")"
        local rss_bytes=$((rss_kib * 1024))
    fi

    if [[ -z "${rss_bytes:-}" || "${rss_bytes}" == "0" ]]; then
        echo "could not parse peak RSS for ${backend}/${operation}" >&2
        sed -n '1,40p' "${stats_file}" >&2
        rm -f "${stats_file}"
        return 1
    fi
    local rss_mib
    rss_mib="$(awk -v bytes="${rss_bytes}" 'BEGIN { printf "%.3f", bytes / 1048576 }')"
    printf '%s\t%s\t%s\t%s\n' "${backend}" "${operation}" "${rss_bytes}" "${rss_mib}" | tee -a "${OUTPUT}"
    rm -f "${stats_file}"
}

for backend in raw ml-matrix; do
    measure "${backend}" vector2 --size 1 --iterations "${VECTOR_ITERATIONS}"
    measure "${backend}" vector3 --size 1 --iterations "${VECTOR_ITERATIONS}"
    measure "${backend}" affine2 --size "${POINTS}" --iterations 1
    measure "${backend}" affine3 --size "${POINTS}" --iterations 1
    measure "${backend}" matvec --size "${MATVEC_SIZE}" --iterations 1
    measure "${backend}" matmul --size "${MATMUL_SIZE}" --iterations 1
    measure "${backend}" cosine1024 --size 1024 --iterations "${COSINE_ITERATIONS}"
    measure "${backend}" eigh --size "${EIGH_SIZE}" --iterations 1
done

echo "Memory results written to ${OUTPUT}"
