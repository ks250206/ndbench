#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/.build}"
BIN="${BUILD_DIR}/ndbench-go"
RESULT_DIR="${RESULT_DIR:-${SCRIPT_DIR}/results}"
OUTPUT="${RESULT_DIR}/memory.tsv"

POINTS="${POINTS:-100000}"
MATVEC_SIZE="${MATVEC_SIZE:-512}"
MATMUL_SIZE="${MATMUL_SIZE:-256}"
COSINE_ITERATIONS="${COSINE_ITERATIONS:-1000}"
EIGH_SIZE="${EIGH_SIZE:-128}"
VECTOR_ITERATIONS="${VECTOR_ITERATIONS:-1000000}"

command -v go >/dev/null || {
	echo "Go is required: https://go.dev/dl/" >&2
	exit 1
}

mkdir -p "${BUILD_DIR}" "${RESULT_DIR}"
(
	cd "${SCRIPT_DIR}"
	go build -trimpath -ldflags="-s -w" -o "${BIN}" .
)
export GOMAXPROCS="${GOMAXPROCS:-1}"
printf 'backend\toperation\tpeak_rss_bytes\tpeak_rss_mib\n' > "${OUTPUT}"

measure() {
	local operation="$1"
	shift
	local stats_file
	stats_file="$(mktemp "${TMPDIR:-/tmp}/ndbench-go-memory.XXXXXX")"
	local rss_bytes

	if [[ "$(uname -s)" == "Darwin" ]]; then
		/usr/bin/time -l "${BIN}" --op "${operation}" "$@" \
			>/dev/null 2>"${stats_file}"
		rss_bytes="$(awk '/maximum resident set size/ { print $1; exit }' "${stats_file}")"
	else
		/usr/bin/time -v "${BIN}" --op "${operation}" "$@" \
			>/dev/null 2>"${stats_file}"
		local rss_kib
		rss_kib="$(awk -F: '/Maximum resident set size/ { gsub(/^[[:space:]]+/, "", $2); print $2; exit }' "${stats_file}")"
		rss_bytes=$((rss_kib * 1024))
	fi

	if [[ -z "${rss_bytes:-}" || "${rss_bytes}" == 0 ]]; then
		echo "could not parse peak RSS for gonum/${operation}" >&2
		sed -n '1,40p' "${stats_file}" >&2
		rm -f "${stats_file}"
		return 1
	fi
	local rss_mib
	rss_mib="$(awk -v bytes="${rss_bytes}" 'BEGIN { printf "%.3f", bytes / 1048576 }')"
	printf 'gonum\t%s\t%s\t%s\n' "${operation}" "${rss_bytes}" "${rss_mib}" | tee -a "${OUTPUT}"
	rm -f "${stats_file}"
}

measure vector2 --size 1 --iterations "${VECTOR_ITERATIONS}"
measure vector3 --size 1 --iterations "${VECTOR_ITERATIONS}"
measure affine2 --size "${POINTS}" --iterations 1
measure affine3 --size "${POINTS}" --iterations 1
measure matvec --size "${MATVEC_SIZE}" --iterations 1
measure matmul --size "${MATMUL_SIZE}" --iterations 1
measure cosine1024 --size 1024 --iterations "${COSINE_ITERATIONS}"
measure eigh --size "${EIGH_SIZE}" --iterations 1

echo "Memory results written to ${OUTPUT}"
