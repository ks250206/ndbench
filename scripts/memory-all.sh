#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

run_project() {
    local name="$1"
    local directory="$2"
    local script="${directory}/memory.sh"

    if [[ ! -x "${script}" ]]; then
        echo "${name}: executable memory script not found: ${script}" >&2
        exit 1
    fi

    echo "== ${name} =="
    "${script}"
}

run_project "Rust" "${REPO_DIR}/scripts"
run_project "Python (uv)" "${REPO_DIR}/python"
run_project "Julia" "${REPO_DIR}/julia"
run_project "Go" "${REPO_DIR}/go"
run_project "Mojo raw" "${REPO_DIR}/mojo"
run_project "Node.js / TypeScript (pnpm)" "${REPO_DIR}/js"
run_project "Swift" "${REPO_DIR}/swift"

echo "All memory results have been written to each project's results/memory.tsv."
