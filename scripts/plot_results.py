#!/usr/bin/env python3
"""Generate the README benchmark charts from hyperfine and RSS results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


OPERATIONS = (
    "vector2",
    "vector3",
    "affine2",
    "affine3",
    "matvec",
    "matmul",
    "cosine1024",
    "eigh",
)
NON_EIGH_OPERATIONS = OPERATIONS[:-1]
OPERATION_LABELS = {
    "vector2": "vector2",
    "vector3": "vector3",
    "affine2": "affine2",
    "affine3": "affine3",
    "matvec": "matvec",
    "matmul": "matmul",
    "cosine1024": "cosine1024",
    "eigh": "eigh",
}

RUST_BACKENDS = ("ndarray", "faer", "nalgebra", "candle", "burn", "raw")
RUST_EIGH_BACKENDS = ("ndarray", "faer", "nalgebra", "raw")
PYTHON_BACKENDS = ("numpy", "pytorch", "raw")
PYTHON_LIBRARY_BACKENDS = ("numpy", "pytorch")
BACKEND_LABELS = {
    "ndarray": "ndarray",
    "faer": "faer",
    "nalgebra": "nalgebra",
    "candle": "Candle",
    "burn": "Burn",
    "raw": "raw Rust",
    "numpy": "NumPy",
    "pytorch": "PyTorch",
}
COLORS = {
    "ndarray": "#4C78A8",
    "faer": "#F58518",
    "nalgebra": "#54A24B",
    "candle": "#E45756",
    "burn": "#B279A2",
    "raw": "#FFBF79",
    "numpy": "#4C78A8",
    "pytorch": "#E45756",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root containing results/ and python/results/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for PNGs (default: <root>/docs/benchmarks)",
    )
    return parser.parse_args()


def read_speed_results(results_dir: Path, operations: Iterable[str]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for operation in operations:
        with (results_dir / f"{operation}.json").open(encoding="utf-8") as stream:
            data = json.load(stream)
        values[operation] = {
            item["command"]: float(item["median"]) * 1000.0 for item in data["results"]
        }
    return values


def read_rss_results(path: Path) -> dict[tuple[str, str], float]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {
            (row["backend"], row["operation"]): float(row["peak_rss_mib"])
            for row in csv.DictReader(stream, delimiter="\t")
        }


def series_from_results(
    values: dict[str, dict[str, float]],
    operations: Iterable[str],
    backends: Iterable[str],
) -> dict[str, list[float | None]]:
    return {
        backend: [values[operation].get(backend) for operation in operations]
        for backend in backends
    }


def series_from_rss(
    values: dict[tuple[str, str], float],
    operations: Iterable[str],
    backends: Iterable[str],
) -> dict[str, list[float | None]]:
    return {
        backend: [values.get((backend, operation)) for operation in operations]
        for backend in backends
    }


def finite_values(series: dict[str, list[float | None]]) -> list[float]:
    return [value for values in series.values() for value in values if value is not None]


def plot_grouped_bars(
    output_path: Path,
    title: str,
    operations: Iterable[str],
    series: dict[str, list[float | None]],
    ylabel: str,
    labels: dict[str, str] | None = None,
) -> None:
    operation_list = list(operations)
    backend_list = list(series)
    x_positions = list(range(len(operation_list)))
    group_width = 0.82
    bar_width = group_width / len(backend_list)

    fig, axis = plt.subplots(figsize=(14, 7.2))
    for index, backend in enumerate(backend_list):
        offset = (index - (len(backend_list) - 1) / 2) * bar_width
        positions = []
        heights = []
        for x_position, value in zip(x_positions, series[backend]):
            if value is not None:
                positions.append(x_position + offset)
                heights.append(value)
        axis.bar(
            positions,
            heights,
            width=bar_width * 0.9,
            label=(labels or {}).get(backend, BACKEND_LABELS[backend]),
            color=COLORS[backend],
            edgecolor="#333333",
            linewidth=0.35,
        )

    fig.suptitle(title, fontsize=16, y=0.98)
    axis.set_ylabel(ylabel)
    axis.set_xticks(x_positions)
    axis.set_xticklabels([OPERATION_LABELS[operation] for operation in operation_list])
    axis.yaxis.set_major_locator(MaxNLocator(nbins=8))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=min(3, len(backend_list)),
        frameon=False,
    )

    maximum = max(finite_values(series))
    axis.set_ylim(0, maximum * 1.16)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.74)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "docs" / "benchmarks").resolve()

    rust_speed = read_speed_results(root / "results", OPERATIONS)
    python_speed = read_speed_results(root / "python" / "results", OPERATIONS)
    rust_rss = read_rss_results(root / "results" / "memory.tsv")
    python_rss = read_rss_results(root / "python" / "results" / "memory.tsv")

    plot_grouped_bars(
        output_dir / "rust-speed.png",
        "Rust CPU speed: median ms (lower is better)",
        NON_EIGH_OPERATIONS,
        series_from_results(rust_speed, NON_EIGH_OPERATIONS, RUST_BACKENDS),
        "milliseconds",
    )
    plot_grouped_bars(
        output_dir / "rust-eigh.png",
        "Rust eigh: median ms (lower is better)",
        ("eigh",),
        series_from_results(rust_speed, ("eigh",), RUST_EIGH_BACKENDS),
        "milliseconds",
    )
    plot_grouped_bars(
        output_dir / "python-speed.png",
        "Python CPU speed: median ms (lower is better)",
        OPERATIONS,
        series_from_results(python_speed, OPERATIONS, PYTHON_LIBRARY_BACKENDS),
        "milliseconds",
    )
    plot_grouped_bars(
        output_dir / "python-raw-speed.png",
        "Raw Python speed: median ms (lower is better)",
        OPERATIONS,
        series_from_results(python_speed, OPERATIONS, ("raw",)),
        "milliseconds",
        labels={"raw": "raw Python"},
    )
    plot_grouped_bars(
        output_dir / "rust-rss.png",
        "Rust peak RSS",
        OPERATIONS,
        series_from_rss(rust_rss, OPERATIONS, RUST_BACKENDS),
        "MiB",
    )
    plot_grouped_bars(
        output_dir / "python-rss.png",
        "Python peak RSS",
        OPERATIONS,
        series_from_rss(python_rss, OPERATIONS, PYTHON_BACKENDS),
        "MiB",
        labels={"raw": "raw Python"},
    )

    print(f"Wrote benchmark plots to {output_dir}")


if __name__ == "__main__":
    main()
