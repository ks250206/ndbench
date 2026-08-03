#!/usr/bin/env python3
"""Small, reproducible NumPy/PyTorch CPU companion for the Rust ndbench CLI."""

from __future__ import annotations

import argparse
from typing import Any


DEFAULT_VECTOR_ITERATIONS = 1_000_000
DEFAULT_POINT_COUNT = 100_000
DEFAULT_MATRIX_SIZE = 128
COSINE_DIMENSION = 1024
DEFAULT_COSINE_ITERATIONS = 1_000

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


def scalar(i: int, j: int, salt: int) -> float:
    """Return the same deterministic input value as the Rust benchmark."""

    value = (i * 37 + j * 17 + salt * 13) % 101
    return 0.125 + value / 101.0


def embedding_scalar(index: int, lane: int) -> float:
    """Return one deterministic f64 embedding component, matching Rust."""

    return scalar(index, lane, COSINE_DIMENSION + lane) * 2.0 - 1.0


def symmetric_scalar(i: int, j: int, size: int) -> float:
    if i == j:
        return float(size + 2)
    distance = abs(i - j)
    return 0.01 * scalar(min(i, j), max(i, j), size) / (1.0 + distance)


def affine_values(dimension: int) -> tuple[float, ...]:
    if dimension == 2:
        return (1.1, -0.2, 0.5, 0.3, 0.9, -0.7, 0.0, 0.0, 1.0)
    if dimension == 3:
        return (
            1.1,
            -0.2,
            0.1,
            0.5,
            0.3,
            0.9,
            -0.15,
            -0.7,
            0.05,
            0.2,
            1.05,
            0.4,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    raise ValueError(f"unsupported affine dimension: {dimension}")


def tiny_term(iteration: int) -> float:
    return (iteration & 7) * 1.0e-12


def _numpy_scalar_line(np: Any, length: int, j: int, salt: int) -> Any:
    indices = np.arange(length, dtype=np.int64)
    values = indices * 37 + j * 17 + salt * 13
    return 0.125 + np.remainder(values, 101).astype(np.float64) / 101.0


def _numpy_scalar_matrix(np: Any, size: int, salt: int) -> Any:
    indices = np.arange(size, dtype=np.int64)
    rows = indices[:, None]
    columns = indices[None, :]
    values = rows * 37 + columns * 17 + salt * 13
    matrix = 0.125 + np.remainder(values, 101).astype(np.float64) / 101.0
    return np.asfortranarray(matrix)


def _numpy_affine_input(np: Any, points: int, dimension: int) -> Any:
    homogeneous = dimension + 1
    result = np.empty((homogeneous, points), dtype=np.float64, order="F")
    for row in range(dimension):
        result[row, :] = _numpy_scalar_line(np, points, row, dimension)
    result[dimension, :] = 1.0
    return result


def _numpy_affine_transform(np: Any, dimension: int) -> Any:
    homogeneous = dimension + 1
    values = np.asarray(affine_values(dimension), dtype=np.float64)
    return np.asfortranarray(values.reshape((homogeneous, homogeneous), order="C"))


def _numpy_symmetric_matrix(np: Any, size: int) -> Any:
    indices = np.arange(size, dtype=np.int64)
    rows = indices[:, None]
    columns = indices[None, :]
    minimum = np.minimum(rows, columns)
    maximum = np.maximum(rows, columns)
    distance = np.abs(rows - columns).astype(np.float64)
    values = minimum * 37 + maximum * 17 + size * 13
    off_diagonal = (
        0.01
        * (0.125 + np.remainder(values, 101).astype(np.float64) / 101.0)
        / (1.0 + distance)
    )
    matrix = np.where(rows == columns, float(size + 2), off_diagonal)
    return np.asfortranarray(matrix)


def _numpy_vector2(np: Any, iterations: int) -> float:
    a = np.asarray((1.25, -2.5), dtype=np.float64)
    b = np.asarray((-0.75, 3.0), dtype=np.float64)
    checksum = 0.0

    for iteration in range(iterations):
        summed = a + b
        dot = np.dot(a, b)
        norm = np.sqrt(np.dot(summed, summed))
        checksum += float(summed[iteration & 1] + dot + norm + tiny_term(iteration))

    return checksum


def _numpy_vector3(np: Any, iterations: int) -> float:
    a = np.asarray((1.25, -2.5, 0.75), dtype=np.float64)
    b = np.asarray((-0.75, 3.0, 1.5), dtype=np.float64)
    checksum = 0.0

    for iteration in range(iterations):
        summed = a + b
        dot = np.dot(a, b)
        norm = np.sqrt(np.dot(summed, summed))
        cross = np.asarray(
            (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ),
            dtype=np.float64,
        )
        checksum += float(
            summed[iteration % 3]
            + dot
            + norm
            + cross[(iteration + 1) % 3]
            + tiny_term(iteration)
        )

    return checksum


def _numpy_affine(np: Any, points: int, iterations: int, dimension: int) -> float:
    transform = _numpy_affine_transform(np, dimension)
    input_points = _numpy_affine_input(np, points, dimension)
    checksum = 0.0

    for _ in range(iterations):
        output = transform @ input_points
        checksum += float(output.sum() + output[0, 0] + output[dimension, points - 1])

    return checksum


def _numpy_matvec(np: Any, size: int, iterations: int) -> float:
    matrix = _numpy_scalar_matrix(np, size, size)
    vector = _numpy_scalar_line(np, size, 0, size + 1)
    checksum = 0.0

    for _ in range(iterations):
        output = matrix @ vector
        checksum += float(output.sum() + output[0])

    return checksum


def _numpy_matmul(np: Any, size: int, iterations: int) -> float:
    left = _numpy_scalar_matrix(np, size, size)
    right = _numpy_scalar_matrix(np, size, size + 1)
    checksum = 0.0

    for _ in range(iterations):
        output = left @ right
        checksum += float(output.sum() + output[0, 0])

    return checksum


def _numpy_eigh(np: Any, size: int, iterations: int) -> float:
    matrix = _numpy_symmetric_matrix(np, size)
    checksum = 0.0

    for _ in range(iterations):
        eigenvalues, eigenvectors = np.linalg.eigh(matrix, UPLO="L")
        eigenvector_norm_squared = np.square(eigenvectors).sum()
        checksum += float(eigenvalues.sum() + eigenvector_norm_squared)

    return checksum


def _numpy_cosine1024(np: Any, iterations: int) -> float:
    a = np.asarray(
        [embedding_scalar(index, 0) for index in range(COSINE_DIMENSION)],
        dtype=np.float64,
    )
    b = np.asarray(
        [embedding_scalar(index, 1) for index in range(COSINE_DIMENSION)],
        dtype=np.float64,
    )
    checksum = 0.0

    for iteration in range(iterations):
        dot = np.dot(a, b)
        norm_a = np.sqrt(np.dot(a, a))
        norm_b = np.sqrt(np.dot(b, b))
        similarity = dot / (norm_a * norm_b)
        checksum += float(similarity + tiny_term(iteration))

    return checksum


def run_numpy(operation: str, size: int, iterations: int) -> float:
    """Run one operation with NumPy, importing it only for this backend."""

    import numpy as np

    if operation == "vector2":
        return _numpy_vector2(np, iterations)
    if operation == "vector3":
        return _numpy_vector3(np, iterations)
    if operation == "affine2":
        return _numpy_affine(np, size, iterations, 2)
    if operation == "affine3":
        return _numpy_affine(np, size, iterations, 3)
    if operation == "matvec":
        return _numpy_matvec(np, size, iterations)
    if operation == "matmul":
        return _numpy_matmul(np, size, iterations)
    if operation == "cosine1024":
        return _numpy_cosine1024(np, iterations)
    if operation == "eigh":
        return _numpy_eigh(np, size, iterations)
    raise ValueError(f"unknown operation: {operation}")


def _configure_torch(torch: Any) -> None:
    # These must be set before doing tensor work. The benchmark is CPU-only,
    # and the shell scripts also constrain common BLAS environment variables.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _torch_scalar_line(torch: Any, length: int, j: int, salt: int) -> Any:
    indices = torch.arange(length, dtype=torch.int64, device="cpu")
    values = indices * 37 + j * 17 + salt * 13
    return 0.125 + torch.remainder(values, 101).to(torch.float64) / 101.0


def _torch_scalar_matrix(torch: Any, size: int, salt: int) -> Any:
    indices = torch.arange(size, dtype=torch.int64, device="cpu")
    rows = indices[:, None]
    columns = indices[None, :]
    values = rows * 37 + columns * 17 + salt * 13
    return 0.125 + torch.remainder(values, 101).to(torch.float64) / 101.0


def _torch_affine_input(torch: Any, points: int, dimension: int) -> Any:
    homogeneous = dimension + 1
    result = torch.empty((homogeneous, points), dtype=torch.float64, device="cpu")
    for row in range(dimension):
        result[row, :] = _torch_scalar_line(torch, points, row, dimension)
    result[dimension, :] = 1.0
    return result


def _torch_affine_transform(torch: Any, dimension: int) -> Any:
    homogeneous = dimension + 1
    return torch.tensor(affine_values(dimension), dtype=torch.float64, device="cpu").reshape(
        (homogeneous, homogeneous)
    )


def _torch_symmetric_matrix(torch: Any, size: int) -> Any:
    indices = torch.arange(size, dtype=torch.int64, device="cpu")
    rows = indices[:, None]
    columns = indices[None, :]
    minimum = torch.minimum(rows, columns)
    maximum = torch.maximum(rows, columns)
    distance = (rows - columns).abs().to(torch.float64)
    values = minimum * 37 + maximum * 17 + size * 13
    matrix = (
        0.01
        * (0.125 + torch.remainder(values, 101).to(torch.float64) / 101.0)
        / (1.0 + distance)
    )
    matrix.diagonal().fill_(float(size + 2))
    return matrix


def _torch_vector2(torch: Any, iterations: int) -> float:
    a = torch.tensor((1.25, -2.5), dtype=torch.float64, device="cpu")
    b = torch.tensor((-0.75, 3.0), dtype=torch.float64, device="cpu")
    checksum = 0.0

    for iteration in range(iterations):
        summed = a + b
        dot = torch.dot(a, b)
        norm = torch.sqrt(torch.dot(summed, summed))
        checksum += float((summed[iteration & 1] + dot + norm + tiny_term(iteration)).item())

    return checksum


def _torch_vector3(torch: Any, iterations: int) -> float:
    a = torch.tensor((1.25, -2.5, 0.75), dtype=torch.float64, device="cpu")
    b = torch.tensor((-0.75, 3.0, 1.5), dtype=torch.float64, device="cpu")
    checksum = 0.0

    for iteration in range(iterations):
        summed = a + b
        dot = torch.dot(a, b)
        norm = torch.sqrt(torch.dot(summed, summed))
        cross = torch.stack(
            (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )
        )
        checksum += float(
            (
                summed[iteration % 3]
                + dot
                + norm
                + cross[(iteration + 1) % 3]
                + tiny_term(iteration)
            ).item()
        )

    return checksum


def _torch_affine(torch: Any, points: int, iterations: int, dimension: int) -> float:
    transform = _torch_affine_transform(torch, dimension)
    input_points = _torch_affine_input(torch, points, dimension)
    checksum = 0.0

    for _ in range(iterations):
        output = transform @ input_points
        checksum += float(
            (torch.sum(output) + output[0, 0] + output[dimension, points - 1]).item()
        )

    return checksum


def _torch_matvec(torch: Any, size: int, iterations: int) -> float:
    matrix = _torch_scalar_matrix(torch, size, size)
    vector = _torch_scalar_line(torch, size, 0, size + 1)
    checksum = 0.0

    for _ in range(iterations):
        output = matrix @ vector
        checksum += float((torch.sum(output) + output[0]).item())

    return checksum


def _torch_matmul(torch: Any, size: int, iterations: int) -> float:
    left = _torch_scalar_matrix(torch, size, size)
    right = _torch_scalar_matrix(torch, size, size + 1)
    checksum = 0.0

    for _ in range(iterations):
        output = left @ right
        checksum += float((torch.sum(output) + output[0, 0]).item())

    return checksum


def _torch_eigh(torch: Any, size: int, iterations: int) -> float:
    matrix = _torch_symmetric_matrix(torch, size)
    checksum = 0.0

    for _ in range(iterations):
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix, UPLO="L")
        eigenvector_norm_squared = torch.sum(eigenvectors * eigenvectors)
        checksum += float((torch.sum(eigenvalues) + eigenvector_norm_squared).item())

    return checksum


def _torch_cosine1024(torch: Any, iterations: int) -> float:
    a = torch.tensor(
        [embedding_scalar(index, 0) for index in range(COSINE_DIMENSION)],
        dtype=torch.float64,
        device="cpu",
    )
    b = torch.tensor(
        [embedding_scalar(index, 1) for index in range(COSINE_DIMENSION)],
        dtype=torch.float64,
        device="cpu",
    )
    checksum = 0.0

    for iteration in range(iterations):
        dot = torch.dot(a, b)
        norm_a = torch.sqrt(torch.dot(a, a))
        norm_b = torch.sqrt(torch.dot(b, b))
        similarity = dot / (norm_a * norm_b)
        checksum += float((similarity + tiny_term(iteration)).item())

    return checksum


def run_pytorch(operation: str, size: int, iterations: int) -> float:
    """Run one operation with CPU-only PyTorch and no autograd bookkeeping."""

    import torch

    _configure_torch(torch)
    with torch.inference_mode():
        if operation == "vector2":
            return _torch_vector2(torch, iterations)
        if operation == "vector3":
            return _torch_vector3(torch, iterations)
        if operation == "affine2":
            return _torch_affine(torch, size, iterations, 2)
        if operation == "affine3":
            return _torch_affine(torch, size, iterations, 3)
        if operation == "matvec":
            return _torch_matvec(torch, size, iterations)
        if operation == "matmul":
            return _torch_matmul(torch, size, iterations)
        if operation == "cosine1024":
            return _torch_cosine1024(torch, iterations)
        if operation == "eigh":
            return _torch_eigh(torch, size, iterations)
    raise ValueError(f"unknown operation: {operation}")


def normalize_operation(value: str) -> str:
    aliases = {"vec2": "vector2", "vec3": "vector3", "diagonalize": "eigh"}
    operation = aliases.get(value, value)
    if operation not in OPERATIONS:
        choices = ", ".join(OPERATIONS)
        raise argparse.ArgumentTypeError(f"unknown operation {value!r}; expected {choices}")
    return operation


def default_size(operation: str) -> int:
    if operation in {"vector2", "vector3"}:
        return 1
    if operation in {"affine2", "affine3"}:
        return DEFAULT_POINT_COUNT
    if operation == "cosine1024":
        return COSINE_DIMENSION
    return DEFAULT_MATRIX_SIZE


def default_iterations(operation: str) -> int:
    if operation in {"vector2", "vector3"}:
        return DEFAULT_VECTOR_ITERATIONS
    if operation == "cosine1024":
        return DEFAULT_COSINE_ITERATIONS
    return 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ndbench NumPy/PyTorch CPU comparison for one operation."
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=("numpy", "pytorch", "torch"),
        help="numeric backend (torch is an alias for pytorch)",
    )
    parser.add_argument(
        "--op",
        "--operation",
        required=True,
        type=normalize_operation,
        help="operation: vector2, vector3, affine2, affine3, matvec, matmul, cosine1024, or eigh",
    )
    parser.add_argument("--size", type=int, help="point count or square matrix order")
    parser.add_argument("--iterations", type=int, help="number of repeated operations")
    args = parser.parse_args(argv)
    args.backend = "pytorch" if args.backend == "torch" else args.backend
    if args.size is None:
        args.size = default_size(args.op)
    if args.iterations is None:
        args.iterations = default_iterations(args.op)
    if args.size <= 0:
        parser.error("--size must be greater than zero")
    if args.iterations <= 0:
        parser.error("--iterations must be greater than zero")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.backend == "numpy":
        checksum = run_numpy(args.op, args.size, args.iterations)
    else:
        checksum = run_pytorch(args.op, args.size, args.iterations)
    print(f"checksum={checksum:.17e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
