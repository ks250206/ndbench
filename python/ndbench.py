#!/usr/bin/env python3
"""Small, reproducible NumPy/PyTorch CPU companion for the Rust ndbench CLI."""

from __future__ import annotations

import argparse
import math
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


JACOBI_TOLERANCE = 1.0e-12


def _raw_vector2(iterations: int) -> float:
    a = [1.25, -2.5]
    b = [-0.75, 3.0]
    checksum = 0.0

    for iteration in range(iterations):
        summed = [a[0] + b[0], a[1] + b[1]]
        dot = a[0] * b[0] + a[1] * b[1]
        norm = math.sqrt(summed[0] * summed[0] + summed[1] * summed[1])
        checksum += summed[iteration & 1] + dot + norm + tiny_term(iteration)

    return checksum


def _raw_vector3(iterations: int) -> float:
    a = [1.25, -2.5, 0.75]
    b = [-0.75, 3.0, 1.5]
    checksum = 0.0

    for iteration in range(iterations):
        summed = [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
        dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        norm = math.sqrt(
            summed[0] * summed[0] + summed[1] * summed[1] + summed[2] * summed[2]
        )
        cross = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
        checksum += (
            summed[iteration % 3]
            + dot
            + norm
            + cross[(iteration + 1) % 3]
            + tiny_term(iteration)
        )

    return checksum


def _raw_affine(points: int, iterations: int, dimension: int) -> float:
    homogeneous = dimension + 1
    transform = list(affine_values(dimension))
    input_points = [0.0] * (homogeneous * points)
    for row in range(homogeneous):
        for column in range(points):
            input_points[row * points + column] = (
                1.0 if row == dimension else scalar(column, row, dimension)
            )
    checksum = 0.0

    for _ in range(iterations):
        output = [0.0] * (homogeneous * points)
        for row in range(homogeneous):
            for column in range(points):
                value = 0.0
                for inner in range(homogeneous):
                    value += transform[row * homogeneous + inner] * input_points[
                        inner * points + column
                    ]
                output[row * points + column] = value
        output_sum = 0.0
        for value in output:
            output_sum += value
        checksum += output_sum + output[0] + output[dimension * points + points - 1]

    return checksum


def _raw_scalar_matrix(size: int, salt: int) -> list[float]:
    matrix = [0.0] * (size * size)
    for row in range(size):
        for column in range(size):
            matrix[row * size + column] = scalar(row, column, salt)
    return matrix


def _raw_matvec(size: int, iterations: int) -> float:
    matrix = _raw_scalar_matrix(size, size)
    vector = [scalar(row, 0, size + 1) for row in range(size)]
    checksum = 0.0

    for _ in range(iterations):
        output = [0.0] * size
        for row in range(size):
            value = 0.0
            for column in range(size):
                value += matrix[row * size + column] * vector[column]
            output[row] = value
        output_sum = 0.0
        for value in output:
            output_sum += value
        checksum += output_sum + output[0]

    return checksum


def _raw_matmul(size: int, iterations: int) -> float:
    left = _raw_scalar_matrix(size, size)
    right = _raw_scalar_matrix(size, size + 1)
    checksum = 0.0

    for _ in range(iterations):
        output = [0.0] * (size * size)
        for row in range(size):
            for column in range(size):
                value = 0.0
                for inner in range(size):
                    value += left[row * size + inner] * right[inner * size + column]
                output[row * size + column] = value
        output_sum = 0.0
        for value in output:
            output_sum += value
        checksum += output_sum + output[0]

    return checksum


def _raw_cosine1024(iterations: int) -> float:
    a = [embedding_scalar(index, 0) for index in range(COSINE_DIMENSION)]
    b = [embedding_scalar(index, 1) for index in range(COSINE_DIMENSION)]
    checksum = 0.0

    for iteration in range(iterations):
        dot = 0.0
        norm_a_squared = 0.0
        norm_b_squared = 0.0
        for index in range(COSINE_DIMENSION):
            dot += a[index] * b[index]
            norm_a_squared += a[index] * a[index]
            norm_b_squared += b[index] * b[index]
        similarity = dot / (math.sqrt(norm_a_squared) * math.sqrt(norm_b_squared))
        checksum += similarity + tiny_term(iteration)

    return checksum


def _raw_symmetric_matrix(size: int) -> list[float]:
    matrix = [0.0] * (size * size)
    for row in range(size):
        for column in range(size):
            if row == column:
                matrix[row * size + column] = float(size + 2)
            else:
                distance = float(abs(row - column))
                matrix[row * size + column] = (
                    0.01
                    * scalar(min(row, column), max(row, column), size)
                    / (1.0 + distance)
                )
    return matrix


def _raw_jacobi_eigh(matrix_input: list[float], size: int) -> tuple[list[float], list[float]]:
    matrix = list(matrix_input)
    eigenvectors = [0.0] * (size * size)
    for index in range(size):
        eigenvectors[index * size + index] = 1.0

    for _ in range(8 * max(size, 1)):
        max_off_diagonal = 0.0
        for row in range(size):
            for column in range(row + 1, size):
                max_off_diagonal = max(
                    max_off_diagonal, abs(matrix[row * size + column])
                )
        if max_off_diagonal < JACOBI_TOLERANCE:
            break

        for p in range(size):
            for q in range(p + 1, size):
                apq = matrix[p * size + q]
                if abs(apq) < JACOBI_TOLERANCE:
                    continue

                app = matrix[p * size + p]
                aqq = matrix[q * size + q]
                tau = (aqq - app) / (2.0 * apq)
                if tau >= 0.0:
                    t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
                cosine = 1.0 / math.sqrt(1.0 + t * t)
                sine = t * cosine

                for k in range(size):
                    if k == p or k == q:
                        continue
                    akp = matrix[k * size + p]
                    akq = matrix[k * size + q]
                    new_kp = cosine * akp - sine * akq
                    new_kq = sine * akp + cosine * akq
                    matrix[k * size + p] = new_kp
                    matrix[p * size + k] = new_kp
                    matrix[k * size + q] = new_kq
                    matrix[q * size + k] = new_kq

                matrix[p * size + p] = (
                    cosine * cosine * app
                    - 2.0 * sine * cosine * apq
                    + sine * sine * aqq
                )
                matrix[q * size + q] = (
                    sine * sine * app
                    + 2.0 * sine * cosine * apq
                    + cosine * cosine * aqq
                )
                matrix[p * size + q] = 0.0
                matrix[q * size + p] = 0.0

                for k in range(size):
                    vkp = eigenvectors[k * size + p]
                    vkq = eigenvectors[k * size + q]
                    eigenvectors[k * size + p] = cosine * vkp - sine * vkq
                    eigenvectors[k * size + q] = sine * vkp + cosine * vkq

    eigenvalues = [matrix[index * size + index] for index in range(size)]
    return eigenvalues, eigenvectors


def _raw_eigh(size: int, iterations: int) -> float:
    matrix = _raw_symmetric_matrix(size)
    checksum = 0.0

    for _ in range(iterations):
        eigenvalues, eigenvectors = _raw_jacobi_eigh(matrix, size)
        eigenvalue_sum = 0.0
        for value in eigenvalues:
            eigenvalue_sum += value
        eigenvector_norm_squared = 0.0
        for value in eigenvectors:
            eigenvector_norm_squared += value * value
        checksum += eigenvalue_sum + eigenvector_norm_squared

    return checksum


def run_raw(operation: str, size: int, iterations: int) -> float:
    """Run the dependency-free Python baseline using lists and ``math`` only."""

    if operation == "vector2":
        return _raw_vector2(iterations)
    if operation == "vector3":
        return _raw_vector3(iterations)
    if operation == "affine2":
        return _raw_affine(size, iterations, 2)
    if operation == "affine3":
        return _raw_affine(size, iterations, 3)
    if operation == "matvec":
        return _raw_matvec(size, iterations)
    if operation == "matmul":
        return _raw_matmul(size, iterations)
    if operation == "cosine1024":
        return _raw_cosine1024(iterations)
    if operation == "eigh":
        return _raw_eigh(size, iterations)
    raise ValueError(f"unknown operation: {operation}")


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
        choices=("numpy", "pytorch", "torch", "raw", "native"),
        help="numeric backend (torch is an alias for pytorch; native is an alias for raw)",
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
    if args.backend == "torch":
        args.backend = "pytorch"
    elif args.backend == "native":
        args.backend = "raw"
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
    elif args.backend == "pytorch":
        checksum = run_pytorch(args.op, args.size, args.iterations)
    else:
        checksum = run_raw(args.op, args.size, args.iterations)
    print(f"checksum={checksum:.17e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
