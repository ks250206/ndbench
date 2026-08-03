//! Burn's CPU `burn-ndarray` backend for the ndbench operations.
//!
//! The parent crate should add `burn = { version = "0.21.0", default-features = false,
//! features = ["std", "ndarray", "simd"] }` and include this file as
//! `mod burn_backend;`.
//! `NdArray<f64>` is used explicitly so the benchmark does not silently fall back to
//! Burn's default `f32` dtype.

use std::hint::black_box;

use burn::backend::{NdArray, ndarray::NdArrayDevice};
use burn::tensor::{DType, Tensor, TensorData};

type Backend = NdArray<f64>;
type Vector = Tensor<Backend, 1>;
type Matrix = Tensor<Backend, 2>;

const COSINE_DIMENSION: usize = 1024;

fn scalar(i: usize, j: usize, salt: usize) -> f64 {
    let value = (i
        .wrapping_mul(37)
        .wrapping_add(j.wrapping_mul(17))
        .wrapping_add(salt.wrapping_mul(13))
        % 101) as f64;
    0.125 + value / 101.0
}

fn device() -> NdArrayDevice {
    // burn-ndarray currently has one device, but naming it explicitly documents that
    // this module is the CPU comparison rather than a generic/default backend run.
    NdArrayDevice::Cpu
}

fn vector(values: Vec<f64>) -> Vector {
    let length = values.len();
    Tensor::from_data(TensorData::new(values, [length]), (&device(), DType::F64))
}

fn matrix(values: Vec<f64>, rows: usize, columns: usize) -> Matrix {
    Tensor::from_data(
        TensorData::new(values, [rows, columns]),
        (&device(), DType::F64),
    )
}

fn sum_all<const D: usize>(tensor: Tensor<Backend, D>) -> f64 {
    tensor.sum().into_scalar()
}

fn vector_element(tensor: Vector, index: usize) -> f64 {
    tensor.slice_dim(0, index).into_scalar()
}

fn matrix_element(tensor: Matrix, row: usize, column: usize) -> f64 {
    tensor
        .slice([row..row + 1, column..column + 1])
        .into_scalar()
}

/// Run 2D vector addition, dot product, and L2 norm.
pub fn vector2(iterations: usize) -> Result<f64, String> {
    let a = black_box(vector(vec![1.25, -2.5]));
    let b = black_box(vector(vec![-0.75, 3.0]));
    let mut checksum = 0.0;

    for i in 0..iterations {
        let sum = a.clone() + b.clone();
        let dot = a.clone().dot(b.clone()).into_scalar();
        let norm = sum.clone().square().sum().sqrt().into_scalar();
        checksum += vector_element(sum, i & 1) + dot + norm + black_box((i & 7) as f64 * 1.0e-12);
    }

    Ok(black_box(checksum))
}

/// Run 3D vector addition, dot product, L2 norm, and cross product.
pub fn vector3(iterations: usize) -> Result<f64, String> {
    let a = black_box(vector(vec![1.25, -2.5, 0.75]));
    let b = black_box(vector(vec![-0.75, 3.0, 1.5]));
    let a0 = a.clone().slice_dim(0, 0);
    let a1 = a.clone().slice_dim(0, 1);
    let a2 = a.clone().slice_dim(0, 2);
    let b0 = b.clone().slice_dim(0, 0);
    let b1 = b.clone().slice_dim(0, 1);
    let b2 = b.clone().slice_dim(0, 2);
    let mut checksum = 0.0;

    for i in 0..iterations {
        let sum = a.clone() + b.clone();
        let dot = a.clone().dot(b.clone()).into_scalar();
        let norm = sum.clone().square().sum().sqrt().into_scalar();
        let cross = Tensor::cat(
            vec![
                a1.clone() * b2.clone() - a2.clone() * b1.clone(),
                a2.clone() * b0.clone() - a0.clone() * b2.clone(),
                a0.clone() * b1.clone() - a1.clone() * b0.clone(),
            ],
            0,
        );

        checksum += vector_element(sum, i % 3)
            + dot
            + norm
            + vector_element(cross, (i + 1) % 3)
            + black_box((i & 7) as f64 * 1.0e-12);
    }

    Ok(black_box(checksum))
}

fn affine_values(dimension: usize) -> Vec<f64> {
    match dimension {
        2 => vec![1.1, -0.2, 0.5, 0.3, 0.9, -0.7, 0.0, 0.0, 1.0],
        3 => vec![
            1.1, -0.2, 0.1, 0.5, 0.3, 0.9, -0.15, -0.7, 0.05, 0.2, 1.05, 0.4, 0.0, 0.0, 0.0, 1.0,
        ],
        _ => unreachable!(),
    }
}

fn affine(points: usize, iterations: usize, dimension: usize) -> Result<f64, String> {
    if points == 0 {
        return Err("affine point count must be greater than zero".to_owned());
    }

    let homogeneous = dimension + 1;
    let transform = matrix(affine_values(dimension), homogeneous, homogeneous);
    let mut input_values = Vec::with_capacity(homogeneous * points);
    for row in 0..homogeneous {
        for column in 0..points {
            input_values.push(if row == dimension {
                1.0
            } else {
                scalar(column, row, dimension)
            });
        }
    }
    let input = matrix(input_values, homogeneous, points);
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let output = transform.clone().matmul(input.clone());
        checksum += sum_all(output.clone())
            + matrix_element(output.clone(), 0, 0)
            + matrix_element(output, dimension, points - 1);
    }

    Ok(black_box(checksum))
}

/// Run a 2D homogeneous affine transform over `points` columns.
pub fn affine2(points: usize, iterations: usize) -> Result<f64, String> {
    affine(points, iterations, 2)
}

/// Run a 3D homogeneous affine transform over `points` columns.
pub fn affine3(points: usize, iterations: usize) -> Result<f64, String> {
    affine(points, iterations, 3)
}

fn scalar_matrix(size: usize, salt: usize) -> Vec<f64> {
    (0..size)
        .flat_map(|row| (0..size).map(move |column| scalar(row, column, salt)))
        .collect()
}

/// Run a dense matrix-vector product.
pub fn matvec(size: usize, iterations: usize) -> Result<f64, String> {
    if size == 0 {
        return Err("matvec size must be greater than zero".to_owned());
    }

    let matrix = matrix(scalar_matrix(size, size), size, size);
    let vector = vector((0..size).map(|row| scalar(row, 0, size + 1)).collect()).reshape([size, 1]);
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let output = matrix.clone().matmul(vector.clone());
        checksum += sum_all(output.clone()) + matrix_element(output, 0, 0);
    }

    Ok(black_box(checksum))
}

/// Run a dense square matrix product.
pub fn matmul(size: usize, iterations: usize) -> Result<f64, String> {
    if size == 0 {
        return Err("matmul size must be greater than zero".to_owned());
    }

    let left = matrix(scalar_matrix(size, size), size, size);
    let right = matrix(scalar_matrix(size, size + 1), size, size);
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let output = left.clone().matmul(right.clone());
        checksum += sum_all(output.clone()) + matrix_element(output, 0, 0);
    }

    Ok(black_box(checksum))
}

fn embedding_scalar(index: usize, lane: usize) -> f64 {
    scalar(index, lane, COSINE_DIMENSION + lane) * 2.0 - 1.0
}

fn embedding_values(lane: usize) -> Vec<f64> {
    (0..COSINE_DIMENSION)
        .map(|index| embedding_scalar(index, lane))
        .collect()
}

/// Run cosine similarity for two deterministic f64 vectors of length 1024.
pub fn cosine1024(iterations: usize) -> Result<f64, String> {
    let a = black_box(vector(embedding_values(0)));
    let b = black_box(vector(embedding_values(1)));
    let mut checksum = 0.0;

    for i in 0..iterations {
        let dot = a.clone().dot(b.clone()).into_scalar();
        let norm_a = a.clone().square().sum().sqrt().into_scalar();
        let norm_b = b.clone().square().sum().sqrt().into_scalar();
        let cosine = dot / (norm_a * norm_b);
        checksum += cosine + black_box((i & 7) as f64 * 1.0e-12);
    }

    Ok(black_box(checksum))
}

/// Explain why the Burn backend is not included in the symmetric eigendecomposition benchmark.
pub fn eigh_error() -> String {
    "Burn 0.21.0's burn-ndarray backend does not provide a general symmetric eigendecomposition API; eigh is unavailable".to_owned()
}
