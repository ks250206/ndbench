//! Dependency-free Rust baseline implemented with `std::vec::Vec` and loops.
//!
//! This is intentionally a low-level reference point rather than a production
//! linear-algebra implementation. Its symmetric eigendecomposition uses a
//! cyclic Jacobi method because the Rust standard library has no eigensolver.

use std::hint::black_box;

const COSINE_DIMENSION: usize = 1024;
const JACOBI_TOLERANCE: f64 = 1.0e-12;

fn scalar(i: usize, j: usize, salt: usize) -> f64 {
    let value = (i
        .wrapping_mul(37)
        .wrapping_add(j.wrapping_mul(17))
        .wrapping_add(salt.wrapping_mul(13))
        % 101) as f64;
    0.125 + value / 101.0
}

fn tiny_term(iteration: usize) -> f64 {
    (iteration & 7) as f64 * 1.0e-12
}

fn embedding_scalar(index: usize, lane: usize) -> f64 {
    scalar(index, lane, COSINE_DIMENSION + lane) * 2.0 - 1.0
}

/// Run 2D vector addition, dot product, and L2 norm using plain `f64`s.
pub fn vector2(iterations: usize) -> Result<f64, String> {
    let a = black_box([1.25_f64, -2.5]);
    let b = black_box([-0.75_f64, 3.0]);
    let mut checksum = 0.0;

    for iteration in 0..iterations {
        let sum = [a[0] + b[0], a[1] + b[1]];
        let dot = a[0] * b[0] + a[1] * b[1];
        let norm = (sum[0] * sum[0] + sum[1] * sum[1]).sqrt();
        checksum += sum[iteration & 1] + dot + norm + black_box(tiny_term(iteration));
    }

    Ok(black_box(checksum))
}

/// Run 3D vector addition, dot product, L2 norm, and cross product.
pub fn vector3(iterations: usize) -> Result<f64, String> {
    let a = black_box([1.25_f64, -2.5, 0.75]);
    let b = black_box([-0.75_f64, 3.0, 1.5]);
    let mut checksum = 0.0;

    for iteration in 0..iterations {
        let sum = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
        let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let norm = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
        let cross = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        checksum += sum[iteration % 3]
            + dot
            + norm
            + cross[(iteration + 1) % 3]
            + black_box(tiny_term(iteration));
    }

    Ok(black_box(checksum))
}

fn affine_values(dimension: usize) -> &'static [f64] {
    match dimension {
        2 => &[1.1, -0.2, 0.5, 0.3, 0.9, -0.7, 0.0, 0.0, 1.0],
        3 => &[
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
    let transform = affine_values(dimension);
    let mut input = vec![0.0; homogeneous * points];
    for row in 0..homogeneous {
        for column in 0..points {
            input[row * points + column] = if row == dimension {
                1.0
            } else {
                scalar(column, row, dimension)
            };
        }
    }
    let input = black_box(input);
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let mut output = vec![0.0; homogeneous * points];
        for row in 0..homogeneous {
            for column in 0..points {
                let mut value = 0.0;
                for inner in 0..homogeneous {
                    value += transform[row * homogeneous + inner] * input[inner * points + column];
                }
                output[row * points + column] = value;
            }
        }
        checksum +=
            output.iter().sum::<f64>() + output[0] + output[dimension * points + points - 1];
    }

    Ok(black_box(checksum))
}

pub fn affine2(points: usize, iterations: usize) -> Result<f64, String> {
    affine(points, iterations, 2)
}

pub fn affine3(points: usize, iterations: usize) -> Result<f64, String> {
    affine(points, iterations, 3)
}

fn scalar_matrix(size: usize, salt: usize) -> Vec<f64> {
    let mut matrix = vec![0.0; size * size];
    for row in 0..size {
        for column in 0..size {
            matrix[row * size + column] = scalar(row, column, salt);
        }
    }
    matrix
}

pub fn matvec(size: usize, iterations: usize) -> Result<f64, String> {
    if size == 0 {
        return Err("matvec size must be greater than zero".to_owned());
    }

    let matrix = black_box(scalar_matrix(size, size));
    let mut vector = vec![0.0; size];
    for (row, value) in vector.iter_mut().enumerate() {
        *value = scalar(row, 0, size + 1);
    }
    let vector = black_box(vector);
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let mut output = vec![0.0; size];
        for row in 0..size {
            let mut value = 0.0;
            for column in 0..size {
                value += matrix[row * size + column] * vector[column];
            }
            output[row] = value;
        }
        checksum += output.iter().sum::<f64>() + output[0];
    }

    Ok(black_box(checksum))
}

pub fn matmul(size: usize, iterations: usize) -> Result<f64, String> {
    if size == 0 {
        return Err("matmul size must be greater than zero".to_owned());
    }

    let left = black_box(scalar_matrix(size, size));
    let right = black_box(scalar_matrix(size, size + 1));
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let mut output = vec![0.0; size * size];
        for row in 0..size {
            for column in 0..size {
                let mut value = 0.0;
                for inner in 0..size {
                    value += left[row * size + inner] * right[inner * size + column];
                }
                output[row * size + column] = value;
            }
        }
        checksum += output.iter().sum::<f64>() + output[0];
    }

    Ok(black_box(checksum))
}

pub fn cosine1024(iterations: usize) -> Result<f64, String> {
    let a = black_box(
        (0..COSINE_DIMENSION)
            .map(|index| embedding_scalar(index, 0))
            .collect::<Vec<_>>(),
    );
    let b = black_box(
        (0..COSINE_DIMENSION)
            .map(|index| embedding_scalar(index, 1))
            .collect::<Vec<_>>(),
    );
    let mut checksum = 0.0;

    for iteration in 0..iterations {
        let mut dot = 0.0;
        let mut norm_a_squared = 0.0;
        let mut norm_b_squared = 0.0;
        for index in 0..COSINE_DIMENSION {
            dot += a[index] * b[index];
            norm_a_squared += a[index] * a[index];
            norm_b_squared += b[index] * b[index];
        }
        let similarity = dot / (norm_a_squared.sqrt() * norm_b_squared.sqrt());
        checksum += similarity + black_box(tiny_term(iteration));
    }

    Ok(black_box(checksum))
}

fn symmetric_matrix(size: usize) -> Vec<f64> {
    let mut matrix = vec![0.0; size * size];
    for row in 0..size {
        for column in 0..size {
            matrix[row * size + column] = if row == column {
                size as f64 + 2.0
            } else {
                let distance = row.abs_diff(column) as f64;
                0.01 * scalar(row.min(column), row.max(column), size) / (1.0 + distance)
            };
        }
    }
    matrix
}

fn jacobi_eigh(input: &[f64], size: usize) -> (Vec<f64>, Vec<f64>) {
    let mut matrix = input.to_vec();
    let mut eigenvectors = vec![0.0; size * size];
    for index in 0..size {
        eigenvectors[index * size + index] = 1.0;
    }

    // The benchmark input is strongly diagonally dominant, so a bounded cyclic
    // Jacobi sweep gives a practical dependency-free baseline at the sizes used.
    for _ in 0..(8 * size.max(1)) {
        let mut max_off_diagonal = 0.0_f64;
        for row in 0..size {
            for column in (row + 1)..size {
                max_off_diagonal = max_off_diagonal.max(matrix[row * size + column].abs());
            }
        }
        if max_off_diagonal < JACOBI_TOLERANCE {
            break;
        }

        for p in 0..size {
            for q in (p + 1)..size {
                let apq = matrix[p * size + q];
                if apq.abs() < JACOBI_TOLERANCE {
                    continue;
                }

                let app = matrix[p * size + p];
                let aqq = matrix[q * size + q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let cosine = 1.0 / (1.0 + t * t).sqrt();
                let sine = t * cosine;

                for k in 0..size {
                    if k == p || k == q {
                        continue;
                    }
                    let akp = matrix[k * size + p];
                    let akq = matrix[k * size + q];
                    let new_kp = cosine * akp - sine * akq;
                    let new_kq = sine * akp + cosine * akq;
                    matrix[k * size + p] = new_kp;
                    matrix[p * size + k] = new_kp;
                    matrix[k * size + q] = new_kq;
                    matrix[q * size + k] = new_kq;
                }

                matrix[p * size + p] =
                    cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
                matrix[q * size + q] =
                    sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
                matrix[p * size + q] = 0.0;
                matrix[q * size + p] = 0.0;

                for k in 0..size {
                    let vkp = eigenvectors[k * size + p];
                    let vkq = eigenvectors[k * size + q];
                    eigenvectors[k * size + p] = cosine * vkp - sine * vkq;
                    eigenvectors[k * size + q] = sine * vkp + cosine * vkq;
                }
            }
        }
    }

    let eigenvalues = (0..size)
        .map(|index| matrix[index * size + index])
        .collect();
    (eigenvalues, eigenvectors)
}

pub fn eigh(size: usize, iterations: usize) -> Result<f64, String> {
    if size == 0 {
        return Err("eigh size must be greater than zero".to_owned());
    }

    let matrix = black_box(symmetric_matrix(size));
    let mut checksum = 0.0;
    for _ in 0..iterations {
        let (eigenvalues, eigenvectors) = jacobi_eigh(&matrix, size);
        let eigenvector_norm_squared = eigenvectors.iter().map(|value| value * value).sum::<f64>();
        checksum += eigenvalues.iter().sum::<f64>() + eigenvector_norm_squared;
    }

    Ok(black_box(checksum))
}
