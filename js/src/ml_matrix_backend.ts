import { EigenvalueDecomposition, Matrix } from "ml-matrix";

import {
  affineValues,
  embeddingScalar,
  COSINE_DIMENSION,
  scalar,
  scalarMatrix,
  symmetricMatrix,
  tinyTerm,
} from "./shared";
import * as raw from "./raw_backend";

function matrixFromFlat(
  values: ArrayLike<number>,
  rows: number,
  columns: number,
): Matrix {
  const matrix = Matrix.zeros(rows, columns);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      matrix.set(row, column, values[row * columns + column]);
    }
  }
  return matrix;
}

function matrixSum(matrix: Matrix): number {
  let sum = 0.0;
  for (let row = 0; row < matrix.rows; row += 1) {
    for (let column = 0; column < matrix.columns; column += 1) {
      sum += matrix.get(row, column);
    }
  }
  return sum;
}

function affine(points: number, iterations: number, dimension: number): number {
  const homogeneous = dimension + 1;
  const transform = matrixFromFlat(affineValues(dimension), homogeneous, homogeneous);
  const input = new Float64Array(homogeneous * points);
  for (let row = 0; row < homogeneous; row += 1) {
    for (let column = 0; column < points; column += 1) {
      input[row * points + column] =
        row === dimension ? 1.0 : scalar(column, row, dimension);
    }
  }
  const inputMatrix = matrixFromFlat(input, homogeneous, points);

  let checksum = 0.0;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const output = transform.mmul(inputMatrix);
    checksum +=
      matrixSum(output) + output.get(0, 0) + output.get(dimension, points - 1);
  }
  return checksum;
}

export function vector2(iterations: number): number {
  // ml-matrix is a dense matrix library. Keep these fixed-size vector kernels
  // identical to the raw Float64Array path instead of manufacturing matrices.
  return raw.vector2(iterations);
}

export function vector3(iterations: number): number {
  return raw.vector3(iterations);
}

export function affine2(points: number, iterations: number): number {
  return affine(points, iterations, 2);
}

export function affine3(points: number, iterations: number): number {
  return affine(points, iterations, 3);
}

export function matvec(size: number, iterations: number): number {
  const matrix = matrixFromFlat(scalarMatrix(size, size), size, size);
  const vectorValues = new Float64Array(size);
  for (let row = 0; row < size; row += 1) {
    vectorValues[row] = scalar(row, 0, size + 1);
  }
  const vector = matrixFromFlat(vectorValues, size, 1);

  let checksum = 0.0;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const output = matrix.mmul(vector);
    let outputSum = 0.0;
    for (let row = 0; row < size; row += 1) {
      outputSum += output.get(row, 0);
    }
    checksum += outputSum + output.get(0, 0);
  }
  return checksum;
}

export function matmul(size: number, iterations: number): number {
  const left = matrixFromFlat(scalarMatrix(size, size), size, size);
  const right = matrixFromFlat(scalarMatrix(size, size + 1), size, size);
  let checksum = 0.0;

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const output = left.mmul(right);
    checksum += matrixSum(output) + output.get(0, 0);
  }
  return checksum;
}

export function cosine1024(iterations: number): number {
  // This is the same raw f64 dot/norm loop as the Float64Array backend; the
  // ml-matrix package does not provide a useful fixed-size vector primitive.
  const a = new Float64Array(COSINE_DIMENSION);
  const b = new Float64Array(COSINE_DIMENSION);
  for (let index = 0; index < COSINE_DIMENSION; index += 1) {
    a[index] = embeddingScalar(index, 0);
    b[index] = embeddingScalar(index, 1);
  }

  let checksum = 0.0;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    let dot = 0.0;
    let normASquared = 0.0;
    let normBSquared = 0.0;
    for (let index = 0; index < COSINE_DIMENSION; index += 1) {
      dot += a[index] * b[index];
      normASquared += a[index] * a[index];
      normBSquared += b[index] * b[index];
    }
    checksum +=
      dot / (Math.sqrt(normASquared) * Math.sqrt(normBSquared)) +
      tinyTerm(iteration);
  }
  return checksum;
}

export function eigh(size: number, iterations: number): number {
  const matrix = matrixFromFlat(symmetricMatrix(size), size, size);
  let checksum = 0.0;

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const decomposition = new EigenvalueDecomposition(matrix);
    let eigenvalueSum = 0.0;
    for (const eigenvalue of decomposition.realEigenvalues) {
      eigenvalueSum += eigenvalue;
    }

    const eigenvectors = decomposition.eigenvectorMatrix;
    let eigenvectorNormSquared = 0.0;
    for (let row = 0; row < eigenvectors.rows; row += 1) {
      for (let column = 0; column < eigenvectors.columns; column += 1) {
        const value = eigenvectors.get(row, column);
        eigenvectorNormSquared += value * value;
      }
    }
    checksum += eigenvalueSum + eigenvectorNormSquared;
  }

  return checksum;
}
