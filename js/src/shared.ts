export const DEFAULT_VECTOR_ITERATIONS = 1_000_000;
export const DEFAULT_POINT_COUNT = 100_000;
export const DEFAULT_MATRIX_SIZE = 128;
export const COSINE_DIMENSION = 1024;
export const DEFAULT_COSINE_ITERATIONS = 1_000;
export const JACOBI_TOLERANCE = 1.0e-12;

export const OPERATIONS = [
  "vector2",
  "vector3",
  "affine2",
  "affine3",
  "matvec",
  "matmul",
  "cosine1024",
  "eigh",
] as const;

export type Operation = (typeof OPERATIONS)[number];

export function scalar(i: number, j: number, salt: number): number {
  const value = (i * 37 + j * 17 + salt * 13) % 101;
  return 0.125 + value / 101.0;
}

export function tinyTerm(iteration: number): number {
  return (iteration & 7) * 1.0e-12;
}

export function embeddingScalar(index: number, lane: number): number {
  return scalar(index, lane, COSINE_DIMENSION + lane) * 2.0 - 1.0;
}

export function affineValues(dimension: number): readonly number[] {
  if (dimension === 2) {
    return [1.1, -0.2, 0.5, 0.3, 0.9, -0.7, 0.0, 0.0, 1.0];
  }
  if (dimension === 3) {
    return [
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
    ];
  }
  throw new Error(`unsupported affine dimension: ${dimension}`);
}

export function scalarMatrix(size: number, salt: number): Float64Array {
  const matrix = new Float64Array(size * size);
  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column < size; column += 1) {
      matrix[row * size + column] = scalar(row, column, salt);
    }
  }
  return matrix;
}

export function symmetricMatrix(size: number): Float64Array {
  const matrix = new Float64Array(size * size);
  for (let row = 0; row < size; row += 1) {
    for (let column = 0; column < size; column += 1) {
      if (row === column) {
        matrix[row * size + column] = size + 2.0;
      } else {
        const distance = Math.abs(row - column);
        matrix[row * size + column] =
          (0.01 * scalar(Math.min(row, column), Math.max(row, column), size)) /
          (1.0 + distance);
      }
    }
  }
  return matrix;
}

export function sumValues(values: ArrayLike<number>): number {
  let result = 0.0;
  for (let index = 0; index < values.length; index += 1) {
    result += values[index];
  }
  return result;
}

export function defaultSize(operation: Operation): number {
  if (operation === "vector2" || operation === "vector3") {
    return 1;
  }
  if (operation === "affine2" || operation === "affine3") {
    return DEFAULT_POINT_COUNT;
  }
  if (operation === "cosine1024") {
    return COSINE_DIMENSION;
  }
  return DEFAULT_MATRIX_SIZE;
}

export function defaultIterations(operation: Operation): number {
  if (operation === "vector2" || operation === "vector3") {
    return DEFAULT_VECTOR_ITERATIONS;
  }
  if (operation === "cosine1024") {
    return DEFAULT_COSINE_ITERATIONS;
  }
  return 1;
}
