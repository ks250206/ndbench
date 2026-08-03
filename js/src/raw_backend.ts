import {
  affineValues,
  COSINE_DIMENSION,
  embeddingScalar,
  JACOBI_TOLERANCE,
  scalar,
  scalarMatrix,
  sumValues,
  symmetricMatrix,
  tinyTerm,
} from "./shared";

export function vector2(iterations: number): number {
  const a = new Float64Array([1.25, -2.5]);
  const b = new Float64Array([-0.75, 3.0]);
  let checksum = 0.0;

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const summed0 = a[0] + b[0];
    const summed1 = a[1] + b[1];
    const dot = a[0] * b[0] + a[1] * b[1];
    const norm = Math.sqrt(summed0 * summed0 + summed1 * summed1);
    checksum += (iteration & 1) === 0 ? summed0 + dot + norm + tinyTerm(iteration) : summed1 + dot + norm + tinyTerm(iteration);
  }

  return checksum;
}

export function vector3(iterations: number): number {
  const a = new Float64Array([1.25, -2.5, 0.75]);
  const b = new Float64Array([-0.75, 3.0, 1.5]);
  let checksum = 0.0;

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const summed0 = a[0] + b[0];
    const summed1 = a[1] + b[1];
    const summed2 = a[2] + b[2];
    const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const norm = Math.sqrt(
      summed0 * summed0 + summed1 * summed1 + summed2 * summed2,
    );
    const cross0 = a[1] * b[2] - a[2] * b[1];
    const cross1 = a[2] * b[0] - a[0] * b[2];
    const cross2 = a[0] * b[1] - a[1] * b[0];
    const selectedSum =
      iteration % 3 === 0 ? summed0 : iteration % 3 === 1 ? summed1 : summed2;
    const selectedCross =
      (iteration + 1) % 3 === 0
        ? cross0
        : (iteration + 1) % 3 === 1
          ? cross1
          : cross2;
    checksum += selectedSum + dot + norm + selectedCross + tinyTerm(iteration);
  }

  return checksum;
}

function affine(points: number, iterations: number, dimension: number): number {
  const homogeneous = dimension + 1;
  const transform = new Float64Array(affineValues(dimension));
  const input = new Float64Array(homogeneous * points);

  for (let row = 0; row < homogeneous; row += 1) {
    for (let column = 0; column < points; column += 1) {
      input[row * points + column] =
        row === dimension ? 1.0 : scalar(column, row, dimension);
    }
  }

  let checksum = 0.0;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const output = new Float64Array(homogeneous * points);
    for (let row = 0; row < homogeneous; row += 1) {
      for (let column = 0; column < points; column += 1) {
        let value = 0.0;
        for (let inner = 0; inner < homogeneous; inner += 1) {
          value +=
            transform[row * homogeneous + inner] * input[inner * points + column];
        }
        output[row * points + column] = value;
      }
    }
    checksum +=
      sumValues(output) + output[0] + output[dimension * points + points - 1];
  }

  return checksum;
}

export function affine2(points: number, iterations: number): number {
  return affine(points, iterations, 2);
}

export function affine3(points: number, iterations: number): number {
  return affine(points, iterations, 3);
}

export function matvec(size: number, iterations: number): number {
  const matrix = scalarMatrix(size, size);
  const vector = new Float64Array(size);
  for (let row = 0; row < size; row += 1) {
    vector[row] = scalar(row, 0, size + 1);
  }

  let checksum = 0.0;
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const output = new Float64Array(size);
    for (let row = 0; row < size; row += 1) {
      let value = 0.0;
      for (let column = 0; column < size; column += 1) {
        value += matrix[row * size + column] * vector[column];
      }
      output[row] = value;
    }
    checksum += sumValues(output) + output[0];
  }

  return checksum;
}

export function matmul(size: number, iterations: number): number {
  const left = scalarMatrix(size, size);
  const right = scalarMatrix(size, size + 1);
  let checksum = 0.0;

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const output = new Float64Array(size * size);
    for (let row = 0; row < size; row += 1) {
      for (let column = 0; column < size; column += 1) {
        let value = 0.0;
        for (let inner = 0; inner < size; inner += 1) {
          value += left[row * size + inner] * right[inner * size + column];
        }
        output[row * size + column] = value;
      }
    }
    checksum += sumValues(output) + output[0];
  }

  return checksum;
}

export function cosine1024(iterations: number): number {
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
    const similarity = dot / (Math.sqrt(normASquared) * Math.sqrt(normBSquared));
    checksum += similarity + tinyTerm(iteration);
  }

  return checksum;
}

function jacobiEigh(input: Float64Array, size: number): {
  eigenvalues: Float64Array;
  eigenvectors: Float64Array;
} {
  const matrix = new Float64Array(input);
  const eigenvectors = new Float64Array(size * size);
  for (let index = 0; index < size; index += 1) {
    eigenvectors[index * size + index] = 1.0;
  }

  for (let sweep = 0; sweep < 8 * Math.max(size, 1); sweep += 1) {
    let maxOffDiagonal = 0.0;
    for (let row = 0; row < size; row += 1) {
      for (let column = row + 1; column < size; column += 1) {
        maxOffDiagonal = Math.max(
          maxOffDiagonal,
          Math.abs(matrix[row * size + column]),
        );
      }
    }
    if (maxOffDiagonal < JACOBI_TOLERANCE) {
      break;
    }

    for (let p = 0; p < size; p += 1) {
      for (let q = p + 1; q < size; q += 1) {
        const apq = matrix[p * size + q];
        if (Math.abs(apq) < JACOBI_TOLERANCE) {
          continue;
        }
        const app = matrix[p * size + p];
        const aqq = matrix[q * size + q];
        const tau = (aqq - app) / (2.0 * apq);
        const t =
          tau >= 0.0
            ? 1.0 / (tau + Math.sqrt(1.0 + tau * tau))
            : -1.0 / (-tau + Math.sqrt(1.0 + tau * tau));
        const cosine = 1.0 / Math.sqrt(1.0 + t * t);
        const sine = t * cosine;

        for (let k = 0; k < size; k += 1) {
          if (k === p || k === q) {
            continue;
          }
          const akp = matrix[k * size + p];
          const akq = matrix[k * size + q];
          const newKP = cosine * akp - sine * akq;
          const newKQ = sine * akp + cosine * akq;
          matrix[k * size + p] = newKP;
          matrix[p * size + k] = newKP;
          matrix[k * size + q] = newKQ;
          matrix[q * size + k] = newKQ;
        }

        matrix[p * size + p] =
          cosine * cosine * app -
          2.0 * sine * cosine * apq +
          sine * sine * aqq;
        matrix[q * size + q] =
          sine * sine * app +
          2.0 * sine * cosine * apq +
          cosine * cosine * aqq;
        matrix[p * size + q] = 0.0;
        matrix[q * size + p] = 0.0;

        for (let k = 0; k < size; k += 1) {
          const vKP = eigenvectors[k * size + p];
          const vKQ = eigenvectors[k * size + q];
          eigenvectors[k * size + p] = cosine * vKP - sine * vKQ;
          eigenvectors[k * size + q] = sine * vKP + cosine * vKQ;
        }
      }
    }
  }

  const eigenvalues = new Float64Array(size);
  for (let index = 0; index < size; index += 1) {
    eigenvalues[index] = matrix[index * size + index];
  }
  return { eigenvalues, eigenvectors };
}

export function eigh(size: number, iterations: number): number {
  const matrix = symmetricMatrix(size);
  let checksum = 0.0;

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const { eigenvalues, eigenvectors } = jacobiEigh(matrix, size);
    let eigenvalueSum = 0.0;
    for (let index = 0; index < eigenvalues.length; index += 1) {
      eigenvalueSum += eigenvalues[index];
    }
    let eigenvectorNormSquared = 0.0;
    for (let index = 0; index < eigenvectors.length; index += 1) {
      eigenvectorNormSquared += eigenvectors[index] * eigenvectors[index];
    }
    checksum += eigenvalueSum + eigenvectorNormSquared;
  }

  return checksum;
}
