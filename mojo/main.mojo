"""Dependency-free CPU baseline for ndbench.

This module deliberately uses only Mojo's standard library, Float64, List, and
small SIMD values. The matrix kernels use row-major flat Lists so that their
input construction and loop order match the Rust and Python raw backends.
"""

from std.collections import List
from std.math import sqrt
from std.sys import argv


comptime COSINE_DIMENSION: Int = 1024
comptime JACOBI_TOLERANCE: Float64 = 1.0e-12


def scalar(i: Int, j: Int, salt: Int) -> Float64:
    var value = (i * 37 + j * 17 + salt * 13) % 101
    return 0.125 + Float64(value) / 101.0


def tiny_term(iteration: Int) -> Float64:
    return Float64(iteration % 8) * 1.0e-12


def embedding_scalar(index: Int, lane: Int) -> Float64:
    return scalar(index, lane, COSINE_DIMENSION + lane) * 2.0 - 1.0


def zeros(count: Int) -> List[Float64]:
    var result: List[Float64] = []
    for _ in range(count):
        result.append(0.0)
    return result^


def copy_values(values: List[Float64]) -> List[Float64]:
    var result: List[Float64] = []
    for index in range(len(values)):
        result.append(values[index])
    return result^


def vector2(iterations: Int) -> Float64:
    # Keep the inputs in Float64 SIMD values, while reading lanes explicitly so
    # the checksum and operation order are easy to compare with Rust/Python.
    var a = SIMD[DType.float64, 2](1.25, -2.5)
    var b = SIMD[DType.float64, 2](-0.75, 3.0)
    var checksum: Float64 = 0.0

    for iteration in range(iterations):
        var summed = a + b
        var dot = a[0] * b[0] + a[1] * b[1]
        var norm = sqrt(summed[0] * summed[0] + summed[1] * summed[1])
        checksum += summed[iteration % 2] + dot + norm + tiny_term(iteration)

    return checksum


def vector3(iterations: Int) -> Float64:
    var a = SIMD[DType.float64, 3](1.25, -2.5, 0.75)
    var b = SIMD[DType.float64, 3](-0.75, 3.0, 1.5)
    var checksum: Float64 = 0.0

    for iteration in range(iterations):
        var summed = a + b
        var dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        var norm = sqrt(
            summed[0] * summed[0]
            + summed[1] * summed[1]
            + summed[2] * summed[2]
        )
        var cross = SIMD[DType.float64, 3](
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )
        checksum += (
            summed[iteration % 3]
            + dot
            + norm
            + cross[(iteration + 1) % 3]
            + tiny_term(iteration)
        )

    return checksum


def affine_values(dimension: Int) -> List[Float64]:
    if dimension == 2:
        return [1.1, -0.2, 0.5, 0.3, 0.9, -0.7, 0.0, 0.0, 1.0]
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
    ]


def affine(points: Int, iterations: Int, dimension: Int) -> Float64:
    var homogeneous = dimension + 1
    var transform = affine_values(dimension)
    var input = zeros(homogeneous * points)

    for row in range(homogeneous):
        for column in range(points):
            if row == dimension:
                input[row * points + column] = 1.0
            else:
                input[row * points + column] = scalar(column, row, dimension)

    var checksum: Float64 = 0.0
    for _ in range(iterations):
        var output = zeros(homogeneous * points)
        for row in range(homogeneous):
            for column in range(points):
                var value: Float64 = 0.0
                for inner in range(homogeneous):
                    value += transform[row * homogeneous + inner] * input[
                        inner * points + column
                    ]
                output[row * points + column] = value

        var output_sum: Float64 = 0.0
        for index in range(len(output)):
            output_sum += output[index]
        checksum += output_sum + output[0] + output[dimension * points + points - 1]

    return checksum


def affine2(points: Int, iterations: Int) -> Float64:
    return affine(points, iterations, 2)


def affine3(points: Int, iterations: Int) -> Float64:
    return affine(points, iterations, 3)


def scalar_matrix(size: Int, salt: Int) -> List[Float64]:
    var matrix = zeros(size * size)
    for row in range(size):
        for column in range(size):
            matrix[row * size + column] = scalar(row, column, salt)
    return matrix^


def matvec(size: Int, iterations: Int) -> Float64:
    var matrix = scalar_matrix(size, size)
    var vector = zeros(size)
    for row in range(size):
        vector[row] = scalar(row, 0, size + 1)

    var checksum: Float64 = 0.0
    for _ in range(iterations):
        var output = zeros(size)
        for row in range(size):
            var value: Float64 = 0.0
            for column in range(size):
                value += matrix[row * size + column] * vector[column]
            output[row] = value

        var output_sum: Float64 = 0.0
        for row in range(size):
            output_sum += output[row]
        checksum += output_sum + output[0]

    return checksum


def matmul(size: Int, iterations: Int) -> Float64:
    var left = scalar_matrix(size, size)
    var right = scalar_matrix(size, size + 1)
    var checksum: Float64 = 0.0

    for _ in range(iterations):
        var output = zeros(size * size)
        for row in range(size):
            for column in range(size):
                var value: Float64 = 0.0
                for inner in range(size):
                    value += left[row * size + inner] * right[inner * size + column]
                output[row * size + column] = value

        var output_sum: Float64 = 0.0
        for index in range(size * size):
            output_sum += output[index]
        checksum += output_sum + output[0]

    return checksum


def cosine1024(iterations: Int) -> Float64:
    var a = zeros(COSINE_DIMENSION)
    var b = zeros(COSINE_DIMENSION)
    for index in range(COSINE_DIMENSION):
        a[index] = embedding_scalar(index, 0)
        b[index] = embedding_scalar(index, 1)

    var checksum: Float64 = 0.0
    for iteration in range(iterations):
        var dot: Float64 = 0.0
        var norm_a_squared: Float64 = 0.0
        var norm_b_squared: Float64 = 0.0
        for index in range(COSINE_DIMENSION):
            dot += a[index] * b[index]
            norm_a_squared += a[index] * a[index]
            norm_b_squared += b[index] * b[index]
        var similarity = dot / (sqrt(norm_a_squared) * sqrt(norm_b_squared))
        checksum += similarity + tiny_term(iteration)

    return checksum


def symmetric_matrix(size: Int) -> List[Float64]:
    var matrix = zeros(size * size)
    for row in range(size):
        for column in range(size):
            if row == column:
                matrix[row * size + column] = Float64(size) + 2.0
            else:
                var minimum = row
                if column < minimum:
                    minimum = column
                var maximum = row
                if column > maximum:
                    maximum = column
                var distance = row - column
                if distance < 0:
                    distance = -distance
                matrix[row * size + column] = (
                    0.01
                    * scalar(minimum, maximum, size)
                    / (1.0 + Float64(distance))
                )
    return matrix^


def jacobi_eigh_checksum(input: List[Float64], size: Int) -> Float64:
    var matrix = copy_values(input)
    var eigenvectors = zeros(size * size)
    for index in range(size):
        eigenvectors[index * size + index] = 1.0

    # This is the same bounded cyclic Jacobi implementation as the Rust and
    # Python raw baselines. The input is strongly diagonally dominant.
    for _ in range(8 * size):
        var max_off_diagonal: Float64 = 0.0
        for row in range(size):
            for column in range(row + 1, size):
                var absolute = matrix[row * size + column]
                if absolute < 0.0:
                    absolute = -absolute
                if absolute > max_off_diagonal:
                    max_off_diagonal = absolute
        if max_off_diagonal < JACOBI_TOLERANCE:
            break

        for p in range(size):
            for q in range(p + 1, size):
                var apq = matrix[p * size + q]
                var absolute_apq = apq
                if absolute_apq < 0.0:
                    absolute_apq = -absolute_apq
                if absolute_apq < JACOBI_TOLERANCE:
                    continue

                var app = matrix[p * size + p]
                var aqq = matrix[q * size + q]
                var tau = (aqq - app) / (2.0 * apq)
                var t: Float64
                if tau >= 0.0:
                    t = 1.0 / (tau + sqrt(1.0 + tau * tau))
                else:
                    t = -1.0 / (-tau + sqrt(1.0 + tau * tau))
                var cosine = 1.0 / sqrt(1.0 + t * t)
                var sine = t * cosine

                for k in range(size):
                    if k == p or k == q:
                        continue
                    var akp = matrix[k * size + p]
                    var akq = matrix[k * size + q]
                    var new_kp = cosine * akp - sine * akq
                    var new_kq = sine * akp + cosine * akq
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
                    var vkp = eigenvectors[k * size + p]
                    var vkq = eigenvectors[k * size + q]
                    eigenvectors[k * size + p] = cosine * vkp - sine * vkq
                    eigenvectors[k * size + q] = sine * vkp + cosine * vkq

    var eigenvalue_sum: Float64 = 0.0
    for index in range(size):
        eigenvalue_sum += matrix[index * size + index]
    var eigenvector_norm_squared: Float64 = 0.0
    for index in range(size * size):
        eigenvector_norm_squared += eigenvectors[index] * eigenvectors[index]
    return eigenvalue_sum + eigenvector_norm_squared


def eigh(size: Int, iterations: Int) -> Float64:
    var matrix = symmetric_matrix(size)
    var checksum: Float64 = 0.0
    for _ in range(iterations):
        checksum += jacobi_eigh_checksum(matrix, size)
    return checksum


def format_checksum(value: Float64) -> String:
    """Format a finite Float64 like printf's %.17e without a numeric library."""
    if value == 0.0:
        return "checksum=0.00000000000000000e+00"

    var sign: String = ""
    var magnitude = value
    if magnitude < 0.0:
        sign = "-"
        magnitude = -magnitude

    var exponent: Int = 0
    while magnitude >= 10.0:
        magnitude /= 10.0
        exponent += 1
    while magnitude < 1.0:
        magnitude *= 10.0
        exponent -= 1

    # 17 digits after the decimal point, rounded using the next binary-derived
    # decimal digit. This is sufficient for the finite benchmark checksums.
    var scale: Int = 100000000000000000
    var scaled = Int(magnitude * Float64(scale) + 0.5)
    if scaled >= scale * 10:
        scaled = scale
        exponent += 1

    var leading = scaled / scale
    var fraction = scaled % scale
    var fraction_text: String = ""
    var divisor = scale / 10
    for _ in range(17):
        var digit = fraction / divisor
        fraction_text += String(digit)
        fraction = fraction % divisor
        divisor /= 10

    var exponent_sign: String = "+"
    var exponent_magnitude = exponent
    if exponent < 0:
        exponent_sign = "-"
        exponent_magnitude = -exponent
    var exponent_text = String(exponent_magnitude)
    if exponent_magnitude < 10:
        exponent_text = "0" + exponent_text

    return String(
        "checksum=",
        sign,
        leading,
        ".",
        fraction_text,
        "e",
        exponent_sign,
        exponent_text,
    )


def usage() raises:
    print(
        "usage: ndbench-mojo --op OP --size N --iterations N\n"
        "operations: vector2 vector3 affine2 affine3 matvec matmul cosine1024 eigh"
    )


def main() raises:
    var operation: String = "vector2"
    var size: Int = 8
    var iterations: Int = 1
    var args = argv()
    var index: Int = 1

    while index < len(args):
        var option = args[index]
        if option == "--help" or option == "-h":
            usage()
            return
        elif option == "--op" or option == "--operation":
            index += 1
            operation = String(args[index])
        elif option == "--size":
            index += 1
            size = Int(args[index])
        elif option == "--iterations":
            index += 1
            iterations = Int(args[index])
        else:
            print("unknown option: {}".format(option))
            usage()
            return
        index += 1

    var checksum: Float64
    if operation == "vector2":
        checksum = vector2(iterations)
    elif operation == "vector3":
        checksum = vector3(iterations)
    elif operation == "affine2":
        checksum = affine2(size, iterations)
    elif operation == "affine3":
        checksum = affine3(size, iterations)
    elif operation == "matvec":
        checksum = matvec(size, iterations)
    elif operation == "matmul":
        checksum = matmul(size, iterations)
    elif operation == "cosine1024":
        checksum = cosine1024(iterations)
    elif operation == "eigh":
        checksum = eigh(size, iterations)
    else:
        print("unknown operation: {}".format(operation))
        usage()
        return

    print(format_checksum(checksum))
