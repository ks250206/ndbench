import Accelerate
import Darwin
import Foundation
import simd

private let cosineDimension = 1024
private let defaultVectorIterations = 1_000_000
private let defaultPointCount = 100_000
private let defaultMatrixSize = 128
private let defaultCosineIterations = 1_000

private enum NdbenchError: Error, CustomStringConvertible {
    case missingOption(String)
    case invalidOption(String)
    case invalidValue(String)
    case lapackFailure(Int)

    var description: String {
        switch self {
        case .missingOption(let option):
            return "missing value for \(option)"
        case .invalidOption(let option):
            return "unknown option or argument: \(option)"
        case .invalidValue(let message):
            return message
        case .lapackFailure(let info):
            return "Accelerate dsyevd failed with info=\(info)"
        }
    }
}

private struct Options {
    let backend: String
    let operation: String
    let size: Int
    let iterations: Int
}

private let operationNames = [
    "vector2",
    "vector3",
    "affine2",
    "affine3",
    "matvec",
    "matmul",
    "cosine1024",
    "eigh",
]

private func printUsage() {
    print("""
    Usage: ndbench [--backend swift] --op OP [--size N] [--iterations N]

    Options:
      --backend NAME       Optional: swift, accelerate, or native (default: swift)
      --op NAME            vector2, vector3, affine2, affine3, matvec, matmul,
                           cosine1024, or eigh
      --operation NAME     Alias for --op
      --size N             Point count or square matrix order
      --iterations N       Number of repeated operations
      --help               Show this help
    """)
}

private func normalizeOperation(_ value: String) throws -> String {
    let aliases = [
        "vec2": "vector2",
        "vec3": "vector3",
        "diagonalize": "eigh",
    ]
    let operation = aliases[value] ?? value
    guard operationNames.contains(operation) else {
        throw NdbenchError.invalidValue(
            "unknown operation \(value); expected one of \(operationNames.joined(separator: ", "))"
        )
    }
    return operation
}

private func defaultSize(for operation: String) -> Int {
    switch operation {
    case "vector2", "vector3":
        return 1
    case "affine2", "affine3":
        return defaultPointCount
    case "cosine1024":
        return cosineDimension
    default:
        return defaultMatrixSize
    }
}

private func defaultIterations(for operation: String) -> Int {
    switch operation {
    case "vector2", "vector3":
        return defaultVectorIterations
    case "cosine1024":
        return defaultCosineIterations
    default:
        return 1
    }
}

private func parsePositiveInt(_ value: String, option: String) throws -> Int {
    guard let parsed = Int(value), parsed > 0 else {
        throw NdbenchError.invalidValue("\(option) must be a positive integer")
    }
    return parsed
}

private func parseArguments() throws -> Options {
    var backend = "swift"
    var operation: String?
    var size: Int?
    var iterations: Int?
    let arguments = Array(CommandLine.arguments.dropFirst())
    var index = 0

    while index < arguments.count {
        let argument = arguments[index]
        if argument == "--help" || argument == "-h" {
            printUsage()
            exit(0)
        }

        func nextValue(for option: String) throws -> String {
            let nextIndex = index + 1
            guard nextIndex < arguments.count else {
                throw NdbenchError.missingOption(option)
            }
            index = nextIndex
            return arguments[nextIndex]
        }

        switch argument {
        case "--backend":
            backend = try nextValue(for: argument)
        case "--op", "--operation":
            operation = try normalizeOperation(try nextValue(for: argument))
        case "--size":
            size = try parsePositiveInt(try nextValue(for: argument), option: argument)
        case "--iterations":
            iterations = try parsePositiveInt(try nextValue(for: argument), option: argument)
        default:
            if argument.hasPrefix("--backend=") {
                backend = String(argument.dropFirst("--backend=".count))
            } else if argument.hasPrefix("--op=") {
                operation = try normalizeOperation(String(argument.dropFirst("--op=".count)))
            } else if argument.hasPrefix("--operation=") {
                operation = try normalizeOperation(String(argument.dropFirst("--operation=".count)))
            } else if argument.hasPrefix("--size=") {
                size = try parsePositiveInt(
                    String(argument.dropFirst("--size=".count)),
                    option: "--size"
                )
            } else if argument.hasPrefix("--iterations=") {
                iterations = try parsePositiveInt(
                    String(argument.dropFirst("--iterations=".count)),
                    option: "--iterations"
                )
            } else {
                throw NdbenchError.invalidOption(argument)
            }
        }
        index += 1
    }

    guard let operation else {
        throw NdbenchError.missingOption("--op")
    }
    let acceptedBackends = ["swift", "accelerate", "native"]
    guard acceptedBackends.contains(backend) else {
        throw NdbenchError.invalidValue(
            "unknown backend \(backend); expected one of \(acceptedBackends.joined(separator: ", "))"
        )
    }

    return Options(
        backend: backend,
        operation: operation,
        size: size ?? defaultSize(for: operation),
        iterations: iterations ?? defaultIterations(for: operation)
    )
}

@inline(__always)
private func scalar(_ i: Int, _ j: Int, _ salt: Int) -> Double {
    let value = (i &* 37 &+ j &* 17 &+ salt &* 13) % 101
    return 0.125 + Double(value) / 101.0
}

@inline(__always)
private func tinyTerm(_ iteration: Int) -> Double {
    Double(iteration & 7) * 1.0e-12
}

@inline(__always)
private func embeddingScalar(_ index: Int, _ lane: Int) -> Double {
    scalar(index, lane, cosineDimension + lane) * 2.0 - 1.0
}

@inline(never)
@_optimize(none)
private func benchmarkBarrier(_ value: Double) -> Double {
    // Keep deterministic inputs observable to the optimizer while preserving
    // the same arithmetic as the Rust/Python raw baselines.
    return value
}

@inline(never)
private func vector2(_ iterations: Int) -> Double {
    let a = SIMD2<Double>(benchmarkBarrier(1.25), benchmarkBarrier(-2.5))
    let b = SIMD2<Double>(benchmarkBarrier(-0.75), benchmarkBarrier(3.0))
    var checksum = 0.0

    for iteration in 0..<iterations {
        let sum = a + b
        let dot = a.x * b.x + a.y * b.y
        let norm = sqrt(sum.x * sum.x + sum.y * sum.y)
        checksum += sum[iteration & 1] + dot + norm + tinyTerm(iteration)
    }

    return benchmarkBarrier(checksum)
}

@inline(never)
private func vector3(_ iterations: Int) -> Double {
    let a = SIMD3<Double>(benchmarkBarrier(1.25), benchmarkBarrier(-2.5), benchmarkBarrier(0.75))
    let b = SIMD3<Double>(benchmarkBarrier(-0.75), benchmarkBarrier(3.0), benchmarkBarrier(1.5))
    var checksum = 0.0

    for iteration in 0..<iterations {
        let sum = a + b
        let dot = a.x * b.x + a.y * b.y + a.z * b.z
        let norm = sqrt(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z)
        let cross = SIMD3<Double>(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        )
        checksum += sum[iteration % 3]
            + dot
            + norm
            + cross[(iteration + 1) % 3]
            + tinyTerm(iteration)
    }

    return benchmarkBarrier(checksum)
}

private func affineValues(_ dimension: Int) -> [Double] {
    switch dimension {
    case 2:
        return [1.1, -0.2, 0.5, 0.3, 0.9, -0.7, 0.0, 0.0, 1.0]
    case 3:
        return [
            1.1, -0.2, 0.1, 0.5,
            0.3, 0.9, -0.15, -0.7,
            0.05, 0.2, 1.05, 0.4,
            0.0, 0.0, 0.0, 1.0,
        ]
    default:
        preconditionFailure("unsupported affine dimension")
    }
}

@inline(never)
private func affine(_ points: Int, _ iterations: Int, _ dimension: Int) -> Double {
    let homogeneous = dimension + 1
    let transform = affineValues(dimension)
    var input = [Double](repeating: 0.0, count: homogeneous * points)
    for row in 0..<homogeneous {
        for column in 0..<points {
            input[row * points + column] = row == dimension
                ? 1.0
                : scalar(column, row, dimension)
        }
    }
    var checksum = 0.0

    for _ in 0..<iterations {
        var output = [Double](repeating: 0.0, count: homogeneous * points)
        for row in 0..<homogeneous {
            for column in 0..<points {
                var value = 0.0
                for inner in 0..<homogeneous {
                    value += transform[row * homogeneous + inner]
                        * input[inner * points + column]
                }
                output[row * points + column] = value
            }
        }

        var outputSum = 0.0
        for value in output {
            outputSum += value
        }
        checksum += outputSum
            + output[0]
            + output[dimension * points + points - 1]
    }

    return benchmarkBarrier(checksum)
}

@inline(never)
private func affine2(_ points: Int, _ iterations: Int) -> Double {
    affine(points, iterations, 2)
}

@inline(never)
private func affine3(_ points: Int, _ iterations: Int) -> Double {
    affine(points, iterations, 3)
}

private func scalarMatrixColumnMajor(_ size: Int, _ salt: Int) -> [Double] {
    var matrix = [Double](repeating: 0.0, count: size * size)
    for column in 0..<size {
        for row in 0..<size {
            matrix[row + column * size] = scalar(row, column, salt)
        }
    }
    return matrix
}

@inline(never)
private func matvec(_ size: Int, _ iterations: Int) -> Double {
    let matrix = scalarMatrixColumnMajor(size, size)
    var vector = [Double](repeating: 0.0, count: size)
    for row in 0..<size {
        vector[row] = scalar(row, 0, size + 1)
    }
    var checksum = 0.0

    for _ in 0..<iterations {
        var output = [Double](repeating: 0.0, count: size)
        matrix.withUnsafeBufferPointer { matrixBuffer in
            vector.withUnsafeBufferPointer { vectorBuffer in
                output.withUnsafeMutableBufferPointer { outputBuffer in
                    cblas_dgemv(
                        CblasColMajor,
                        CblasNoTrans,
                        size,
                        size,
                        1.0,
                        matrixBuffer.baseAddress,
                        size,
                        vectorBuffer.baseAddress,
                        1,
                        0.0,
                        outputBuffer.baseAddress,
                        1
                    )
                }
            }
        }

        var outputSum = 0.0
        for value in output {
            outputSum += value
        }
        checksum += outputSum + output[0]
    }

    return benchmarkBarrier(checksum)
}

@inline(never)
private func matmul(_ size: Int, _ iterations: Int) -> Double {
    let left = scalarMatrixColumnMajor(size, size)
    let right = scalarMatrixColumnMajor(size, size + 1)
    var checksum = 0.0

    for _ in 0..<iterations {
        var output = [Double](repeating: 0.0, count: size * size)
        left.withUnsafeBufferPointer { leftBuffer in
            right.withUnsafeBufferPointer { rightBuffer in
                output.withUnsafeMutableBufferPointer { outputBuffer in
                    cblas_dgemm(
                        CblasColMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        size,
                        size,
                        size,
                        1.0,
                        leftBuffer.baseAddress,
                        size,
                        rightBuffer.baseAddress,
                        size,
                        0.0,
                        outputBuffer.baseAddress,
                        size
                    )
                }
            }
        }

        var outputSum = 0.0
        for value in output {
            outputSum += value
        }
        checksum += outputSum + output[0]
    }

    return benchmarkBarrier(checksum)
}

@inline(never)
private func cosine1024(_ iterations: Int) -> Double {
    let a = (0..<cosineDimension).map { embeddingScalar($0, 0) }
    let b = (0..<cosineDimension).map { embeddingScalar($0, 1) }
    var checksum = 0.0

    for iteration in 0..<iterations {
        var dot = 0.0
        var normASquared = 0.0
        var normBSquared = 0.0
        for index in 0..<cosineDimension {
            dot += a[index] * b[index]
            normASquared += a[index] * a[index]
            normBSquared += b[index] * b[index]
        }
        let similarity = dot / (sqrt(normASquared) * sqrt(normBSquared))
        checksum += similarity + tinyTerm(iteration)
    }

    return benchmarkBarrier(checksum)
}

private func symmetricMatrixColumnMajor(_ size: Int) -> [Double] {
    var matrix = [Double](repeating: 0.0, count: size * size)
    for column in 0..<size {
        for row in 0..<size {
            if row == column {
                matrix[row + column * size] = Double(size + 2)
            } else {
                let distance = Double(abs(row - column))
                matrix[row + column * size] = 0.01
                    * scalar(min(row, column), max(row, column), size)
                    / (1.0 + distance)
            }
        }
    }
    return matrix
}

private func lapackWorkspaceSizes(_ input: [Double], _ size: Int) throws -> (Int, Int) {
    var matrix = input
    var eigenvalues = [Double](repeating: 0.0, count: size)
    var work = [Double](repeating: 0.0, count: 1)
    var iwork = [Int](repeating: 0, count: 1)
    var jobz: CChar = 86 // 'V': eigenvalues and eigenvectors
    var uplo: CChar = 76 // 'L': lower triangle
    var n = size
    var lda = size
    var lwork = -1
    var liwork = -1
    var info = 0

    matrix.withUnsafeMutableBufferPointer { matrixBuffer in
        eigenvalues.withUnsafeMutableBufferPointer { eigenvalueBuffer in
            work.withUnsafeMutableBufferPointer { workBuffer in
                iwork.withUnsafeMutableBufferPointer { iworkBuffer in
                    dsyevd_(
                        &jobz,
                        &uplo,
                        &n,
                        matrixBuffer.baseAddress,
                        &lda,
                        eigenvalueBuffer.baseAddress,
                        workBuffer.baseAddress!,
                        &lwork,
                        iworkBuffer.baseAddress,
                        &liwork,
                        &info
                    )
                }
            }
        }
    }

    guard info == 0 else {
        throw NdbenchError.lapackFailure(info)
    }
    let workCount = max(1, Int(work[0].rounded(.up)))
    let integerWorkCount = max(1, iwork[0])
    return (workCount, integerWorkCount)
}

private func lapackEigh(
    _ input: [Double],
    _ size: Int,
    workCount: Int,
    integerWorkCount: Int
) throws -> ([Double], [Double]) {
    var matrix = input
    var eigenvalues = [Double](repeating: 0.0, count: size)
    var work = [Double](repeating: 0.0, count: workCount)
    var iwork = [Int](repeating: 0, count: integerWorkCount)
    var jobz: CChar = 86 // 'V': eigenvalues and eigenvectors
    var uplo: CChar = 76 // 'L': lower triangle
    var n = size
    var lda = size
    var lwork = workCount
    var liwork = integerWorkCount
    var info = 0

    matrix.withUnsafeMutableBufferPointer { matrixBuffer in
        eigenvalues.withUnsafeMutableBufferPointer { eigenvalueBuffer in
            work.withUnsafeMutableBufferPointer { workBuffer in
                iwork.withUnsafeMutableBufferPointer { iworkBuffer in
                    dsyevd_(
                        &jobz,
                        &uplo,
                        &n,
                        matrixBuffer.baseAddress,
                        &lda,
                        eigenvalueBuffer.baseAddress,
                        workBuffer.baseAddress!,
                        &lwork,
                        iworkBuffer.baseAddress,
                        &liwork,
                        &info
                    )
                }
            }
        }
    }

    guard info == 0 else {
        throw NdbenchError.lapackFailure(info)
    }
    return (eigenvalues, matrix)
}

@inline(never)
private func eigh(_ size: Int, _ iterations: Int) throws -> Double {
    let matrix = symmetricMatrixColumnMajor(size)
    let (workCount, integerWorkCount) = try lapackWorkspaceSizes(matrix, size)
    var checksum = 0.0

    for _ in 0..<iterations {
        let (eigenvalues, eigenvectors) = try lapackEigh(
            matrix,
            size,
            workCount: workCount,
            integerWorkCount: integerWorkCount
        )
        var eigenvalueSum = 0.0
        for value in eigenvalues {
            eigenvalueSum += value
        }
        var eigenvectorNormSquared = 0.0
        for value in eigenvectors {
            eigenvectorNormSquared += value * value
        }
        checksum += eigenvalueSum + eigenvectorNormSquared
    }

    return benchmarkBarrier(checksum)
}

private func run(_ options: Options) throws -> Double {
    switch options.operation {
    case "vector2":
        return vector2(options.iterations)
    case "vector3":
        return vector3(options.iterations)
    case "affine2":
        return affine2(options.size, options.iterations)
    case "affine3":
        return affine3(options.size, options.iterations)
    case "matvec":
        return matvec(options.size, options.iterations)
    case "matmul":
        return matmul(options.size, options.iterations)
    case "cosine1024":
        return cosine1024(options.iterations)
    case "eigh":
        return try eigh(options.size, options.iterations)
    default:
        throw NdbenchError.invalidValue("unsupported operation \(options.operation)")
    }
}

private func configureAccelerateThreads() {
    // The benchmark scripts export these before process launch as well. Keep
    // direct CLI runs deterministic when they are invoked without the scripts.
    _ = setenv("VECLIB_MAXIMUM_THREADS", "1", 0)
    _ = setenv("VECLIB_MAXIMUM_NUMBER_OF_THREADS", "1", 0)
    _ = setenv("OMP_NUM_THREADS", "1", 0)
}

do {
    let options = try parseArguments()
    configureAccelerateThreads()
    let checksum = try run(options)
    let formatted = String(
        format: "checksum=%.17e",
        locale: Locale(identifier: "en_US_POSIX"),
        checksum
    ).replacingOccurrences(of: "E", with: "e")
    print(formatted)
} catch {
    fputs("error: \(error)\n", stderr)
    fputs("Run --help for usage.\n", stderr)
    exit(1)
}
