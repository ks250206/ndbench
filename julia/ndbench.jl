#!/usr/bin/env julia

"""Julia CPU backend for the ndbench operation set."""

using LinearAlgebra
using Printf

const DEFAULT_VECTOR_ITERATIONS = 1_000_000
const DEFAULT_POINT_COUNT = 100_000
const DEFAULT_MATRIX_SIZE = 128
const COSINE_DIMENSION = 1024
const DEFAULT_COSINE_ITERATIONS = 1_000

# Keep the Julia BLAS/LAPACK layer serial so this backend has the same CPU
# threading policy as the Rust and Python benchmark scripts.
BLAS.set_num_threads(1)

@inline function scalar(i::Int, j::Int, salt::Int)::Float64
    value = mod(i * 37 + j * 17 + salt * 13, 101)
    return 0.125 + Float64(value) / 101.0
end

@inline tiny_term(iteration::Int)::Float64 = Float64(iteration & 7) * 1.0e-12

@inline function embedding_scalar(index::Int, lane::Int)::Float64
    return scalar(index, lane, COSINE_DIMENSION + lane) * 2.0 - 1.0
end

function affine_values(dimension::Int)::Tuple
    if dimension == 2
        return (1.1, -0.2, 0.5, 0.3, 0.9, -0.7, 0.0, 0.0, 1.0)
    elseif dimension == 3
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
    end
    error("unsupported affine dimension: $dimension")
end

"""Sum a vector in index order, matching the scalar checksum reference."""
function sequential_sum(values)::Float64
    total = 0.0
    @inbounds for index in eachindex(values)
        total += values[index]
    end
    return total
end

"""Sum a matrix in row-major logical order, independent of Julia storage order."""
function row_major_sum(matrix)::Float64
    total = 0.0
    @inbounds for row in axes(matrix, 1)
        for column in axes(matrix, 2)
            total += matrix[row, column]
        end
    end
    return total
end

function row_major_sum_squares(matrix)::Float64
    total = 0.0
    @inbounds for row in axes(matrix, 1)
        for column in axes(matrix, 2)
            value = matrix[row, column]
            total += value * value
        end
    end
    return total
end

function vector2(iterations::Int)::Float64
    a = [1.25, -2.5]
    b = [-0.75, 3.0]
    checksum = 0.0

    for iteration in 0:(iterations - 1)
        summed = a + b
        dot_product = dot(a, b)
        norm_value = norm(summed)
        checksum += summed[(iteration & 1) + 1] + dot_product + norm_value + tiny_term(iteration)
    end

    return checksum
end

function vector3(iterations::Int)::Float64
    a = [1.25, -2.5, 0.75]
    b = [-0.75, 3.0, 1.5]
    checksum = 0.0

    for iteration in 0:(iterations - 1)
        summed = a + b
        dot_product = dot(a, b)
        norm_value = norm(summed)
        cross_product = cross(a, b)
        checksum +=
            summed[mod(iteration, 3) + 1] +
            dot_product +
            norm_value +
            cross_product[mod(iteration + 1, 3) + 1] +
            tiny_term(iteration)
    end

    return checksum
end

function affine_transform(dimension::Int)::Matrix{Float64}
    homogeneous = dimension + 1
    values = affine_values(dimension)
    transform = Matrix{Float64}(undef, homogeneous, homogeneous)
    @inbounds for row in 1:homogeneous
        for column in 1:homogeneous
            # affine_values follows the row-major Rust/Python literal order.
            transform[row, column] = values[(row - 1) * homogeneous + column]
        end
    end
    return transform
end

function affine_input(points::Int, dimension::Int)::Matrix{Float64}
    homogeneous = dimension + 1
    input = Matrix{Float64}(undef, homogeneous, points)
    @inbounds for row in 0:(homogeneous - 1)
        for column in 0:(points - 1)
            input[row + 1, column + 1] =
                row == dimension ? 1.0 : scalar(column, row, dimension)
        end
    end
    return input
end

function affine(points::Int, iterations::Int, dimension::Int)::Float64
    transform = affine_transform(dimension)
    input = affine_input(points, dimension)
    checksum = 0.0
    homogeneous = dimension + 1

    for _ in 1:iterations
        output = transform * input
        checksum += row_major_sum(output) + output[1, 1] + output[homogeneous, points]
    end

    return checksum
end

affine2(points::Int, iterations::Int)::Float64 = affine(points, iterations, 2)
affine3(points::Int, iterations::Int)::Float64 = affine(points, iterations, 3)

function scalar_matrix(size::Int, salt::Int)::Matrix{Float64}
    matrix = Matrix{Float64}(undef, size, size)
    @inbounds for row in 0:(size - 1)
        for column in 0:(size - 1)
            matrix[row + 1, column + 1] = scalar(row, column, salt)
        end
    end
    return matrix
end

function matvec(size::Int, iterations::Int)::Float64
    matrix = scalar_matrix(size, size)
    vector = Vector{Float64}(undef, size)
    @inbounds for row in 0:(size - 1)
        vector[row + 1] = scalar(row, 0, size + 1)
    end
    checksum = 0.0

    for _ in 1:iterations
        output = matrix * vector
        checksum += sequential_sum(output) + output[1]
    end

    return checksum
end

function matmul(size::Int, iterations::Int)::Float64
    left = scalar_matrix(size, size)
    right = scalar_matrix(size, size + 1)
    checksum = 0.0

    for _ in 1:iterations
        output = left * right
        checksum += row_major_sum(output) + output[1, 1]
    end

    return checksum
end

function cosine1024(iterations::Int)::Float64
    a = [embedding_scalar(index, 0) for index in 0:(COSINE_DIMENSION - 1)]
    b = [embedding_scalar(index, 1) for index in 0:(COSINE_DIMENSION - 1)]
    checksum = 0.0

    for iteration in 0:(iterations - 1)
        dot_product = dot(a, b)
        norm_a_squared = dot(a, a)
        norm_b_squared = dot(b, b)
        similarity = dot_product / (sqrt(norm_a_squared) * sqrt(norm_b_squared))
        checksum += similarity + tiny_term(iteration)
    end

    return checksum
end

function symmetric_matrix(size::Int)::Matrix{Float64}
    matrix = Matrix{Float64}(undef, size, size)
    @inbounds for row in 0:(size - 1)
        for column in 0:(size - 1)
            if row == column
                matrix[row + 1, column + 1] = Float64(size + 2)
            else
                distance = Float64(abs(row - column))
                matrix[row + 1, column + 1] =
                    0.01 * scalar(min(row, column), max(row, column), size) / (1.0 + distance)
            end
        end
    end
    return matrix
end

"""Run a full real symmetric eigendecomposition using Julia's LAPACK-backed API."""
function eigh_checksum(size::Int, iterations::Int)::Float64
    matrix = symmetric_matrix(size)
    checksum = 0.0

    for _ in 1:iterations
        decomposition = eigen(Symmetric(matrix))
        checksum += sequential_sum(decomposition.values) + row_major_sum_squares(decomposition.vectors)
    end

    return checksum
end

function usage()::String
    return """Usage:
  julia --project=. ndbench.jl --op <operation> [options]

Operations:
  vector2, vector3    low-dimensional vector arithmetic repeated --iterations times
  affine2, affine3    homogeneous affine transform of --size points
  matvec              dense matrix-vector product of a --size square matrix
  matmul              dense square matrix product of order --size
  cosine1024          cosine similarity of two f64 vectors of length 1024
  eigh                full symmetric eigendecomposition of order --size

Options:
  --op, --operation <name>  operation to run
  --size <N>                point count or matrix order, depending on operation
  --iterations <N>          repeat the operation N times
  --help                    show this message
"""
end

function parse_positive_integer(flag::String, value::String)::Int
    parsed = tryparse(Int, value)
    parsed === nothing && error("invalid value for $flag: $value")
    parsed > 0 || error("$flag must be greater than zero")
    return parsed
end

function normalize_operation(value::String)::Symbol
    if value == "vector2" || value == "vec2"
        return :vector2
    elseif value == "vector3" || value == "vec3"
        return :vector3
    elseif value == "affine2"
        return :affine2
    elseif value == "affine3"
        return :affine3
    elseif value == "matvec"
        return :matvec
    elseif value == "matmul"
        return :matmul
    elseif value == "cosine1024" || value == "cosine"
        return :cosine1024
    elseif value == "eigh" || value == "diagonalize"
        return :eigh
    end
    error("unknown operation `$value`; expected vector2, vector3, affine2, affine3, matvec, matmul, cosine1024, or eigh")
end

function default_size(operation::Symbol)::Int
    if operation === :vector2 || operation === :vector3
        return 1
    elseif operation === :affine2 || operation === :affine3
        return DEFAULT_POINT_COUNT
    elseif operation === :cosine1024
        return COSINE_DIMENSION
    end
    return DEFAULT_MATRIX_SIZE
end

function default_iterations(operation::Symbol)::Int
    if operation === :vector2 || operation === :vector3
        return DEFAULT_VECTOR_ITERATIONS
    elseif operation === :cosine1024
        return DEFAULT_COSINE_ITERATIONS
    end
    return 1
end

function next_option_value(args::Vector{String}, index::Int, flag::String)
    index == length(args) && error("missing value after $flag")
    return args[index + 1], index + 2
end

function parse_args(args::Vector{String})
    operation_name = nothing
    size = nothing
    iterations = nothing
    index = 1

    while index <= length(args)
        argument = args[index]
        if argument == "--help" || argument == "-h"
            return nothing
        elseif argument == "--op" || argument == "--operation"
            operation_name, index = next_option_value(args, index, argument)
        elseif startswith(argument, "--op=")
            operation_name = argument[6:end]
            index += 1
        elseif startswith(argument, "--operation=")
            operation_name = argument[13:end]
            index += 1
        elseif argument == "--size"
            value, index = next_option_value(args, index, argument)
            size = parse_positive_integer(argument, value)
        elseif startswith(argument, "--size=")
            size = parse_positive_integer("--size", argument[8:end])
            index += 1
        elseif argument == "--iterations"
            value, index = next_option_value(args, index, argument)
            iterations = parse_positive_integer(argument, value)
        elseif startswith(argument, "--iterations=")
            iterations = parse_positive_integer("--iterations", argument[14:end])
            index += 1
        else
            error("unknown argument `$argument`\n\n$(usage())")
        end
    end

    operation_name === nothing && error("missing --op or --operation\n\n$(usage())")
    operation = normalize_operation(String(operation_name))
    resolved_size = size === nothing ? default_size(operation) : size
    resolved_iterations = iterations === nothing ? default_iterations(operation) : iterations
    return (operation=operation, size=resolved_size, iterations=resolved_iterations)
end

function run_operation(operation::Symbol, size::Int, iterations::Int)::Float64
    if operation === :vector2
        return vector2(iterations)
    elseif operation === :vector3
        return vector3(iterations)
    elseif operation === :affine2
        return affine2(size, iterations)
    elseif operation === :affine3
        return affine3(size, iterations)
    elseif operation === :matvec
        return matvec(size, iterations)
    elseif operation === :matmul
        return matmul(size, iterations)
    elseif operation === :cosine1024
        return cosine1024(iterations)
    elseif operation === :eigh
        return eigh_checksum(size, iterations)
    end
    error("unsupported operation: $operation")
end

function main(args::Vector{String}=ARGS)::Int
    try
        config = parse_args(args)
        if config === nothing
            println(usage())
            return 0
        end
        checksum = run_operation(config.operation, config.size, config.iterations)
        @printf("checksum=%.17e\n", checksum)
        return 0
    catch exception
        println(stderr, "error: ", sprint(showerror, exception))
        return 1
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    exit(main())
end
