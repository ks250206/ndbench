use std::env;
use std::hint::black_box;
use std::process::ExitCode;

use candle_core::{Device as CandleDevice, Tensor as CandleTensor};
use faer::{Accum, Mat, Par, Side};
use nalgebra::{DMatrix, DVector, SymmetricEigen, Vector2, Vector3};
use ndarray::{Array1, Array2, ShapeBuilder};

mod burn_backend;
mod raw_backend;

const DEFAULT_VECTOR_ITERATIONS: usize = 1_000_000;
const DEFAULT_POINT_COUNT: usize = 100_000;
const DEFAULT_MATRIX_SIZE: usize = 128;
const COSINE_DIMENSION: usize = 1024;
const DEFAULT_COSINE_ITERATIONS: usize = 1_000;

#[derive(Clone, Copy, Debug)]
enum Backend {
    Ndarray,
    Faer,
    Nalgebra,
    Candle,
    Burn,
    Raw,
}

impl Backend {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "ndarray" => Ok(Self::Ndarray),
            "faer" => Ok(Self::Faer),
            "nalgebra" => Ok(Self::Nalgebra),
            "candle" => Ok(Self::Candle),
            "burn" => Ok(Self::Burn),
            "raw" | "native" => Ok(Self::Raw),
            _ => Err(format!(
                "unknown backend `{value}`; expected ndarray, faer, nalgebra, candle, burn, or raw"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Operation {
    Vector2,
    Vector3,
    Affine2,
    Affine3,
    MatVec,
    MatMul,
    Cosine1024,
    Eigh,
}

impl Operation {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "vector2" | "vec2" => Ok(Self::Vector2),
            "vector3" | "vec3" => Ok(Self::Vector3),
            "affine2" => Ok(Self::Affine2),
            "affine3" => Ok(Self::Affine3),
            "matvec" => Ok(Self::MatVec),
            "matmul" => Ok(Self::MatMul),
            "cosine1024" | "cosine" => Ok(Self::Cosine1024),
            "eigh" | "diagonalize" => Ok(Self::Eigh),
            _ => Err(format!(
                "unknown operation `{value}`; expected vector2, vector3, affine2, affine3, matvec, matmul, cosine1024, or eigh"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Config {
    backend: Backend,
    operation: Operation,
    size: usize,
    iterations: usize,
}

fn usage() -> &'static str {
    "Usage:
  ndbench --backend <ndarray|faer|nalgebra|candle|burn|raw> --op <operation> [options]

Operations:
  vector2, vector3    low-dimensional vector arithmetic repeated --iterations times
  affine2, affine3    homogeneous affine transform of --size points
  matvec              dense matrix-vector product of a --size square matrix
  matmul              dense square matrix product of order --size
  cosine1024          cosine similarity of two f64 vectors of length 1024
  eigh                full symmetric eigendecomposition of order --size

Options:
  --size <N>          point count or matrix order, depending on operation
  --iterations <N>    repeat the operation N times
  --help              show this message

Examples:
  ndbench --backend nalgebra --op vector3 --iterations 1000000
  ndbench --backend faer --op matmul --size 256
  ndbench --backend ndarray --op eigh --size 128
  ndbench --backend candle --op matmul --size 256
  ndbench --backend nalgebra --op cosine1024 --iterations 1000
  ndbench --backend burn --op cosine1024 --iterations 1000
  ndbench --backend raw --op matmul --size 256
"
}

fn parse_usize(flag: &str, value: Option<String>) -> Result<usize, String> {
    let value = value.ok_or_else(|| format!("missing value after {flag}"))?;
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid value for {flag}: {error}"))
}

fn parse_args() -> Result<Option<Config>, String> {
    let mut args = env::args().skip(1);
    let mut backend = None;
    let mut operation = None;
    let mut size = None;
    let mut iterations = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--backend" => {
                backend =
                    Some(Backend::parse(&args.next().ok_or_else(|| {
                        "missing value after --backend".to_owned()
                    })?)?)
            }
            "--op" | "--operation" => {
                operation = Some(Operation::parse(
                    &args
                        .next()
                        .ok_or_else(|| "missing value after --op".to_owned())?,
                )?)
            }
            "--size" => size = Some(parse_usize("--size", args.next())?),
            "--iterations" => iterations = Some(parse_usize("--iterations", args.next())?),
            other => return Err(format!("unknown argument `{other}`\n\n{}", usage())),
        }
    }

    let backend = backend.ok_or_else(|| format!("missing --backend\n\n{}", usage()))?;
    let operation = operation.ok_or_else(|| format!("missing --op\n\n{}", usage()))?;
    let default_size = match operation {
        Operation::Vector2 | Operation::Vector3 => 1,
        Operation::Affine2 | Operation::Affine3 => DEFAULT_POINT_COUNT,
        Operation::MatVec | Operation::MatMul | Operation::Eigh => DEFAULT_MATRIX_SIZE,
        Operation::Cosine1024 => COSINE_DIMENSION,
    };
    let default_iterations = match operation {
        Operation::Vector2 | Operation::Vector3 => DEFAULT_VECTOR_ITERATIONS,
        Operation::Cosine1024 => DEFAULT_COSINE_ITERATIONS,
        _ => 1,
    };

    let config = Config {
        backend,
        operation,
        size: size.unwrap_or(default_size),
        iterations: iterations.unwrap_or(default_iterations),
    };

    if config.size == 0 {
        return Err("--size must be greater than zero".to_owned());
    }
    if config.iterations == 0 {
        return Err("--iterations must be greater than zero".to_owned());
    }

    Ok(Some(config))
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let Some(config) = parse_args()? else {
        println!("{}", usage());
        return Ok(());
    };

    // Keep CPU comparisons single-threaded. The benchmark scripts also set
    // BLAS/OpenBLAS thread limits for the optional ndarray eigensolver.
    faer::set_global_parallelism(Par::Seq);

    let checksum = match (config.backend, config.operation) {
        (Backend::Ndarray, Operation::Vector2) => Ok(ndarray_vector2(config.iterations)),
        (Backend::Faer, Operation::Vector2) => Ok(faer_vector2(config.iterations)),
        (Backend::Nalgebra, Operation::Vector2) => Ok(nalgebra_vector2(config.iterations)),
        (Backend::Candle, Operation::Vector2) => candle_vector2(config.iterations),
        (Backend::Burn, Operation::Vector2) => burn_backend::vector2(config.iterations),
        (Backend::Raw, Operation::Vector2) => raw_backend::vector2(config.iterations),
        (Backend::Ndarray, Operation::Vector3) => Ok(ndarray_vector3(config.iterations)),
        (Backend::Faer, Operation::Vector3) => Ok(faer_vector3(config.iterations)),
        (Backend::Nalgebra, Operation::Vector3) => Ok(nalgebra_vector3(config.iterations)),
        (Backend::Candle, Operation::Vector3) => candle_vector3(config.iterations),
        (Backend::Burn, Operation::Vector3) => burn_backend::vector3(config.iterations),
        (Backend::Raw, Operation::Vector3) => raw_backend::vector3(config.iterations),
        (Backend::Ndarray, Operation::Affine2) => {
            Ok(ndarray_affine2(config.size, config.iterations))
        }
        (Backend::Faer, Operation::Affine2) => Ok(faer_affine2(config.size, config.iterations)),
        (Backend::Nalgebra, Operation::Affine2) => {
            Ok(nalgebra_affine2(config.size, config.iterations))
        }
        (Backend::Candle, Operation::Affine2) => candle_affine2(config.size, config.iterations),
        (Backend::Burn, Operation::Affine2) => {
            burn_backend::affine2(config.size, config.iterations)
        }
        (Backend::Raw, Operation::Affine2) => raw_backend::affine2(config.size, config.iterations),
        (Backend::Ndarray, Operation::Affine3) => {
            Ok(ndarray_affine3(config.size, config.iterations))
        }
        (Backend::Faer, Operation::Affine3) => Ok(faer_affine3(config.size, config.iterations)),
        (Backend::Nalgebra, Operation::Affine3) => {
            Ok(nalgebra_affine3(config.size, config.iterations))
        }
        (Backend::Candle, Operation::Affine3) => candle_affine3(config.size, config.iterations),
        (Backend::Burn, Operation::Affine3) => {
            burn_backend::affine3(config.size, config.iterations)
        }
        (Backend::Raw, Operation::Affine3) => raw_backend::affine3(config.size, config.iterations),
        (Backend::Ndarray, Operation::MatVec) => Ok(ndarray_matvec(config.size, config.iterations)),
        (Backend::Faer, Operation::MatVec) => Ok(faer_matvec(config.size, config.iterations)),
        (Backend::Nalgebra, Operation::MatVec) => {
            Ok(nalgebra_matvec(config.size, config.iterations))
        }
        (Backend::Candle, Operation::MatVec) => candle_matvec(config.size, config.iterations),
        (Backend::Burn, Operation::MatVec) => burn_backend::matvec(config.size, config.iterations),
        (Backend::Raw, Operation::MatVec) => raw_backend::matvec(config.size, config.iterations),
        (Backend::Ndarray, Operation::MatMul) => Ok(ndarray_matmul(config.size, config.iterations)),
        (Backend::Faer, Operation::MatMul) => Ok(faer_matmul(config.size, config.iterations)),
        (Backend::Nalgebra, Operation::MatMul) => {
            Ok(nalgebra_matmul(config.size, config.iterations))
        }
        (Backend::Candle, Operation::MatMul) => candle_matmul(config.size, config.iterations),
        (Backend::Burn, Operation::MatMul) => burn_backend::matmul(config.size, config.iterations),
        (Backend::Raw, Operation::MatMul) => raw_backend::matmul(config.size, config.iterations),
        (Backend::Ndarray, Operation::Cosine1024) => Ok(ndarray_cosine1024(config.iterations)),
        (Backend::Faer, Operation::Cosine1024) => Ok(faer_cosine1024(config.iterations)),
        (Backend::Nalgebra, Operation::Cosine1024) => Ok(nalgebra_cosine1024(config.iterations)),
        (Backend::Candle, Operation::Cosine1024) => candle_cosine1024(config.iterations),
        (Backend::Burn, Operation::Cosine1024) => burn_backend::cosine1024(config.iterations),
        (Backend::Raw, Operation::Cosine1024) => raw_backend::cosine1024(config.iterations),
        (Backend::Ndarray, Operation::Eigh) => ndarray_eigh(config.size, config.iterations),
        (Backend::Faer, Operation::Eigh) => faer_eigh(config.size, config.iterations),
        (Backend::Nalgebra, Operation::Eigh) => Ok(nalgebra_eigh(config.size, config.iterations)),
        (Backend::Candle, Operation::Eigh) => candle_eigh(config.size, config.iterations),
        (Backend::Burn, Operation::Eigh) => Err(burn_backend::eigh_error()),
        (Backend::Raw, Operation::Eigh) => raw_backend::eigh(config.size, config.iterations),
    }?;

    println!("checksum={checksum:.17e}");
    Ok(())
}

fn scalar(i: usize, j: usize, salt: usize) -> f64 {
    let value = (i
        .wrapping_mul(37)
        .wrapping_add(j.wrapping_mul(17))
        .wrapping_add(salt.wrapping_mul(13))
        % 101) as f64;
    0.125 + value / 101.0
}

fn symmetric_scalar(i: usize, j: usize, n: usize) -> f64 {
    if i == j {
        n as f64 + 2.0
    } else {
        let distance = i.abs_diff(j) as f64;
        0.01 * scalar(i.min(j), i.max(j), n) / (1.0 + distance)
    }
}

#[allow(clippy::op_ref)]
fn ndarray_vector2(iterations: usize) -> f64 {
    let a = black_box(Array1::<f64>::from_vec(vec![1.25, -2.5]));
    let b = black_box(Array1::<f64>::from_vec(vec![-0.75, 3.0]));
    let mut checksum = 0.0;

    for i in 0..iterations {
        let sum = &a + &b;
        let dot = a.dot(&b);
        let norm = sum.dot(&sum).sqrt();
        checksum += sum[i & 1] + dot + norm + black_box((i & 7) as f64 * 1.0e-12);
    }

    black_box(checksum)
}

fn faer_vector2(iterations: usize) -> f64 {
    let a = black_box(Mat::from_fn(2, 1, |i, _| [1.25, -2.5][i]));
    let b = black_box(Mat::from_fn(2, 1, |i, _| [-0.75, 3.0][i]));
    let mut checksum = 0.0;

    for i in 0..iterations {
        let sum = &a + &b;
        let dot = a[(0, 0)] * b[(0, 0)] + a[(1, 0)] * b[(1, 0)];
        let norm = sum.as_ref().norm_l2();
        checksum += sum[(i & 1, 0)] + dot + norm + black_box((i & 7) as f64 * 1.0e-12);
    }

    black_box(checksum)
}

fn nalgebra_vector2(iterations: usize) -> f64 {
    let a = black_box(Vector2::new(1.25, -2.5));
    let b = black_box(Vector2::new(-0.75, 3.0));
    let mut checksum = 0.0;

    for i in 0..iterations {
        let sum = a + b;
        let dot = a.dot(&b);
        let norm = sum.norm();
        checksum += sum[i & 1] + dot + norm + black_box((i & 7) as f64 * 1.0e-12);
    }

    black_box(checksum)
}

#[allow(clippy::op_ref)]
fn ndarray_vector3(iterations: usize) -> f64 {
    let a = black_box(Array1::<f64>::from_vec(vec![1.25, -2.5, 0.75]));
    let b = black_box(Array1::<f64>::from_vec(vec![-0.75, 3.0, 1.5]));
    let mut checksum = 0.0;

    for i in 0..iterations {
        let sum = &a + &b;
        let dot = a.dot(&b);
        let norm = sum.dot(&sum).sqrt();
        let cross = Array1::from_vec(vec![
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]);
        checksum +=
            sum[i % 3] + dot + norm + cross[(i + 1) % 3] + black_box((i & 7) as f64 * 1.0e-12);
    }

    black_box(checksum)
}

fn faer_vector3(iterations: usize) -> f64 {
    let a = black_box(Mat::from_fn(3, 1, |i, _| [1.25, -2.5, 0.75][i]));
    let b = black_box(Mat::from_fn(3, 1, |i, _| [-0.75, 3.0, 1.5][i]));
    let mut checksum = 0.0;

    for i in 0..iterations {
        let sum = &a + &b;
        let dot = (0..3).map(|j| a[(j, 0)] * b[(j, 0)]).sum::<f64>();
        let norm = sum.as_ref().norm_l2();
        let cross = [
            a[(1, 0)] * b[(2, 0)] - a[(2, 0)] * b[(1, 0)],
            a[(2, 0)] * b[(0, 0)] - a[(0, 0)] * b[(2, 0)],
            a[(0, 0)] * b[(1, 0)] - a[(1, 0)] * b[(0, 0)],
        ];
        checksum +=
            sum[(i % 3, 0)] + dot + norm + cross[(i + 1) % 3] + black_box((i & 7) as f64 * 1.0e-12);
    }

    black_box(checksum)
}

fn nalgebra_vector3(iterations: usize) -> f64 {
    let a = black_box(Vector3::new(1.25, -2.5, 0.75));
    let b = black_box(Vector3::new(-0.75, 3.0, 1.5));
    let mut checksum = 0.0;

    for i in 0..iterations {
        let sum = a + b;
        let dot = a.dot(&b);
        let norm = sum.norm();
        let cross = a.cross(&b);
        checksum +=
            sum[i % 3] + dot + norm + cross[(i + 1) % 3] + black_box((i & 7) as f64 * 1.0e-12);
    }

    black_box(checksum)
}

fn embedding_scalar(index: usize, lane: usize) -> f64 {
    scalar(index, lane, COSINE_DIMENSION + lane) * 2.0 - 1.0
}

fn ndarray_cosine1024(iterations: usize) -> f64 {
    let a = Array1::from_shape_fn(COSINE_DIMENSION, |index| embedding_scalar(index, 0));
    let b = Array1::from_shape_fn(COSINE_DIMENSION, |index| embedding_scalar(index, 1));
    let mut checksum = 0.0;

    for iteration in 0..iterations {
        let dot = a.dot(&b);
        let norm_a = a.dot(&a).sqrt();
        let norm_b = b.dot(&b).sqrt();
        let similarity = dot / (norm_a * norm_b);
        checksum += similarity + black_box((iteration & 7) as f64 * 1.0e-12);
    }

    black_box(checksum)
}

fn faer_cosine1024(iterations: usize) -> f64 {
    let a = Mat::from_fn(COSINE_DIMENSION, 1, |index, _| embedding_scalar(index, 0));
    let b = Mat::from_fn(COSINE_DIMENSION, 1, |index, _| embedding_scalar(index, 1));
    let mut checksum = 0.0;

    for iteration in 0..iterations {
        let dot = (0..COSINE_DIMENSION)
            .map(|index| a[(index, 0)] * b[(index, 0)])
            .sum::<f64>();
        let norm_a = a.as_ref().norm_l2();
        let norm_b = b.as_ref().norm_l2();
        let similarity = dot / (norm_a * norm_b);
        checksum += similarity + black_box((iteration & 7) as f64 * 1.0e-12);
    }

    black_box(checksum)
}

fn nalgebra_cosine1024(iterations: usize) -> f64 {
    let a = DVector::from_fn(COSINE_DIMENSION, |index, _| embedding_scalar(index, 0));
    let b = DVector::from_fn(COSINE_DIMENSION, |index, _| embedding_scalar(index, 1));
    let mut checksum = 0.0;

    for iteration in 0..iterations {
        let dot = a.dot(&b);
        let similarity = dot / (a.norm() * b.norm());
        checksum += similarity + black_box((iteration & 7) as f64 * 1.0e-12);
    }

    black_box(checksum)
}

fn candle_error(error: candle_core::Error) -> String {
    format!("candle operation failed: {error}")
}

fn candle_tensor_1(values: Vec<f64>, device: &CandleDevice) -> Result<CandleTensor, String> {
    let size = values.len();
    CandleTensor::from_vec(values, size, device).map_err(candle_error)
}

fn candle_tensor_2(
    values: Vec<f64>,
    rows: usize,
    columns: usize,
    device: &CandleDevice,
) -> Result<CandleTensor, String> {
    CandleTensor::from_vec(values, (rows, columns), device).map_err(candle_error)
}

fn candle_sum_all(tensor: &CandleTensor) -> Result<f64, String> {
    tensor
        .sum_all()
        .map_err(candle_error)?
        .to_scalar::<f64>()
        .map_err(candle_error)
}

fn candle_vector_element(tensor: &CandleTensor, index: usize) -> Result<f64, String> {
    candle_sum_all(&tensor.narrow(0, index, 1).map_err(candle_error)?)
}

fn candle_matrix_element(tensor: &CandleTensor, row: usize, column: usize) -> Result<f64, String> {
    let row = tensor.narrow(0, row, 1).map_err(candle_error)?;
    let element = row.narrow(1, column, 1).map_err(candle_error)?;
    candle_sum_all(&element)
}

fn candle_vector2(iterations: usize) -> Result<f64, String> {
    let device = CandleDevice::Cpu;
    let a = black_box(candle_tensor_1(vec![1.25, -2.5], &device)?);
    let b = black_box(candle_tensor_1(vec![-0.75, 3.0], &device)?);
    let mut checksum = 0.0;

    for i in 0..iterations {
        let sum = a.add(&b).map_err(candle_error)?;
        let dot = candle_sum_all(&a.mul(&b).map_err(candle_error)?)?;
        let norm = candle_sum_all(&sum.sqr().map_err(candle_error)?)?.sqrt();
        checksum +=
            candle_vector_element(&sum, i & 1)? + dot + norm + black_box((i & 7) as f64 * 1.0e-12);
    }

    Ok(black_box(checksum))
}

fn candle_vector3(iterations: usize) -> Result<f64, String> {
    let device = CandleDevice::Cpu;
    let a = black_box(candle_tensor_1(vec![1.25, -2.5, 0.75], &device)?);
    let b = black_box(candle_tensor_1(vec![-0.75, 3.0, 1.5], &device)?);
    let a0 = a.narrow(0, 0, 1).map_err(candle_error)?;
    let a1 = a.narrow(0, 1, 1).map_err(candle_error)?;
    let a2 = a.narrow(0, 2, 1).map_err(candle_error)?;
    let b0 = b.narrow(0, 0, 1).map_err(candle_error)?;
    let b1 = b.narrow(0, 1, 1).map_err(candle_error)?;
    let b2 = b.narrow(0, 2, 1).map_err(candle_error)?;
    let mut checksum = 0.0;

    for i in 0..iterations {
        let sum = a.add(&b).map_err(candle_error)?;
        let dot = candle_sum_all(&a.mul(&b).map_err(candle_error)?)?;
        let norm = candle_sum_all(&sum.sqr().map_err(candle_error)?)?.sqrt();

        let a1b2 = a1.mul(&b2).map_err(candle_error)?;
        let a2b1 = a2.mul(&b1).map_err(candle_error)?;
        let a2b0 = a2.mul(&b0).map_err(candle_error)?;
        let a0b2 = a0.mul(&b2).map_err(candle_error)?;
        let a0b1 = a0.mul(&b1).map_err(candle_error)?;
        let a1b0 = a1.mul(&b0).map_err(candle_error)?;
        let cross_x = a1b2.sub(&a2b1).map_err(candle_error)?;
        let cross_y = a2b0.sub(&a0b2).map_err(candle_error)?;
        let cross_z = a0b1.sub(&a1b0).map_err(candle_error)?;
        let cross = CandleTensor::cat(&[&cross_x, &cross_y, &cross_z], 0).map_err(candle_error)?;

        checksum += candle_vector_element(&sum, i % 3)?
            + dot
            + norm
            + candle_vector_element(&cross, (i + 1) % 3)?
            + black_box((i & 7) as f64 * 1.0e-12);
    }

    Ok(black_box(checksum))
}

fn candle_cosine1024(iterations: usize) -> Result<f64, String> {
    let device = CandleDevice::Cpu;
    let a = candle_tensor_1(
        (0..COSINE_DIMENSION)
            .map(|index| embedding_scalar(index, 0))
            .collect(),
        &device,
    )?;
    let b = candle_tensor_1(
        (0..COSINE_DIMENSION)
            .map(|index| embedding_scalar(index, 1))
            .collect(),
        &device,
    )?;
    let mut checksum = 0.0;

    for iteration in 0..iterations {
        let dot = candle_sum_all(&a.mul(&b).map_err(candle_error)?)?;
        let norm_a = candle_sum_all(&a.sqr().map_err(candle_error)?)?.sqrt();
        let norm_b = candle_sum_all(&b.sqr().map_err(candle_error)?)?.sqrt();
        let similarity = dot / (norm_a * norm_b);
        checksum += similarity + black_box((iteration & 7) as f64 * 1.0e-12);
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

fn ndarray_affine2(points: usize, iterations: usize) -> f64 {
    ndarray_affine(points, iterations, 2)
}

fn ndarray_affine3(points: usize, iterations: usize) -> f64 {
    ndarray_affine(points, iterations, 3)
}

fn ndarray_affine(points: usize, iterations: usize, dimension: usize) -> f64 {
    let homogeneous = dimension + 1;
    let values = affine_values(dimension);
    let transform = Array2::from_shape_fn((homogeneous, homogeneous).f(), |(row, column)| {
        values[row * homogeneous + column]
    });
    let input = Array2::from_shape_fn((homogeneous, points).f(), |(row, column)| {
        if row == dimension {
            1.0
        } else {
            scalar(column, row, dimension)
        }
    });
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let output = transform.dot(&input);
        checksum += output.sum() + output[(0, 0)] + output[(dimension, points - 1)];
    }

    black_box(checksum)
}

fn faer_affine2(points: usize, iterations: usize) -> f64 {
    faer_affine(points, iterations, 2)
}

fn faer_affine3(points: usize, iterations: usize) -> f64 {
    faer_affine(points, iterations, 3)
}

fn faer_affine(points: usize, iterations: usize, dimension: usize) -> f64 {
    let homogeneous = dimension + 1;
    let values = affine_values(dimension);
    let transform = Mat::from_fn(homogeneous, homogeneous, |row, column| {
        values[row * homogeneous + column]
    });
    let input = Mat::from_fn(homogeneous, points, |row, column| {
        if row == dimension {
            1.0
        } else {
            scalar(column, row, dimension)
        }
    });
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let mut output = Mat::zeros(homogeneous, points);
        faer::linalg::matmul::matmul(
            &mut output,
            Accum::Replace,
            &transform,
            &input,
            1.0,
            Par::Seq,
        );
        checksum += output.as_ref().sum() + output[(0, 0)] + output[(dimension, points - 1)];
    }

    black_box(checksum)
}

fn nalgebra_affine2(points: usize, iterations: usize) -> f64 {
    nalgebra_affine(points, iterations, 2)
}

fn nalgebra_affine3(points: usize, iterations: usize) -> f64 {
    nalgebra_affine(points, iterations, 3)
}

fn nalgebra_affine(points: usize, iterations: usize, dimension: usize) -> f64 {
    let homogeneous = dimension + 1;
    let values = affine_values(dimension);
    let transform = DMatrix::from_fn(homogeneous, homogeneous, |row, column| {
        values[row * homogeneous + column]
    });
    let input = DMatrix::from_fn(homogeneous, points, |row, column| {
        if row == dimension {
            1.0
        } else {
            scalar(column, row, dimension)
        }
    });
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let output = &transform * &input;
        checksum +=
            output.iter().copied().sum::<f64>() + output[(0, 0)] + output[(dimension, points - 1)];
    }

    black_box(checksum)
}

fn candle_affine2(points: usize, iterations: usize) -> Result<f64, String> {
    candle_affine(points, iterations, 2)
}

fn candle_affine3(points: usize, iterations: usize) -> Result<f64, String> {
    candle_affine(points, iterations, 3)
}

fn candle_affine(points: usize, iterations: usize, dimension: usize) -> Result<f64, String> {
    let device = CandleDevice::Cpu;
    let homogeneous = dimension + 1;
    let transform = candle_tensor_2(affine_values(dimension), homogeneous, homogeneous, &device)?;
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
    let input = candle_tensor_2(input_values, homogeneous, points, &device)?;
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let output = transform.matmul(&input).map_err(candle_error)?;
        checksum += candle_sum_all(&output)?
            + candle_matrix_element(&output, 0, 0)?
            + candle_matrix_element(&output, dimension, points - 1)?;
    }

    Ok(black_box(checksum))
}

fn ndarray_matvec(size: usize, iterations: usize) -> f64 {
    let matrix = Array2::from_shape_fn((size, size).f(), |(row, column)| scalar(row, column, size));
    let vector = Array1::from_shape_fn(size, |row| scalar(row, 0, size + 1));
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let output = matrix.dot(&vector);
        checksum += output.sum() + output[0];
    }

    black_box(checksum)
}

fn faer_matvec(size: usize, iterations: usize) -> f64 {
    let matrix = Mat::from_fn(size, size, |row, column| scalar(row, column, size));
    let vector = Mat::from_fn(size, 1, |row, _| scalar(row, 0, size + 1));
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let mut output = Mat::zeros(size, 1);
        faer::linalg::matmul::matmul(&mut output, Accum::Replace, &matrix, &vector, 1.0, Par::Seq);
        checksum += output.as_ref().sum() + output[(0, 0)];
    }

    black_box(checksum)
}

fn nalgebra_matvec(size: usize, iterations: usize) -> f64 {
    let matrix = DMatrix::from_fn(size, size, |row, column| scalar(row, column, size));
    let vector = DVector::from_fn(size, |row, _| scalar(row, 0, size + 1));
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let output = &matrix * &vector;
        checksum += output.iter().copied().sum::<f64>() + output[0];
    }

    black_box(checksum)
}

fn candle_matvec(size: usize, iterations: usize) -> Result<f64, String> {
    let device = CandleDevice::Cpu;
    let matrix_values = (0..size)
        .flat_map(|row| (0..size).map(move |column| scalar(row, column, size)))
        .collect();
    let vector_values = (0..size).map(|row| scalar(row, 0, size + 1)).collect();
    let matrix = candle_tensor_2(matrix_values, size, size, &device)?;
    let vector = candle_tensor_1(vector_values, &device)?;
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let output = matrix.mv(&vector).map_err(candle_error)?;
        checksum += candle_sum_all(&output)? + candle_vector_element(&output, 0)?;
    }

    Ok(black_box(checksum))
}

fn ndarray_matmul(size: usize, iterations: usize) -> f64 {
    let left = Array2::from_shape_fn((size, size).f(), |(row, column)| scalar(row, column, size));
    let right = Array2::from_shape_fn((size, size).f(), |(row, column)| {
        scalar(row, column, size + 1)
    });
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let output = left.dot(&right);
        checksum += output.sum() + output[(0, 0)];
    }

    black_box(checksum)
}

fn faer_matmul(size: usize, iterations: usize) -> f64 {
    let left = Mat::from_fn(size, size, |row, column| scalar(row, column, size));
    let right = Mat::from_fn(size, size, |row, column| scalar(row, column, size + 1));
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let mut output = Mat::zeros(size, size);
        faer::linalg::matmul::matmul(&mut output, Accum::Replace, &left, &right, 1.0, Par::Seq);
        checksum += output.as_ref().sum() + output[(0, 0)];
    }

    black_box(checksum)
}

fn nalgebra_matmul(size: usize, iterations: usize) -> f64 {
    let left = DMatrix::from_fn(size, size, |row, column| scalar(row, column, size));
    let right = DMatrix::from_fn(size, size, |row, column| scalar(row, column, size + 1));
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let output = &left * &right;
        checksum += output.iter().copied().sum::<f64>() + output[(0, 0)];
    }

    black_box(checksum)
}

fn candle_matmul(size: usize, iterations: usize) -> Result<f64, String> {
    let device = CandleDevice::Cpu;
    let left_values = (0..size)
        .flat_map(|row| (0..size).map(move |column| scalar(row, column, size)))
        .collect();
    let right_values = (0..size)
        .flat_map(|row| (0..size).map(move |column| scalar(row, column, size + 1)))
        .collect();
    let left = candle_tensor_2(left_values, size, size, &device)?;
    let right = candle_tensor_2(right_values, size, size, &device)?;
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let output = left.matmul(&right).map_err(candle_error)?;
        checksum += candle_sum_all(&output)? + candle_matrix_element(&output, 0, 0)?;
    }

    Ok(black_box(checksum))
}

#[cfg(feature = "ndarray-eigh")]
fn ndarray_symmetric_matrix(size: usize) -> Array2<f64> {
    Array2::from_shape_fn((size, size).f(), |(row, column)| {
        symmetric_scalar(row, column, size)
    })
}

fn faer_symmetric_matrix(size: usize) -> Mat<f64> {
    Mat::from_fn(size, size, |row, column| {
        symmetric_scalar(row, column, size)
    })
}

fn nalgebra_symmetric_matrix(size: usize) -> DMatrix<f64> {
    DMatrix::from_fn(size, size, |row, column| {
        symmetric_scalar(row, column, size)
    })
}

fn faer_eigh(size: usize, iterations: usize) -> Result<f64, String> {
    let matrix = faer_symmetric_matrix(size);
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let eigen = matrix
            .as_ref()
            .self_adjoint_eigen(Side::Lower)
            .map_err(|error| format!("faer eigendecomposition failed: {error:?}"))?;
        let eigenvalues = eigen.S().column_vector();
        let eigenvector_norm_squared = eigen
            .U()
            .col_iter()
            .flat_map(|column| column.iter())
            .map(|value| value * value)
            .sum::<f64>();
        checksum += eigenvalues.iter().copied().sum::<f64>() + eigenvector_norm_squared;
    }

    Ok(black_box(checksum))
}

fn nalgebra_eigh(size: usize, iterations: usize) -> f64 {
    let matrix = nalgebra_symmetric_matrix(size);
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let eigen = SymmetricEigen::new(matrix.clone());
        let eigenvector_norm_squared = eigen
            .eigenvectors
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        checksum += eigen.eigenvalues.iter().copied().sum::<f64>() + eigenvector_norm_squared;
    }

    black_box(checksum)
}

#[cfg(feature = "ndarray-eigh")]
fn ndarray_eigh(size: usize, iterations: usize) -> Result<f64, String> {
    use ndarray_linalg::{Eigh, UPLO};

    let matrix = ndarray_symmetric_matrix(size);
    let mut checksum = 0.0;

    for _ in 0..iterations {
        let (eigenvalues, eigenvectors) = matrix
            .eigh(UPLO::Lower)
            .map_err(|error| format!("ndarray eigendecomposition failed: {error:?}"))?;
        let eigenvector_norm_squared = eigenvectors.iter().map(|value| value * value).sum::<f64>();
        checksum += eigenvalues.sum() + eigenvector_norm_squared;
    }

    Ok(black_box(checksum))
}

#[cfg(not(feature = "ndarray-eigh"))]
fn ndarray_eigh(_size: usize, _iterations: usize) -> Result<f64, String> {
    Err("ndarray eigh requires a LAPACK backend; rebuild with --features ndarray-eigh-openblas-static (or ndarray-eigh-openblas-system)".to_owned())
}

fn candle_eigh(_size: usize, _iterations: usize) -> Result<f64, String> {
    Err("candle-core 0.11.0 does not provide a general symmetric eigendecomposition API".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_operation_aliases() {
        assert!(matches!(Operation::parse("vec2"), Ok(Operation::Vector2)));
        assert!(matches!(
            Operation::parse("diagonalize"),
            Ok(Operation::Eigh)
        ));
        assert!(matches!(Backend::parse("native"), Ok(Backend::Raw)));
    }

    #[test]
    fn vector_checksums_are_finite() {
        for checksum in [
            ndarray_vector2(8),
            faer_vector2(8),
            nalgebra_vector2(8),
            candle_vector2(8).unwrap(),
            burn_backend::vector2(8).unwrap(),
            raw_backend::vector2(8).unwrap(),
            ndarray_vector3(8),
            faer_vector3(8),
            nalgebra_vector3(8),
            candle_vector3(8).unwrap(),
            burn_backend::vector3(8).unwrap(),
            raw_backend::vector3(8).unwrap(),
        ] {
            assert!(checksum.is_finite());
        }

        assert_close(candle_vector2(8).unwrap(), ndarray_vector2(8));
        assert_close(candle_vector3(8).unwrap(), ndarray_vector3(8));
        assert_close(burn_backend::vector2(8).unwrap(), ndarray_vector2(8));
        assert_close(burn_backend::vector3(8).unwrap(), ndarray_vector3(8));
        assert_close(raw_backend::vector2(8).unwrap(), ndarray_vector2(8));
        assert_close(raw_backend::vector3(8).unwrap(), ndarray_vector3(8));
    }

    #[test]
    fn matrix_operations_are_finite() {
        for checksum in [
            ndarray_affine2(8, 1),
            faer_affine2(8, 1),
            nalgebra_affine2(8, 1),
            candle_affine2(8, 1).unwrap(),
            burn_backend::affine2(8, 1).unwrap(),
            raw_backend::affine2(8, 1).unwrap(),
            candle_affine3(8, 1).unwrap(),
            burn_backend::affine3(8, 1).unwrap(),
            raw_backend::affine3(8, 1).unwrap(),
            ndarray_matvec(8, 1),
            faer_matvec(8, 1),
            nalgebra_matvec(8, 1),
            candle_matvec(8, 1).unwrap(),
            burn_backend::matvec(8, 1).unwrap(),
            raw_backend::matvec(8, 1).unwrap(),
            ndarray_matmul(8, 1),
            faer_matmul(8, 1),
            nalgebra_matmul(8, 1),
            candle_matmul(8, 1).unwrap(),
            burn_backend::matmul(8, 1).unwrap(),
            raw_backend::matmul(8, 1).unwrap(),
            nalgebra_eigh(8, 1),
        ] {
            assert!(checksum.is_finite());
        }
        let expected_eigh_checksum = 8.0 * (8.0 + 2.0) + 8.0;
        assert!((faer_eigh(8, 1).unwrap() - expected_eigh_checksum).abs() < 1.0e-8);
        assert!((nalgebra_eigh(8, 1) - expected_eigh_checksum).abs() < 1.0e-8);

        #[cfg(feature = "ndarray-eigh")]
        assert!((ndarray_eigh(8, 1).unwrap() - expected_eigh_checksum).abs() < 1.0e-8);

        assert_close(candle_affine2(8, 1).unwrap(), ndarray_affine2(8, 1));
        assert_close(candle_affine3(8, 1).unwrap(), ndarray_affine3(8, 1));
        assert_close(candle_matvec(8, 1).unwrap(), ndarray_matvec(8, 1));
        assert_close(candle_matmul(8, 1).unwrap(), ndarray_matmul(8, 1));
        assert_close(burn_backend::affine2(8, 1).unwrap(), ndarray_affine2(8, 1));
        assert_close(burn_backend::affine3(8, 1).unwrap(), ndarray_affine3(8, 1));
        assert_close(burn_backend::matvec(8, 1).unwrap(), ndarray_matvec(8, 1));
        assert_close(burn_backend::matmul(8, 1).unwrap(), ndarray_matmul(8, 1));
        assert_close(raw_backend::affine2(8, 1).unwrap(), ndarray_affine2(8, 1));
        assert_close(raw_backend::affine3(8, 1).unwrap(), ndarray_affine3(8, 1));
        assert_close(raw_backend::matvec(8, 1).unwrap(), ndarray_matvec(8, 1));
        assert_close(raw_backend::matmul(8, 1).unwrap(), ndarray_matmul(8, 1));
    }

    #[test]
    fn candle_does_not_claim_eigendecomposition_support() {
        let error = candle_eigh(8, 1).unwrap_err();
        assert!(error.contains("does not provide a general symmetric eigendecomposition API"));
    }

    #[test]
    fn raw_eigh_is_a_finite_jacobi_baseline() {
        let checksum = raw_backend::eigh(8, 1).unwrap();
        assert!(checksum.is_finite());
    }

    #[test]
    fn burn_does_not_claim_eigendecomposition_support() {
        let error = burn_backend::eigh_error();
        assert!(error.contains("does not provide a general symmetric eigendecomposition API"));
    }

    #[test]
    fn cosine1024_matches_across_rust_backends() {
        let expected = ndarray_cosine1024(2);
        assert_close(faer_cosine1024(2), expected);
        assert_close(nalgebra_cosine1024(2), expected);
        assert_close(candle_cosine1024(2).unwrap(), expected);
        assert_close(burn_backend::cosine1024(2).unwrap(), expected);
        assert_close(raw_backend::cosine1024(2).unwrap(), expected);
    }

    fn assert_close(actual: f64, expected: f64) {
        let scale = actual.abs().max(expected.abs()).max(1.0);
        assert!((actual - expected).abs() <= 1.0e-10 * scale);
    }
}
