# ndbench Swift backend

This directory is a standalone Swift Package Manager executable for the same
deterministic CPU operations used by the Rust and Python backends in the parent
repository. It targets Apple Silicon/macOS and has no external package
dependencies. GPU APIs are intentionally not used.

## Requirements

- macOS 14 or later
- Apple Silicon recommended
- Swift 6.0 or later with Swift Package Manager
- `hyperfine` for timing and the macOS `/usr/bin/time -l` for RSS measurement

The package links the system `Accelerate.framework`. The source uses Swift
`SIMD2<Double>`/`SIMD3<Double>` for the small vector operations, explicit loops
for affine transforms and cosine similarity, and Accelerate CBLAS/LAPACK for
matrix operations and eigendecomposition.

## Build and run

```sh
cd swift
swift build -c release

swift run -c release -- --op vector2 --iterations 1
swift run -c release -- --backend swift --op matmul --size 8 --iterations 1
swift run -c release -- --operation eigh --size 8 --iterations 1
```

The release executable is `.build/release/ndbench`. `--backend` is optional;
`swift`, `accelerate`, and `native` are accepted names for this backend.

Every operation prints one machine-readable line:

```text
checksum=8.80000000000001137e+01
```

The accepted operations are:

| Operation | `--size` | Calculation |
| --- | --- | --- |
| `vector2` | unused | 2D addition, dot product, L2 norm |
| `vector3` | unused | 3D addition, dot product, L2 norm, cross product |
| `affine2` | point count | 3×3 homogeneous transform of 2D points |
| `affine3` | point count | 4×4 homogeneous transform of 3D points |
| `matvec` | matrix order | dense square matrix × vector |
| `matmul` | matrix order | dense square matrix × matrix |
| `cosine1024` | fixed at 1024 | cosine similarity of two deterministic `f64` vectors |
| `eigh` | matrix order | all eigenvalues and eigenvectors of a real symmetric matrix |

`--op` and `--operation` are aliases. `vec2`, `vec3`, and `diagonalize` are
accepted operation aliases. If `--size` or `--iterations` is omitted, the
defaults match the parent Rust/Python CLI.

## Accelerate layout and LAPACK API

The logical matrices use the same values as `src/raw_backend.rs` and
`python/ndbench.py`. For BLAS/LAPACK calls they are stored in column-major
order: element `(row, column)` is at `data[row + column * size]`. This is the
layout required by Accelerate's CBLAS and LAPACK interfaces. The output of
`cblas_dgemv` and `cblas_dgemm` is therefore interpreted with the same logical
row/column indexing as the other backends.

`matvec` calls `cblas_dgemv` and `matmul` calls `cblas_dgemm` with
`CblasColMajor`, `CblasNoTrans`, `alpha = 1`, and `beta = 0`.

`eigh` calls Accelerate's `dsyevd_` divide-and-conquer LAPACK routine with
`jobz = 'V'` to request both all eigenvalues and all eigenvectors, and
`uplo = 'L'` to read the lower triangle. Workspace sizes are obtained with a
LAPACK workspace query before the repeated benchmark calls. The returned
eigenvectors are the columns of the overwritten matrix, and the checksum is
the sum of all eigenvalues plus the squared norm of all eigenvector entries.

SwiftPM passes `ACCELERATE_NEW_LAPACK` and `ACCELERATE_LAPACK_ILP64` to the C
headers. Consequently Accelerate's integer parameters are the platform-sized
`Int` type in this package. This uses the current non-deprecated Accelerate
BLAS/LAPACK declarations available on macOS 13.3 and later.

The benchmark is single-threaded. `benchmark.sh` and `memory.sh` export
`VECLIB_MAXIMUM_THREADS=1`, `VECLIB_MAXIMUM_NUMBER_OF_THREADS=1`, and
`OMP_NUM_THREADS=1`; the executable also sets them if they are absent. This
keeps Accelerate's BLAS/LAPACK work comparable to the single-threaded Rust and
Python runs.

## Checksum comparison

The following commands exercise all operations with the small cross-check
configuration used by the parent benchmark:

```sh
for op in vector2 vector3 affine2 affine3 matvec matmul cosine1024 eigh; do
  size=8
  [[ "${op}" == vector2 || "${op}" == vector3 ]] && size=1
  [[ "${op}" == cosine1024 ]] && size=1024
  .build/release/ndbench --op "${op}" --size "${size}" --iterations 1
done
```

The vector/affine/matrix/cosine checksums should match the dependency-free Rust
raw backend to floating-point tolerance. `eigh` uses Accelerate's LAPACK
solver rather than the raw Jacobi solver, so its eigenvalue/eigenvector result
is compared by checksum tolerance, not by bit-for-bit string equality.

## Hyperfine and peak RSS

`benchmark.sh` builds the release executable and measures each operation with
`hyperfine`. The environment variables below can be overridden for a smaller
or larger run:

```sh
HYPERFINE_RUNS=10 HYPERFINE_WARMUP=2 ./benchmark.sh
```

JSON and Markdown reports are written to `swift/results/`.

`memory.sh` measures the process peak resident set size. On macOS it parses
`/usr/bin/time -l`'s `maximum resident set size`; on Linux it parses
`/usr/bin/time -v`'s `Maximum resident set size` and converts KiB to bytes.
The Swift executable itself is macOS-only because it links Accelerate.

```sh
./memory.sh
```
