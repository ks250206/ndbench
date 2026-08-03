# Go backend

This directory contains the Go CPU backend for the parent `ndbench` benchmark.
It targets Go 1.26 and uses [Gonum](https://gonum.org/) v0.17's
`gonum.org/v1/gonum/mat` package for dense matrix operations and symmetric
eigendecomposition. GPU execution is intentionally out of scope.

## Operations

The CLI implements the same deterministic operations and input formulas as the
parent Rust `src/raw_backend.rs` and Python `python/ndbench.py` backends:

| Operation | Calculation |
| --- | --- |
| `vector2` | 2D addition, dot product, L2 norm |
| `vector3` | 3D addition, dot product, L2 norm, cross product |
| `affine2` | 3×3 homogeneous transform of `--size` points |
| `affine3` | 4×4 homogeneous transform of `--size` points |
| `matvec` | `--size` square matrix × vector |
| `matmul` | `--size` square matrix × square matrix |
| `cosine1024` | cosine similarity of two deterministic `f64` vectors of length 1024 |
| `eigh` | all eigenvalues and eigenvectors of a real symmetric matrix |

Every result is printed as:

```text
checksum=-7.23039321881345209e+00
```

The checksum includes the same output sums as the existing backends, so it is
used to catch accidental changes to the input data or operation. Floating-point
rounding can differ slightly between implementations and BLAS paths; compare
with a small tolerance rather than byte-for-byte equality.

## Gonum representation

The deterministic input formulas produce row-major `f64` slices. Gonum's
`mat.NewDense` also accepts row-major data, so `Dense` is used directly for
affine transforms, matrix-vector products, and matrix multiplication. The
affine points are stored as a `(dimension+1) × points` matrix, matching the
Rust/Python layout.

For `eigh`, the same symmetric matrix is constructed as a `mat.SymDense` and
factorized with `mat.EigenSym.Factorize(matrix, true)`. The `true` argument is
important: it requests the complete orthonormal eigenvector matrix in addition
to all eigenvalues. The checksum sums all eigenvalues and all squared
eigenvector elements, just like the existing implementations. Gonum's default
pure-Go BLAS/LAPACK path is therefore the measured Go backend; no external
vendor BLAS is required by this directory.

Small vector and cosine operations use explicit `float64` loops because there
is no matrix allocation to amortize for those cases. They still use exactly the
same deterministic scalar expressions as the reference backends.

## Run the CLI

From the repository root:

```sh
cd go
go mod download
go run . --op vector2 --size 1 --iterations 1
go run . --operation matmul --size 8 --iterations 1
go run . --op eigh --size 8 --iterations 1
```

Build a release-equivalent binary and run it directly:

```sh
go build -trimpath -ldflags='-s -w' -o /tmp/ndbench-go .
/tmp/ndbench-go --op cosine1024 --size 1024 --iterations 1
```

The Go compiler's normal optimized build is used; `-trimpath` removes local
paths and `-s -w` removes symbol/debug tables from the benchmark binary.

## Benchmark and memory measurement

Both scripts resolve paths relative to their own location, so they can be run
from the repository root or from this directory. They build the measured
binary before timing it.

```sh
./go/benchmark.sh
./go/memory.sh
```

`benchmark.sh` uses `hyperfine` and writes JSON/Markdown files to
`go/results/`. `memory.sh` writes `go/results/memory.tsv`. On macOS it parses
`/usr/bin/time -l`; on Linux it parses `/usr/bin/time -v`.

The same size/iteration knobs as the parent scripts can be overridden, for
example:

```sh
HYPERFINE_RUNS=5 HYPERFINE_WARMUP=1 \
POINTS=1000 MATVEC_SIZE=64 MATMUL_SIZE=32 EIGH_SIZE=16 \
./go/benchmark.sh
```

`GOMAXPROCS` defaults to `1` in both scripts to keep the CPU comparison
single-threaded. Set it explicitly if a separate multithreaded experiment is
desired.
