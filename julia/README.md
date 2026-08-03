# Julia backend

This directory contains the Julia CPU implementation of the same deterministic
operations used by the Rust and Python backends in the parent repository.
Julia 1.12 is the intended runtime. GPU execution is intentionally out of
scope for this backend.

The implementation uses only the Julia standard libraries `LinearAlgebra` and
`Printf`. `Project.toml` records those dependencies and the Julia compatibility
range, so the environment can be instantiated reproducibly:

```sh
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

## CLI

Run from the repository root:

```sh
julia --project=julia --startup-file=no --threads=1 julia/ndbench.jl \
  --op matmul --size 128 --iterations 1
```

The CLI accepts `--op` or `--operation`, plus `--size` and `--iterations`.
It prints one machine-readable result:

```text
checksum=1.91503970199000094e+02
```

Supported operations are:

- `vector2`: 2D vector addition, dot product, and L2 norm
- `vector3`: 3D vector addition, dot product, L2 norm, and cross product
- `affine2`, `affine3`: homogeneous affine transforms over `--size` points
- `matvec`: dense square matrix-vector multiplication
- `matmul`: dense square matrix multiplication
- `cosine1024`: cosine similarity for two deterministic `f64` vectors of length 1024
- `eigh`: full real symmetric eigendecomposition, including eigenvalues and eigenvectors

The deterministic scalar and matrix construction follows
`src/raw_backend.rs` and `python/ndbench.py`. The Julia `eigh` operation uses
the standard-library `eigen(Symmetric(matrix))` LAPACK-backed implementation;
its checksum includes the sum of all eigenvalues and the squared norm of the
full eigenvector matrix. Checksum accumulation is kept in logical row-major
order where the reference backend exposes a flat row-major buffer, despite
Julia's column-major array storage.

The CLI sets BLAS threads to one in code. The scripts additionally set
`JULIA_NUM_THREADS=1` and invoke Julia with `--threads=1` so that Julia and
BLAS/LAPACK remain single-threaded for CPU comparisons.

## Benchmark and memory scripts

Both scripts resolve paths from their own directory, so they can be run from
the repository root or from `julia/`:

```sh
julia/benchmark.sh
julia/memory.sh
```

`benchmark.sh` uses `hyperfine` and writes JSON/Markdown results under
`julia/results/`. `memory.sh` writes `julia/results/memory.tsv` and parses
macOS `/usr/bin/time -l` as well as Linux `/usr/bin/time -v` output.

The workload sizes and repeat counts can be overridden with the same style of
environment variables used by the parent benchmark scripts, for example:

```sh
HYPERFINE_RUNS=5 HYPERFINE_WARMUP=1 VECTOR_ITERATIONS=10000 julia/benchmark.sh
POINTS=10000 MATMUL_SIZE=128 julia/memory.sh
```

## Small checksum smoke test

For `--size 8 --iterations 1`, the scalar reference values are:

| operation | checksum |
| --- | ---: |
| `vector2` | `-7.23039321881345209e+00` |
| `vector3` | `-6.89150471698584965e+00` |
| `affine2` | `1.96184405940594040e+01` |
| `affine3` | `2.86269801980198011e+01` |
| `matvec` | `3.03813964317223792e+01` |
| `matmul` | `1.91503970199000094e+02` |
| `cosine1024` | `-6.87357264609637503e-02` |
| `eigh` | `8.80000000000001137e+01` |

`size` is ignored by the fixed-size vector and cosine operations, as in the
other backends. The values above are the Rust/Python raw reference output;
BLAS-backed matrix/eigenvalue implementations can differ in the final few
floating-point bits while computing the same operation.
