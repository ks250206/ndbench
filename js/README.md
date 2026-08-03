# ndbench JavaScript / TypeScript

This directory is an independent CPU benchmark for the same deterministic
operations used by the Rust and Python implementations in the parent
repository. TypeScript is compiled to JavaScript first, and the benchmark runs
the compiled CLI with Node.js:

```text
TypeScript source -> dist JavaScript -> node dist/ndbench.js
```

GPU execution is intentionally out of scope.

## Backends

| backend | implementation | `eigh` |
| --- | --- | --- |
| `raw` | explicit loops over `Float64Array` values | dependency-free cyclic Jacobi method |
| `ml-matrix` | `ml-matrix` dense `Matrix` operations | `EigenvalueDecomposition` |

The `raw` backend is the primary JavaScript baseline. It uses JavaScript
`number` values, which are IEEE-754 binary64 values, and `Float64Array` storage
for every array and matrix.

The `ml-matrix` backend uses the library for `affine2`, `affine3`, `matvec`,
`matmul`, and `eigh`. `vector2`, `vector3`, and `cosine1024` intentionally use
the same fixed-size `Float64Array` kernels as `raw`: `ml-matrix` is a dense
matrix library rather than a useful fixed-size vector primitive. Therefore the
two backends are expected to have identical timings and checksums for those
three operations. The matrix/eigendecomposition results are not a pure
algorithm-for-algorithm comparison: `raw` uses flat typed arrays and explicit
loops, while `ml-matrix` allocates `Matrix` objects and calls its own dense
algorithms.

The raw `eigh` checksum is directly comparable with the Rust/Python raw
baseline because it follows the same input construction, cyclic Jacobi sweep,
eigenvector update, and checksum accumulation order. `ml-matrix` computes all
real eigenvalues and the full eigenvector matrix with
`EigenvalueDecomposition`; small floating-point differences are expected.

## Setup and CLI

Requirements: Node.js 20 or newer, pnpm, and hyperfine for timing benchmarks.

```sh
pnpm install --frozen-lockfile
pnpm build

node dist/ndbench.js --backend raw --op vector3 --iterations 1000000
node dist/ndbench.js --backend ml-matrix --op matmul --size 256 --iterations 5
node dist/ndbench.js --backend ml-matrix --op eigh --size 128 --iterations 1
```

`--backend` is optional and defaults to `raw`. The other options are:

```text
--op, --operation  vector2, vector3, affine2, affine3, matvec, matmul,
                   cosine1024, or eigh
--size             affine point count or square matrix order
--iterations       number of repeated operations
```

Every successful invocation prints one machine-readable line such as:

```text
checksum=3.03813964317223792e+01
```

The exponent is formatted in the same 17-decimal-place style as the Rust and
Python CLIs. `vector2` and `vector3` use the fixed vector defaults, affine
operations use 100,000 points, matrix operations use order 128, and
`cosine1024` uses two fixed 1,024-dimensional vectors.

## Checksum cross-check

For `size=8` and `iterations=1`, the raw backend produces the same values as
the Rust and Python raw implementations (ignoring the different exponent
zero-padding convention):

| operation | raw checksum |
| --- | ---: |
| `vector2` | `-7.23039321881345209e+00` |
| `vector3` | `-6.89150471698584965e+00` |
| `affine2` | `1.96184405940594040e+01` |
| `affine3` | `2.86269801980198011e+01` |
| `matvec` | `3.03813964317223792e+01` |
| `matmul` | `1.91503970199000094e+02` |
| `cosine1024` | `-6.87357264609637503e-02` |
| `eigh` | `8.80000000000001137e+01` |

The same command can be used to inspect both JavaScript backends:

```sh
for backend in raw ml-matrix; do
  for op in vector2 vector3 affine2 affine3 matvec matmul cosine1024 eigh; do
    node dist/ndbench.js --backend "${backend}" --op "${op}" --size 8 --iterations 1
  done
done
```

The `ml-matrix` `eigh` result for this input is within floating-point roundoff
of the raw/Rust/Python checksum; the other matrix operation checksums match in
the current Node.js run.

## Speed and memory benchmarks

`benchmark.sh` first runs `pnpm install --frozen-lockfile` and `pnpm build`,
then measures the compiled `dist/ndbench.js` with hyperfine. It measures both
`raw` and `ml-matrix` for all eight operations. The timed command includes
Node.js process startup, V8 startup/JIT behavior, module loading, input
construction, and the operation; it is a CLI end-to-end benchmark rather than
an in-process kernel-only measurement.

```sh
HYPERFINE_RUNS=10 HYPERFINE_WARMUP=2 ./benchmark.sh
```

The default sizes and iteration counts can be overridden with environment
variables such as `MATMUL_SIZE`, `EIGH_SIZE`, `VECTOR_ITERATIONS`, and
`COSINE_ITERATIONS`. JSON, Markdown, and TSV outputs are written to the ignored
`js/results/` directory.

`memory.sh` builds the same `dist` output and measures peak process RSS using
macOS `/usr/bin/time -l` or Linux `/usr/bin/time -v`; it does not measure a
separate TypeScript or package-install process.

```sh
./memory.sh
```

The generated memory values are process RSS, not the logical size of the
typed-array or `ml-matrix` data structures.
