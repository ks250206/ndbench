# Mojo raw backend

This directory contains the dependency-free Mojo CPU baseline for `ndbench`.
It uses Mojo's standard library only: `Float64`, `List[Float64]`, and
`SIMD[DType.float64, 2/3]`. There are no BLAS, LAPACK, or other numerical
packages, and GPU execution is intentionally out of scope.

The operation and input formulas match [`src/raw_backend.rs`](../src/raw_backend.rs)
and the Python raw implementation:

- `vector2`, `vector3`
- `affine2`, `affine3`
- `matvec`, `matmul`
- `cosine1024`
- `eigh` using the same bounded cyclic Jacobi method

## Prerequisites

The Mojo SDK must be available through the `mojo` executable. This checkout
uses a small uv project so the SDK can be installed without changing the root
Python environment:

```sh
uv sync --project mojo
uv run --project mojo mojo --version
```

Mojo is distributed as a platform-specific SDK package. It is not available
from the configured conda-forge pixi channel on the benchmark machine, so no
pixi environment is declared here. The `pyproject.toml` intentionally keeps
the dependency as `mojo` and lets uv resolve the current compatible SDK.

## CLI

Build and run a deterministic smoke test:

```sh
uv run --project mojo mojo build --optimization-level 3 -D ASSERT=none mojo/main.mojo \
  -o mojo/target/ndbench-mojo
mojo/target/ndbench-mojo --op vector2 --size 8 --iterations 1
```

The output is formatted as:

```text
checksum=-7.23039321881345152e+00
```

`--operation` is accepted as an alias for `--op`. `--size` is used for affine,
matrix, and eigendecomposition operations; `--iterations` controls repetition.

With `--size 8 --iterations 1`, the Mojo checksums are within `1.5e-14` of
the Rust/Python raw baselines. The small differences are expected from the
Mojo optimizer and SIMD expression evaluation order:

| operation | Mojo | Rust/Python raw |
| --- | ---: | ---: |
| vector2 | `-7.23039321881345152e+00` | `-7.23039321881345209e+00` |
| vector3 | `-6.89150471698584960e+00` | `-6.89150471698584965e+00` |
| affine2 | `1.96184405940594016e+01` | `1.96184405940594040e+01` |
| affine3 | `2.86269801980198016e+01` | `2.86269801980198011e+01` |
| matvec | `3.03813964317223808e+01` | `3.03813964317223792e+01` |
| matmul | `1.91503970199000096e+02` | `1.91503970199000094e+02` |
| cosine1024 | `-6.87357264609637248e-02` | `-6.87357264609637503e-02` |
| eigh | `8.80000000000001280e+01` | `8.80000000000001137e+01` |

## Benchmark and memory measurement

The scripts build with `mojo build --optimization-level 3 -D ASSERT=none` and use `hyperfine` for
timing. Override the defaults when matching the root benchmark:

```sh
HYPERFINE_RUNS=5 HYPERFINE_WARMUP=2 VECTOR_ITERATIONS=10000 \
  bash mojo/benchmark.sh
bash mojo/memory.sh
```

`memory.sh` uses `/usr/bin/time -l` on macOS and `/usr/bin/time -v` on Linux.
It writes the normalized peak-RSS rows to `mojo/results/memory.tsv`. The
benchmark JSON/Markdown files and generated executable are kept under `mojo/`
and are ignored by the local benchmark configuration.

## Validation status

The implementation was written against the current Mojo API (`argv()`,
`List`, `SIMD[DType.float64, N]`, and `String.format`). Mojo 0.26.2.0 does not
accept Python-style `:.17e` fields in `String.format` (the compiler reports
`Index :.17e not in kwargs`), so `format_checksum` implements the required
scientific notation using integer digit extraction. If the SDK cannot be
resolved or its API changes, run the commands above first and record the exact
diagnostic before comparing timing results. A Mojo SDK was not installed
system-wide on the original machine, but uv resolved Mojo 0.26.2.0 for local
compilation validation.
