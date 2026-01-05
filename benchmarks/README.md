# VECTORIA Benchmarks

This directory contains reproducible benchmark harnesses for measuring kernel performance and detecting regression.

## Purpose
Benchmarks in VECTORIA exist to:
1. Ensure that optimizations provide a tangible benefit over the reference kernels.
2. Detect performance regressions in core dispatch and memory management.

## Available Benchmarks
- `gemm_bench.cpp`: Measures matrix multiplication throughput (GFLOPS).

## Running Benchmarks
```bash
# Build the benchmark
g++ -std=c++17 -O3 -DVECTORIA_USE_ASM -I../core/include \
    ../core/src/*.cpp ../core/src/kernels/*.cpp ../asm/arm64/gemm_neon.S \
    gemm_bench.cpp -o gemm_bench

# Run
./gemm_bench
```

For more details on our performance philosophy, see [Benchmarking Policy](../docs/benchmarks.md).