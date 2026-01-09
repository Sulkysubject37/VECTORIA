# VECTORIA Benchmarks

This directory contains reproducible benchmark harnesses for measuring kernel performance and detecting regression.

## Purpose
Benchmarks in VECTORIA exist solely to:
1. Ensure that optimizations provide a tangible benefit over the reference kernels.
2. Detect performance regressions in core dispatch and memory management.

**Measurements are intended for internal validation only.** Results are non-portable and depend on specific CPU microarchitecture, compiler versions, and build flags.

## Available Benchmarks
- `gemm_bench.cpp`: Measures matrix multiplication throughput (GFLOPS).
- `elementwise_bench.cpp`: Measures `Add`, `Mul`, `Sub`, `Div`, and `ReLU` performance.
- `reduction_bench.cpp`: Measures `ReduceSum` and `ReduceMax` throughput.

## Running Benchmarks

### ARM64 (macOS/Linux)
```bash
g++ -std=c++17 -O3 -DVECTORIA_USE_ASM -I../core/include \
    ../core/src/*.cpp ../core/src/kernels/*.cpp ../core/src/graph/*.cpp ../asm/arm64/*.S \
    elementwise_bench.cpp -o bench_el
./bench_el
```

### x86_64 (AVX2)
```bash
g++ -std=c++17 -O3 -DVECTORIA_USE_ASM -I../core/include \
    ../core/src/*.cpp ../core/src/kernels/*.cpp ../core/src/graph/*.cpp ../asm/x86_64/*.S \
    reduction_bench.cpp -o bench_red
./bench_red
```

For more details on our performance philosophy, see [Benchmarking Policy](../docs/benchmarks.md).
