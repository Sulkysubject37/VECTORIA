# Benchmarking Policy

## No Marketing Claims
VECTORIA does not publish "fastest in the world" charts. We measure performance solely to:
1. Ensure we don't get slower (Regression Testing).
2. Validate that SIMD optimizations provide *some* benefit over scalar reference.

## Reproducibility
Benchmarks must be reproducible on local developer machines.
- **Fixed Inputs**: Matrix sizes are hardcoded.
- **Fixed Iterations**: Loop counts are static.
- **Single Threaded**: Currently, all benchmarks run on a single core to reduce variance.

## Interpretation
Results from `benchmarks/gemm_bench.cpp` represent the raw kernel execution time, including C++ dispatch overhead. They do NOT include Python FFI overhead.

## Elementwise Benchmarks
- **File**: `benchmarks/elementwise_bench.cpp`
- **Kernels**: `Add`, `Mul`, `Sub`, `Div`, `ReLU`.
- **Metrics**: Throughput (GB/s), Latency (us).

## Reduction Benchmarks
- **File**: `benchmarks/reduction_bench.cpp`
- **Kernels**: `ReduceSum`, `ReduceMax`.
- **Metrics**: Throughput (GB/s).

## Methodology
- **Warmup**: 10 iterations.
- **Measurement**: 100 iterations, averaged.
- **Cache**: Inputs larger than L2 cache (1MB+) are tested to measure memory bandwidth limits.
- **Comparison**: Strictly compares SIMD kernel vs Reference C++ kernel on the same machine.
