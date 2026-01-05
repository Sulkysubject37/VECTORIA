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
