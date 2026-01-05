# VECTORIA Benchmarks

This directory contains reproducible benchmark harnesses.

## Purpose
These benchmarks exist to **detect regression**, not to generate marketing numbers. 

## Running
```bash
g++ -std=c++17 -O3 -I../core/include \
    ../core/src/engine.cpp \
    ../core/src/memory.cpp \
    ../core/src/kernels/gemm_ref.cpp \
    ../core/src/trace.cpp \
    gemm_bench.cpp -o gemm_bench

./gemm_bench
```

## Methodology
- **Warmup**: 5 iterations are discarded.
- **Timing**: `std::chrono::high_resolution_clock` around the execution loop.
- **Metric**: Average ms/iter and estimated GFLOPS.

```