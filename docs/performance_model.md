# Performance Model & Boundaries

VECTORIA aims to be **predictably fast**, not necessarily the fastest on Earth. We prioritize:
1. **Determinism**: Identical results across runs.
2. **Traceability**: Knowing exactly *why* a calculation took time X.
3. **Reproducibility**: Benchmarks that don't fluctuate wildly.

## What "Fast" Means
- **Memory Bound**: For elementwise ops (`Add`, `Mul`), we aim to saturate memory bandwidth. Speedups of 1.1x-1.5x over scalar C++ are typical and acceptable.
- **Compute Bound**: For `MatMul`, we aim for >80% of peak theoretical FLOPS for the given SIMD instruction set (NEON/AVX2).
- **Latency**: We minimize overhead for small graphs, but we do NOT optimize for micro-op latency (< 5us) at the expense of safety checks.

## Benchmarking Boundaries
- **No Cross-Framework Comparisons**: We do not publish charts comparing VECTORIA to PyTorch, TensorFlow, or NumPy. Our goals are different (determinism vs throughput).
- **No "Hero" Runs**: We report the average of stable runs, not the single fastest outlier.
- **Local Only**: Benchmarks are intended to be run by developers on their own hardware to validate improvements, not to generate marketing material.

## Optimization Strategy
1. **Reference First**: Always implement in C++ scalar first.
2. **SIMD Second**: Implement NEON/AVX2 only if:
   - Correctness is proven.
   - Determinism is preserved (within tolerance).
   - Speedup is measurable (>10%).
3. **No Fusion**: We do NOT fuse kernels (e.g. `MatMul+ReLU`) implicitly. This hides performance characteristics and complicates tracing. Fusion must be an explicit new kernel if added.
