# Performance Model

**Purpose:** Provides a framework for interpreting benchmarks and understanding performance boundaries.

Performance in VECTORIA is interpreted through the lens of **Memory Bandwidth** vs. **Compute Intensity**. We do not optimize for peak theoretical FLOPS if it compromises traceability.

## Compute-Bound Operations

*   **Primary Op:** `MatMul` (Matrix Multiplication).
*   **Goal:** Maximize FMA (Fused Multiply-Add) throughput.
*   **Scaling:** Performance scales linearly with the number of available SIMD lanes (NEON 128-bit vs AVX2 256-bit).

## Memory-Bound Operations

*   **Primary Ops:** `Add`, `Mul`, `Relu`, `BiasAdd`.
*   **Constraint:** Performance is limited by how fast data can be moved from RAM to L1 cache.
*   **Interpretation:** SIMD implementations for these ops typically show marginal gains (1.1x - 1.5x) over autovectorized C++, primarily due to reduced instruction overhead rather than calculation speed.

## Benchmarking Policy

1.  **No "Hero" Runs:** We report average stable throughput, not the single fastest outlier.
2.  **No Cross-Framework Comparison:** VECTORIA benchmarks track internal regression, not competition with PyTorch or TensorFlow.
3.  **Latency Floors:** We accept a baseline latency for `Engine` dispatch and tracing overhead. We do not optimize for sub-microsecond execution of single scalars.

## Optimization Strategy

1.  **Reference First:** Always implement in C++ scalar first.
2.  **SIMD Second:** Implement NEON/AVX2 only if correctness is proven and speedup is measurable (>10%).
3.  **No Fusion:** We do NOT fuse kernels (e.g. `MatMul+ReLU`) implicitly. This hides performance characteristics and complicates tracing.

## References

*   `benchmarks/`
*   [docs/performance_model.md](../performance_model.md)