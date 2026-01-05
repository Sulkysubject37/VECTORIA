# Optimization Governance

## Core Principle
**Correctness is non-negotiable.** No performance gain justifies a loss of determinism or correctness.

## Acceptance Criteria for New Kernels
Any new SIMD kernel (e.g., AVX-512 GEMM, Neon Conv2D) must meet the following criteria before merging:

1. **Equivalence**: Output must match the Reference Kernel within `1e-5` relative error for FP32.
2. **Deterministic**: Repeated runs must yield bitwise identical results.
3. **Explicit**: Must be gated behind `VECTORIA_USE_ASM` and `KernelPolicy::SIMD`.
4. **Observable**: Must log `KernelDispatch` events to the tracer.
5. **Tested**: Must utilize the `core/tests/test_gemm_simd.cpp` harness (or equivalent).

## Rejection Criteria
PRs will be rejected if they:
- Use heuristics ("If size > 100, use threads") without explicit configuration.
- Introduce library dependencies (e.g., BLAS, MKL) in the core path.
- Break the build on any supported architecture (x86_64, ARM64).
