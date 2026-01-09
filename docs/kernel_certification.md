# Kernel Certification Policy

VECTORIA employs a strict certification process for all computational kernels. This ensures that performance optimizations (SIMD) never compromise correctness, determinism, or cross-platform reproducibility.

## Certification Tiers

### 1. Reference (Gold Standard)
*   **Definition**: A pure C++ scalar implementation.
*   **Role**: The absolute ground truth for correctness.
*   **Requirements**:
    *   FP32 precision.
    *   Deterministic loop order.
    *   No external dependencies.
    *   Must handle all edge cases (broadcasting, unaligned access).
*   **Validation**: Checked into `core/src/kernels/*_ref.cpp` and covered by unit tests.

### 2. Experimental (Opt-In)
*   **Definition**: A candidate SIMD implementation (ASM or Intrinsics).
*   **Role**: A proving ground for performance optimizations.
*   **Requirements**:
    *   Must pass basic correctness tests locally.
    *   Hidden behind a compile-time flag or runtime feature gate.
*   **Validation**: Not yet required to pass strict CI on all architectures.

### 3. Validated (Production)
*   **Definition**: A SIMD implementation that is fully integrated and enabled by default (when architecture permits).
*   **Role**: High-performance execution path.
*   **Requirements**:
    *   **Bitwise Accuracy**: Must match the Reference kernel within `1e-5` (or strict bitwise if possible) on target hardware.
    *   **CI Coverage**: Must be tested on native hardware (e.g., Apple Silicon runner for ARM64, Linux runner for x86_64).
    *   **Traceability**: Must explicitly report its usage in the execution trace (e.g., "SIMD [AVX2]").
*   **Promotion**: A kernel promotes from Experimental to Validated ONLY after passing the full CI matrix.

## Promotion Checklist

To promote a kernel from Reference to Validated SIMD:
1.  [ ] Implement the ASM/Intrinsic kernel.
2.  [ ] Add a specific test case in `core/tests/test_gemm_simd.cpp` (or equivalent).
3.  [ ] Verify it against the Reference implementation with random data.
4.  [ ] Enable it in CI and ensure it passes on the target architecture.
5.  [ ] Ensure the `Engine` trace logs the specific kernel variant used.

## Current Kernel Status

| Operation | Reference | ARM64 (NEON) | x86_64 (AVX2) |
| :--- | :---: | :---: | :---: |
| **MatMul** | ✅ | ✅ | ✅ |
| **BiasAdd** | ✅ | ❌ | ❌ |
| **ReLU** | ✅ | ✅ | ✅ |
| **Add** | ✅ | ✅ | ✅ |
| **Mul** | ✅ | ✅ | ✅ |
| **Sub** | ✅ | ✅ | ✅ |
| **Div** | ✅ | ✅ | ✅ |
| **ReduceSum** | ✅ | ✅ | ✅ |
| **ReduceMax** | ✅ | ✅ | ✅ |
| **Exp** | ✅ | ❌ | ❌ |
