# Kernel Philosophy and Certification

**Purpose:** Defines the hierarchy of kernels and the rigorous process for validating SIMD optimizations.

VECTORIA uses a tiered system to manage the trade-off between portability and performance. Correctness is derived exclusively from the **Reference** implementation.

## Kernel Tiers

### 1. Reference (The Truth)
*   **Implementation:** Pure C++ scalar code.
*   **Role:** Defines the exact expected output for every operation.
*   **Characteristics:** Portable, strictly serial, and bitwise deterministic across all IEEE 754 compliant platforms.

### 2. Experimental (Opt-In)
*   **Implementation:** Architecture-specific assembly or intrinsics (NEON, AVX2).
*   **Role:** Sandbox for performance work.
*   **Status:** Not enabled by default; may not pass full cross-platform CI.

### 3. Validated (Production)
*   **Implementation:** Optimized SIMD kernels (`asm/`).
*   **Requirements:**
    *   **Bitwise Identity:** Must match the Reference kernel within precision limits (often bit-exact) on native hardware.
    *   **CI Verification:** Must be verified by CI on the target architecture (e.g., Apple Silicon runners for NEON).
    *   **Traceability:** Usage must be explicitly logged in the trace.
*   **Status:** Enabled by default when `KernelPolicy::SIMD` is active.

## Dispatch Logic

The `Engine` attempts to use the most efficient kernel available. If a SIMD kernel fails or is unavailable for the current architecture, it **silently falls back** to the Reference kernel to ensure execution continuity.

## Current Support Matrix (v1.2.1-sigma)

| Op | Reference | ARM64 (NEON) | x86_64 (AVX2) |
|----|:---------:|:------------:|:-------------:|
| MatMul | ✅ | ✅ | ✅ |
| ReduceSum | ✅ | ✅ | ✅ |
| ReduceMax | ✅ | ✅ | ✅ |
| ReLU | ✅ | ✅ | ✅ |
| Add | ✅ | ✅ | ✅ |
| Mul | ✅ | ✅ | ✅ |
| Sub | ✅ | ✅ | ✅ |
| Div | ✅ | ✅ | ✅ |
| Exp | ✅ | ❌ | ❌ |
| Sqrt | ✅ | ❌ | ❌ |
| Log | ✅ | ❌ | ❌ |
| Transpose | ✅ | ❌ | ❌ |
| Reshape | ✅ | ❌ | ❌ |
| Concat | ✅ | ❌ | ❌ |
| Slice | ✅ | ❌ | ❌ |

## Composed Semantic Operations (Reference-Only)

Phase 7 introduced a suite of high-level operations built entirely from the reference primitives above. These operations prioritize numerical stability and semantic correctness over fused performance.

*   **LayerNorm**: Broadcast-aware normalization using `Mean` and `Variance` composition.
*   **LogSoftmax**: Numerically stable implementation using the log-sum-exp trick.
*   **StableSoftmax**: Defined as `Exp(LogSoftmax(x))` to prevent overflow.
*   **CrossEntropy**: Inference-only evaluation metric.
*   **Attention (Scaled Dot-Product)**: Semantic expansion for Transformer-style attention. **This is not a fused kernel;** it expands into explicit `MatMul` and `StableSoftmax` nodes.
*   **MultiHeadAttention**: High-level semantic composition for multi-head subspaces.
*   **TransformerEncoderBlock**: The highest level of semantic composition, integrating MHA and FFN blocks with residual connections.

*Note: `Exp`, `Sqrt`, `Log`, `Transpose`, `Reshape`, `Concat`, `Slice`, `Softmax` (composed), `LayerNorm` (composed), `LogSoftmax` (composed), `StableSoftmax` (composed), `CrossEntropy` (composed), `Attention` (composed), `MultiHeadAttention` (composed), and `TransformerEncoderBlock` (composed) currently rely on Reference implementations.*

## References

*   [docs/kernel_certification.md](../kernel_certification.md)
*   [asm/README.md](../../asm/README.md)
*   `core/src/kernels/`