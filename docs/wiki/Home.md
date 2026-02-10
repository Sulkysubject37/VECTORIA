# Home

**Purpose:** This document defines the project's core mission, boundaries, and intended audience to prevent misuse.

## What is VECTORIA?

VECTORIA is a **deterministic, inspectable, and high-performance computational kernel framework**. It serves as the "engine room" for higher-level applications that require absolute control over numerical execution, memory layout, and auditability.

Unlike general-purpose machine learning frameworks, VECTORIA prioritizes **correctness over peak throughput** and **transparency over abstraction**.

### Core Tenets

*   **Strict Determinism:** Execution results are bitwise identical across repeated runs on the same hardware.
*   **Semantic Inference Stack:** Includes stable implementations of LayerNorm, Softmax, CrossEntropy, Attention, Multi-Head Attention, and the full Transformer Encoder Block, all validated against reference math.
*   **Semantic Truth:** The C++ Reference implementation defines the mathematical truth. SIMD kernels (ARM64 NEON, x86_64 AVX2) are validated against this truth.
*   **Explicit Control:** Memory is managed via a static arena. There is no garbage collection, no hidden threads, and no dynamic graph mutation during execution.
*   **Auditable:** Every decision—from kernel dispatch to memory allocation—is traceable.

## What VECTORIA is NOT

*   **NOT an ML Framework:** It has no auto-differentiation (autograd), no optimizers, and no training loops.
*   **NOT a Black Box:** It contains no heuristics that swap algorithms at runtime based on opaque criteria.
*   **NOT "Fastest at All Costs":** We will disable SIMD optimizations if they drift from the reference implementation's numerical results.

## Intended Audience

*   **Systems Engineers:** Building verified compute pipelines.
*   **Researchers:** Requiring reproducible numerical experiments.
*   **Developers:** Needing a bridge between research code and constrained deployment environments (CoreML).

## References

*   [TRUTH.md (Authoritative Manifesto)](../../TRUTH.md)
*   [README.md](../../README.md)
*   [Tooling & Introspection](Tooling-and-Introspection.md)
*   [docs/kernel_certification.md](../kernel_certification.md)