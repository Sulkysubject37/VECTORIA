# Release v1.2.1-sigma

**Semantic Inference Stack Complete**

This release marks the completion of Phase 7, introducing a full suite of numerically stable, composed operations for inference and evaluation. It consolidates the project's documentation and formalizes the mathematical definitions of the new high-level operators.

## 🌟 New Features (Semantic Stack)

These operations are implemented as **Composed Graphs** using the validated Reference kernels. They prioritize numerical correctness, determinism, and traceability over fused performance.

*   **Layer Normalization (`LayerNorm`)**:
    *   Broadcast-aware normalization over the last axis.
    *   Supports learnable $\gamma$ and $\beta$.
*   **LogSoftmax**:
    *   Numerically stable implementation using the $\log(\sum \exp(x - \max(x)))$ trick.
    *   Prevents underflow/overflow for extreme input values.
*   **Stable Softmax**:
    *   Defined as $\exp(\text{LogSoftmax}(x))$.
    *   Recommended over the naïve Softmax implementation for all production use cases.
*   **CrossEntropy (Inference-Only)**:
    *   Computes $-\sum (t \cdot \log(\text{softmax}(x)))$.
    *   Strictly for evaluation (no gradients).

## 🛠 Core Improvements

*   **New Primitives**: Added `Sqrt`, `Log` to the IR and Reference backend.
*   **Broadcasting**: Enhanced `Mul` and `Add` kernels to support scalar and row-vector broadcasting needed for the semantic stack.
*   **Documentation**:
    *   Added **TRUTH.md**: The authoritative manifesto on numerical determinism.
    *   Standardized mathematical notation across all specs.

## ⚠️ Notes

*   **No SIMD Changes**: This release does not introduce new SIMD kernels. The new operations run on the Reference backend (scalar C++).
*   **No Training**: VECTORIA remains an inference-forward engine.
*   **Determinism**: All new operations fully respect the project's strict determinism guarantees.
